from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn

from experts.kad import (
    KAD512_LABELS,
    KAD512_PROMPTS,
    KAD512_EMBED_DIM,
    KADCheckpointError,
    KADQueryDecoder,
    KADResNet512Encoder,
    KADTextEncoder,
    build_kad512_query_pack,
    kad512_query_pack_semantic_sha256,
    load_kad512_checkpoint,
    preflight_kad512_checkpoint,
    preflight_kad512_query_pack,
    preprocess_kad512,
)
from scripts.export_kad_query_pack import (
    PHASE1_FEATURES_SHA256,
    PHASE1_LABELS,
    PHASE1_PROMPTS,
    PHASE1_QUERY_SPECS,
    load_phase1_query_features,
    main as export_query_pack_main,
)


class _SyntheticEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(31, KAD512_EMBED_DIM)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class _SyntheticBert(nn.Module):
    """Tiny computation with BERT-like names; no transformers/network dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _SyntheticEmbeddings()
        self.pooler = nn.Linear(KAD512_EMBED_DIM, KAD512_EMBED_DIM)

    def forward(self, *, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return SimpleNamespace(pooler_output=torch.tanh(self.pooler(pooled)))


class KADPreprocessTests(unittest.TestCase):
    def test_preprocess_is_512_rgb_and_imagenet_normalized(self) -> None:
        image = torch.linspace(0, 1, 13 * 17).reshape(1, 13, 17)

        output = preprocess_kad512(image)

        self.assertEqual(tuple(output.shape), (3, 512, 512))
        self.assertEqual(output.dtype, torch.float32)
        # A repeated grayscale input remains identical across channels before each
        # channel's distinct ImageNet normalization.
        recovered = [
            output[channel] * std + mean
            for channel, (mean, std) in enumerate(
                zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            )
        ]
        torch.testing.assert_close(recovered[0], recovered[1])
        torch.testing.assert_close(recovered[1], recovered[2])

    def test_preprocess_rejects_ambiguous_channel_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects 1 or 3 channels"):
            preprocess_kad512(torch.zeros(2, 10, 10))

    def test_preprocess_matches_official_pil_bicubic_path(self) -> None:
        source = np.arange(320 * 320, dtype=np.uint8).reshape(320, 320)
        image = Image.fromarray(source, mode="L").convert("RGB")
        resampling = getattr(Image, "Resampling", Image)
        resized = image.resize((512, 512), resample=resampling.BICUBIC)
        expected = (
            torch.from_numpy(np.asarray(resized, dtype=np.float32).copy())
            .permute(2, 0, 1)
            / 255.0
        )
        expected = (
            expected
            - torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        ) / torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

        actual = preprocess_kad512(image)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class KADComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # These are the released architecture widths, but every tensor is randomly
        # initialized locally.  No model or checkpoint is downloaded.
        cls.image_encoder = KADResNet512Encoder().eval()
        cls.text_encoder = KADTextEncoder(_SyntheticBert()).eval()
        cls.query_decoder = KADQueryDecoder().eval()
        cls.checkpoint = {
            "image_encoder": cls.image_encoder.state_dict(),
            "text_encoder": cls.text_encoder.state_dict(),
            "model": cls.query_decoder.state_dict(),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.checkpoint
        del cls.image_encoder
        del cls.text_encoder
        del cls.query_decoder

    def test_component_forward_shapes(self) -> None:
        with torch.inference_mode():
            patches, pooled = self.image_encoder(torch.randn(1, 3, 64, 64))
            text = self.text_encoder.encode_text(
                {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                }
            )
            logits, attention = self.query_decoder(
                patches,
                text,
                return_attention=True,
            )

        self.assertEqual(tuple(patches.shape), (1, 4, KAD512_EMBED_DIM))
        self.assertEqual(tuple(pooled.shape), (1, KAD512_EMBED_DIM))
        self.assertEqual(tuple(text.shape), (2, KAD512_EMBED_DIM))
        self.assertEqual(tuple(logits.shape), (1, 2, 2))
        self.assertEqual(tuple(attention.shape), (1, 2, 4))

    def test_query_context_changes_score_and_single_query_shape_is_isolated(self) -> None:
        image_features = torch.linspace(
            -0.5,
            0.5,
            steps=4 * KAD512_EMBED_DIM,
        ).reshape(1, 4, KAD512_EMBED_DIM)
        first_query = torch.linspace(
            -1.0,
            1.0,
            steps=KAD512_EMBED_DIM,
        ).unsqueeze(0)
        second_query = torch.flip(first_query, dims=(1,))

        with torch.inference_mode():
            isolated = self.query_decoder(image_features, first_query)
            joint = self.query_decoder(
                image_features,
                torch.cat((first_query, second_query), dim=0),
            )

        self.assertEqual(tuple(isolated.shape), (1, 1, 2))
        self.assertEqual(tuple(joint.shape), (1, 2, 2))
        self.assertGreater(
            float((isolated[:, 0] - joint[:, 0]).abs().max()),
            1e-7,
            "KAD query self-attention unexpectedly left the first score unchanged",
        )

    def test_state_dict_uses_official_component_names(self) -> None:
        image_keys = self.checkpoint["image_encoder"]
        text_keys = self.checkpoint["text_encoder"]
        model_keys = self.checkpoint["model"]

        self.assertIn("resnet.conv1.weight", image_keys)
        self.assertIn("res_features.7.2.conv3.weight", image_keys)
        self.assertIn("res_l1.weight", image_keys)
        self.assertIn("res_l2.weight", image_keys)
        self.assertIn("bert_model.embeddings.word_embeddings.weight", text_keys)
        self.assertIn("mlp_embed.2.weight", text_keys)
        self.assertIn("decoder.layers.3.multihead_attn.in_proj_weight", model_keys)
        self.assertIn("decoder_norm.weight", model_keys)
        self.assertIn("decoder.norm.weight", model_keys)
        self.assertIn("mlp_head.0.weight", model_keys)

    def test_synthetic_official_checkpoint_preflights_and_loads(self) -> None:
        sections = preflight_kad512_checkpoint(self.checkpoint)
        self.assertEqual(set(sections), {"image_encoder", "text_encoder", "model"})

        # Loading into the matching modules is the strongest offline compatibility
        # check: key sets and every tensor shape must agree.
        load_kad512_checkpoint(
            self.checkpoint,
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
            query_decoder=self.query_decoder,
        )

    def test_preflight_identifies_wrong_resnet_variant(self) -> None:
        checkpoint = dict(self.checkpoint)
        image_state = dict(checkpoint["image_encoder"])
        image_state["res_l1.weight"] = torch.empty(1024, 1024)
        checkpoint["image_encoder"] = image_state

        with self.assertRaisesRegex(KADCheckpointError, "not KAD-512"):
            preflight_kad512_checkpoint(checkpoint)

    def test_query_pack_drops_text_encoder_and_keeps_provenance(self) -> None:
        prompts = ("atelectasis", "pleural effusion")
        labels = ("Atelectasis", "Effusion")
        pack = build_kad512_query_pack(
            self.checkpoint,
            text_features=torch.randn(2, KAD512_EMBED_DIM),
            labels=labels,
            prompts=prompts,
            source_metadata={"checkpoint_sha256": "synthetic"},
        )

        self.assertNotIn("text_encoder", pack)
        checked = preflight_kad512_query_pack(pack)
        self.assertEqual(checked["labels"], list(labels))
        self.assertEqual(checked["prompts"], list(prompts))
        self.assertEqual(checked["source"]["checkpoint_sha256"], "synthetic")
        self.assertEqual(tuple(checked["text_features"].shape), (2, KAD512_EMBED_DIM))

    def test_query_pack_rejects_feature_prompt_mismatch(self) -> None:
        with self.assertRaisesRegex(KADCheckpointError, "query features"):
            build_kad512_query_pack(
                self.checkpoint,
                text_features=torch.randn(1, KAD512_EMBED_DIM),
                labels=("A", "B"),
                prompts=("a", "b"),
            )

    def test_query_pack_canonicalizes_last_bit_text_encoder_differences(self) -> None:
        base_features = torch.full((1, KAD512_EMBED_DIM), 0.123456)
        first = build_kad512_query_pack(
            self.checkpoint,
            text_features=base_features,
            labels=("A",),
            prompts=("a",),
        )
        second = build_kad512_query_pack(
            self.checkpoint,
            text_features=base_features + 1e-7,
            labels=("A",),
            prompts=("a",),
        )

        self.assertEqual(first["format_version"], 3)
        self.assertEqual(
            first["text_feature_canonicalization"],
            "torch_bfloat16_roundtrip_then_float32_v1",
        )
        self.assertTrue(torch.equal(first["text_features"], second["text_features"]))
        self.assertEqual(
            kad512_query_pack_semantic_sha256(first),
            kad512_query_pack_semantic_sha256(second),
        )

    def test_query_pack_semantic_hash_ignores_only_local_checkpoint_path(self) -> None:
        base = build_kad512_query_pack(
            self.checkpoint,
            text_features=torch.randn(1, KAD512_EMBED_DIM),
            labels=("A",),
            prompts=("a",),
            source_metadata={
                "checkpoint": "/tmp/first.pt",
                "checkpoint_sha256": "synthetic",
            },
        )
        moved = dict(base)
        moved["source"] = {
            **base["source"],
            "checkpoint": "/content/same-weights.pt",
        }
        changed = dict(base)
        changed["text_features"] = base["text_features"].clone()
        changed["text_features"][0, 0] += 1.0

        self.assertEqual(
            kad512_query_pack_semantic_sha256(base),
            kad512_query_pack_semantic_sha256(moved),
        )
        self.assertNotEqual(
            kad512_query_pack_semantic_sha256(base),
            kad512_query_pack_semantic_sha256(changed),
        )


class KADQueryPackExporterTests(unittest.TestCase):
    def test_canonical_phase1_features_are_checksum_pinned_and_aligned(self) -> None:
        features = load_phase1_query_features()

        self.assertEqual(tuple(features), PHASE1_LABELS)
        self.assertEqual(
            PHASE1_FEATURES_SHA256,
            "54c74d20a5bcecab770ce6a0d84bc0b14caa40e798693d1af6bb0b022a4cf094",
        )
        for feature in features.values():
            self.assertEqual(tuple(feature.shape), (1, KAD512_EMBED_DIM))
            self.assertTrue(
                torch.equal(
                    feature,
                    feature.to(torch.bfloat16).to(torch.float32),
                )
            )

    def test_phase1_specs_are_aligned_and_have_stable_endpoint_ids(self) -> None:
        self.assertEqual(tuple(PHASE1_QUERY_SPECS), PHASE1_LABELS)
        self.assertEqual(
            tuple(spec["prompt"] for spec in PHASE1_QUERY_SPECS.values()),
            PHASE1_PROMPTS,
        )
        self.assertEqual(
            {
                label: spec["query_set"]
                for label, spec in PHASE1_QUERY_SPECS.items()
            },
            {
                "Pneumothorax": "doctor_assistant.phase1.pneumothorax.v1",
                "Nodule_or_mass": "doctor_assistant.phase1.nodule_or_mass.v1",
                "Airspace_opacity": "doctor_assistant.phase1.airspace_opacity.v1",
            },
        )

    def test_phase1_requires_an_explicit_active_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    export_query_pack_main(
                        [
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(Path(directory) / "pack.pt"),
                            "--query-set",
                            "phase1",
                        ]
                    )

    def test_nih14_rejects_an_active_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    export_query_pack_main(
                        [
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(Path(directory) / "pack.pt"),
                            "--query-set",
                            "nih14",
                            "--active-target",
                            "Pneumothorax",
                        ]
                    )

    @patch(
        "scripts.export_kad_query_pack.sha256_file",
        return_value="a" * 64,
    )
    @patch("scripts.export_kad_query_pack.export_kad512_query_pack")
    def test_phase1_exports_exactly_one_endpoint_query(
        self,
        export_pack,
        _hash_file,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            for target, spec in PHASE1_QUERY_SPECS.items():
                output = Path(directory) / f"{target}.pt"
                export_pack.reset_mock()
                export_pack.return_value = output

                with redirect_stdout(io.StringIO()):
                    result = export_query_pack_main(
                        [
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(output),
                            "--query-set",
                            "phase1",
                            "--active-target",
                            target,
                        ]
                    )

                self.assertEqual(result, 0)
                export_pack.assert_called_once()
                call = export_pack.call_args
                self.assertEqual(call.kwargs["labels"], (target,))
                self.assertEqual(call.kwargs["prompts"], (spec["prompt"],))
                self.assertTrue(
                    torch.equal(
                        call.kwargs["text_features"],
                        load_phase1_query_features()[target],
                    )
                )
                self.assertEqual(
                    call.kwargs["source_metadata"]["query_set"],
                    spec["query_set"],
                )

    @patch(
        "scripts.export_kad_query_pack.sha256_file",
        return_value="a" * 64,
    )
    @patch("scripts.export_kad_query_pack.export_kad512_query_pack")
    def test_nih14_retains_the_joint_smoke_query_set(
        self,
        export_pack,
        _hash_file,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            output = Path(directory) / "nih14.pt"
            export_pack.return_value = output

            with redirect_stdout(io.StringIO()):
                result = export_query_pack_main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(output),
                        "--query-set",
                        "nih14",
                    ]
                )

            self.assertEqual(result, 0)
            call = export_pack.call_args
            self.assertEqual(call.kwargs["labels"], KAD512_LABELS)
            self.assertEqual(call.kwargs["prompts"], KAD512_PROMPTS)
            self.assertIsNone(call.kwargs["text_features"])
            self.assertEqual(
                call.kwargs["source_metadata"]["query_set"],
                "doctor_assistant.nih14_smoke.v1",
            )


if __name__ == "__main__":
    unittest.main()
