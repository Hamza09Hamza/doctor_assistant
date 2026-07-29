from __future__ import annotations

import csv
import os
import random
import tempfile
import unittest

import torch
from torch import nn

from data.chest_xray14 import _stratified_val_split, load_chest_xray14
from training.trainer import TrainConfig, Trainer


class DatasetManifestTests(unittest.TestCase):
    def test_missing_official_split_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            images = os.path.join(root, "images")
            os.makedirs(images)
            rows = [
                ("00000001_000.png", "Effusion"),
                ("00000002_000.png", "No Finding"),
            ]
            with open(
                os.path.join(root, "Data_Entry_2017.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Image Index", "Finding Labels"]
                )
                writer.writeheader()
                for name, labels in rows:
                    writer.writerow(
                        {"Image Index": name, "Finding Labels": labels}
                    )
                    open(os.path.join(images, name), "wb").close()

            with self.assertRaisesRegex(FileNotFoundError, "Official NIH split manifest"):
                load_chest_xray14(root, split="train", val_fraction=0.5)

    def test_loader_preserves_official_filename_and_patient_identity(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            images = os.path.join(root, "images")
            os.makedirs(images)
            rows = [
                ("00000001_000.png", "Effusion"),
                ("00000002_000.png", "No Finding"),
                ("00000003_000.png", "Mass"),
            ]
            with open(
                os.path.join(root, "Data_Entry_2017.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Image Index", "Finding Labels"]
                )
                writer.writeheader()
                for name, labels in rows:
                    writer.writerow(
                        {"Image Index": name, "Finding Labels": labels}
                    )
                    open(os.path.join(images, name), "wb").close()
            with open(
                os.path.join(root, "train_val_list.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write("00000001_000.png\n00000002_000.png\n")
            with open(
                os.path.join(root, "test_list.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write("00000003_000.png\n")

            test = load_chest_xray14(root, split="test", val_fraction=0.5)

            self.assertEqual(len(test), 1)
            self.assertEqual(test[0].meta["sample_id"], "00000003_000.png")
            self.assertEqual(test[0].meta["filename"], "00000003_000.png")
            self.assertEqual(test[0].meta["patient_id"], "00000003")
            self.assertEqual(test[0].meta["official_partition"], "test")
            self.assertEqual(test[0].meta["split_provenance"], "official_test_manifest")

    def test_stratification_never_splits_one_patient_across_folds(self) -> None:
        names = {
            "00000001_000.png",
            "00000001_001.png",
            "00000002_000.png",
            "00000002_001.png",
            "00000003_000.png",
            "00000003_001.png",
        }
        vectors = {
            name: [1.0, 0.0] if name.startswith("00000001") else [0.0, 1.0]
            for name in names
        }
        validation = _stratified_val_split(
            names, vectors, n_labels=2, val_fraction=0.34, seed=42
        )

        for patient in ("00000001", "00000002", "00000003"):
            patient_names = {name for name in names if name.startswith(patient)}
            self.assertIn(
                patient_names & validation,
                (set(), patient_names),
                f"patient {patient} leaked across train and validation",
            )

    def test_max_samples_is_seeded_sample_not_csv_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            images = os.path.join(root, "images")
            os.makedirs(images)
            train_name = "00000001_000.png"
            test_names = [f"{patient:08d}_000.png" for patient in range(2, 8)]
            rows = [(train_name, "No Finding")] + [
                (name, "Pneumothorax" if index % 2 else "No Finding")
                for index, name in enumerate(test_names)
            ]
            with open(
                os.path.join(root, "Data_Entry_2017.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Image Index", "Finding Labels"]
                )
                writer.writeheader()
                for name, labels in rows:
                    writer.writerow({"Image Index": name, "Finding Labels": labels})
                    open(os.path.join(images, name), "wb").close()
            with open(
                os.path.join(root, "train_val_list.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write(f"{train_name}\n")
            with open(
                os.path.join(root, "test_list.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write("\n".join(test_names) + "\n")

            sampled = load_chest_xray14(
                root,
                split="test",
                max_samples=2,
                seed=42,
            )
            sampled_names = [sample.meta["filename"] for sample in sampled]

            self.assertEqual(sampled_names, random.Random(42).sample(test_names, k=2))
            self.assertNotEqual(sampled_names, test_names[:2])

    def test_max_samples_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            images = os.path.join(root, "images")
            os.makedirs(images)
            with open(
                os.path.join(root, "Data_Entry_2017.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Image Index", "Finding Labels"]
                )
                writer.writeheader()
            with self.assertRaisesRegex(ValueError, "max_samples"):
                load_chest_xray14(root, max_samples=0)


class FakeEvaluator:
    class_names = ["negative", "positive"]

    def reset(self) -> None:
        pass


class RecordingTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(1, 1),
            FakeEvaluator(),
            TrainConfig(epochs=1, mixed_precision=False, resume=False),
            device="cpu",
        )
        self.saved: list[tuple[str, float]] = []

    def _train_one_epoch(self, loader) -> float:
        return 0.1

    def _validate(self, loader) -> dict:
        return {"auc": 0.8, "val_loss": 0.2}

    def _save(self, name: str, epoch: int) -> None:
        self.saved.append((name, self.best_metric))


class TrainerCheckpointTests(unittest.TestCase):
    def test_last_checkpoint_records_updated_best_metric(self) -> None:
        trainer = RecordingTrainer()
        trainer.fit(None, None)
        self.assertEqual(
            trainer.saved,
            [("best.pt", 0.8), ("last.pt", 0.8)],
        )


if __name__ == "__main__":
    unittest.main()
