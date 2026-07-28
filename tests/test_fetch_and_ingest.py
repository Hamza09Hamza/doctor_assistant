from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from ingest.loaders import _spacing_from_affine


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_samples.py"
_SPEC = importlib.util.spec_from_file_location("fetch_samples_for_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_FETCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FETCH)


class FetchSampleTests(unittest.TestCase):
    def test_per_class_selects_distinct_examples(self) -> None:
        stream = []
        for pathology in _FETCH.TARGET_PATHOLOGIES:
            for index in range(2):
                stream.append(
                    {
                        "label": [pathology],
                        "image": {"path": f"{pathology}_{index}.png"},
                    }
                )
        stream.extend(
            {"label": ["No Finding"], "image": {"path": f"normal_{i}.png"}}
            for i in range(2)
        )

        selected = _FETCH._collect(
            stream, per_class=2, normals=2, max_scan=len(stream)
        )

        self.assertEqual(len(selected), len(_FETCH.TARGET_PATHOLOGIES) * 2 + 2)
        self.assertEqual(len({id(example) for example, _ in selected}), len(selected))


class AffineSpacingTests(unittest.TestCase):
    def test_rotated_affine_uses_basis_norms_not_diagonal(self) -> None:
        # 90-degree in-plane rotation with 2, 3, and 4 mm voxel sizes. The first two
        # diagonal entries are zero, so the former diagonal-only calculation failed.
        affine = torch.tensor(
            [
                [0.0, -3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.assertEqual(_spacing_from_affine(affine, 3), (2.0, 3.0, 4.0))


if __name__ == "__main__":
    unittest.main()
