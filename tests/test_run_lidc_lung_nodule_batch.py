"""Regression test for scripts/run_lidc_lung_nodule_batch.py's candidate-filtering logic.

filter_clean_candidates() is deliberately split out from discover_clean_candidates() so
this can be tested against a small synthetic index -- no live idc-index/network call,
and no dependency on LIDC-IDRI/LUNA16's actual current contents. The real end-to-end
count (27 candidates as of 2026-08-05) was independently verified once against the live
idc-index + LUNA16 candidates.csv before this script was trusted; that number can drift
as IDC's index is revised, so it is not asserted here.
"""

from __future__ import annotations

import unittest

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

if PANDAS_AVAILABLE:
    from scripts.run_lidc_lung_nodule_batch import filter_clean_candidates


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas not installed in this environment")
class FilterCleanCandidatesTests(unittest.TestCase):
    def _index(self, rows):
        """rows: (PatientID, SeriesInstanceUID, Modality, instanceCount)."""
        return pd.DataFrame(
            rows, columns=["PatientID", "SeriesInstanceUID", "Modality", "instanceCount"]
        )

    def test_accepts_case_matching_verified_shape(self):
        """One CT series + exactly 4 SEG series, CT not in LUNA16 -- must be kept, and
        the CT's own instanceCount carried through (used later to detect a partial/
        interrupted download instead of just checking "folder is non-empty")."""
        rows = [
            ("P1", "CT-1", "CT", 113),
            ("P1", "SEG-1", "SEG", 1),
            ("P1", "SEG-2", "SEG", 1),
            ("P1", "SEG-3", "SEG", 1),
            ("P1", "SEG-4", "SEG", 1),
        ]
        candidates = filter_clean_candidates(self._index(rows), luna_series=set())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["patient_id"], "P1")
        self.assertEqual(candidates[0]["ct_series_uid"], "CT-1")
        self.assertEqual(candidates[0]["ct_instance_count"], 113)
        self.assertEqual(set(candidates[0]["seg_series_uids"]), {"SEG-1", "SEG-2", "SEG-3", "SEG-4"})

    def test_rejects_case_in_luna16(self):
        rows = [
            ("P1", "CT-1", "CT", 113),
            ("P1", "SEG-1", "SEG", 1),
            ("P1", "SEG-2", "SEG", 1),
            ("P1", "SEG-3", "SEG", 1),
            ("P1", "SEG-4", "SEG", 1),
        ]
        candidates = filter_clean_candidates(self._index(rows), luna_series={"CT-1"})
        self.assertEqual(candidates, [])

    def test_rejects_wrong_seg_count(self):
        """Fewer or more than 4 SEG series -- e.g. per-nodule rather than per-patient
        SEGs -- is a different case shape, out of scope for this batch, not an error."""
        too_few = self._index(
            [("P1", "CT-1", "CT", 113), ("P1", "SEG-1", "SEG", 1), ("P1", "SEG-2", "SEG", 1)]
        )
        self.assertEqual(filter_clean_candidates(too_few, luna_series=set()), [])

        too_many = self._index(
            [("P1", "CT-1", "CT", 113)]
            + [("P1", f"SEG-{i}", "SEG", 1) for i in range(6)]
        )
        self.assertEqual(filter_clean_candidates(too_many, luna_series=set()), [])

    def test_rejects_multiple_ct_series(self):
        """A patient with more than one CT series doesn't match the single-series
        assumption the rest of the pipeline (_ct_dicom_to_nifti etc.) makes."""
        rows = [
            ("P1", "CT-1", "CT", 113),
            ("P1", "CT-2", "CT", 90),
            ("P1", "SEG-1", "SEG", 1),
            ("P1", "SEG-2", "SEG", 1),
            ("P1", "SEG-3", "SEG", 1),
            ("P1", "SEG-4", "SEG", 1),
        ]
        candidates = filter_clean_candidates(self._index(rows), luna_series=set())
        self.assertEqual(candidates, [])

    def test_multiple_patients_independent(self):
        rows = [
            ("P1", "CT-1", "CT", 113),
            ("P1", "SEG-1", "SEG", 1),
            ("P1", "SEG-2", "SEG", 1),
            ("P1", "SEG-3", "SEG", 1),
            ("P1", "SEG-4", "SEG", 1),
            ("P2", "CT-2", "CT", 90),
            ("P2", "SEG-5", "SEG", 1),
            ("P2", "SEG-6", "SEG", 1),
            ("P2", "SEG-7", "SEG", 1),
            ("P2", "SEG-8", "SEG", 1),
        ]
        candidates = filter_clean_candidates(self._index(rows), luna_series={"CT-2"})
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["patient_id"], "P1")


if __name__ == "__main__":
    unittest.main()
