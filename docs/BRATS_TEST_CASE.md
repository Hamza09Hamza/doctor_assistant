# Real DICOM brain-tumour MRI test case for BraTS wiring

`BraTSExpert` (`experts/mri_brats.py`, MONAI `brats_mri_segmentation`) needs four
co-registered MRI sequences -- T1, T1c, T2, FLAIR -- as four separate DICOM series.
`api/persistence.py::_match_brats_sequences` groups a study's `Series` rows into those
four sequences by reading each series' real DICOM `SeriesDescription` tag and matching
it through `experts/mri_brats.py::canonical_sequence_name`. That wiring had only been
tested against a fake `Series` double. This case is a real, DICOM-native, publicly
licensed brain-tumour MRI study staged locally so the real matching code can be
exercised end to end.

## Why not the Medical Segmentation Decathlon

`mri_brats.py`'s own validation numbers come from MSD Task01_BrainTumour, but that
dataset ships one combined 4-channel NIfTI file per case, not four separate DICOM
series with distinguishable `SeriesDescription` tags -- unusable for testing
`_match_brats_sequences`, which only exists because real studies arrive as separate
series.

## Collections surveyed on NCI Imaging Data Commons (IDC)

Queried via `idc_index.IDCClient().sql_query(...)` against the `index` table (the same
tool/pattern `scripts/run_lidc_lung_nodule_colab.py` used for the LIDC-IDRI CT case):

| Collection | DICOM `Modality='MR'` in IDC? | Outcome |
|---|---|---|
| `tcga_gbm` | No (only `SM`/`SEG`/`ANN`/`OT`) | Ruled out -- no MR at all in IDC's index |
| `tcga_lgg` | No (only `SM`) | Ruled out -- no MR at all in IDC's index |
| `cptac_gbm` | No (only `SM`) | Ruled out |
| `icdc_glioma` | Yes, 650 series | Real but messy site-specific naming (`BRAIN/T1_TRANS+C`, `AX FSE T2`, `T1/D/SE +C`, ...); not used |
| `upenn_gbm` | Yes, 3680 series across 630 patients | **Used** |

A direct sweep of every `Modality='MR'` row in the whole IDC index for series whose
`SeriesDescription`, once whitespace/underscore/hyphen-normalized, exactly equals one
of `canonical_sequence_name`'s bare vocabulary words (`t1`, `t1c`, `t2`, `flair`, ...)
returned **zero** brain-MRI hits anywhere in IDC -- only two unrelated breast-MRI
series. This is the honest finding: no public DICOM-native brain-tumour MRI collection
in IDC uses the bare canonical strings `_MODALITY_ALIASES` shipped with. Real PACS/
post-processing pipelines stamp protocol-specific strings instead (see below), which is
exactly why the acceptance test in this doc runs the matcher twice -- once against the
strings as shipped, once after a small, documented, collection-specific alias addition.

## The chosen case: UPENN-GBM-00020

- **Collection**: [`UPENN-GBM`](https://www.cancerimagingarchive.net/collection/upenn-gbm/)
  on IDC ([portal listing](https://portal.imaging.datacommons.cancer.gov/explore/filters/?collection_id=upenn_gbm)),
  pre-operative multi-parametric MRI of glioblastoma patients (Bakas et al.).
- **License**: `CC BY 4.0`.
- **Patient/case**: `UPENN-GBM-00020`, four CaPTk-post-processed, co-registered series:

  | Sequence | Real `SeriesDescription` (verified with pydicom) | Instances | Series size |
  |---|---|---:|---:|
  | T1 | `t1 axial: Processed_CaPTk` | 192 | 19.4 MB |
  | T1c (post-contrast) | `t1 axial stealth-post : Processed_CaPTk` | 192 | 19.4 MB |
  | T2 | `Axial T2 tse: Processed_CaPTk` | 64 | 7.0 MB |
  | FLAIR | `t2_Flair_axial: Processed_CaPTk` | 60 | 6.1 MB |

  Total download: ~52 MB (chosen over the perfusion/DTI series in the same patient,
  which run to 900 instances each and aren't part of the BraTS 4-sequence protocol).

- **Quality check**: the four sequences share a common resampled z-grid (CaPTk's
  co-registration step), so picking the physically nearest slice to FLAIR's middle
  slice in each of the other three sequences lines all four panels up on the same
  anatomy. The rendered preview shows a clear rim-enhancing left-hemisphere mass on
  T1c with matching T2/FLAIR hyperintensity -- sharp, correctly labeled, and
  anatomically consistent:

  ```text
  data/validation/upenn_gbm_00020/preview_t1_t1c_t2_flair.png
  ```

- **A real quirk, documented rather than hidden**: each of the four series carries its
  *own* distinct DICOM `StudyInstanceUID` -- IDC's `upenn_gbm` packaging is per-series
  CaPTk output, not one shared PACS study the way `LIDC-IDRI-0686` was. This doesn't
  block staging: `_match_brats_sequences` groups series by the *local* `Series.study_id`
  foreign key, never by raw DICOM StudyInstanceUID, so one local `Study` row legitimately
  owns all four. `Study.dicom_study_uid` is left `None` in the staged database rather
  than picking one of the four UIDs and misrepresenting it as the whole case's UID.

## Acceptance test result

Running `api.persistence._match_brats_sequences` / `_build_brats_scan` directly
(imported, no server) against the staged study:

- **As shipped** (`_MODALITY_ALIASES` before this task): **0/4 sequences matched.**
  All four real `SeriesDescription` strings raised `ValueError` in
  `canonical_sequence_name` and were logged/skipped, exactly as designed for an
  unrecognized series -- `_match_brats_sequences` correctly returned `None`.
- **After the fix**: `experts/mri_brats.py::_MODALITY_ALIASES` gained four literal,
  documented entries for this case's real normalized strings (`t1axial:processedcaptk`
  -> `t1`, `t1axialstealthpost:processedcaptk` -> `t1c`,
  `axialt2tse:processedcaptk` -> `t2`, `t2flairaxial:processedcaptk` -> `flair`). All
  four series matched, `_match_brats_sequences` returned all four paths, and
  `_build_brats_scan` produced a `Scan` with `meta.modality == Modality.MRI`,
  `meta.body_part == BodyPart.BRAIN`, and `meta.extra["sequence_paths"]` populated with
  the four staged series directories.

This addition is deliberately narrow -- the literal, collection-specific strings
(including the `Processed_CaPTk` suffix), not a generalized fuzzy parser. The same
reasoning that keeps `canonical_sequence_name` exact-match-only (documented in its own
docstring: `t1gd` starts with `t1` but is a different channel) applies here: guessing at
substrings/tokens across arbitrary real PACS naming risks the identical silent
misclassification bug, just with different strings. See the comment block above the new
entries in `experts/mri_brats.py` for the full reasoning.

All 22 existing `tests/test_mri_brats.py` + `tests/test_persistence_brats.py` unit
tests (which use a fake `Series` double, unaffected by the new literal aliases) still
pass.

## Reproduce it

From the repository root, in the MLX virtualenv (has `idc-index`, `pydicom`,
`sqlalchemy`; any venv with those three works):

```bash
source .venv-mlx/bin/activate
python scripts/prepare_brats_test_case.py
```

The script is idempotent: it re-verifies (by real instance count and real
`SeriesDescription` read via pydicom, not just IDC's index metadata) or re-downloads
each of the four pinned series, stages them under
`api_storage/brats_test_case/studies/<study-id>/series/<series-uid>/`, creates/updates
one `Study` row and four `Series` rows in a local SQLite database, and renders the
anatomically-aligned preview PNG. Raw DICOM, the database, and staged storage are all
gitignored (`data/`, `api_storage/`).

```text
Manifest:  data/validation/upenn_gbm_00020/brats_test_case_manifest.json
Preview:   data/validation/upenn_gbm_00020/preview_t1_t1c_t2_flair.png
Database:  api_storage/brats_test_case.sqlite
```

Then run the real acceptance test -- imports `api.persistence` directly, no server
needed:

```bash
source .venv-mlx/bin/activate
python3 - <<'EOF'
import logging
logging.basicConfig(level=logging.INFO)

from api.db import build_engine, build_session_factory
from api.persistence import _match_brats_sequences, _build_brats_scan
from api.models import Study

engine = build_engine("sqlite:///api_storage/brats_test_case.sqlite")
session_factory = build_session_factory(engine)

with session_factory() as db:
    study = db.query(Study).first()
    sequence_paths = _match_brats_sequences(db, study.id)
    print("Matched:", sequence_paths)
    assert sequence_paths is not None and len(sequence_paths) == 4
    scan = _build_brats_scan(study, sequence_paths)
    print("Scan modality/body_part:", scan.meta.modality, scan.meta.body_part)
    print("sequence_paths:", scan.meta.extra["sequence_paths"])
EOF
```

Expected output: `Matched:` a dict with all four of `t1`/`t1c`/`t2`/`flair`, followed by
`Modality.MRI BodyPart.BRAIN` and the four staged series directories.

To exercise this through the real API instead of calling `api.persistence` directly,
point the app at the staged database/storage and submit a study-level analysis (no
`series_id`) for `api_study_id` from the manifest -- that's the code path
`_match_brats_sequences` was written for
(`POST /v1/studies/{id}/analyses` with no series scoping):

```bash
export DATABASE_URL="sqlite:///$(pwd)/api_storage/brats_test_case.sqlite"
export STORAGE_DIR="$(pwd)/api_storage/brats_test_case"
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000
```

Running the actual `brats_mri_segmentation` bundle end to end additionally requires
downloading the MONAI bundle (`BraTSExpert` does this on first `predict()`) and enough
memory/compute to run `sliding_window_inference` on the four volumes -- out of scope for
this task, which is about verifying the *matching and scan-building* wiring against real
data, not benchmarking segmentation accuracy on this particular patient.

## Sources and licenses

- [UPENN-GBM on TCIA](https://www.cancerimagingarchive.net/collection/upenn-gbm/) (`CC BY 4.0`)
- [UPENN-GBM on IDC portal](https://portal.imaging.datacommons.cancer.gov/explore/filters/?collection_id=upenn_gbm)
- [IDC download documentation](https://learn.canceridc.dev/data/downloading-data)
