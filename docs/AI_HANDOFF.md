# AI handoff — full context for whoever picks this up next

Written for another AI agent (or a human) with zero prior context on this project.
Read this before touching anything. It's long on purpose — the goal is that you never
have to guess at *why* something is built the way it is, because guessing wrong here
tends to either quietly break a safety property or waste real time re-deriving
something already settled. Where I'm not sure of something, I say so explicitly rather
than presenting a guess as fact.

## 0. The one-paragraph version

`doctor_assistant` is a non-clinical, experimental radiology AI-assistance system with
two mostly-independent halves that will eventually connect: (1) a rigorous,
statistics-first evaluation pipeline for a chest X-ray classifier (KAD-512), built
around a multi-expert plug-in architecture that already has several other pretrained
models scaffolded in for other findings/modalities; and (2) a vendored, rebranded OHIF
DICOM viewer ("Clinique Amina") that's meant to eventually display those models'
output to a non-radiologist audience. The defining trait of this whole codebase is
**refusing to overclaim** — every gate, every license check, every "diagnostic only"
label exists because something upstream (a paper, a past mistake, a legal constraint)
made overclaiming a real risk. If you're tempted to relax a threshold, loosen a gate,
or skip a check "just to get a result," don't — that instinct has already been
pushed back on multiple times in this project's history and the answer was always no.

## 1. Who you're working with (read this, it changes how you should communicate)

The user is not a radiologist and is explicit that the target audience for the
eventual product isn't radiologists either — it's presented as "Clinique Amina," a
consumer/patient-facing framing, though the underlying rigor (statistical evaluation,
license diligence) is done at a professional-grade level regardless of who the
end-viewer is. Concretely:

- They get **frustrated by a string of "not accepted yet" results** and will ask you
  to relax targets or find a shortcut. The correct response every time so far has been
  to explain *why* the strict result is actually a good sign (the system refusing to
  overclaim), not to weaken the bar. Don't be the one who finally caves on this.
- They explicitly asked, more than once, to **not hold back** on scope/ambition (UI
  redesign, chasing new AI capabilities) — but the actual constraint has never been
  ambition, it's been real things like licensing (MAIRA-2) or infrastructure
  (Docker/Orthanc unavailable in past sandboxes). Be ambitious about *what to try*, be
  rigorous about *what to claim works*.
- They **pasted a live Hugging Face token in plaintext chat once**. I refused to use
  it and made them revoke and rotate it before continuing. If this happens again with
  any credential, same response: don't use it, tell them why, wait for rotation.
- They want plain-language explanations when confused ("explain it to an outsider") —
  several turns in this project were pure explanation, no code. Don't skip those or
  rush back to code when what's actually being asked for is understanding.
- They are receptive to being told "no, that's not a good idea" when it's backed by a
  concrete reason (license terms, a real architectural bug) — but not when it's vague
  caution. Be specific.

## 2. Repository map (what's real, what's stock, what's ours)

```
core/            — shared vocabulary (Modality, BodyPart enums) + Protocol interfaces
                   (ExpertModel, Loader) that every expert/loader satisfies structurally,
                   no inheritance required. Read core/interfaces.py + core/enums.py first.
experts/         — the multi-expert architecture. Each file wraps ONE pretrained model
                   for ONE modality/body-part niche behind the ExpertModel contract:
                     kad.py                — KAD-512, chest X-ray, zero-shot (§4, the
                                              main active workstream)
                     torchxrayvision.py    — different chest X-ray model; explicitly
                                              DISQUALIFIED as an NIH comparator because
                                              its own weights were trained on NIH data
                                              (contamination) — see its own docstring
                     chest_xray.py         — this project's own retired DenseNet
                                              classifier (see docs/CHEST_CLASSIFIER_RESET.md)
                     maira2.py             — Microsoft MAIRA-2, grounded chest X-ray
                                              report generation with bounding boxes.
                                              GATED on HF + MSRLA license = non-commercial,
                                              research-only, CANNOT be wired into anything
                                              other people use. Personal-research use only.
                     ct_totalsegmentator.py— TotalSegmentator, CT organ segmentation,
                                              104-117 structures, Apache 2.0 (no license
                                              problem), pretrained, no training needed.
                                              This is the current best "quick win" candidate.
                     msk_fracture.py       — YOLOv8 wrist fracture detection, MIT-licensed
                                              weights, chosen specifically because the
                                              "usual" academic benchmark (MURA) never
                                              publishes weights
reporting/       — findings.py (the Finding dataclass + localizers), reporter.py (LLM
                   "typist" that verbalizes findings, never invents them), verifier.py
                   (deterministic grounding check: every number/pathology in the prose
                   must trace back to a Finding), guidelines.py (curated lookup, not
                   RAG, mapping findings -> recommendations + urgency tiers)
explainability/  — gradcam.py: in-house Grad-CAM saliency, works generically on any
                   BaseExpert-shaped backbone. Currently wired to the OLD chest_xray.py
                   classifier only — NOT yet wired to KAD-512 (real, scoped follow-up
                   work, not done).
pipeline.py      — the orchestrator: ingest -> route -> expert(s) -> findings -> report
                   -> verify -> guidelines. Nothing invents facts another stage didn't
                   supply — that discipline is the whole point of this design.
api/             — FastAPI backend the OHIF findings panel talks to (CORS-enabled,
                   Orthanc-backed). Untested end-to-end against real Orthanc in any
                   sandbox so far — Docker has never been available.
deployments/     — docker-compose.yml for Orthanc + Postgres.
viewer/ohif/     — vendored (history-stripped, source committed) OHIF Viewers monorepo,
                   customized via its OWN sanctioned extension/mode/CustomizationService
                   surface. See §5.
notebooks/       — see §4 and §6, this is where almost all real compute happens (Colab).
scripts/         — the actual behavior notebooks orchestrate; see §4.
docs/            — CHEST_CLASSIFIER_RESET.md is the single most load-bearing doc in the
                   repo (the full Gate 1-4 protocol + rationale). Read it before touching
                   anything classifier-related. CLINIQUE_AMINA_REDESIGN.md documents the
                   viewer rebrand in similar depth. This file (AI_HANDOFF.md) is the
                   cross-cutting one.
```

## 3. The core design discipline (applies everywhere, not just one module)

Read the docstrings in `reporting/reporter.py`, `reporting/verifier.py`, and
`experts/maira2.py` — they all independently restate the same rule: **a stage that
generates language is a typist, not a decision-maker.** Vision/classification models
decide *what* is true and *where*; the LLM only verbalizes structured `Finding`
objects it's handed; the verifier re-checks the prose against those same findings and
hard-flags anything that doesn't trace back to a number. This is why `Maira2Expert`
parses MAIRA-2's grounded output into structured `Finding`s rather than just storing
the free-text report — same discipline, applied consistently.

The second cross-cutting discipline is **evidentiary honesty over impressiveness**.
Concrete instances so far: refusing to lower KAD's acceptance thresholds even under
user frustration (§4); refusing to wire MAIRA-2 into the viewer despite it being the
most visually impressive option, because its license forbids that exact use (§4b);
correcting a stale doc section that described an aspirational artifact layout that
didn't match what the code actually produces (§6); calling out a real leaked-credential
incident instead of quietly using the token (§1).

## 4. The KAD-512 evaluation workstream (the main active thread)

**What KAD-512 is:** a pretrained (MIMIC-CXR-trained, externally validated on NIH),
zero-shot vision-language chest X-ray classifier. `experts/kad.py` is a small,
dependency-clean reimplementation of its inference path (deliberately doesn't import
the original repo — that code does global object-storage setup at import time).
**We are not training this model.** We are grading whether its raw output can be
trusted enough to act on, for specific findings, one at a time.

**Why "one endpoint at a time" (query isolation) is non-negotiable:** KAD's decoder
uses self-attention across every query in the prompt pack, so a score for
"Pneumothorax" computed alongside other queries is NOT the same number as computed
alone. Combining/reusing scores across query-pack configurations is explicitly
forbidden throughout the codebase (`export_kad_query_pack.py`, `benchmark_kad.py`, the
notebook). Each endpoint (`Pneumothorax`, `Nodule_or_mass`, `Airspace_opacity`) has its
own frozen, checksum-pinned, singleton query pack.

**The Gate 1-4 protocol** (full detail in `docs/CHEST_CLASSIFIER_RESET.md`, this is
the compressed version):
- **Gate 1 (zero-shot)** — run the frozen model on the development cohort, defer
  ranking metrics until patient roles are frozen (prevents peeking).
- Patient-disjoint role split: `model_selection` 30% / `calibration` 20% /
  `threshold_selection` 20% / `acceptance` 30%, via a deterministic seeded
  (`20250729`) hash search over 512 attempts, endpoint-specific (not shared across
  endpoints — Gate 3's own text is explicit that one endpoint's acceptance patients
  must never leak into another endpoint's training).
- **Gate 3 (calibration + threshold)** — fit a calibrator on the `calibration` role,
  pick an operating threshold on `threshold_selection` against **prespecified**
  targets (currently: sensitivity ≥ 0.85, specificity ≥ 0.60), freeze it *before*
  looking at acceptance data.
- **Gate 4 (locked acceptance)** — the threshold is accepted only if the **two-sided
  95% Wilson confidence interval's lower bound** — not the point estimate — clears
  both targets, on the untouched `acceptance` role (minimum 22 positive + 20 negative
  *patients*, not images; that minimum is mathematically derived — see the doc for why
  21/21 isn't enough but 22/22 is).
- Image provenance matters independently of all the above: schema-2
  `doctor_assistant.nih_image_provenance` (multi-source SHA-256 consensus, ≥2
  independently-operated sources agreeing per file) is required before acceptance
  scoring is even attempted; schema-1 (self-declared) can never claim
  `original_nih_pixels=true`, by hard-coded design.

**Results so far, all three phase-1 endpoints run, and the honest read of each:**

| Endpoint | AUROC (model-selection) | Result | Why |
|---|---|---|---|
| Pneumothorax | 0.907 | `diagnostic_not_deployable` | **Data-support failure.** Only 43 positive images / 33 positive patients total in this cohort — the 4-way split can't leave enough positives anywhere; calibration alone needed ≥10 positives and got 7. Needs CANDID-PTX (external dataset, separate access process) before this can ever pass, independent of model quality. |
| Nodule_or_mass | 0.835 | `diagnostic_not_deployable` | **Confidence-bound failure**, not data-support (106 positives in model-selection alone, calibration *passed*). `sensitivity_lower_95_percent_confidence_bound_below_target` AND `specificity_lower_95_percent_confidence_bound_below_floor`. |
| Airspace_opacity | 0.890 | `diagnostic_not_deployable` | Same failure shape as Nodule_or_mass — best-balanced data of the three (299 pos / 438 neg), calibration passed, still failed both confidence bounds. |

**The important pattern across all three:** 2 of 3 endpoints show good raw
discrimination (AUROC 0.83-0.89) but fail the *simultaneous* sensitivity+specificity
confidence-bound requirement. That's not "the model is bad" — it's "raw zero-shot
scores aren't calibrated tightly enough to hit one specific operating point with
statistical confidence." This is exactly the documented trigger for **Gate 2
controlled adaptation** (`docs/CHEST_CLASSIFIER_RESET.md`: *"Only if Gate 1 shows
useful separation"*) — supervised fine-tuning specific to that endpoint, not a bigger
pretrained model swap (there isn't a better swap available — see below). **Gate 2 is
not built yet.** This is the most concrete, well-motivated next engineering task in
the whole project.

**Why "find a different pretrained model" doesn't work, and don't suggest it again:**
already checked. `experts/torchxrayvision.py` was the obvious candidate and its own
docstring disqualifies it — its weights were trained partly on NIH ChestX-ray14 data,
so testing it against NIH-labeled data is contamination, not a fair test, and an
earlier informal check scored it worse anyway (0.758 vs KAD's 0.835+). There's a
structural reason no shortcut model exists: any model that's already "proven" against
NIH data got that proof by training on NIH data, which disqualifies it from being
independently tested on NIH data. KAD-512 was chosen *because* it's MIMIC-trained, not
NIH-trained — that independence is the whole point, and it means the rigor can't be
skipped by picking a different model.

**Original-pixel sourcing — already solved, don't re-litigate this.** There was a
period in this project's history (visible in git history and possibly in stale prior
context) where the working assumption was "no download automation exists, user must
manually source two independent copies via academictorrents + the NIH host." That
assumption is **outdated**. `notebooks/nih_original_pixel_ingestion_colab.ipynb` is a
real, working, one-time infrastructure notebook that already solved this: **Source A
is the official Kaggle dataset (`nih-chest-xrays/data`) via the Kaggle API, Source B is
a pinned Hugging Face mirror commit** — both plain HTTPS, no torrent client, nothing
Colab's free-tier policy restricts. This has already been run successfully — the
`original_nih_pixels: true` / `acceptance_eligible: true` results on Nodule_or_mass and
Airspace_opacity above are the proof. If a stale doc, comment, or your own prior
context says otherwise, trust this section instead.

**Colab setup, current state:**
- `notebooks/chest_classifier_build_colab.ipynb` — the canonical Gate 1 runner.
  `ACTIVE_TARGET` currently defaults to `Pneumothorax` (fixed from a stale
  `Nodule_or_mass` default). GPU lock is **dynamic** — accepts whatever accelerator
  Colab assigns (was hard-locked to T4 only; relaxed because `benchmark_kad.py`
  independently records exact GPU/CUDA/cuDNN identity into every result's runtime
  contract, so evidence stays traceable per-accelerator without needing to force one
  GPU class). Has an optional cell (4a) to run original-pixel ingestion inline if you
  haven't run the dedicated ingestion notebook yet.
- `notebooks/nih_original_pixel_ingestion_colab.ipynb` — the one-time archive
  ingestion notebook described above. Run once, reused by all endpoints thereafter.
- The user has **Colab Pro with an L4 GPU** and **5TB of Google Drive** — plenty of
  headroom for bigger jobs; don't design around scarcity that no longer applies.

**A real, documented gotcha if you touch `modes/doctor-assistant` or similar
customization code (this is OHIF-side, but worth knowing):** `appInit.js` eagerly
registers every mode's `customizations` export at Default scope, keyed by its own
top-level name. If that value contains nested `$apply`/`$set` commands (immutability-
helper syntax) referencing *other* customization IDs, it crashes — `$apply` only
works against an *already-registered* top-level customization ID, not a fresh wrapper.
Fix pattern: put such overrides in an extension's `getCustomizationModule.tsx`
(Default scope, registered after the extension whose customization you're patching),
not in the mode's own `customizations` block. Full trace in
`docs/CLINIQUE_AMINA_REDESIGN.md` §"Gate 1" of that doc (search "CustomizationService
crash").

## 5. The OHIF/Clinique Amina viewer

Full detail already lives in `docs/CLINIQUE_AMINA_REDESIGN.md` — that document is
itself written for a future-agent audience, so read it directly rather than trusting a
summary here to be complete. The short version: OHIF is real, mature, vendored
unmodified except for two single-line, individually-justified core edits (a hardcoded
`bg-black` and a hardcoded untranslatable string, both listed explicitly in that doc).
Everything else — palette, typography (Playfair Display + Inter), the findings panel,
toolbar trims, WorkList customizations, elevation/shadow system, the brand monogram —
goes through OHIF's own sanctioned `CustomizationService`/extension/mode surface. Two
real Tailwind-config gotchas are documented there (a non-`extend` `theme.fontFamily`
key silently shadowing a preset's same key; the app-level `tailwind.config.js`, not
the `ui-next` one, is the one that actually wins).

**Status: built and compiles clean, never visually verified by any AI in this
project's history** — every sandbox so far lacked a working browser (Playwright/
Chromium install fails: unsupported OS, no sudo). All verification has been
compile-status + served-bundle content matching via `curl | grep`. If you have browser
access, use it — this is a real, meaningful gap.

**The API/viewer connection is unbuilt in practice.** The findings panel
(`extensions/extension-doctor-assistant/src/DoctorAssistantPanel.tsx`) has real,
working code that calls the FastAPI backend (`api/`), but nothing has ever been
exercised end-to-end against a real running Orthanc — Docker has never been available
in any sandbox this project has run in. This is a real gap, not a solved problem.

## 6. AI-capability exploration threads (in progress, not finished)

**MAIRA-2** (`experts/maira2.py`, `scripts/demo_maira2.py`) — grounded chest X-ray
report generation with bounding boxes, the most visually impressive option
investigated. **Gated on HF, licensed MSRLA: non-commercial, research-only, explicitly
forbids "a stand-alone hosted solution for others to use."** This rules it out for
anything patient/radiologist-facing, permanently, not just "for now" — don't
re-propose wiring it into the viewer. It's fine for the user's own private
exploration only. `scripts/demo_maira2.py` exists for that purpose (untested by me —
this sandbox's GPU, 6GB VRAM, can't fit a 7B model; needs to run in Colab).

**TotalSegmentator** (`experts/ct_totalsegmentator.py`,
`scripts/demo_totalsegmentator.py`, `notebooks/totalsegmentator_demo_colab.ipynb`) —
CT organ segmentation, **Apache 2.0, no gate, no restriction** — the actual "real win"
candidate. Confirmed installed and importable in a local sandbox (though full runs
need a real GPU/Colab). The visual-proof demo was rebuilt on 2026-08-04: it pins
TotalSegmentator 2.17.0, verifies Zenodo record `10047263`'s exact v2.0.1 filename and
published MD5, extracts only the selected CT, stages inference under `/content`, and
writes a geometry-checked `segmentation.nii.gz`, colored `preview.png`, organ-volume
`measurements.json`, hashes/runtime provenance, and TotalSegmentator's own statistics
and run report. The old script incorrectly passed an output *directory* with `ml=True`
(the API requires a NIfTI file path), so that path could not have completed. The rebuilt
workflow completed on an L4 for subject `s0011`, producing 108 non-empty anatomy labels
with matching source geometry.

**OHIF bridge is now implemented but still needs its first real DICOM end-to-end run.**
`notebooks/totalsegmentator_dicom_seg_colab.ipynb` accepts a user-confirmed
de-identified DICOM CT and emits direct `dicom_seg` plus a portable viewer bundle.
`scripts/publish_dicom_seg_to_orthanc.py` validates the references, uploads source CT +
SEG to Orthanc, verifies QIDO visibility, and prints the exact Clinique Amina URL. The
doctor-assistant OHIF mode now exposes a read-only segmentation panel, defaults to the
Orthanc data source, and `deployments/docker-compose.yml` includes the viewer with a
same-origin DICOMweb proxy. Unit/static checks pass; Docker is unavailable in the agent
sandbox and no de-identified source DICOM was supplied, so do not claim the live OHIF
overlay has been observed yet. The runbook is `docs/TOTALSEGMENTATOR_OHIF.md`.

**Grad-CAM on KAD-512** — the licensing-clean alternative to MAIRA-2 for "AI
highlights where it's looking" in the viewer. `explainability/gradcam.py` already
exists and works generically on `BaseExpert`-shaped backbones, currently wired to the
old retired classifier only. Extending it to KAD-512 is real, unstarted work (KAD's
decoder-with-cross-attention architecture is different enough from a plain
classification head that this isn't a copy-paste job). This was the recommended path
once an endpoint clears Gate 4 acceptance — tie the visual "wow" feature to a model
that's actually been proven, not a borrowed one that can't be deployed.

**A licensing decision framework, since this will come up again for any new
model/dataset:** Apache 2.0 / MIT with weights actually included = safe, wrap it
(`ct_totalsegmentator.py`, `msk_fracture.py`, the breast-MRI candidate researched but
not yet integrated — CC BY 4.0, `lkshrsch/BreastCancerDiagnosisMRI`). Anything
CC-BY-NC(-SA), MSRLA, or "research purposes only" = fine for the user's personal
exploration, **never** for anything another person (radiologist, patient) would
interact with. A repo with training code but no downloadable weights = not usable at
all (checked and rejected: `tariqshaban/yolov7-vinbigdata-chest-x-ray`, an unvalidated
student project with neither). Always verify licenses by actually reading the LICENSE
file / HF model card, not by assuming from a model's popularity or your own training
data — MAIRA-2's real terms were only caught by actually fetching and reading the
license text.

## 7. Git/branch state (as of the write-up)

Branch `classifier-evaluation-colab`, pushed to `origin` (SSH remote
`git@github.com:Hamza09Hamza/doctor_assistant.git`) at commit `c8a9fca` — a large
single commit that added the vendored OHIF viewer, the API backend, the notebook
fixes, and the two new demo scripts/notebook, all in one push (explicitly confirmed
with the user beforehand given the size). Working tree was clean immediately after.
Verify with `git log --oneline -5` and `git status` before assuming this is still
current — time may have passed and more work may have landed since this was written.

**A note on SSH auth in a fresh sandbox:** the push above required
`GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_github_hamza"` explicitly — the key existed
but wasn't wired into `~/.ssh/config` for github.com by default. If you're in a new
sandbox and `git push` fails with "Permission denied (publickey)", check
`~/.ssh/` for an existing key before assuming none exists.

## 8. What to do first if you're picking this up cold

1. Run the test suite (`.venv/bin/python -m unittest discover -s tests`) to confirm
   the baseline is still green before changing anything.
2. Re-read `docs/CHEST_CLASSIFIER_RESET.md` in full if you're touching anything
   classifier-related — it's more authoritative and detailed than this file's
   compressed summary in §4.
3. If the user is asking for a "quick win": TotalSegmentator (§6) is genuinely the
   fastest legitimate path, has zero license risk, and just needs its demo notebook
   actually executed in Colab, then the NiFTI→DICOM→DICOM-SEG→OHIF wiring built.
4. If the user is frustrated about KAD not passing: re-read §4's pattern analysis
   before agreeing to lower any target. Gate 2 adaptation is the honest fix, and it's
   unstarted, real, scoped work — that's a legitimate thing to start building.
5. Don't re-propose MAIRA-2 for the viewer, TorchXRayVision as an NIH comparator, or
   any dataset/model without weights actually downloadable under a permissive license
   — all three have already been checked and ruled out for specific, documented
   reasons above.
