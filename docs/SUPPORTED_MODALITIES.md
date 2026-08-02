# Supported modality / body-region combinations

This is a hand-maintained snapshot of the `(modality, body_part)` niches the current
expert adapters advertise via `core.interfaces.ExpertModel`. It is **not** generated from
code — verify against each `experts/*.py` file's `self.modality`/`self.body_part`
assignment before relying on it for anything load-bearing. There is no central "register
every expert" builder yet; `scripts/smoke_system.py` wires up an ad hoc registry for
its own smoke run, and any other caller assembles its own `routing.ExpertRegistry`.

| Modality | Body part | Expert(s) | Notes |
|---|---|---|---|
| `XRAY` | `CHEST` | `experts/torchxrayvision.py::TorchXRayVisionExpert`, `experts/maira2.py::Maira2Expert`, `experts/kad.py::KAD512Expert` | All three can be registered together under the same niche; the router returns all matches and the orchestrator pools their findings. TorchXRayVision's `all` weights include NIH training data, so it is an engineering control, not an independent NIH comparator. |
| `XRAY` | `BONE` | `experts/msk_fracture.py::MSKFractureExpert` | Trained on pediatric wrist trauma X-rays (GRAZPEDWRI-DX); not validated for adult or non-wrist bone imaging. |
| `CT` | `ABDOMEN` (default) | `experts/ct_totalsegmentator.py::TotalSegmentatorExpert` | `body_part` is a constructor argument, not fixed — the same weights also read chest CT. Register the same instance under multiple niches via `ExpertRegistry.register_niche` rather than constructing it twice. |

Everything else in `core.enums.Modality`/`BodyPart` (MRI, ultrasound, mammography,
fundus, OCT; brain, spine, breast, heart, eye) has no registered expert today.
`routing.ModalityRouter` raises `RoutingError` for any niche with no match — this is
intentional fail-closed behavior, not a bug to work around.

## Keeping this current

When adding, removing, or re-scoping an expert's niche, update this table in the same
change. If this ever drifts noticeably out of sync with reality, the fix is a small
introspection test (each expert would need a cheap, weights-free way to report its
niche — most already do via constructor-time `self.modality`/`self.body_part`, but this
hasn't been formalized into a contract test yet) rather than continuing to hand-edit
prose indefinitely.
