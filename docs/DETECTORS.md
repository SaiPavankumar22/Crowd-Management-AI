# Detector families

Godavari is not tied to one detector. Four families are registered and
switchable from the **Engine** panel, `PUT /api/models`, or the `CROWD_MODEL`
environment variable.

| Key | Family | Licence | Dependencies | Status here |
|---|---|---|---|---|
| `yolo26n.pt` | YOLO26 | AGPL-3.0 | built in | measured |
| `yolo11n.pt`, `yolov8n.pt`, `yolov8x.pt` | YOLO | AGPL-3.0 | built in | fallbacks |
| `rtdetr-l.pt`, `rtdetr-x.pt` | RT-DETR | Apache-2.0 | built in (Ultralytics) | measured |
| `rf-detr-base.pth`, `rf-detr-large.pth` | RF-DETR | Apache-2.0 | `pip install rfdetr` | adapter, untested |
| `dfine_m_coco.pt`, `dfine_l_coco.pt` | D-FINE | Apache-2.0 | `dfine` package | adapter, untested |

## Which one runs

- Default: **YOLO26n** on every device (`CROWD_MODEL=yolo26n.pt`).
- `CROWD_MODEL=rtdetr-l.pt`: RT-DETR — best at medium density, weak on packed frames.
- `CROWD_MODEL_CHOICES=a,b,c`: narrow or extend what the picker offers.

Latency: RT-DETR-l needs about 1.7 s per 640 px frame on CPU against 143 ms for
YOLO26n, which is part of why it is an opt-in rather than the default.

## Measured on the labelled benchmark

CUHK Mall, first 60 annotated frames, 640 px, confidence 0.30, CPU:

| Model | MAE (people/frame) | Prediction mean | Ground-truth mean |
|---|---:|---:|---:|
| RT-DETR-l | **9.23** | 20.97 | 30.07 |
| YOLO26n | 15.33 | 14.70 | 30.07 |

Both undercount head-only crowds, which is why the dense estimators exist; this
comparison only decides which detector is the better starting point. These are
research-dataset numbers, not ghat validation.

The ranking reverses on packed footage. On the 1080p sample clip, same detail
pass, same 640 px full-frame pass:

| Model | Full-frame count | Final count | Mode |
|---|---:|---:|---|
| YOLO26n | ≤6, so it escalates | **156** | `detector_lower_bound` |
| RT-DETR-l | 36, above the trigger | 36 | `detector_crosscheck` |

RT-DETR's full-frame count never falls below the escalation trigger, so the
detail pass never runs and a packed scene is under-reported. Raising
`CROWD_DETAIL_PROBE_BELOW` makes RT-DETR escalate too, at a higher frame cost.
That is why the default stays YOLO26n until the trigger is tuned per family.

## Installing the optional families

```bash
pip install rfdetr            # RF-DETR, Apache-2.0 code and weights
pip install dfine             # D-FINE; if the package exposes no load_model(),
                              # use the Peterande/D-FINE repo on PYTHONPATH
```

Until the package imports, the registry lists the family with
`available: false` and the picker disables it — the switch can never half-load a
model and leave the camera without a detector.

## Adding a family

1. Add a `ModelSpec` to `KNOWN_MODELS` in `app/adapters.py` (`backend`, licence,
   expected VRAM, optional `requires` package).
2. Add an adapter subclass, or let `build_detector()` map the backend to one.
3. Cover it in `tests/test_detector_families.py`.
4. Never mark it `commercial_use=True` without checking the weights licence.
