# Person detection and dense-crowd model selection

Research checked 4 November 2026. This is a practical shortlist for Godavari's live camera pipeline, where detection boxes are also used for tracking, virtual-line counts, and zone movement.

## Recommendation

There is no single publicly available model that can be called best for every 200–300-person frame. View angle, pixel size per head, lighting, compression, and occlusion matter more than the advertised benchmark score. Use a two-path design:

1. Keep a person detector and tracker for people who are individually visible. The current YOLO26n remains the low-latency default. Test larger YOLO26 variants or a crowd-trained detector only on the deployment hardware and representative ghat footage.
2. Add a crowd-density/counting model as a separate estimate for tightly packed views. DM-Count is the practical open-source candidate to evaluate first: it produces a density map and total count, not person boxes or persistent IDs. Use it to cross-check zone totals and flag detector undercount; do not feed its count into line-crossing or identity logic.
3. For operationally meaningful totals, fine-tune on legally usable, camera-matched footage and validate against manually counted frames. General COCO benchmark scores do not establish accuracy for a particular ghat camera.

## Candidate comparison

| Candidate | What it is good at | Fit for this application | Caveat |
|---|---|---|---|
| YOLO26n/s/m/l/x | Fast person boxes; multi-scale family | Existing pipeline; preserves boxes for tracking and virtual lines. Larger variants can be evaluated when GPU capacity permits. | Released checkpoints are COCO-trained. Public mAP and latency numbers are COCO/T4 results, not dense-ghat accuracy. Ultralytics code/models use AGPL-3.0 or Enterprise licensing. |
| CrowdHuman-trained detector (for example, a detector fine-tuned from YOLO) | Person boxes under overlap and partial occlusion | Strong next step for improving box recall while retaining the pipeline's interfaces. CrowdHuman is closer to crowded street views than COCO. | A dataset is not a ready-to-run model. Build a labelled training set, preserve head/visible/body annotations appropriately, train, and check checkpoint/framework terms. |
| DM-Count (MIT implementation) | Total count and spatial density in dense images | Best first complementary model to evaluate for severe crowding; use density output for a count cross-check/heat layer. Official deployment model variants are described for SHA, SHB, and QNRF crowd domains. | No individual boxes or IDs, so it cannot replace detector/tracker/line counting. Choose a checkpoint by camera viewpoint and validate domain shift. |
| P2PNet | Point localization and counting in dense scenes | Research candidate if point-level crowd localization is specifically needed. | Official license restricts use to academic research; not a suitable default for an operational system without separate permission. |
| CrowdDet | Multiple detections in heavily overlapping scenes | Interesting research baseline for crowded person boxes. | Older research stack; integration and checkpoint terms need careful review. Avoid making it the production default without a benchmark and licensing review. |

## What changed in the application

The Engine card now offers two live profiles:

- **Balanced** uses confidence 0.30 and image size 640.
- **Dense crowd** uses confidence 0.15 and image size 1280. It may recover weak/small detections, but increases inference cost and false positives. On a CPU-only host, verify throughput before enabling it for many cameras.
- **Configured defaults** restores the environment-configured values.

The API endpoint `PUT /api/detector-profile` applies the setting without reloading weights. Inference reads settings under the detector lock so a profile change cannot split one inference call across settings. This does not magically count completely occluded people; it is a safe inference-tuning option while the crowd-density path is evaluated separately.

## Sources

- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26) — model sizes, COCO mAP/latency, and AGPL/Enterprise note.
- [CrowdHuman benchmark paper](https://arxiv.org/abs/1805.00123) — 470k annotated person instances, about 22.6 people per image, and head/visible/full-body boxes.
- [DM-Count official repository](https://github.com/cvlab-stonybrook/DM-Count) and [model card](https://models.luxonis.com/luxonis/dm-count/aim_N8aaf6PqmzAbnj6eheTKLy?backTo=%2F%3F) — density-map crowd counting, dataset-specific variants, and MIT license information.
- [P2PNet official repository](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) and [license text](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/LICENSE) — point-based crowd counting/localization and academic-research-only restriction.
- [CrowdDet official repository](https://github.com/megvii-model/CrowdDetection) — crowded-scene detector implementation.

## Validation plan before operational deployment

1. Record representative camera clips with permission, spanning ordinary flow, packed surges, night, rain, backlight, and each camera elevation.
2. Manually count a time-stratified set of frames and label line crossings for the cameras that need entry/exit totals.
3. Compare YOLO26 profiles and a CrowdHuman-fine-tuned checkpoint for precision/recall, count MAE, line-count error, and actual per-camera inference FPS.
4. Compare DM-Count against the same ground truth by camera/domain. Keep its confidence and count error visible to operators; do not fuse estimates until calibration on that camera is demonstrated.
5. Set per-zone safety thresholds using conservative error bounds and test alert latency with simultaneous cameras.

Existing sample footage notes in `FOOTAGE.md` already show severe undercounting in overhead and tightly packed scenes (for example, 7–9 detected in a frame estimated at roughly 40 visible people). That evidence means the current COCO-pretrained YOLO26n should not be described as best-in-class for 200–300 tightly packed people. It is a practical real-time baseline, not a validated crowd-counting solution for that case.
