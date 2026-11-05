# Crowd intelligence research and implementation notes

## Operational design principles

Current crowd-safety guidance treats ingress, circulation, egress, queueing, pinch points,
capacity, communications, and staff observation as one operating picture. HSE recommends
planning arrival and departure flows, using queue systems and barriers, monitoring arrivals,
and keeping emergency routes clear. It also calls out pinch points, surges, and a central
control point that gives decision-makers the overall picture. The app should surface evidence
for those decisions and keep camera health and human confirmation visible.

Sources: [HSE crowd management](https://www.hse.gov.uk/event-safety/crowd-management.htm),
[HSE crowd controls](https://www.hse.gov.uk/event-safety/crowd-management-controls.htm),
[HSE controls inside venues](https://www.hse.gov.uk/event-safety/crowd-management-control-inside.htm).

Computer-vision queue research tracks people joining and leaving a queue over time to estimate
wait. That depends on reliable track continuity and measured service/exit flow; a queue-zone
count alone cannot produce a defensible wait estimate. The same principle applies to forecasts:
expose the horizon and sample sufficiency, and label a simple trend projection as a trend rather
than a safety prediction. [Example queue tracking and wait-time research](https://ieeexplore.ieee.org/abstract/document/9864749).

## Integrated in this iteration

| Capability | Implementation | Interpretation |
|---|---|---|
| Control-room wall and fullscreen | Existing camera wall view plus fullscreen action | Browser fullscreen is operator initiated |
| Mobile operator dashboard | Existing responsive layout tightened for small screens; analytics tiles reflow | Camera/alert controls remain available on narrow screens |
| Virtual entry/exit lines | Per-camera normalized line API and dashboard endpoint prompts; tracked crossings are classified by side | Configure the line across a visible threshold; crossings depend on stable track IDs |
| Direction-based flow | Existing normalized velocity arrows per zone, plus directed line crossings | Image-plane direction unless ground-plane calibration is configured |
| Crowd growth alerts / forecasts | Durable 15-second samples, recent growth rate, projections at 5/10/15/30 minutes | Forecast is a linear trend estimate; sparse history is explicitly low confidence |
| Bottleneck signals | High occupancy combined with fewer than three moving tracks | A review cue, not an automatic declaration of unsafe conditions |
| Alert timeline / performance | Existing persisted alert stream, KPI and Prometheus metrics; analytics endpoint includes period alerts and sample coverage | Historical analytics begin accumulating after deployment |
| Heatmap over time | Versioned 48×27 density snapshots every 5 minutes, 90-day retention, timeline slider and playback | Snapshots begin after the first five minutes of a camera session |
| Zone-to-zone movement | Track IDs produce camera-local directed transitions between configured polygons; totals appear in event analytics | Cross-camera hand-off is not inferred |
| Daily and event reports | Dependency-free PDF routes for a 24-hour daily report and configurable 1–168 hour event report | Report covers sampled counts and persisted alerts |

## Follow-up work for operational-grade analytics

1. Add calibrated queue polygons and service/exit virtual lines, then estimate wait from the
   rolling departure rate and queue population. Until there is enough exit history, show
   “insufficient data” rather than a numeric wait.
2. Add explicit cross-camera hand-off only after camera geometry and overlap are configured.
3. Compare events by operator-selected event windows and comparable phases (arrival, peak,
   dispersal), accounting for camera set and missing-data coverage.
4. Calibrate growth-alert thresholds per venue and zone; support staffing/response playbooks,
   acknowledgement, and escalation policy.
5. Validate detection/flow performance on each camera angle and dense-crowd footage. Box
   detectors undercount packed crowds, so density-model cross-checking remains important.

The software is decision support. Venue-specific capacity plans, trained stewards, emergency
procedures, independent camera checks, and locally applicable safety guidance remain the basis
for operational decisions.
