"""A second opinion on a photo's count, to catch the point model's known failure.

PET can put dots on things that are not heads: railway gravel, waves at the horizon, rows of advertising
boards. The density model (DM-Count) makes different mistakes, so when PET counts far more than DM-Count
does, the photo is worth a second look. Measured on Kaggle with the first point model
(results/second_opinion.json): with DM-Count on the photo shrunk to 768 px and a ratio above 2.5, it
flagged none of the 498 ShanghaiTech test photos and none of the 9 well-counted hand-labelled Indian
photos, and it flagged the Mumbai railway photo (PET 1,282 against 415 true).

Re-run with the current point model (results/model_v2/second_opinion.json): still none of the 498
ShanghaiTech photos, and none of the 10 hand-labelled Indian photos, because it now counts the Mumbai
photo about right. Of the 95 Indian photos without a hand count it flagged one, a false alarm: a crowd of
about 8,000 where both models agree at full size, but DM-Count undercounts once the photo is shrunk.

It never changes the count: the app still reports PET's number, with a warning to look at the dots.
"""

from __future__ import annotations

import os

from .config import memory_in_use_bytes, memory_limit_bytes

SIDE = 768          # DM-Count sees the photo at this size (longer side): 3 s on a CPU, little memory
RATIO = 2.5         # PET more than this many times DM-Count -> ask for a second look
MIN_COUNT = 50      # below this, a ratio says little (5 against 15 is noise, not a failure)
HEADROOM_MB = 700   # run only if at least this much memory is free, so the check can never crash a small server


def enabled() -> bool:
    return os.environ.get("CROWDSHIELD_SECOND_OPINION", "1") != "0"


def has_headroom(limit: int | None = None, used: int | None = None) -> bool:
    limit = memory_limit_bytes() if limit is None else limit
    if limit is None or limit >= 64 * 1024 ** 3:
        return True
    used = memory_in_use_bytes() if used is None else used
    return (limit - (used or 0)) / 1e6 >= HEADROOM_MB


def second_opinion(img_bgr, pet_count: float) -> dict | None:
    """{"dm_count", "ratio", "disagree"} for a photo PET counted, or None when the check was skipped."""
    if not enabled() or pet_count < MIN_COUNT or not has_headroom():
        return None
    try:
        import cv2

        from .density import density_from_model, load_density_model

        model, device = load_density_model()
        scale = min(1.0, SIDE / max(img_bgr.shape[:2]))
        small = img_bgr if scale == 1.0 else cv2.resize(img_bgr, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_AREA)
        dm = float(density_from_model(model, device, small).sum())
    except Exception:  # no weights, no memory, anything: the check is optional, the count is not
        return None
    ratio = pet_count / max(dm, 1.0)
    return {"dm_count": round(dm), "ratio": round(ratio, 2), "disagree": ratio > RATIO}
