# Crowd Shield — third-party notice

The network definitions in this directory come from the **Crowd_shield** project
(`https://github.com/khushi-git-7/Crowd_shield`). They are redistributed here
unmodified so that the PET head-point and DM-Count density estimators can be
loaded offline, with every checkpoint verified by SHA-256 before use.

Files:

- `points.py`, `pet/` — PET (point-based head counting, VGG16-BN backbone)
- `density.py` — DM-Count (density-map counting, VGG19 backbone)
- `models/__init__.py`, `misc/`, `datasets/` — helpers required by the above

## Licence

The upstream project is published for **academic and research use**. It carries
no explicit open-source licence file, so it must not be treated as
permissively licensed. Before any operational or commercial deployment:

1. obtain written permission from the upstream authors, or
2. replace these definitions with an implementation you are licensed to use.

The application treats both estimators as optional: with
`CROWD_POINT_BACKEND` / `CROWD_DENSITY_BACKEND` empty, the service runs on the
detector alone and none of this code is imported.

## Checkpoints

The weights (`models/pet_shab.pth`, `models/dmcount_shab.pth`) are **not** part
of this repository — they are large binaries and are git-ignored. Place them in
the project's `models/` directory (or point `CROWD_MODELS_DIR` at them). The
expected SHA-256 digests are pinned in `app/crowd_shield.py`; a mismatch is a
hard error, so a substituted or corrupted file can never run silently.
