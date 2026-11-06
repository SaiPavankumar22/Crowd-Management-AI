# Dense counting service definitions

The network definitions in this directory implement PET head-point counting and
DM-Count density estimation used by `backend/app/dense.py`.

Expected SHA-256 digests for operator-supplied checkpoints are pinned in
`backend/app/dense.py`. A mismatch refuses to load the file.

Weights themselves are not stored in this package: place verified checkpoints
in the repository `models/` directory (or point `CROWD_MODELS_DIR` at them).
