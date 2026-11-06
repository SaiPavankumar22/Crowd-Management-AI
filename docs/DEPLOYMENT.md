# Deploying Godavari Crowd Monitor

This guide takes the app from `bash tools/start_dev.sh` on a laptop to a
long-running deployment on a server with a GPU. It covers the container image,
environment variables, storage, a reverse proxy with TLS and password, backups,
upgrades, and the failure modes you will actually hit.

Everything here assumes the repository, including `Dockerfile`,
`docker-compose.yml` and `docker-compose.gpu.yml`.

---

## 1. What you are deploying

One Python process: FastAPI + uvicorn, serving the dashboard, the REST API, the
MJPEG streams and the WebSocket feed, plus one capture/inference thread per
camera.

| Component | Where the state lives | Survives restart? |
|---|---|---|
| Camera list + zones | `.data/data/cameras.json` | yes |
| Alert history | `.data/data/crowd.db` (SQLite) | yes |
| Uploaded clips | `.data/uploads/` | yes |
| Sample clip library | `.data/samples/` | yes, if you keep the folder |
| Model weights | `/app/models/yolo26n.pt` (image) or `.data/...` cache | yes |
| Live counts, tracks, heatmap | process memory | no — rebuilt from video in seconds |

That single-process design is deliberate and it constrains deployment:

- **Run exactly one uvicorn worker.** Camera workers, the model, the WebSocket
  client set and the registry all live in process memory; a second worker would
  start its own cameras and show an empty dashboard half the time.
- **Scale by container, not by replica, until you split the design.** Two
  containers pointing at the same `.data` volume will fight over
  `cameras.json`. For many cameras (100+), use the sharded architecture in
  [`PRODUCTION_ARCHITECTURE.md`](PRODUCTION_ARCHITECTURE.md); this app is the
  single-node version of it.
- **The app is stateless per request but stateful per process** — so restarts
  are cheap (seconds), but `docker stop` must be graceful to let SQLite and the
  JSON writer finish (compose sets `stop_grace_period: 20s`).

---

## 2. Deployment options

| Option | Best for | Cameras (YOLO26n) | Effort |
|---|---|---|---|
| **A. Local, no container** (`tools/start_dev.sh`) | Development, demo on this laptop | 2–4 on CPU | none |
| **B. Docker, CPU host** (`Dockerfile` default) | Small venue, single server, 1–4 cameras | 2–4 | 5 min |
| **C. Docker, GPU VM** (L40S-class) — **recommended for a real event** | Many RTSP cameras, 24/7 | ~10–20 with this app; 100+ with the sharded design | 20 min |
| **D. Bare metal + systemd** | You already run the host and dislike containers | same as B/C | 15 min |
| **E. Kubernetes** | Multiple sites, central ops | — | not yet: one process, one volume |

Option C is what the rest of this document walks through, because it is the
path from "works on my machine" to "runs at the ghat".

---

## 3. The image

One `Dockerfile` builds both variants. The only difference is which PyTorch
wheel index it installs from — the CUDA wheels bundle their own CUDA runtime,
so the base image stays `python:3.11-slim` and the host only needs a driver.

```bash
# CPU image (0.67 GB): no GPU, no driver, works anywhere
docker build -t godavari:cpu .

# GPU image (4.4 GB): CUDA 12.8 wheels, needs --gpus all at run time
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
             -t godavari:gpu .
```

To include PET head points and DM-Count in an edge image, add
`--build-arg INSTALL_CROWD_SHIELD=1`. Enable the paths at runtime with
`CROWD_POINT_BACKEND=dense` and/or `CROWD_DENSITY_BACKEND=dense`.
They start as independent advisory estimates. The dashboard records hand-count
checks per camera and only permits a selected dense count after calibration,
the configured sample count and error threshold are met.

Other CUDA lines: `cu126`, `cu124`, `cu121` — match your host driver.
`nvidia-smi` shows the driver's CUDA version; pick a wheel that is **≤** it.

Both variants were built and run during development on Windows + Docker Desktop
with an RTX 3050 (driver 592.82): the CPU image reports
`{"device":"cpu","torch":"2.15.0+cpu"}` and the CUDA image reports
`{"device":"gpu","torch":"2.11.0+cu128","cuda_available":true}` with
`--gpus all`. Note the CUDA wheel indexes lag the CPU index, so the two images
legitimately resolve different torch versions — that is expected, both are
supported by ultralytics.

> **Disk:** the first GPU build downloads ~3 GB of NVIDIA wheels and leaves a
> sizeable build cache behind. `docker builder prune` reclaims it once the image
> is built. On Docker Desktop the freed blocks stay inside the WSL disk image
> until it is compacted, so Windows free space may not move until then.

What the image does for you:

- installs exactly the pinned dependency set from `requirements.txt` (the same
  versions the local venv runs), plus OpenCV's shared-library deps (`libgl1`,
  `libglib2.0-0`) so `import cv2` works in a bare container;
- pre-installs PyTorch from one index **before** ultralytics, so a GPU image
  never ends up with a CPU torch build (the most common "why is it so slow"
  bug);
- ships `yolo26n.pt`, so first boot works with no internet access;
- runs as non-root `app` (uid 10001) and keeps every cache in `/app/.data`;
- declares a healthcheck on `GET /api/system`, which answers before the model
  has finished loading (model load takes 30–90 s cold, hence a 120 s
  `start-period`).

### Run it

```bash
docker run -d --name godavari \
  -p 127.0.0.1:8000:8000 \
  -v godavari-data:/app/.data \
  --restart unless-stopped \
  godavari:cpu
```

GPU:

```bash
docker run -d --name godavari \
  --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v godavari-data:/app/.data \
  --restart unless-stopped \
  godavari:gpu
```

`127.0.0.1:8000` publishes only on loopback — that is intentional. See
[§6](#6-put-a-proxy-and-a-password-in-front) before exposing it.

### Or with compose

```bash
docker compose up -d --build            # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # GPU
docker compose logs -f                  # follow
docker compose down                     # stop (volume is kept)
```

Verify what actually loaded:

```bash
docker exec godavari python -c "import torch,ultralytics; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('ultralytics',ultralytics.__version__)"
curl -s localhost:8000/api/system
curl -s localhost:8000/api/samples | head -c 400
```

`"cuda": false` on a GPU host means the image is the CPU build, `--gpus` was
not passed, or the driver/toolkit is missing — in that order.

---

## 4. GPU VM, step by step

The target shape: one GPU VM running the container, a small disk, and the
dashboard reachable over HTTPS.

1. **Create the VM** — any GPU cloud or on-prem host: a 1×L40S-class GPU
   (48 GB VRAM), 8 vCPU / 32 GB RAM. Ubuntu 24.04 with the NVIDIA driver
   image, 100 GB disk (OS + image + a few GB of clips).
2. **Attach a public IP** and allow inbound **443** (and 22 for SSH) in the
   security group. Do not open 8000 to the internet.
3. **Install Docker + NVIDIA container toolkit:**

   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker "$USER" && newgrp docker
   sudo nvidia-smi                     # driver present?
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
   ```

4. **Get the code onto the VM** (either way works):

   ```bash
   # git
   git clone <your-repo-url> godavari && cd godavari
   # or from this machine (Git Bash):
   rsync -av --exclude .venv --exclude .data --exclude .tmp ./ user@vm:/home/user/godavari/
   ```

   Copy `yolo26n.pt` too if you want a fully offline first boot (it is in the
   repo root, and `.dockerignore` keeps it in the build context).

5. **Build and start on GPU:**

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
   docker compose logs -f | tail -40
   ```

6. **Load the sample library (optional).** The pilgrim clips are 70 MB and are
   not in git. From this machine:

   ```bash
   rsync -av .data/samples/ user@vm:/home/user/godavari/.data/samples/
   ```

   …then run the container with that folder mounted read-only on top of the
   volume:

   ```yaml
   volumes:
     - godavari-data:/app/.data
     - ./data/samples:/app/.data/samples:ro
   ```

7. **Add real cameras.** In the dashboard: *Add source* → paste
   `rtsp://user:pass@camera-ip:554/Streaming/Channels/101`, name it, then draw
   zones. Reconnect-on-drop is built in; a dead camera shows `RECONNECTING`
   rather than killing the process.

---

## 5. Configuration

Every knob is an environment variable. In compose, the ones worth changing are
already parameterised from your shell (`CROWD_MODEL=... docker compose up -d`).

| Variable | Default | What it changes |
|---|---|---|
| `CROWD_MODEL` | `yolo26n.pt` | `yolo26s.pt` for better far-field recall at ~3× the compute; `yolo11n.pt` to compare |
| `CROWD_CONF` | `0.30` | Detector confidence. Lower (0.2) in dense scenes, higher in a scene with statues/mannequins |
| `CROWD_IMGSZ` | `640` | Inference size. 960–1280 for aerial/wide shots with tiny people; costs FPS |
| `CROWD_TARGET_INFER_FPS` | `5` | Per-camera inference rate. 3 is plenty for counting; raise for fast flow |
| `CROWD_MAX_FRAME_WIDTH` | `1280` | Downscale before detection — the main CPU lever |
| `CROWD_STREAM_FPS` | `12` | MJPEG rate per viewer stream |
| `CROWD_WS_PERIOD` | `0.6` | Dashboard snapshot period (s) |
| `CROWD_LOOP` | `1` | Loop file sources (sample clips) |
| `CROWD_WARN_PCT` | `75` | Zone warning threshold (% of capacity) |
| `CROWD_ZONE_CAPACITY` | `100` | Capacity for auto-created zones |
| `CROWD_ALERT_SUSTAIN` | `5` | Seconds at ≥100 % before a capacity alert |
| `CROWD_ALERT_COOLDOWN` | `60` | Minimum gap between repeat alerts |
| `HOST` / `PORT` | `0.0.0.0` / `8000` in Docker | Bind address inside the container |
| `CROWD_DATA_DIR` | `/app/.data/data` | `cameras.json`, `crowd.db` |
| `CROWD_UPLOADS_DIR` | `/app/.data/uploads` | Uploaded clips |
| `CROWD_SAMPLES_DIR` | `/app/.data/samples` | Sample library (supports one level of subfolders) |
| `CROWD_POINT_BACKEND` / `CROWD_DENSITY_BACKEND` | empty | Set either to `dense` to enable PET / DM-Count on a dense node |
| `CROWD_POINT_MODEL_VALIDATED` / `CROWD_DENSITY_VALIDATED` | `0` | Allow that checkpoint to pass a per-camera promotion gate |
| `CROWD_VALIDATION_MIN_SAMPLES` | `20` | Manual checks required before dense-source promotion |
| `CROWD_VALIDATION_MAX_MAPE` | `0.20` | Maximum mean absolute percentage error accepted at promotion |

Sizing, measured here and estimated for GPU (see §9 for what was actually
measured):

| Hardware | Measured / estimated | Notes |
|---|---|---|
| This dev machine, CPU torch, `imgsz=1280` | 3.6 inferences/s total (1 camera), 1.8/s each with 2 cameras | Local venv, no container |
| CPU container, `imgsz=640` | 3.2 FPS for one camera | Verified in the `godavari:cpu` image |
| RTX 3050 8 GB, `imgsz=640`, cap lifted | **32 inferences/s aggregate**, i.e. 4 cameras at 8.0 FPS each | Verified in the `godavari:gpu` image; 1.8 GB VRAM, ~130 % CPU (JPEG encoding) |
| One L40S | ~10–20 cameras at 5 FPS | Estimate, roughly 6–10× the 3050; one process, model behind a lock, MJPEG is CPU-bound |
| L40S, sharded design | 100+ cameras | [`PRODUCTION_ARCHITECTURE.md`](PRODUCTION_ARCHITECTURE.md) |

Two numbers matter more than camera count: total inferences/s (a single shared
model serialises every camera) and JPEG encoding CPU (one MJPEG stream per
viewer). Raise `CROWD_TARGET_INFER_FPS` only as far as the GPU keeps up, and
watch `docker stats` CPU for the encoding ceiling.

GPU memory is not the limit; JPEG encoding and Python threads are. If you plan
more than ~20 cameras, read the architecture document before buying more GPUs.

---

## 6. Put a proxy and a password in front

**The dashboard has no authentication.** Anyone who can reach it can add
cameras, read feeds and delete zones. Never publish it directly.

Minimum viable: Caddy on the same VM, automatic TLS, HTTP basic auth.

```caddyfile
crowd.example.org {
    basic_auth {
        ops $2a$14$REPLACE_WITH_BCRYPT_HASH   # caddy hash-password --plaintext '...'
    }
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1          # MJPEG + WebSocket need unbuffered streaming
    }
}
```

nginx equivalent, with the two directives people forget:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;      # WebSocket /ws
    proxy_set_header Connection "upgrade";
    proxy_buffering off;                          # MJPEG /stream
    proxy_read_timeout 3600s;
}
```

Then start the container bound to loopback only (the default in
`docker-compose.yml`) and expose 443. Optionally restrict by IP at the cloud
security-group level.

Hardening checklist:

- [ ] `8000` reachable only from the proxy host.
- [ ] TLS (cookies/credentials in RTSP URLs otherwise travel in clear text).
- [ ] Password (or an SSO proxy) in front of the dashboard.
- [ ] Container runs as uid 10001, non-root.
- [ ] Only the `.data` volume is writable.
- [ ] Alert history backed up (see below).
- [ ] **RTSP credentials live in plain text in `cameras.json`** — protect that
      volume, and prefer camera accounts that can only read the stream.

---

## 7. Backups and data recovery

Everything durable is under `/app/.data`:

```
.data/
├── data/
│   ├── cameras.json     # cameras + zones (the thing you never want to lose)
│   └── crowd.db         # alert history (SQLite)
├── uploads/             # uploaded clips (disposable)
├── samples/             # sample library (disposable, re-fetchable)
└── {...cache dirs...}   # ultralytics/torch/matplotlib — disposable
```

Back up just the two files that matter:

```bash
docker run --rm -v godavari-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/godavari-$(date +%F).tgz -C /data data/cameras.json data/crowd.db
```

That works on Linux with a *named* volume. On Docker Desktop (Windows/macOS) the
alpine shell mangles `/data` into a host path unless you stop the path
conversion — either run the same command from PowerShell, or use
`docker cp` from the running container instead:

```bash
docker cp godavari:/app/.data/data/cameras.json ./backup/
docker cp godavari:/app/.data/data/crowd.db ./backup/
```

Restore the same way (`tar xzf` into the volume while the container is
stopped). Zone polygons can also be re-exported at any time via
`GET /api/cameras`.

Bind mounts instead of a named volume are fine on Linux, but then ownership is
yours to fix:

```bash
sudo chown -R 10001:10001 ./data
```

Without that the container starts and then fails to write `cameras.json` the
first time you add a camera.

---

## 8. Operations

```bash
docker compose ps                       # health: starting → healthy
docker compose logs -f --tail=100
docker stats godavari
docker exec godavari python -c "import torch;print(torch.cuda.is_available())"
watch -n2 nvidia-smi                    # on the host, GPU build
curl -s localhost:8000/api/alerts?limit=5
```

**Upgrade:**

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The volume is untouched, so cameras and zones survive. Tag images
(`godavari:gpu-2026-11-04`) if you want a rollback target; rollback is
`docker tag` + `up -d`, not a rebuild.

**Restart behaviour:** `restart: unless-stopped` survives reboots (Docker
starts on boot). The healthcheck flips to `unhealthy` after ~2.5 min of failed
`/api/system` probes; it does not restart the container by itself — add an
external watchdog or use `--restart` semantics you are comfortable with.

**Shutdown:** `docker compose down` (not `kill`) so `cameras.json` and SQLite
finish writing. A hard kill mid-save costs you the last camera edit, not the
database.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cuda: false` on a GPU box | CPU wheel in the image (`TORCH_INDEX_URL` build arg), missing `--gpus all`, or missing driver/toolkit. Check `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` |
| Inference ~1 FPS with a GPU | Same as above, plus `CROWD_IMGSZ=640` and `CROWD_TARGET_INFER_FPS=5` sanity check |
| `Permission denied` writing `cameras.json` | Bind mount owned by root. `sudo chown -R 10001:10001 ./data`, or use a named volume |
| Dashboard connects, feed stays black | RTSP unreachable from inside the container (VLAN/firewall), wrong stream path, or an expired camera password. Check `docker compose logs` for the worker's reconnect line |
| Dashboard shows stale numbers | Camera frozen (loop/stream stall). See the failure-mode table in `PRODUCTION_ARCHITECTURE.md` — detect frozen frames, do not trust "no change" |
| WebSocket never connects behind a proxy | Missing `Upgrade`/`Connection` headers or a proxy that buffers; see §6 |
| MJPEG stream stutters through nginx | `proxy_buffering off` missing |
| `Address already in use` | Another process on 8000; change the published port (`PORT=8080 docker compose up -d`) |
| Container unhealthy on first start | Model download blocked by a firewall. Ship `yolo26n.pt` in the build context, or allow egress to `github.com`/`ultralytics.com` once |
| Browser tab using 100 % CPU with the wall view | Each tile is its own MJPEG stream; at scale use the server-side mosaic described in the architecture doc |

---

## 10. Bare metal alternative (systemd)

If you would rather not use containers, the repo's own venv works:

```ini
# /etc/systemd/system/godavari.service
[Unit]
Description=Godavari Crowd Monitor
After=network-online.target

[Service]
User=crowd
WorkingDirectory=/opt/godavari
Environment=CROWD_DATA_DIR=/var/lib/godavari/data
Environment=CROWD_UPLOADS_DIR=/var/lib/godavari/uploads
Environment=CROWD_SAMPLES_DIR=/var/lib/godavari/samples
Environment=HOST=127.0.0.1
Environment=PORT=8000
ExecStart=/opt/godavari/.venv/bin/python run.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now godavari
```

For a GPU host, install CUDA torch into that venv first:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.

---

## 11. Licensing note for production

Ultralytics YOLO (the detector this app loads) is **AGPL-3.0**. Serving it as a
network service for others can oblige you to publish the corresponding source
of your service. For an internal deployment this is usually fine; for a
commercial product, buy an Ultralytics licence or swap in a permissively
licensed detector. Also: the sample test clips from YouTube in
[`FOOTAGE.md`](FOOTAGE.md) are **evaluation-only** and must not be shipped with
a customer-facing deployment — the CC BY 2.0 Archive.org clips are the ones you
can redistribute.
