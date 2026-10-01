#!/bin/bash
# Startup script for the VM that scripts/gcp_run.sh creates; runs as root at boot.
# Pulls the uploaded checkout and the Spider data from GCS, installs VERTO's pinned
# requirements, runs scripts/train.py once per line of TRAIN_ARGS, uploads the checkpoint,
# the evaluations and the log, then deletes the VM it runs on.
set -uo pipefail
export HOME=/root

md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME=$(md name)
ZONE=$(md zone)
ZONE=${ZONE##*/}
BUCKET=$(md attributes/verto-bucket)
DEST="gs://$BUCKET/runs/$(md attributes/verto-run)"
LOG=/var/log/verto-run.log

if [[ -e /var/lib/verto-ran ]]; then  # startup scripts run on every boot; training runs once
  echo "verto already ran on this VM; powering off"
  poweroff
  exit 0
fi
touch /var/lib/verto-ran

(
  set -e
  mkdir -p /opt/verto && cd /opt/verto
  gcloud storage cp "$DEST/run.env" .
  gcloud storage cp "$DEST/src.tar.gz" - | tar xz
  mkdir -p data
  gcloud storage cp "gs://$BUCKET/data/spider.tar.gz" - | tar xz -C data
  if gcloud storage ls "$DEST/best_model.pt" > /dev/null 2>&1; then  # an evaluation-only run
    mkdir -p checkpoints
    gcloud storage cp "$DEST/best_model.pt" checkpoints/
  fi
  source ./run.env
  echo "== waiting for the NVIDIA driver"
  for _ in $(seq 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done
  nvidia-smi
  echo "== installing uv and VERTO's pinned requirements"
  curl -LsSf https://astral.sh/uv/0.11.15/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  uv venv /opt/venv --python 3.12
  uv pip install --python /opt/venv/bin/python -r requirements.txt
  uv pip install --python /opt/venv/bin/python -e .
  export PATH="/opt/venv/bin:$PATH"
  while IFS= read -r args; do
    [[ -n $args ]] || continue
    echo "== python scripts/train.py --device cuda $args"
    # shellcheck disable=SC2086  # each line of TRAIN_ARGS is a list of flags
    python scripts/train.py --device cuda $args
  done <<< "$TRAIN_ARGS"
) 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

outcome=$([[ $status == 0 ]] && echo DONE || echo FAILED)
echo "== $outcome (exit $status); uploading to $DEST"
cd /opt/verto || true
# A checkpoint this run trained (an evaluation-only run's checkpoint came from the bucket).
if [[ -f checkpoints/best_model.pt ]] && ! gcloud storage ls "$DEST/best_model.pt" > /dev/null 2>&1; then
  gcloud storage cp checkpoints/best_model.pt "$DEST/trained/"
fi
compgen -G "results/*.json" > /dev/null && gcloud storage cp results/*.json "$DEST/results/"
gcloud storage cp "$LOG" "$DEST/run.log"
echo "$outcome" | gcloud storage cp - "$DEST/$outcome"  # last: gcp_run.sh fetches once this exists
echo "== deleting this VM"
gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || poweroff
