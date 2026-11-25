#!/usr/bin/env bash
# Rebuild the lumbr B0 search server on a fresh Lightsail Ubuntu 24.04 instance (4 GB plan).
#
# Run as the `ubuntu` user from a checkout of this repo on the server:
#   scp ~/.lumbr/backend_key ubuntu@<ip>:/tmp/backend_key
#   ssh ubuntu@<ip> 'git clone https://github.com/JohnAChester/lumbr.git /tmp/lumbr && bash /tmp/lumbr/deploy/lightsail/setup.sh'
#
# Two steps stay manual because they need John's secrets (see README.md):
#   - Hugging Face login for the artifacts download
#   - the DNS A record, before certbot runs
set -euo pipefail

DOMAIN=${DOMAIN:-api.johnchester.dev}
REPO=${REPO:-https://github.com/JohnAChester/lumbr.git}
REF=${REF:-main}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== swap (2 GB safety net)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

echo "== packages"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  openjdk-21-jdk-headless python3-venv python3-dev build-essential git curl \
  nginx certbot python3-certbot-nginx vmtouch >/dev/null

echo "== service user and code"
sudo useradd --system --create-home --home-dir /srv/lumbr --shell /usr/sbin/nologin lumbr 2>/dev/null || true
[ -d /srv/lumbr/app ] || sudo -u lumbr git clone -q "$REPO" /srv/lumbr/app
sudo -u lumbr bash -c "cd /srv/lumbr/app && git fetch -q origin $REF && git checkout -q -B $REF origin/$REF"
sudo -u lumbr mkdir -p /srv/lumbr/artifacts
[ -d /srv/lumbr/venv ] || sudo -u lumbr python3 -m venv /srv/lumbr/venv
sudo -u lumbr -H /srv/lumbr/venv/bin/pip install -q --upgrade pip
# CPU-only torch first: otherwise pip pulls the CUDA build, which loads unused GPU libraries,
# costs ~5 GB of disk, and crowds the search index out of RAM on the 4 GB plan.
sudo -u lumbr -H /srv/lumbr/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
sudo -u lumbr -H bash -c "cd /srv/lumbr/app && /srv/lumbr/venv/bin/pip install -q -e '.[demo,demo-b0]'"
sudo -u lumbr -H /srv/lumbr/venv/bin/pip cache purge -q >/dev/null 2>&1 || true

echo "== backend key (root-only env file; never printed)"
if [ -f /tmp/backend_key ]; then
  sudo install -d -m 700 -o root -g root /etc/lumbr
  { printf "LUMBR_BACKEND_KEY="; tr -d "\r\n" < /tmp/backend_key; printf "\n"; } | sudo tee /etc/lumbr/backend.env >/dev/null
  sudo chmod 600 /etc/lumbr/backend.env
  shred -u /tmp/backend_key
elif ! sudo test -f /etc/lumbr/backend.env; then
  echo "Missing /tmp/backend_key. Copy ~/.lumbr/backend_key there first." >&2; exit 1
fi

echo "== systemd unit and warm-up"
sudo install -d -m 755 /usr/local/lib/lumbr
sudo install -m 755 "$HERE/warmup.sh" /usr/local/lib/lumbr/warmup.sh
sudo install -m 644 "$HERE/lumbr-pin.service" /etc/systemd/system/lumbr-pin.service
sudo install -m 644 "$HERE/lumbr-search.service" /etc/systemd/system/lumbr-search.service
sudo systemctl daemon-reload
sudo systemctl enable lumbr-pin lumbr-search >/dev/null

echo "== automatic security reboots (09:00 UTC = 4 am US Eastern)"
sudo install -m 644 "$HERE/52lumbr-auto-reboot" /etc/apt/apt.conf.d/52lumbr-auto-reboot

echo "== nginx"
sudo install -m 644 "$HERE/nginx-ratelimit.conf" /etc/nginx/conf.d/lumbr-ratelimit.conf
sed "s/api.johnchester.dev/$DOMAIN/" "$HERE/nginx-lumbr.conf" | sudo tee /etc/nginx/sites-available/lumbr >/dev/null
sudo ln -sf /etc/nginx/sites-available/lumbr /etc/nginx/sites-enabled/lumbr
sudo rm -f /etc/nginx/sites-enabled/default
sudo sed -i "s/# server_tokens off;/server_tokens off;/" /etc/nginx/nginx.conf
sudo nginx -t && sudo systemctl reload nginx

cat <<EOF

Done with the automatic part. Remaining manual steps (README.md):
  1. ssh -t ubuntu@<ip> "sudo -u lumbr -H /srv/lumbr/venv/bin/hf auth login --no-add-to-git-credential"
  2. sudo -u lumbr -H /srv/lumbr/venv/bin/hf download johnchester/lumbr-demo-b0 --repo-type dataset --local-dir /srv/lumbr/artifacts
  3. sudo systemctl start lumbr-pin lumbr-search   # locks the FAISS index in RAM; ~3.5 min warm-up
  4. Point the $DOMAIN A record at this server (Cloudflare: DNS only), then:
     sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --register-unsafely-without-email --redirect
EOF
