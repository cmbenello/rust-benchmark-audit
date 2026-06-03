#!/bin/bash
set -euxo pipefail

# Bootstrap script for rustbench eval VM (Ubuntu 22.04, c6i.8xlarge)
# Idempotent: safe to re-run.

export DEBIAN_FRONTEND=noninteractive

# --- System packages ---
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl gnupg lsb-release \
  git build-essential pkg-config \
  python3.11 python3.11-venv python3-pip \
  htop tmux jq unzip rsync

# --- Docker CE ---
if ! command -v docker >/dev/null; then
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  ARCH=$(dpkg --print-architecture)
  CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
  echo "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $CODENAME stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker ubuntu
fi

# --- Docker daemon tuning for heavy parallel builds ---
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "3"},
  "default-ulimits": {"nofile": {"Name": "nofile", "Hard": 65536, "Soft": 65536}}
}
EOF
sudo systemctl restart docker || sudo systemctl start docker
sudo systemctl enable docker

# --- Raise file descriptor limits for this user ---
sudo tee -a /etc/security/limits.conf > /dev/null <<'EOF'
ubuntu soft nofile 65536
ubuntu hard nofile 65536
EOF

# --- Python venv for rustbench ---
if [ ! -d ~/rb-venv ]; then
  python3.11 -m venv ~/rb-venv
fi
source ~/rb-venv/bin/activate
pip install --upgrade pip setuptools wheel

# --- Disk / memory sanity print ---
echo "=== disk ==="
df -h /
echo "=== mem ==="
free -h
echo "=== cpu ==="
nproc
echo "=== docker ==="
sudo docker info 2>&1 | head -20 || true

echo "BOOTSTRAP DONE"
