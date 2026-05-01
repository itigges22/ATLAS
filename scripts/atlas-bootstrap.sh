#!/usr/bin/env bash
#
# atlas-bootstrap.sh — one-shot installer for ATLAS on a fresh Linux host.
#
# Targets the Docker Compose deployment path (the most common). For the
# K3s deployment path use scripts/install.sh instead.
#
# What this does:
#   1. Detects distro (RHEL/Fedora/Rocky/Alma, Ubuntu/Debian).
#   2. Installs Docker Engine + Compose plugin if missing.
#   3. Detects NVIDIA GPU and installs nvidia-container-toolkit if needed.
#   4. Sets vm.overcommit_memory=1 (PC-011 — Redis silent-write killer).
#   5. RHEL-family: enables EPEL, opens firewalld ports, blacklists nouveau.
#   6. Copies .env.example to .env if missing.
#   7. Downloads model GGUFs and Lens weights from HuggingFace.
#   8. `docker compose up -d` and waits for all services healthy.
#   9. Prints a green "ATLAS ready" banner and the next-step command.
#
# Idempotent — safe to re-run. Each step checks "already done" before acting.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/itigges22/ATLAS/main/scripts/atlas-bootstrap.sh | bash
#   # or, from a checkout:
#   ./scripts/atlas-bootstrap.sh
#
# Flags (env vars):
#   ATLAS_BOOTSTRAP_SKIP_DOCKER=1     skip Docker install (already managed)
#   ATLAS_BOOTSTRAP_SKIP_NVIDIA=1     skip GPU/nvidia-container-toolkit
#   ATLAS_BOOTSTRAP_SKIP_MODELS=1     skip model download
#   ATLAS_BOOTSTRAP_SKIP_COMPOSE=1    skip `docker compose up`
#   ATLAS_BOOTSTRAP_NO_SUDO=1         fail instead of attempting sudo
#   ATLAS_REPO_URL=...                clone source if no local repo (default: GitHub)
#   ATLAS_INSTALL_DIR=...             where to clone/install (default: /opt/atlas)
#
# Exit codes:
#   0   success
#   1   user-recoverable error (missing prereq, network failure, etc.)
#   2   unsupported platform
#   3   internal error (file system, permissions)

set -euo pipefail

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Colors only when stdout is a TTY — avoid escape codes in piped output.
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'
    CYAN=$'\033[0;36m'
    NC=$'\033[0m'
else
    BOLD='' DIM='' RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

log_step()  { echo -e "${CYAN}${BOLD}==>${NC} ${BOLD}$*${NC}"; }
log_info()  { echo -e "    $*"; }
log_ok()    { echo -e "    ${GREEN}✓${NC} $*"; }
log_warn()  { echo -e "    ${YELLOW}!${NC} $*"; }
log_err()   { echo -e "    ${RED}✗${NC} $*" >&2; }
log_skip()  { echo -e "    ${DIM}⊘ $*${NC}"; }

die() {
    log_err "$*"
    echo
    echo -e "${RED}${BOLD}Bootstrap failed.${NC} Re-run after addressing the issue above."
    echo -e "${DIM}For help: https://github.com/itigges22/ATLAS/issues${NC}"
    exit 1
}

# ---------------------------------------------------------------------------
# sudo wrapper — uses sudo if we're not root, fails fast if blocked
# ---------------------------------------------------------------------------

if [[ "$(id -u)" == "0" ]]; then
    SUDO=""
elif [[ "${ATLAS_BOOTSTRAP_NO_SUDO:-0}" == "1" ]]; then
    SUDO="false"  # any sudo invocation will exit 1
else
    if ! command -v sudo &>/dev/null; then
        die "Not running as root and 'sudo' is not installed. Install sudo or run as root."
    fi
    SUDO="sudo"
fi

# ---------------------------------------------------------------------------
# Distro detection
# ---------------------------------------------------------------------------

detect_distro() {
    if [[ ! -r /etc/os-release ]]; then
        die "/etc/os-release not found — can't detect distro."
    fi
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO_ID="${ID:-unknown}"
    DISTRO_VERSION_ID="${VERSION_ID:-unknown}"
    DISTRO_LIKE="${ID_LIKE:-}"

    case "$DISTRO_ID" in
        ubuntu|debian)
            DISTRO_FAMILY="debian"
            PKG="apt-get"
            ;;
        rhel|fedora|rocky|almalinux|centos|ol)
            DISTRO_FAMILY="rhel"
            if command -v dnf &>/dev/null; then PKG="dnf"; else PKG="yum"; fi
            ;;
        *)
            # Fall back to ID_LIKE for less common distros (e.g. linuxmint,
            # popos, oraclelinux). If ID_LIKE mentions debian or rhel,
            # treat as that family.
            if [[ "$DISTRO_LIKE" == *debian* || "$DISTRO_LIKE" == *ubuntu* ]]; then
                DISTRO_FAMILY="debian"
                PKG="apt-get"
            elif [[ "$DISTRO_LIKE" == *rhel* || "$DISTRO_LIKE" == *fedora* ]]; then
                DISTRO_FAMILY="rhel"
                if command -v dnf &>/dev/null; then PKG="dnf"; else PKG="yum"; fi
            else
                log_warn "Unknown distro '$DISTRO_ID' (ID_LIKE='$DISTRO_LIKE')."
                log_warn "Bootstrap will attempt the closest match but may fail."
                die "Unsupported distro. Open an issue with your /etc/os-release contents."
            fi
            ;;
    esac
    log_info "Detected: ${BOLD}${DISTRO_ID}${NC} ${DISTRO_VERSION_ID} (${DISTRO_FAMILY} family, pkg=${PKG})"
}

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

detect_gpu() {
    HAS_NVIDIA=0
    GPU_NAME=""
    if command -v lspci &>/dev/null; then
        if lspci 2>/dev/null | grep -qi 'nvidia'; then
            HAS_NVIDIA=1
            GPU_NAME=$(lspci 2>/dev/null | grep -i 'nvidia' | head -1 | sed 's/.*: //')
        fi
    fi
    # nvidia-smi present is also a positive signal even if lspci fails
    if [[ $HAS_NVIDIA -eq 0 ]] && command -v nvidia-smi &>/dev/null; then
        HAS_NVIDIA=1
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    fi

    if [[ $HAS_NVIDIA -eq 1 ]]; then
        log_info "GPU: ${BOLD}${GPU_NAME}${NC}"
    else
        log_warn "No NVIDIA GPU detected. ATLAS can run CPU-only but inference will be very slow."
        log_warn "Set ATLAS_BOOTSTRAP_SKIP_NVIDIA=1 to suppress GPU steps and continue."
        if [[ "${ATLAS_BOOTSTRAP_SKIP_NVIDIA:-0}" != "1" ]]; then
            die "No GPU detected. Re-run with ATLAS_BOOTSTRAP_SKIP_NVIDIA=1 to install CPU-only."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Docker Engine + Compose plugin
# ---------------------------------------------------------------------------

install_docker() {
    log_step "Step 1: Docker Engine"

    if [[ "${ATLAS_BOOTSTRAP_SKIP_DOCKER:-0}" == "1" ]]; then
        log_skip "Skipped (ATLAS_BOOTSTRAP_SKIP_DOCKER=1)"
        return
    fi

    if command -v docker &>/dev/null && docker compose version &>/dev/null; then
        log_ok "Docker + compose plugin already installed ($(docker --version | awk '{print $3}' | tr -d ','))"
        return
    fi

    log_info "Installing Docker via the official convenience script…"
    if ! command -v curl &>/dev/null; then
        if [[ "$DISTRO_FAMILY" == "debian" ]]; then
            $SUDO $PKG update -y >/dev/null
            $SUDO $PKG install -y curl
        else
            $SUDO $PKG install -y curl
        fi
    fi

    # Official Docker convenience script — handles repo setup per distro.
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh || die "Failed to download Docker installer."
    $SUDO sh /tmp/get-docker.sh >/tmp/docker-install.log 2>&1 || {
        log_err "Docker install failed. Last 20 lines of /tmp/docker-install.log:"
        tail -20 /tmp/docker-install.log >&2 || true
        die "Docker installation failed."
    }
    rm -f /tmp/get-docker.sh

    # Make sure compose plugin is present (some distros need it as a separate package)
    if ! docker compose version &>/dev/null; then
        log_info "Installing docker-compose-plugin…"
        if [[ "$DISTRO_FAMILY" == "debian" ]]; then
            $SUDO $PKG install -y docker-compose-plugin >/dev/null
        else
            $SUDO $PKG install -y docker-compose-plugin >/dev/null
        fi
    fi

    # Enable + start the daemon
    $SUDO systemctl enable --now docker >/dev/null 2>&1 || true

    # Add invoking user to docker group so they can run without sudo
    if [[ -n "${SUDO_USER:-}" ]]; then
        $SUDO usermod -aG docker "$SUDO_USER" 2>/dev/null || true
        log_warn "Added $SUDO_USER to the docker group. Log out and back in for it to take effect."
    elif [[ "$(id -u)" != "0" ]]; then
        $SUDO usermod -aG docker "$USER" 2>/dev/null || true
        log_warn "Added $USER to the docker group. Log out and back in for it to take effect."
    fi

    log_ok "Docker installed: $(docker --version | awk '{print $3}' | tr -d ',')"
}

# ---------------------------------------------------------------------------
# Step 2: nvidia-container-toolkit
# ---------------------------------------------------------------------------

install_nvidia_toolkit() {
    log_step "Step 2: NVIDIA Container Toolkit"

    if [[ $HAS_NVIDIA -eq 0 || "${ATLAS_BOOTSTRAP_SKIP_NVIDIA:-0}" == "1" ]]; then
        log_skip "No NVIDIA GPU or skip flag set"
        return
    fi

    # Already installed and working?
    if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        log_ok "nvidia-container-toolkit already configured (Docker can see GPU)"
        return
    fi

    log_info "Installing nvidia-container-toolkit…"
    case "$DISTRO_FAMILY" in
        debian)
            # Add NVIDIA's repo
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null \
                || die "Failed to add NVIDIA GPG key."
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
            $SUDO $PKG update -y >/dev/null
            $SUDO $PKG install -y nvidia-container-toolkit >/dev/null \
                || die "nvidia-container-toolkit install failed."
            ;;
        rhel)
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
                | $SUDO tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
            $SUDO $PKG install -y nvidia-container-toolkit >/dev/null \
                || die "nvidia-container-toolkit install failed."
            ;;
    esac

    log_info "Configuring Docker runtime for NVIDIA…"
    $SUDO nvidia-ctk runtime configure --runtime=docker >/dev/null \
        || die "nvidia-ctk runtime configure failed."
    $SUDO systemctl restart docker
    sleep 2

    # Verify
    if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        log_ok "nvidia-container-toolkit verified — Docker can see GPU"
    else
        log_warn "nvidia-container-toolkit installed but GPU not visible to Docker yet."
        log_warn "Try: $SUDO systemctl restart docker  # then re-run this script"
    fi
}

# ---------------------------------------------------------------------------
# Step 3: Kernel sysctl (PC-011)
# ---------------------------------------------------------------------------

configure_sysctl() {
    log_step "Step 3: Kernel parameters (PC-011 — Redis overcommit)"

    local current
    current=$(sysctl -n vm.overcommit_memory 2>/dev/null || echo "0")
    if [[ "$current" == "1" ]]; then
        log_ok "vm.overcommit_memory=1 already set"
    else
        log_info "Setting vm.overcommit_memory=1 (was $current)…"
        $SUDO sysctl -w vm.overcommit_memory=1 >/dev/null
        # Persist via /etc/sysctl.d so it survives reboot
        if ! $SUDO grep -q '^vm.overcommit_memory' /etc/sysctl.d/99-atlas.conf 2>/dev/null; then
            echo "vm.overcommit_memory=1" | $SUDO tee /etc/sysctl.d/99-atlas.conf >/dev/null
        fi
        log_ok "vm.overcommit_memory=1 (persisted to /etc/sysctl.d/99-atlas.conf)"
    fi
}

# ---------------------------------------------------------------------------
# Step 4: RHEL-family extras (EPEL, firewalld, nouveau)
# ---------------------------------------------------------------------------

configure_rhel_extras() {
    [[ "$DISTRO_FAMILY" != "rhel" ]] && return

    log_step "Step 4: RHEL-family extras"

    # EPEL — many of our dependencies come from EPEL.
    if ! rpm -q epel-release &>/dev/null; then
        log_info "Installing EPEL…"
        $SUDO $PKG install -y epel-release >/dev/null 2>&1 || \
            log_warn "EPEL install failed (may not be needed on Fedora)."
    else
        log_ok "EPEL already installed"
    fi

    # firewalld — open compose ports if firewalld is running
    if systemctl is-active --quiet firewalld 2>/dev/null; then
        log_info "firewalld is active — opening ATLAS ports (8090, 8099, 8070, 30820)…"
        for port in 8090 8099 8070 30820; do
            $SUDO firewall-cmd --permanent --add-port=${port}/tcp >/dev/null 2>&1 || true
        done
        $SUDO firewall-cmd --reload >/dev/null 2>&1 || true
        log_ok "firewalld ports opened (atlas-proxy/lens/v3/sandbox)"
    else
        log_skip "firewalld not active — no ports to open"
    fi

    # nouveau driver conflict check (informational only — we don't blacklist
    # without explicit user opt-in because reboots are disruptive).
    if [[ $HAS_NVIDIA -eq 1 ]] && lsmod 2>/dev/null | grep -q nouveau; then
        log_warn "nouveau driver is loaded but you have an NVIDIA GPU."
        log_warn "If GPU performance is poor, blacklist nouveau and reboot:"
        log_warn "  echo 'blacklist nouveau' | $SUDO tee /etc/modprobe.d/blacklist-nouveau.conf"
        log_warn "  $SUDO dracut --force  # then reboot"
    fi
}

# ---------------------------------------------------------------------------
# Step 5: Repo + .env
# ---------------------------------------------------------------------------

ensure_repo_and_env() {
    log_step "Step 5: Repo & .env"

    # If we're not in a checkout, clone to ATLAS_INSTALL_DIR
    if [[ ! -f "./docker-compose.yml" || ! -d "./atlas-proxy" ]]; then
        local install_dir="${ATLAS_INSTALL_DIR:-/opt/atlas}"
        local repo_url="${ATLAS_REPO_URL:-https://github.com/itigges22/ATLAS.git}"

        log_info "Not in a checkout. Cloning $repo_url to $install_dir…"
        if [[ -d "$install_dir/.git" ]]; then
            log_info "Existing checkout at $install_dir — pulling latest"
            (cd "$install_dir" && git pull --ff-only) || die "git pull failed in $install_dir"
        else
            $SUDO mkdir -p "$(dirname "$install_dir")"
            $SUDO chown "$(id -u):$(id -g)" "$(dirname "$install_dir")"
            git clone "$repo_url" "$install_dir" || die "git clone failed"
        fi
        cd "$install_dir"
        log_ok "Working in $install_dir"
    else
        log_ok "Already in an ATLAS checkout: $(pwd)"
    fi

    # .env
    if [[ -f .env ]]; then
        log_ok ".env exists ($(wc -l < .env) lines)"
    else
        if [[ ! -f .env.example ]]; then
            die ".env.example not found — broken checkout?"
        fi
        cp .env.example .env
        log_ok "Created .env from .env.example (edit ATLAS_MODELS_DIR if needed)"
    fi
}

# ---------------------------------------------------------------------------
# Step 6: Models
# ---------------------------------------------------------------------------

download_models() {
    log_step "Step 6: Model weights (Qwen3.5-9B + Lens)"

    if [[ "${ATLAS_BOOTSTRAP_SKIP_MODELS:-0}" == "1" ]]; then
        log_skip "Skipped (ATLAS_BOOTSTRAP_SKIP_MODELS=1)"
        return
    fi

    if [[ ! -x "./scripts/download-models.sh" ]]; then
        die "scripts/download-models.sh not found or not executable."
    fi

    log_info "Calling scripts/download-models.sh (this can take 10-30 min on first run)…"
    if ./scripts/download-models.sh 2>&1 | tee /tmp/atlas-models.log | grep -E '^\[INFO\]|\[WARN\]|\[ERROR\]'; then
        log_ok "Model download complete. See /tmp/atlas-models.log for details."
    else
        log_err "Model download failed. Last 20 lines of /tmp/atlas-models.log:"
        tail -20 /tmp/atlas-models.log >&2 || true
        die "Model download failed — check HuggingFace credentials or network."
    fi
}

# ---------------------------------------------------------------------------
# Step 7: docker compose up
# ---------------------------------------------------------------------------

start_compose() {
    log_step "Step 7: Starting services (docker compose up -d)"

    if [[ "${ATLAS_BOOTSTRAP_SKIP_COMPOSE:-0}" == "1" ]]; then
        log_skip "Skipped (ATLAS_BOOTSTRAP_SKIP_COMPOSE=1)"
        return
    fi

    # Use current user's docker socket access; if they're not in the docker
    # group yet (newly added in step 1), use sudo for this run.
    local DC="docker compose"
    if ! docker info &>/dev/null; then
        DC="sudo docker compose"
        log_warn "Using sudo for docker compose (user not in docker group yet — log out/in to fix)"
    fi

    # Compose v2 with `image:` set in compose.yml does `pull missing` by
    # default — first run downloads the GHCR images (PC-052), subsequent
    # runs reuse the local cache. To force a rebuild from source instead
    # of pulling, run `docker compose build` before this.
    log_info "Pulling images from GHCR (first run only) and starting containers…"
    $DC up -d 2>&1 | tee /tmp/atlas-compose.log | tail -20
    log_ok "Containers started. See /tmp/atlas-compose.log for details."
}

# ---------------------------------------------------------------------------
# Step 8: Wait for healthy
# ---------------------------------------------------------------------------

wait_for_healthy() {
    log_step "Step 8: Waiting for services to be healthy"

    if [[ "${ATLAS_BOOTSTRAP_SKIP_COMPOSE:-0}" == "1" ]]; then
        log_skip "Skipped (compose was skipped)"
        return
    fi

    local DC="docker compose"
    docker info &>/dev/null || DC="sudo docker compose"

    local services=(redis llama-server geometric-lens v3-service sandbox atlas-proxy)
    local timeout=300  # 5 min — first start can be slow while llama-server warms
    local elapsed=0
    local interval=5

    while [[ $elapsed -lt $timeout ]]; do
        local healthy=0
        local total=${#services[@]}
        for s in "${services[@]}"; do
            local state
            state=$($DC ps --format '{{.Service}} {{.State}} {{.Health}}' 2>/dev/null \
                    | awk -v s="$s" '$1==s {print $2"/"$3; exit}')
            # Healthy = "running/healthy" or "running/" (no healthcheck = running counts)
            if [[ "$state" == running/healthy || "$state" == running/ ]]; then
                healthy=$((healthy + 1))
            fi
        done
        printf "\r    ${DIM}%d/%d services healthy (elapsed: %ds / %ds)${NC}" \
            "$healthy" "$total" "$elapsed" "$timeout"
        if [[ $healthy -eq $total ]]; then
            echo
            log_ok "All $total services healthy"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo
    log_err "Timeout: not all services healthy after ${timeout}s"
    log_err "Run '$DC ps' to see current state, or '$DC logs <service>' for details."
    return 1
}

# ---------------------------------------------------------------------------
# Step 9: Ready banner
# ---------------------------------------------------------------------------

print_ready_banner() {
    echo
    echo -e "${GREEN}${BOLD}╭─────────────────────────────────────────────╮${NC}"
    echo -e "${GREEN}${BOLD}│${NC}  ${BOLD}ATLAS is ready.${NC}                            ${GREEN}${BOLD}│${NC}"
    echo -e "${GREEN}${BOLD}╰─────────────────────────────────────────────╯${NC}"
    echo
    echo -e "  ${BOLD}Quick start${NC}"
    echo -e "    ${DIM}# In any project directory you want to code in:${NC}"
    echo -e "    ${CYAN}cd /path/to/your/project${NC}"
    echo -e "    ${CYAN}atlas${NC}                   ${DIM}# launches the Aider front-end${NC}"
    echo
    echo -e "  ${BOLD}Diagnostics${NC}"
    echo -e "    ${CYAN}atlas doctor${NC}            ${DIM}# verify all services are healthy${NC}"
    echo -e "    ${CYAN}docker compose ps${NC}       ${DIM}# raw container status${NC}"
    echo -e "    ${CYAN}docker compose logs -f${NC}  ${DIM}# stream logs across all services${NC}"
    echo
    echo -e "  ${BOLD}Docs${NC}: https://github.com/itigges22/ATLAS/tree/main/docs"
    echo -e "  ${BOLD}Issues${NC}: https://github.com/itigges22/ATLAS/issues"
    echo
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo
    echo -e "${BOLD}ATLAS bootstrap${NC} — installing on $(uname -s) $(uname -r)"
    echo -e "${DIM}Started at $(date)${NC}"
    echo

    log_step "Detecting system"
    detect_distro
    detect_gpu
    echo

    install_docker
    echo
    install_nvidia_toolkit
    echo
    configure_sysctl
    echo
    configure_rhel_extras
    echo
    ensure_repo_and_env
    echo
    download_models
    echo
    start_compose
    echo
    wait_for_healthy || die "Service health check failed."

    print_ready_banner
}

main "$@"
