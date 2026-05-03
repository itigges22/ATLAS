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
# Target user (who owns the install)
# ---------------------------------------------------------------------------
# The script supports both `curl | bash` (run as regular user, sudo prompts
# for elevation) and `curl | sudo bash` (run as root, $SUDO_USER set to the
# invoking user). In both cases we want the install tree at /opt/atlas to
# be owned by the *human* using the system, not by root — otherwise every
# later `atlas` invocation, `git pull`, or `docker compose build` would
# trip permission-denied. Resolve the target once and use it everywhere.
if [[ "$(id -u)" == "0" ]]; then
    if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
        TARGET_USER="$SUDO_USER"
        TARGET_UID=$(id -u "$SUDO_USER" 2>/dev/null || echo 0)
        TARGET_GID=$(id -g "$SUDO_USER" 2>/dev/null || echo 0)
    else
        # Real root login (no sudo) — own as root and accept the consequence.
        TARGET_USER="root"
        TARGET_UID=0
        TARGET_GID=0
    fi
else
    TARGET_USER="$USER"
    TARGET_UID=$(id -u)
    TARGET_GID=$(id -g)
fi

# ---------------------------------------------------------------------------
# Docker invocation prefix
# ---------------------------------------------------------------------------
# After install_docker adds the user to the docker group, the CURRENT shell
# still doesn't have group membership (it's only refreshed by re-login or
# `newgrp docker`). Every subsequent `docker ...` call in this script has
# to know whether to use sudo. Set DOCKER_PREFIX once after the install,
# then reuse it everywhere — no per-step heuristics. PC-051 follow-up.
DOCKER_PREFIX=""
detect_docker_prefix() {
    if docker info &>/dev/null; then
        DOCKER_PREFIX=""
    elif [[ -n "$SUDO" ]] && $SUDO -n docker info &>/dev/null 2>&1; then
        # `sudo -n` works (no password prompt expected); use sudo silently.
        DOCKER_PREFIX="$SUDO"
    elif [[ -n "$SUDO" ]]; then
        # sudo would prompt — still set prefix; user already approved sudo above.
        DOCKER_PREFIX="$SUDO"
    fi
}

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
                log_warn "$DISTRO_ID isn't on the supported list but ID_LIKE matches Debian — proceeding with apt-get."
            elif [[ "$DISTRO_LIKE" == *rhel* || "$DISTRO_LIKE" == *fedora* ]]; then
                DISTRO_FAMILY="rhel"
                if command -v dnf &>/dev/null; then PKG="dnf"; else PKG="yum"; fi
                log_warn "$DISTRO_ID isn't on the supported list but ID_LIKE matches RHEL — proceeding with $PKG."
            else
                log_warn "Unknown distro '$DISTRO_ID' (ID_LIKE='$DISTRO_LIKE')."
                log_warn "Supported: Ubuntu 20.04+, Debian 11+, RHEL 9+, Rocky 9+, AlmaLinux 9+, Fedora 38+, CentOS Stream 9+"
                die "Unsupported distro. Open an issue with your /etc/os-release contents."
            fi
            ;;
    esac
    log_info "Detected: ${BOLD}${DISTRO_ID}${NC} ${DISTRO_VERSION_ID} (${DISTRO_FAMILY} family, pkg=${PKG})"
}

print_supported_distros() {
    cat <<EOF
    Supported distributions:
      - Ubuntu 20.04+ / Debian 11+ (apt-get)
      - RHEL 9+ / Rocky 9+ / AlmaLinux 9+ / CentOS Stream 9+ (dnf)
      - Fedora 38+ (dnf)
      - Oracle Linux 9+ (dnf)
    Other distros with ID_LIKE matching one of the above are accepted with a warning.
EOF
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

install_nvidia_driver_libs() {
    # Called when libnvidia-ml.so.1 isn't in the ld.so cache. Installs the
    # NVIDIA userspace driver libraries (libnvidia-ml, libcuda, etc) that
    # nvidia-container-cli needs to bind into containers. The kernel module
    # alone (which makes `nvidia-smi` work on the host) isn't enough.
    #
    # Per-distro logic:
    #   - RHEL 9:        add CUDA repo + enable codeready-builder via
    #                    subscription-manager + EPEL, then dnf module install
    #                    nvidia-driver:open-dkms (Blackwell 50xx requires open).
    #   - Rocky/Alma 9:  same but with `dnf config-manager --set-enabled crb`
    #                    instead of subscription-manager.
    #   - Fedora:        rpmfusion-nonfree + akmod-nvidia-open is the standard
    #                    path; we add CUDA repo as the simpler universal route.
    #   - Ubuntu/Debian: matched libnvidia-compute-NN package from the running
    #                    driver's major version.
    case "$DISTRO_FAMILY" in
        rhel)
            local cuda_repo="/etc/yum.repos.d/cuda-rhel9.repo"
            if [[ ! -f "$cuda_repo" ]]; then
                log_info "Adding NVIDIA CUDA repo for RHEL 9…"
                $SUDO dnf config-manager --add-repo \
                    "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo" \
                    >/dev/null 2>&1 \
                    || { log_err "failed to add NVIDIA CUDA repo"; return 1; }
            else
                log_ok "CUDA repo already present"
            fi

            # CodeReady Builder (RHEL with subscription) or CRB (rebuilds)
            # provides the dkms / kernel-devel packages the open-dkms
            # module needs. Try both — only one applies to a given host.
            if [[ "$DISTRO_ID" == "rhel" ]] && command -v subscription-manager &>/dev/null; then
                log_info "Enabling CodeReady Builder repo…"
                $SUDO subscription-manager repos --enable=codeready-builder-for-rhel-9-x86_64-rpms \
                    >/dev/null 2>&1 \
                    || log_warn "couldn't enable codeready-builder (subscription not active?)"
            else
                log_info "Enabling CRB repo…"
                $SUDO dnf config-manager --set-enabled crb >/dev/null 2>&1 \
                    || log_warn "couldn't enable crb (already enabled or unavailable)"
            fi

            # Make sure EPEL is present (dkms lives there).
            if ! rpm -q epel-release &>/dev/null; then
                $SUDO dnf install -y \
                    "https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm" \
                    >/dev/null 2>&1 || log_warn "EPEL install failed (continuing)"
            fi

            # The open-dkms module is REQUIRED for Blackwell GPUs (RTX
            # 5060/70/80/90). Older GPUs work with either open or proprietary;
            # default to open since it's the future and works for both.
            log_info "Installing nvidia-driver:open-dkms (this can take 5-10 min)…"
            if $SUDO dnf module install -y nvidia-driver:open-dkms 2>&1 | tee /tmp/atlas-nvidia-install.log; then
                log_ok "nvidia-driver:open-dkms installed"
                return 0
            else
                log_err "nvidia-driver:open-dkms install failed. Last 20 lines:"
                tail -20 /tmp/atlas-nvidia-install.log >&2 || true
                return 1
            fi
            ;;
        debian)
            local drv_major
            drv_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
                        | head -1 | cut -d. -f1)
            if [[ -z "$drv_major" || "$drv_major" == "0" ]]; then
                log_err "nvidia-smi didn't return a driver version — install the NVIDIA driver first."
                log_err "  $SUDO $PKG install -y nvidia-driver-XXX  (where XXX is your driver branch, e.g. 570)"
                return 1
            fi
            log_info "Installing libnvidia-compute-$drv_major to match driver $drv_major…"
            $SUDO $PKG install -y "libnvidia-compute-$drv_major" \
                || { log_err "libnvidia-compute-$drv_major install failed"; return 1; }
            log_ok "libnvidia-compute-$drv_major installed"
            return 0
            ;;
        *)
            log_err "Don't know how to install NVIDIA driver libs on $DISTRO_FAMILY."
            log_err "Manual fix: install your distro's libnvidia-ml.so.1 provider, then re-run."
            return 1
            ;;
    esac
}

install_nvidia_toolkit() {
    log_step "Step 2: NVIDIA Container Toolkit"

    if [[ $HAS_NVIDIA -eq 0 || "${ATLAS_BOOTSTRAP_SKIP_NVIDIA:-0}" == "1" ]]; then
        log_skip "No NVIDIA GPU or skip flag set"
        return
    fi

    # Already installed and working?
    if $DOCKER_PREFIX docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
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

    # Refresh ld.so cache. On RHEL after a fresh nvidia-driver install the
    # libnvidia-ml.so.1 symlink may exist but not be in /etc/ld.so.cache,
    # which makes nvidia-container-cli fail with "load library failed".
    $SUDO ldconfig 2>/dev/null || true

    # Sanity-check the host has the userspace driver libs the toolkit needs.
    # nvidia-smi working on the host means the kernel module is loaded, but
    # the userspace libs (libnvidia-ml, libcuda) come from a separate package
    # on RHEL and may be missing on minimal installs / fresh CUDA setups.
    if ! $SUDO ldconfig -p 2>/dev/null | grep -q 'libnvidia-ml\.so\.1'; then
        log_warn "libnvidia-ml.so.1 not in ld.so cache — installing missing driver libs."
        install_nvidia_driver_libs || die "could not install NVIDIA driver libraries; see hints above."
        $SUDO ldconfig 2>/dev/null || true
    fi

    $SUDO systemctl restart docker
    sleep 3

    # Verify (using DOCKER_PREFIX since user may not be in docker group yet).
    local verify_log=/tmp/atlas-nvidia-verify.log
    if $DOCKER_PREFIX docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >"$verify_log" 2>&1; then
        log_ok "nvidia-container-toolkit verified — Docker can see GPU"
        return
    fi
    sleep 5
    if $DOCKER_PREFIX docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >"$verify_log" 2>&1; then
        log_ok "nvidia-container-toolkit verified — Docker can see GPU"
        return
    fi

    # Verify failed twice. Show the actual error so the user knows what's wrong.
    log_err "nvidia-container-toolkit installed but Docker can't talk to the GPU."
    log_err "Container error:"
    grep -E 'error|failed|cannot' "$verify_log" | head -5 | sed 's/^/      /' >&2

    # Diagnostic: try CDI mode (newer style — replaces legacy mode for
    # newer drivers). Some setups need CDI generated explicitly.
    if command -v nvidia-ctk &>/dev/null; then
        log_info "Trying CDI mode as a fallback…"
        $SUDO mkdir -p /etc/cdi
        if $SUDO nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1; then
            if $DOCKER_PREFIX docker run --rm --device=nvidia.com/gpu=all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
                log_ok "CDI mode works (legacy mode does not). Compose may need updating to use CDI."
                log_info "  See: https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/docs/cdi.md"
                return
            fi
        fi
    fi

    die "GPU not visible to Docker. Check the container error above and the libnvidia-ml.so.1 hint."
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
    log_step "Step 5: Repo, .env, and ATLAS CLI"

    # If we're not in a checkout, clone to ATLAS_INSTALL_DIR
    if [[ ! -f "./docker-compose.yml" || ! -d "./proxy" ]]; then
        local install_dir="${ATLAS_INSTALL_DIR:-/opt/atlas}"
        local repo_url="${ATLAS_REPO_URL:-https://github.com/itigges22/ATLAS.git}"

        log_info "Not in a checkout. Cloning $repo_url to $install_dir…"
        if [[ -d "$install_dir/.git" ]]; then
            log_info "Existing checkout at $install_dir — pulling latest"
            (cd "$install_dir" && git pull --ff-only) || die "git pull failed in $install_dir"
        else
            $SUDO mkdir -p "$install_dir"
            # Pre-chown the dir so the clone goes in user-owned, then
            # re-chown after to catch any leftover root-owned bits (rare,
            # but safer than assuming git inherits perms cleanly).
            $SUDO chown -R "$TARGET_UID:$TARGET_GID" "$install_dir"
            if [[ "$(id -u)" == "0" && "$TARGET_USER" != "root" ]]; then
                # Run the clone as the target user when bootstrapping via
                # sudo, so .git/config, hooks, etc. get the right ownership.
                $SUDO -u "$TARGET_USER" git clone "$repo_url" "$install_dir" \
                    || die "git clone failed"
            else
                git clone "$repo_url" "$install_dir" || die "git clone failed"
            fi
            $SUDO chown -R "$TARGET_UID:$TARGET_GID" "$install_dir"
        fi
        cd "$install_dir"
        log_ok "Working in $install_dir (owner: $TARGET_USER)"
    else
        log_ok "Already in an ATLAS checkout: $(pwd)"
    fi

    # If the install dir isn't owned by the target user (e.g. user pre-cloned
    # with sudo, or a previous bootstrap left root-owned droppings), take
    # ownership now so downstream steps can write here. Idempotent.
    local owner
    owner=$(stat -c '%u' . 2>/dev/null || echo "$TARGET_UID")
    if [[ "$owner" != "$TARGET_UID" ]]; then
        log_info "Install dir is owned by uid=$owner; chowning to $TARGET_USER…"
        $SUDO chown -R "$TARGET_UID:$TARGET_GID" . \
            || die "chown failed; can't proceed without write access here."
        log_ok "Install dir now owned by $TARGET_USER"
    fi

    # Pin ATLAS_INSTALL_DIR for downstream steps (run_doctor, etc) — pwd
    # is wherever ensure_repo_and_env left us.
    export ATLAS_INSTALL_DIR
    ATLAS_INSTALL_DIR="$(pwd)"

    # .env
    if [[ -f .env ]]; then
        # Make sure .env actually has the keys download-models.sh and
        # docker compose need. A user-supplied .env (e.g. with only an
        # ATLAS_IMAGE_TAG override) will fail downstream without them.
        local missing=()
        for key in ATLAS_MODELS_DIR ATLAS_MODEL_FILE ATLAS_MODEL_NAME ATLAS_CTX_SIZE; do
            grep -q "^${key}=" .env || missing+=("$key")
        done
        if [[ ${#missing[@]} -gt 0 ]]; then
            log_warn ".env is missing required keys: ${missing[*]}"
            log_info "Appending defaults from .env.example…"
            for key in "${missing[@]}"; do
                grep "^${key}=" .env.example >> .env || true
            done
            log_ok ".env patched with $((${#missing[@]})) missing keys"
        else
            log_ok ".env exists with all required keys"
        fi
    else
        if [[ ! -f .env.example ]]; then
            die ".env.example not found — broken checkout?"
        fi
        cp .env.example .env
        log_ok "Created .env from .env.example (edit ATLAS_MODELS_DIR if needed)"
    fi

    install_atlas_cli
}

install_atlas_cli() {
    # The Python CLI (`atlas`, `atlas tui`, `atlas doctor`, `atlas tier`)
    # lives in the `atlas/` Python package. Without `pip install -e .`
    # the user has the repo on disk but no `atlas` command on PATH —
    # they hit "command not found" right after install completes.
    if ! command -v python3 &>/dev/null; then
        log_warn "python3 not found — skipping CLI install. \`atlas\` command will be unavailable."
        log_warn "  Install Python 3.9+ then run: pip install --user -e ."
        return
    fi

    # pip is sometimes a separate package on RHEL minimal installs.
    if ! python3 -m pip --version &>/dev/null; then
        log_info "python3-pip missing — installing…"
        case "$DISTRO_FAMILY" in
            debian) $SUDO $PKG install -y python3-pip >/dev/null 2>&1 ;;
            rhel)   $SUDO $PKG install -y python3-pip >/dev/null 2>&1 ;;
        esac
        if ! python3 -m pip --version &>/dev/null; then
            log_warn "couldn't install pip — skipping CLI install."
            log_warn "  Install pip then run: pip install --user -e ."
            return
        fi
    fi

    # Run pip as the target user so it lands in their ~/.local/bin, not root's.
    local runner=""
    if [[ "$(id -u)" == "0" && "$TARGET_USER" != "root" ]]; then
        runner="$SUDO -u $TARGET_USER -H"
    fi
    log_info "Installing ATLAS Python CLI (pip install --user -e .)…"
    if $runner python3 -m pip install --user -e . --quiet 2>&1 | tee /tmp/atlas-pip.log; then
        log_ok "ATLAS CLI installed"
    else
        log_warn "pip install failed (exit $?). Last 10 lines: /tmp/atlas-pip.log"
        tail -10 /tmp/atlas-pip.log >&2 || true
        log_warn "  Recovery: cd $ATLAS_INSTALL_DIR && pip install --user -e ."
        return
    fi

    # Ensure ~/.local/bin is on PATH for future shells. pip puts the
    # `atlas` script there; without it on PATH, `atlas tui` says
    # "command not found" until the user manually adds it. Append to the
    # target user's .bashrc only if it's not already present.
    local target_home
    if [[ "$TARGET_USER" == "root" ]]; then
        target_home="/root"
    else
        target_home=$(getent passwd "$TARGET_USER" | cut -d: -f6 2>/dev/null)
        [[ -z "$target_home" ]] && target_home="$HOME"
    fi
    local bashrc="$target_home/.bashrc"
    if [[ -f "$bashrc" ]] && ! grep -q '\.local/bin' "$bashrc" 2>/dev/null; then
        if [[ "$(id -u)" == "0" && "$TARGET_USER" != "root" ]]; then
            $SUDO -u "$TARGET_USER" sh -c "echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> $bashrc"
        else
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$bashrc"
        fi
        log_info "Added ~/.local/bin to PATH in $bashrc"
        log_warn "  Run \`source ~/.bashrc\` or open a new shell for \`atlas\` to be on PATH."
    fi

    # Quick check: can we resolve `atlas` for the target user?
    if [[ -x "$target_home/.local/bin/atlas" ]]; then
        log_ok "atlas binary at: $target_home/.local/bin/atlas"
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
    log_info "Progress is shown live below; full output also saved to /tmp/atlas-models.log."
    echo
    # Run as the target user so files end up owned by the human, not root.
    local runner=""
    if [[ "$(id -u)" == "0" && "$TARGET_USER" != "root" ]]; then
        runner="$SUDO -u $TARGET_USER"
    fi
    # Stream output live (no grep filter — that hid curl's progress bar
    # and any error messages that didn't match the [INFO]/[WARN]/[ERROR]
    # pattern). `tee` preserves the log without breaking line buffering.
    set +e
    $runner ./scripts/download-models.sh 2>&1 | tee /tmp/atlas-models.log
    local rc=${PIPESTATUS[0]}
    set -e
    echo
    if [[ $rc -eq 0 ]]; then
        log_ok "Model download complete (log: /tmp/atlas-models.log)"
    else
        log_err "Model download failed (exit $rc)."
        die "Model download failed — check the live output above, /tmp/atlas-models.log, disk space, or network."
    fi

    # Lens weights are a separate fetch — the main script's --lens
    # subcommand drops them in geometric-lens/geometric_lens/models/.
    # Without these, the lens service starts but returns neutral scores.
    log_info "Fetching Geometric Lens weights…"
    set +e
    $runner ./scripts/download-models.sh --lens 2>&1 | tee -a /tmp/atlas-models.log
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -eq 0 ]]; then
        log_ok "Lens weights ready"
    else
        log_warn "Lens weight fetch failed (exit $rc) — service will run with neutral scores."
        log_warn "Recovery: ./scripts/download-models.sh --lens"
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

    # Use the same DOCKER_PREFIX we set up at the top — handles "user just
    # added to docker group, current shell doesn't know yet" transparently.
    local DC="$DOCKER_PREFIX docker compose"
    if [[ -n "$DOCKER_PREFIX" ]]; then
        log_warn "Using sudo for docker compose (user not in docker group yet — log out/in to fix)"
    fi

    # Pull images first as a separate step so the user can see the layer-by-
    # layer download progress (5 images, ~3GB total on first run). Without
    # this split, `up -d` would silently pull during the up call and only
    # surface output if it fails. PC-052.
    log_info "Pulling images from GHCR (first run: ~3GB across 5 services)…"
    echo
    set +e
    $DC pull 2>&1 | tee /tmp/atlas-compose-pull.log
    local rc=${PIPESTATUS[0]}
    set -e
    echo
    if [[ $rc -ne 0 ]]; then
        log_err "docker compose pull failed (exit $rc). Log: /tmp/atlas-compose-pull.log"
        log_err "Common causes: GHCR rate-limit, network, or auth (private package)."
        die "Image pull failed — see live output above."
    fi
    log_ok "All images pulled."

    log_info "Starting containers…"
    echo
    set +e
    $DC up -d 2>&1 | tee /tmp/atlas-compose.log
    rc=${PIPESTATUS[0]}
    set -e
    echo
    if [[ $rc -ne 0 ]]; then
        log_err "docker compose up failed (exit $rc). Log: /tmp/atlas-compose.log"
        die "Compose start failed — see live output above."
    fi
    log_ok "Containers started (log: /tmp/atlas-compose.log)"
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

    local DC="$DOCKER_PREFIX docker compose"

    local services=(redis llama-server geometric-lens v3-service sandbox atlas-proxy)
    local timeout=300  # 5 min — first start can be slow while llama-server warms
    local elapsed=0
    local interval=5

    log_info "Waiting up to ${timeout}s for all services to report healthy."
    log_info "Tip: open another terminal and run \`$DC logs -f llama-server\` to watch model load."
    echo

    local last_status=""
    while [[ $elapsed -lt $timeout ]]; do
        local healthy=0
        local total=${#services[@]}
        local status_line=""
        for s in "${services[@]}"; do
            local state
            state=$($DC ps --format '{{.Service}} {{.State}} {{.Health}}' 2>/dev/null \
                    | awk -v s="$s" '$1==s {print $2"/"$3; exit}')
            if [[ "$state" == running/healthy || "$state" == running/ ]]; then
                healthy=$((healthy + 1))
                status_line+="✓"
            elif [[ "$state" == running/starting ]]; then
                status_line+="⠿"
            elif [[ "$state" == running/unhealthy ]]; then
                status_line+="✗"
            else
                status_line+="·"
            fi
        done
        # Print status line on change OR every 30s, so the user sees life signs
        # without flooding the screen on every 5s tick.
        if [[ "$status_line" != "$last_status" || $((elapsed % 30)) -eq 0 ]]; then
            printf "    ${DIM}[%s] %d/%d healthy after %ds (services: %s)${NC}\n" \
                "$status_line" "$healthy" "$total" "$elapsed" "${services[*]}"
            last_status="$status_line"
        fi
        if [[ $healthy -eq $total ]]; then
            log_ok "All $total services healthy after ${elapsed}s"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo
    log_err "Timeout: not all services healthy after ${timeout}s"
    log_err "Current state:"
    $DC ps 2>&1 | sed 's/^/      /' >&2
    log_err "Inspect a stuck service: $DC logs <service-name>"
    return 1
}

# ---------------------------------------------------------------------------
# Step 8.5: atlas doctor (PC-053) — surface non-container issues
# ---------------------------------------------------------------------------
# Health-poll only checks `docker compose ps` for healthy. Doctor adds the
# "is the install actually correct" layer: lens weights, model file size,
# overcommit, image-tag skew. Run --quick (skip e2e smoke — wait until
# user explicitly invokes for that) and only surface a summary line.

run_doctor() {
    log_step "Step 8.5: atlas doctor (sanity sweep)"

    if [[ "${ATLAS_BOOTSTRAP_SKIP_COMPOSE:-0}" == "1" ]]; then
        log_skip "Skipped (compose was skipped, no stack to check)"
        return
    fi

    if ! command -v python3 &>/dev/null; then
        log_warn "python3 not found — skipping doctor (doctor is Python)"
        return
    fi

    # Doctor lives in the repo. cd there, run --quick, capture exit code.
    # ATLAS_INSTALL_DIR is set by ensure_repo_and_env; fall back to pwd
    # in the unlikely case it wasn't (e.g. compose was skipped earlier).
    local install_dir="${ATLAS_INSTALL_DIR:-$(pwd)}"
    local doctor_out doctor_rc
    set +e
    doctor_out=$(cd "$install_dir" && python3 -m atlas.cli.commands.doctor --quick --no-color 2>&1)
    doctor_rc=$?
    set -e

    if [[ $doctor_rc -ne 0 ]]; then
        log_err "atlas doctor reported failures:"
        echo "$doctor_out" | grep -E "FAIL|WARN" | sed 's/^/      /'
        log_info "Run \`atlas doctor -v\` after install completes for detail."
        return  # Don't block bootstrap; failures are install-time signals
    fi

    # Exit 0 may still include warnings — surface them inline so users
    # don't miss things like vm.overcommit_memory=0 (PC-011).
    local warn_lines
    warn_lines=$(echo "$doctor_out" | grep "WARN" || true)
    if [[ -n "$warn_lines" ]]; then
        log_warn "atlas doctor passed with warnings:"
        echo "$warn_lines" | sed 's/^/      /'
        log_info "Run \`atlas doctor -v\` for the recommended fix."
    else
        log_ok "atlas doctor passed (run \`atlas doctor\` for full check)"
    fi
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
    echo -e "    ${CYAN}atlas${NC}                   ${DIM}# launches the TUI chat UI${NC}"
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
    if [[ "$(id -u)" == "0" ]]; then
        if [[ "$TARGET_USER" == "root" ]]; then
            echo -e "${DIM}Running as: root (no SUDO_USER detected — install will be root-owned)${NC}"
        else
            echo -e "${DIM}Running as: root (via sudo from $TARGET_USER — install owned by $TARGET_USER)${NC}"
        fi
    else
        echo -e "${DIM}Running as: $TARGET_USER (will sudo as needed)${NC}"
    fi
    echo

    print_supported_distros
    echo

    log_step "Detecting system"
    detect_distro
    detect_gpu
    echo
    log_info "Install location: ${BOLD}${ATLAS_INSTALL_DIR:-/opt/atlas}${NC} (override with ATLAS_INSTALL_DIR=...)"
    log_info "  Why /opt/atlas? It's the standard prefix for system-wide third-party"
    log_info "  software (FHS), survives \$HOME purges, and lets multiple users on the"
    log_info "  same box share one install. Set ATLAS_INSTALL_DIR=\$HOME/atlas if you'd"
    log_info "  rather it land in your home dir."
    echo

    install_docker
    echo
    # After Docker is installed (or confirmed present), pin down whether
    # subsequent `docker` calls in this script need sudo. The user may have
    # been added to the docker group in install_docker but their CURRENT
    # shell doesn't see that yet — this prefix is what makes the rest of
    # the script work without "permission denied on /var/run/docker.sock".
    detect_docker_prefix
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
    echo
    run_doctor

    print_ready_banner
}

main "$@"
