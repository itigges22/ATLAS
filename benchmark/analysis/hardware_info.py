"""
Hardware and software environment information collection.

Collects GPU, CPU, OS, and software version information for
benchmark reproducibility.
"""

import os
import platform
import subprocess
import re
from typing import Dict, Any

from ..models import HardwareInfo
from ..config import config


def run_command(cmd: str, default: str = "") -> str:
    """
    Run a shell command and return output.

    Args:
        cmd: Command to run
        default: Default value if command fails

    Returns:
        Command output or default
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else default
    except Exception:
        return default


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information using whichever vendor SMI tool is present.

    V3.1.1: vendor-aware. Returns vendor + model + VRAM + driver version
    + power draw. Power draw is NVIDIA-only (rocm-smi power reporting
    varies wildly by card and isn't reliable enough to use in benchmark
    metadata).

    Returns:
        Dictionary with model, vendor, vram_gb, driver_version, power_draw_watts
    """
    info = {
        "model": "",
        "vendor": "",
        "vram_gb": 0.0,
        "driver_version": "",
        "power_draw_watts": 0.0
    }

    # Vendor-agnostic detection via tier — keeps the SMI parsing logic
    # in one place (tier.py) rather than duplicated across files.
    try:
        from atlas.commands import tier
        gpus = tier.detect_gpu()
    except Exception:
        gpus = []
    primary = None
    for g in gpus:
        if primary is None or g.vram_gb > primary.vram_gb:
            primary = g
    if primary is not None:
        info["model"] = primary.name
        info["vendor"] = primary.vendor
        info["vram_gb"] = round(primary.vram_gb, 2)

    # Per-vendor extras (driver version, power draw) — keep these inline
    # since benchmark metadata wants per-vendor specifics.
    if info["vendor"] == "nvidia":
        driver = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
        if driver:
            info["driver_version"] = driver.split('\n')[0].strip()
        power = run_command("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits")
        if power:
            try:
                info["power_draw_watts"] = float(power.split('\n')[0].strip())
            except ValueError:
                # best-effort: swallow on failure (caller continues)
                pass
    elif info["vendor"] == "amd":
        # rocm-smi --showdriverversion prints "ROCm Driver Version: 6.x.x"
        out = run_command("rocm-smi --showdriverversion")
        m = re.search(r'Driver Version:\s*([\d.]+)', out)
        if m:
            info["driver_version"] = m.group(1)
    return info


def get_cuda_version() -> str:
    """
    Get CUDA version (NVIDIA only — returns "" on non-NVIDIA hosts).
    """
    # Try nvcc first
    nvcc_version = run_command("nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//'")
    if nvcc_version:
        return nvcc_version

    # Try nvidia-smi
    nvidia_smi_output = run_command("nvidia-smi")
    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', nvidia_smi_output)
    if match:
        return match.group(1)

    return ""


def get_rocm_version() -> str:
    """
    Get ROCm version (AMD only — returns "" on non-AMD hosts).

    Tries hipcc, then rocm-smi --version, then /opt/rocm/.info/version.
    """
    # hipcc — bundled with ROCm dev
    hipcc = run_command("hipcc --version 2>&1 | head -1")
    m = re.search(r'HIP version:\s*([\d.]+)', hipcc)
    if m:
        return m.group(1)
    # rocm-smi --version
    smi = run_command("rocm-smi --version 2>&1 | head -1")
    m = re.search(r'ROCm[\s-]*[Vv]ersion:\s*([\d.]+)', smi) or \
        re.search(r'([\d]+\.[\d]+\.[\d]+)', smi)
    if m:
        return m.group(1)
    # Last resort: /opt/rocm/.info/version on most installs
    version_file = run_command("cat /opt/rocm/.info/version 2>/dev/null")
    if version_file:
        return version_file.strip()
    return ""


def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU information.

    Returns:
        Dictionary with CPU model and core count
    """
    info = {
        "model": platform.processor() or "",
        "cores": os.cpu_count() or 0
    }

    # Try to get more detailed CPU info on Linux
    if platform.system() == "Linux":
        model = run_command("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2")
        if model:
            info["model"] = model.strip()

    return info


def get_memory_info() -> float:
    """
    Get total system RAM in GB.

    Returns:
        RAM in GB
    """
    if platform.system() == "Linux":
        mem_kb = run_command("cat /proc/meminfo | grep MemTotal | awk '{print $2}'")
        if mem_kb:
            try:
                return float(mem_kb) / (1024 * 1024)
            except ValueError:
                pass

    # Fallback - try psutil if available
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    return 0.0


def get_os_info() -> Dict[str, str]:
    """
    Get OS and kernel information.

    Returns:
        Dictionary with OS name and kernel version
    """
    return {
        "os_name": f"{platform.system()} {platform.release()}",
        "kernel_version": platform.version()
    }


def get_k3s_version() -> str:
    """
    Get K3s version.

    Returns:
        K3s version string
    """
    version = run_command("k3s --version | head -1")
    if version:
        match = re.search(r'v[\d.]+', version)
        if match:
            return match.group(0)
    return ""


def get_llama_cpp_version() -> str:
    """
    Get llama.cpp version from running server.

    Returns:
        llama.cpp version or commit hash
    """
    # Try to get from llama-server
    # This would require querying the running server
    # For now, return empty - can be populated from server response
    return ""


def get_model_info() -> Dict[str, str]:
    """
    Get model information from config.

    Returns:
        Dictionary with model name and quantization
    """
    model_name = config.model_name
    quantization = ""

    # Extract quantization from model filename
    # e.g., "Qwen3.5-9B-Q6_K.gguf" -> "Q6_K"
    match = re.search(r'(Q\d+_K(?:_[A-Z])?)', model_name, re.IGNORECASE)
    if match:
        quantization = match.group(1).upper()

    return {
        "name": model_name,
        "quantization": quantization
    }


def collect_hardware_info() -> HardwareInfo:
    """
    Collect all hardware and software information.

    Returns:
        HardwareInfo dataclass with all collected data
    """
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    os_info = get_os_info()
    model_info = get_model_info()

    vendor = gpu_info.get("vendor", "")
    return HardwareInfo(
        gpu_model=gpu_info["model"],
        gpu_vram_gb=gpu_info["vram_gb"],
        gpu_vendor=vendor,
        gpu_driver_version=gpu_info["driver_version"],
        # Populate the vendor-specific compute runtime field; leave the
        # other empty so JSON consumers can switch on vendor cleanly.
        cuda_version=get_cuda_version() if vendor == "nvidia" else "",
        rocm_version=get_rocm_version() if vendor == "amd" else "",
        cpu_model=cpu_info["model"],
        cpu_cores=cpu_info["cores"],
        ram_gb=get_memory_info(),
        os_name=os_info["os_name"],
        kernel_version=os_info["kernel_version"],
        k3s_version=get_k3s_version(),
        llama_cpp_version=get_llama_cpp_version(),
        model_name=model_info["name"],
        model_quantization=model_info["quantization"],
        context_length=0,  # Can be populated from server query
        power_draw_watts=gpu_info["power_draw_watts"]
    )


def hardware_info_to_markdown(info: HardwareInfo) -> str:
    """
    Generate Markdown table of hardware information.

    Args:
        info: HardwareInfo to format

    Returns:
        Formatted Markdown string
    """
    # Show whichever compute-runtime line applies to the detected vendor.
    # On NVIDIA hosts: CUDA line + power draw. On AMD: ROCm line, no power
    # (rocm-smi power reporting is too inconsistent to publish in benchmark
    # metadata). Apple/Intel/unknown: skip both, just show vendor tag.
    if info.gpu_vendor == "nvidia":
        runtime_line = f"- CUDA: {info.cuda_version or 'N/A'}"
        power_line = (f"- Power Draw: {info.power_draw_watts:.0f}W"
                      if info.power_draw_watts else "- Power Draw: N/A")
    elif info.gpu_vendor == "amd":
        runtime_line = f"- ROCm: {info.rocm_version or 'N/A'}"
        power_line = "- Power Draw: N/A (rocm-smi power not reported)"
    else:
        runtime_line = f"- Runtime: N/A (vendor={info.gpu_vendor or 'unknown'})"
        power_line = "- Power Draw: N/A"

    lines = [
        "## Hardware Information",
        "",
        "### GPU",
        f"- Model: {info.gpu_model or 'N/A'}",
        f"- Vendor: {info.gpu_vendor or 'N/A'}",
        f"- VRAM: {info.gpu_vram_gb:.1f} GB" if info.gpu_vram_gb else "- VRAM: N/A",
        f"- Driver: {info.gpu_driver_version or 'N/A'}",
        runtime_line,
        power_line,
        "",
        "### CPU",
        f"- Model: {info.cpu_model or 'N/A'}",
        f"- Cores: {info.cpu_cores or 'N/A'}",
        "",
        "### System",
        f"- RAM: {info.ram_gb:.1f} GB" if info.ram_gb else "- RAM: N/A",
        f"- OS: {info.os_name or 'N/A'}",
        f"- Kernel: {info.kernel_version or 'N/A'}",
        "",
        "### Software",
        f"- K3s: {info.k3s_version or 'N/A'}",
        f"- llama.cpp: {info.llama_cpp_version or 'N/A'}",
        "",
        "### Model",
        f"- Name: {info.model_name or 'N/A'}",
        f"- Quantization: {info.model_quantization or 'N/A'}",
        f"- Context Length: {info.context_length or 'N/A'}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Test hardware info collection
    print("Collecting hardware information...")
    info = collect_hardware_info()
    print(hardware_info_to_markdown(info))
    print("\n--- JSON Output ---")
    print(info.to_dict())
