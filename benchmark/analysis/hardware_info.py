"""
Hardware and software environment information collection.

Collects GPU, CPU, OS, and software version information for
benchmark reproducibility.
"""

import os
import platform
import subprocess
import re
from typing import Dict, Any, Optional

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
    Get GPU information using nvidia-smi.

    Returns:
        Dictionary with GPU model, VRAM, driver version, and power draw
    """
    info = {
        "model": "",
        "vram_gb": 0.0,
        "driver_version": "",
        "power_draw_watts": 0.0
    }

    # Try nvidia-smi
    nvidia_smi = run_command("which nvidia-smi")
    if not nvidia_smi:
        return info

    # Get GPU name
    name = run_command("nvidia-smi --query-gpu=name --format=csv,noheader,nounits")
    if name:
        info["model"] = name.split('\n')[0].strip()

    # Get VRAM
    vram = run_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    if vram:
        try:
            # Convert MiB to GB
            info["vram_gb"] = float(vram.split('\n')[0].strip()) / 1024
        except ValueError:
            pass

    # Get driver version
    driver = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
    if driver:
        info["driver_version"] = driver.split('\n')[0].strip()

    # Get current power draw
    power = run_command("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits")
    if power:
        try:
            info["power_draw_watts"] = float(power.split('\n')[0].strip())
        except ValueError:
            pass

    return info


def get_cuda_version() -> str:
    """
    Get CUDA version.

    Returns:
        CUDA version string
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
    # e.g., "Qwen3-14B-Q4_K_M.gguf" -> "Q4_K_M"
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

    return HardwareInfo(
        gpu_model=gpu_info["model"],
        gpu_vram_gb=gpu_info["vram_gb"],
        gpu_driver_version=gpu_info["driver_version"],
        cuda_version=get_cuda_version(),
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
    lines = [
        "## Hardware Information",
        "",
        "### GPU",
        f"- Model: {info.gpu_model or 'N/A'}",
        f"- VRAM: {info.gpu_vram_gb:.1f} GB" if info.gpu_vram_gb else "- VRAM: N/A",
        f"- Driver: {info.gpu_driver_version or 'N/A'}",
        f"- CUDA: {info.cuda_version or 'N/A'}",
        f"- Power Draw: {info.power_draw_watts:.0f}W" if info.power_draw_watts else "- Power Draw: N/A",
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
