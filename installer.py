#!/usr/bin/env python3
# Chat-Linux-Gguf installer

import os
import json
import platform
import subprocess
import sys
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Constants
APP_NAME = "Chat-Linux-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
LLAMA_DIR = BASE_DIR / "llama.cpp"
BUILD_DIR = LLAMA_DIR / "build"
BIN_DIR = BASE_DIR / "data" / "llama-cpp"
CONFIG_PATH = BASE_DIR / "data" / "persistent.json"

# Minimum requirements
MIN_CUDA_VERSION = 11.0
MIN_DRIVER_VERSION = 450.80
MIN_VRAM = 2048  # 2GB VRAM
MIN_COMPUTE_CAPABILITY = 6.0

REQUIREMENTS = [
    "gradio>=4.25.0",
    "requests==2.31.0",
    "pyperclip",
    "yake",
    "psutil",
    "duckduckgo-search",
    "newspaper3k",
    "langchain-community==0.3.18",
    "pygments==2.17.2",
    "lxml_html_clean",
    "llama-cpp-python==0.2.23",
]

CONFIG_TEMPLATE = """{
    "model_settings": {
        "model_dir": "models",
        "model_name": "",
        "context_size": 8192,
        "temperature": 0.66,
        "repeat_penalty": 1.1,
        "mmap": false,
        "mlock": false,
        "n_batch": %d,
        "max_history_slots": 12,
        "max_attach_slots": 6,
        "session_log_height": 500
    },
    "backend_config": {
        "available_gpus": %s,
        "selected_gpu": %d,
        "cuda_version": "CUDA %s",
        "compute_capability": "%s"
    }
}"""

class InstallerError(Exception):
    """Custom installer exception"""
    pass

def log_message(message: str, level: str = "INFO") -> None:
    """Log message"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def run_command(cmd: list, cwd: Optional[Path] = None, timeout: int = 300, 
                env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """Execute command"""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"

def get_cuda_compute_capabilities() -> List[str]:
    """Fetch GPU capabilities"""
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        capabilities = list(set(line.strip() for line in smi_output.split('\n') if line.strip()))
        return capabilities
    except Exception as e:
        log_message(f"Capability detection failed: {e}", "WARNING")
        time.sleep(1)
        return []

def get_cuda_architecture_flags() -> Tuple[List[str], str]:
    """Generate CUDA flags"""
    capabilities = get_cuda_compute_capabilities()
    if not capabilities:
        log_message("Using default architectures", "WARNING")
        time.sleep(1)
        arch_string = "60;61;70;75;80;86;89;90"
        return [f"-DCMAKE_CUDA_ARCHITECTURES={arch_string}"], arch_string
    architectures = sorted(set(cap.replace('.', '') for cap in capabilities if len(cap.replace('.', '')) == 2))
    arch_string = ";".join(architectures) if architectures else "60;61;70;75;80;86;89;90"
    log_message(f"Detected architectures: {arch_string}")
    time.sleep(1)
    return [f"-DCMAKE_CUDA_ARCHITECTURES={arch_string}"], arch_string

def check_unified_memory_support(compute_cap: str) -> bool:
    """Verify memory support"""
    try:
        return float(compute_cap) >= MIN_COMPUTE_CAPABILITY
    except (ValueError, TypeError):
        return False

def check_cuda_compatibility(cuda_version: str, gpus: List[Dict[str, any]]) -> bool:
    """Check CUDA compatibility"""
    cuda_major = float(cuda_version.split('.')[0])
    for gpu in gpus:
        compute_cap = float(gpu["compute_capability"])
        if cuda_major >= 12 and compute_cap < 6.0:
            log_message(f"CUDA {cuda_version} incompatible", "WARNING")
            time.sleep(1)
            return False
        if cuda_major < 12 and compute_cap >= 8.0:
            log_message(f"CUDA {cuda_version} suboptimal", "WARNING")
            time.sleep(1)
    return True

def select_gpu(gpus: List[Dict[str, any]]) -> int:
    """Choose GPU with unified memory support"""
    log_message("Detected GPUs:")
    for gpu in gpus:
        unified = "✓UM" if gpu["unified_memory_capable"] else "✗UM"
        log_message(f"[{gpu['index']}] {gpu['name']} {unified} VRAM: {gpu['vram']}MB")
    time.sleep(1)
    
    suitable_gpus = [g for g in gpus if g["unified_memory_capable"] and g["vram"] >= MIN_VRAM]
    if not suitable_gpus:
        log_message("No suitable GPUs", "ERROR")
        time.sleep(3)
        raise InstallerError("No GPUs with unified memory support")
    
    if len(suitable_gpus) == 1:
        log_message(f"Auto-selected GPU: {suitable_gpus[0]['index']}")
        time.sleep(1)
        return suitable_gpus[0]['index']
    
    if sys.stdin.isatty():
        while True:
            try:
                choice = input(f"Select GPU (0-{len(gpus)-1}, default 0): ").strip()
                choice = int(choice) if choice else 0
                if any(g['index'] == choice for g in suitable_gpus):
                    log_message(f"Selected GPU: {choice}")
                    time.sleep(1)
                    return choice
                log_message(f"Invalid GPU index: {choice}", "ERROR")
                time.sleep(3)
            except ValueError:
                log_message("Invalid input", "ERROR")
                time.sleep(3)
    else:
        best_gpu = max(suitable_gpus, key=lambda x: x["vram"])
        log_message(f"Auto-selected GPU: {best_gpu['index']}")
        time.sleep(1)
        return best_gpu['index']

def check_system() -> Tuple[bool, Dict[str, any]]:
    """Check system requirements"""
    system_info = {
        "cuda_version": None,
        "driver_version": None,
        "gpus": [],
        "selected_gpu": 0,
        "unified_memory_support": False,
        "cuda_architectures": []
    }
    log_message("Verifying system...")
    time.sleep(1)
    if platform.system() != "Linux":
        log_message("Linux required", "ERROR")
        time.sleep(3)
        raise InstallerError("Linux required")
    try:
        with open("/etc/os-release") as f:
            os_info = f.read()
        if "ubuntu" not in os_info.lower():
            log_message("Ubuntu recommended", "WARNING")
            time.sleep(1)
    except Exception:
        log_message("OS check failed", "WARNING")
        time.sleep(1)
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,driver_version,memory.total,name,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        if not smi_output:
            log_message("No GPUs found", "ERROR")
            time.sleep(3)
            raise InstallerError("No GPUs found")
        gpus = []
        driver_version = None
        for line in smi_output.split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_index, driver_ver, vram, gpu_name, compute_cap = parts
                    if driver_version is None:
                        driver_version = driver_ver
                    gpus.append({
                        "index": int(gpu_index),
                        "vram": int(vram),
                        "name": gpu_name,
                        "compute_capability": compute_cap,
                        "unified_memory_capable": check_unified_memory_support(compute_cap)
                    })
        if not gpus:
            log_message("No valid GPUs", "ERROR")
            time.sleep(3)
            raise InstallerError("No valid GPUs")
        driver_num = float('.'.join(driver_version.split('.')[:2]))
        if driver_num < MIN_DRIVER_VERSION:
            log_message(f"Driver {MIN_DRIVER_VERSION}+ needed", "ERROR")
            time.sleep(3)
            raise InstallerError("Driver version too low")
        system_info.update({
            "driver_version": driver_version,
            "gpus": gpus,
            "unified_memory_support": any(gpu["unified_memory_capable"] for gpu in gpus)
        })
    except Exception:
        log_message("Driver check failed", "ERROR")
        time.sleep(3)
        raise InstallerError("Driver issue")
    try:
        cuda_bin = "/usr/local/cuda/bin/nvcc"
        if not os.path.exists(cuda_bin):
            log_message("CUDA missing", "ERROR")
            time.sleep(3)
            raise InstallerError("CUDA missing")
        nvcc = subprocess.check_output([cuda_bin, "--version"], text=True)
        match = re.search(r"release (\d+\.\d+)", nvcc)
        if not match:
            log_message("CUDA version error", "ERROR")
            time.sleep(3)
            raise InstallerError("CUDA version error")
        cuda_version = float(match.group(1))
        if cuda_version < MIN_CUDA_VERSION:
            log_message(f"CUDA {MIN_CUDA_VERSION}+ needed", "ERROR")
            time.sleep(3)
            raise InstallerError("CUDA too old")
        system_info["cuda_version"] = match.group(1)
        if not check_cuda_compatibility(system_info["cuda_version"], gpus):
            log_message("CUDA-GPU mismatch", "ERROR")
            time.sleep(3)
            raise InstallerError("CUDA-GPU mismatch")
    except Exception:
        log_message("CUDA check failed", "ERROR")
        time.sleep(3)
        raise InstallerError("CUDA issue")
    _, arch_string = get_cuda_architecture_flags()
    system_info["cuda_architectures"] = arch_string
    suitable_gpus = [gpu for gpu in gpus if gpu["vram"] >= MIN_VRAM]
    if not suitable_gpus:
        log_message("No sufficient VRAM", "ERROR")
        time.sleep(3)
        raise InstallerError("Insufficient VRAM")
    system_info["selected_gpu"] = select_gpu(suitable_gpus)
    return True, system_info

def install_system_deps() -> bool:
    """Install dependencies"""
    log_message("Installing dependencies...")
    time.sleep(1)
    deps = [
        "git", "cmake", "build-essential", "pkg-config",
        "libssl-dev", "zlib1g-dev", "libbz2-dev",
        "libreadline-dev", "libsqlite3-dev", "wget",
        "curl", "llvm", "libncurses5-dev", "xz-utils",
        "tk-dev", "libxml2-dev", "libxmlsec1-dev",
        "libffi-dev", "liblzma-dev", "python3-dev"
    ]
    success, output = run_command(["sudo", "apt", "update"], timeout=120)
    if not success:
        log_message(f"Update failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("Update failed")
    success, output = run_command(["sudo", "apt", "install", "-y"] + deps, timeout=300)
    if not success:
        log_message(f"Install failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("Install failed")
    time.sleep(1)
    return True

def setup_llama_cpp() -> bool:
    """Setup llama.cpp"""
    log_message("Setting up llama.cpp...")
    time.sleep(1)
    if LLAMA_DIR.exists():
        shutil.rmtree(LLAMA_DIR)
    success, output = run_command([
        "git", "clone", "--depth", "1",
        "https://github.com/ggerganov/llama.cpp.git",
        str(LLAMA_DIR)
    ], timeout=180)
    if not success:
        log_message(f"Clone failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("Clone failed")
    time.sleep(1)
    return True

def compile_llama_cpp(cuda_version: str, arch_string: str) -> bool:
    """Compile llama.cpp"""
    log_message("Compiling llama.cpp...")
    time.sleep(1)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)
    arch_flags, _ = get_cuda_architecture_flags()
    cmake_flags = [
        "-DLLAMA_CUDA=ON",
        "-DLLAMA_CUDA_UNIFIED_MEMORY=ON",
        "-DLLAMA_CUDA_F16=ON",
        "-DCMAKE_BUILD_TYPE=Release"
    ] + arch_flags
    cuda_env = os.environ.copy()
    cuda_env.update({"CUDACXX": "/usr/local/cuda/bin/nvcc"})
    success, output = run_command(
        ["cmake", ".."] + cmake_flags,
        cwd=BUILD_DIR,
        timeout=180,
        env=cuda_env
    )
    if not success:
        log_message(f"CMake failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("CMake failed")
    success, output = run_command(
        ["make", "-j", str(min(os.cpu_count() or 4, 8)), "main"],
        cwd=BUILD_DIR,
        timeout=900,
        env=cuda_env
    )
    if not success:
        log_message(f"Make failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("Make failed")
    possible_locations = [BUILD_DIR / "bin" / "main", BUILD_DIR / "main"]
    main_binary = next((loc for loc in possible_locations if loc.exists()), None)
    if not main_binary:
        log_message("Binary missing", "ERROR")
        time.sleep(3)
        raise InstallerError("Binary missing")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(main_binary, BIN_DIR / "main")
    os.chmod(BIN_DIR / "main", 0o755)
    success, output = run_command([str(BIN_DIR / "main"), "--help"], timeout=10)
    if not success:
        log_message(f"Binary test failed: {output}", "WARNING")
        time.sleep(1)
    time.sleep(1)
    return True

def setup_python_env(cuda_version: str, arch_string: str) -> bool:
    """Setup Python env"""
    log_message("Creating Python env...")
    time.sleep(1)
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)
    success, output = run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
    if not success:
        log_message(f"Venv failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("Venv failed")
    pip = str(VENV_DIR / "bin" / "pip")
    success, output = run_command([pip, "install", "--upgrade", "pip", "wheel", "setuptools"], timeout=120)
    if not success:
        log_message(f"Pip failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("Pip failed")
    success, output = run_command([pip, "install"] + REQUIREMENTS, timeout=600)
    if not success:
        log_message(f"Packages failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("Packages failed")
    env = os.environ.copy()
    env.update({
        "LLAMA_CUDA": "1",
        "CMAKE_ARGS": f"-DLLAMA_CUDA=ON -DLLAMA_CUDA_UNIFIED_MEMORY=ON -DCMAKE_CUDA_ARCHITECTURES={arch_string}",
        "FORCE_CMAKE": "1",
        "CUDACXX": "/usr/local/cuda/bin/nvcc"
    })
    success, output = run_command([
        pip, "install", "--force-reinstall", "--no-cache-dir", "llama-cpp-python==0.2.23"
    ], timeout=1200, env=env)
    if not success:
        log_message(f"llama-cpp failed: {output}", "ERROR")
        time.sleep(3)
        raise InstallerError("llama-cpp failed")
    time.sleep(1)
    return True

def create_config(system_info: Dict[str, any]) -> None:
    """Generate config file"""
    print_status("Creating config...", None)
    time.sleep(1)
    gpu = next(g for g in system_info["gpus"] if g["index"] == system_info["selected_gpu"])
    vram_mb = gpu["vram"]
    if vram_mb >= 12288:
        n_batch = 8192
    elif vram_mb >= 8192:
        n_batch = 4096
    elif vram_mb >= 4096:
        n_batch = 2048
    else:
        n_batch = 1024

    for dir_path in [BASE_DIR / "data", BASE_DIR / "data" / "temp", 
                     BASE_DIR / "data" / "history", BASE_DIR / "data" / "vectors"]:
        dir_path.mkdir(parents=True, exist_ok=True)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        f.write(CONFIG_TEMPLATE % (
            n_batch,
            json.dumps(system_info["gpus"]),
            system_info["selected_gpu"],
            system_info["cuda_version"],
            gpu["compute_capability"],
            vram_mb
        ))

    if not CONFIG_PATH.exists():
        print_status("Config creation failed", False)
        time.sleep(3)
        raise InstallerError("Config creation failed")

    try:
        with open(CONFIG_PATH, "r") as f:
            json.load(f)
    except json.JSONDecodeError:
        print_status("Invalid JSON config", False)
        time.sleep(3)
        raise InstallerError("Invalid JSON config")
    time.sleep(1)

def print_system_summary(system_info: Dict[str, any]) -> None:
    """Display system summary"""
    log_message("Installation Summary:")
    log_message(f"CUDA: {system_info['cuda_version']}")
    log_message(f"Driver: {system_info['driver_version']}")
    log_message(f"GPU: {system_info['selected_gpu']}")
    time.sleep(1)

def cleanup_on_failure():
    """Remove failed setup"""
    log_message("Cleaning up...", "WARNING")
    time.sleep(1)
    for path in [VENV_DIR, LLAMA_DIR, BUILD_DIR, BIN_DIR]:
        if path.exists():
            shutil.rmtree(path)

def main():
    """Run installer"""
    try:
        log_message(f"Installing {APP_NAME}")
        time.sleep(1)
        if sys.stdin.isatty():
            response = input("Continue? (y/N): ").strip().lower()
            if response != 'y':
                sys.exit(0)
        sys_ok, system_info = check_system()
        print_system_summary(system_info)
        install_system_deps()
        setup_llama_cpp()
        compile_llama_cpp(system_info['cuda_version'], system_info['cuda_architectures'])
        setup_python_env(system_info['cuda_version'], system_info['cuda_architectures'])
        create_config(system_info)
        log_message("Installation successful!")
        log_message(f"Run: {VENV_DIR}/bin/python launcher.py")
    except InstallerError as e:
        log_message(f"Failed: {e}", "ERROR")
        cleanup_on_failure()
        sys.exit(1)
    except Exception as e:
        log_message(f"Error: {e}", "ERROR")
        cleanup_on_failure()
        sys.exit(1)

if __name__ == "__main__":
    main()