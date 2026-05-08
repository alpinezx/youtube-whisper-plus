"""
GPU detection helper — called by launch.bat before install.
Prints: GPU_MODE=cuda or GPU_MODE=cpu
Also prints driver and CUDA version if detected.
"""
import subprocess
import sys
import re

def detect():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi failed")

        output = result.stdout

        # Parse driver version
        driver = re.search(r"Driver Version:\s*([\d.]+)", output)
        driver_ver = driver.group(1) if driver else "unknown"

        # Parse CUDA version
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", output)
        cuda_ver = cuda.group(1) if cuda else "unknown"

        print(f"  [OK] NVIDIA GPU detected")
        print(f"       Driver  : {driver_ver}")
        print(f"       CUDA    : {cuda_ver}")
        print(f"       PyTorch : cu126 (compatible with CUDA 12.1+)")
        print(f"GPU_MODE=cuda")

    except Exception:
        print(f"  [--] No NVIDIA GPU detected - running in CPU mode")
        print(f"       Transcription will be slower without a GPU.")
        print(f"GPU_MODE=cpu")

if __name__ == "__main__":
    detect()
