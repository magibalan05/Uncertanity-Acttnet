import os
import sys
import subprocess

def check_environment():
    print("=" * 60)
    print(" Checking Installed Python Runtimes & GPU/CUDA Status")
    print("=" * 60)

    # Check PyLauncher available py versions
    print("[1] Installed Python Environments:")
    try:
        res = subprocess.run(["py", "-0p"], capture_output=True, text=True)
        print(res.stdout if res.stdout else "Could not list via py launcher")
    except Exception as e:
        print(f"Error calling py launcher: {e}")

    # Check CUDA / NVIDIA Driver on System
    print("\n[2] Checking NVIDIA Drivers & CUDA System GPU:")
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if res.returncode == 0:
            print("NVIDIA Driver Detected Successfully!")
            lines = res.stdout.split('\n')
            for line in lines[:12]:
                print(line)
        else:
            print("nvidia-smi returned non-zero exit code or NVIDIA GPU driver is not installed.")
    except Exception as e:
        print("nvidia-smi command not found in PATH.")

    # Check Current active python details
    print("\n[3] Current Active Python Details:")
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version}")

if __name__ == "__main__":
    check_environment()
