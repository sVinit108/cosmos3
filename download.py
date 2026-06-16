"""
=============================================================
  NVIDIA Cosmos3-Nano — Model Downloader
  Saves the model to a LOCAL folder, NOT the HuggingFace cache.
=============================================================

  What this script does:
    1. Installs / checks all required packages
    2. Logs you into HuggingFace (you only need to do this once)
    3. Downloads the full Cosmos3-Nano model (~16B params) to
       the LOCAL path you specify in MODEL_LOCAL_DIR below.

  Prerequisites:
    - Python 3.10+
    - NVIDIA GPU with CUDA (≥ 24 GB VRAM recommended)
    - A HuggingFace account (free) — the model is gated, you
      must accept the license at:
      https://huggingface.co/nvidia/Cosmos3-Nano

  Usage:
    1. Set MODEL_LOCAL_DIR to wherever you want the files.
    2. Run:  python 1_download_cosmos3_nano.py
    3. When prompted, paste your HuggingFace token
       (get it from https://huggingface.co/settings/tokens).
=============================================================
"""

import subprocess
import sys
import os

# ─────────────────────────────────────────────
#  ✏️  CONFIGURE THIS — where to save the model
# ─────────────────────────────────────────────
MODEL_LOCAL_DIR = "./cosmos3-nano-model"   # change to any absolute path you like
MODEL_REPO_ID   = "nvidia/Cosmos3-Nano"
# ─────────────────────────────────────────────


def install_packages():
    """Make sure every required package is present."""
    packages = [
        "huggingface_hub[cli]",
        #"torch",
        "diffusers @ git+https://github.com/huggingface/diffusers.git",  # main branch (Cosmos3 not in stable yet)
        "transformers",
        "accelerate",
        "cosmos_guardrail",   # NVIDIA's mandatory safety checker
    ]
    print("\n[1/3] Installing / verifying packages...\n")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade"] + packages # "--quiet"] + packages
    )
    print("      ✅ Packages ready.\n")


def login_to_huggingface():
    """Interactive HF login so the gated model can be downloaded."""
    from huggingface_hub import HfFolder, login

    if HfFolder.get_token():
        print("[2/3] Already logged in to HuggingFace. ✅\n")
    else:
        print("[2/3] You need a HuggingFace token to download this gated model.")
        print("      Get yours at: https://huggingface.co/settings/tokens\n")
        login()   # opens an interactive prompt
        print("      ✅ Logged in.\n")


def download_model():
    """
    Download every file in the repo to MODEL_LOCAL_DIR.
    We explicitly set HF_HOME to a dummy value so nothing
    goes to the default cache — all files land in MODEL_LOCAL_DIR.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)

    print(f"[3/3] Downloading {MODEL_REPO_ID} → {os.path.abspath(MODEL_LOCAL_DIR)}")
    print("      This is a large model (~16B params). It will take a while.\n")

    # snapshot_download with local_dir bypasses the HF cache completely.
    path = snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=MODEL_LOCAL_DIR,
        local_dir_use_symlinks=False,   # real copies, not symlinks into cache
        ignore_patterns=["*.msgpack", "flax_model*"],  # skip non-PyTorch weights
    )

    print(f"\n✅ Download complete!  Model stored at:\n   {os.path.abspath(path)}\n")
    print("   ➡️  Open  2_run_cosmos3_nano.py  and set:")
    print(f'      MODEL_LOCAL_DIR = "{os.path.abspath(path)}"')
    print("   then run it to generate images / videos.\n")


if __name__ == "__main__":
    # install_packages()
    # login_to_huggingface()
    download_model()