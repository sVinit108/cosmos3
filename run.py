"""
=============================================================
  NVIDIA Cosmos3-Nano — Inference Runner
  Change your prompt → run → get images / videos / sound.
=============================================================

  What this model can do (all covered here):
    MODE 1 → Text  → Image           (single frame)
    MODE 2 → Text  → Video
    MODE 3 → Image → Video           (image-conditioned)
    MODE 4 → Text  → Video + Sound   (audio muxed in)

  How to use:
    1. Make sure you ran  1_download_cosmos3_nano.py  first.
    2. Set MODEL_LOCAL_DIR to the folder where you saved the model.
    3. Choose a MODE (1-4) in the CONFIG section below.
    4. Edit the PROMPT (and NEGATIVE_PROMPT if you like).
    5. Run:  python 2_run_cosmos3_nano.py
=============================================================
"""

import torch
import os

# ╔══════════════════════════════════════════════════════════════╗
# ║                   ✏️  YOUR CONFIG — EDIT HERE               ║
# ╠══════════════════════════════════════════════════════════════╣

# Path where you saved the model in step 1
MODEL_LOCAL_DIR = "./cosmos3-nano-model"

# ── Choose what to generate ──────────────────────────────────
#   1 = Text → Image
#   2 = Text → Video
#   3 = Image → Video
#   4 = Text → Video with Sound
MODE = 2

# ── Your creative prompt ─────────────────────────────────────
#   TIP: For video, use rich narrative paragraphs (~100-200 words).
#   For image, a single detailed sentence works well.
PROMPT = (
    "A cinematic aerial shot of a futuristic city at dusk. Skyscrapers are lit with "
    "glowing blue and purple neon lights reflecting off a calm river below. Flying "
    "vehicles drift silently between towers. The sky transitions from deep orange to "
    "violet, with scattered clouds catching the last rays of sunlight."
)

# ── Optional: strengthen quality by telling the model what to avoid ──
# (Most useful for video modes. Leave as "" to skip.)
NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)

# ── Video settings (only used in MODE 2 / 3 / 4) ─────────────
NUM_FRAMES  = 241     # 49 ≈ 2 s  |  121 ≈ 5 s  |  189 ≈ 7.9 s @ 24 fps
HEIGHT      = 480    # 480 or 720 (720 needs more VRAM)
WIDTH       = 848    # 848 or 1280
FPS         = 24.0

# ── Image conditioning path (only for MODE 3) ─────────────────
# Local file path OR a public URL.
CONDITIONING_IMAGE = "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"

# ── Output files ──────────────────────────────────────────────
OUTPUT_DIR        = "./cosmos3_outputs"
OUTPUT_IMAGE_FILE = "cosmos3_image.jpg"
OUTPUT_VIDEO_FILE = "cosmos3_video.mp4"
OUTPUT_SOUND_FILE = "cosmos3_video_with_sound.mp4"

# ── Memory optimisation ───────────────────────────────────────
# Set True if you have < 24 GB VRAM — trades speed for memory.
ENABLE_LAYERWISE_OFFLOAD = False

# ╚══════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────
#  Internal helpers (no need to touch these)
# ─────────────────────────────────────────────

def check_model_dir():
    if not os.path.isdir(MODEL_LOCAL_DIR):
        raise FileNotFoundError(
            f"\n❌  Model folder not found: {MODEL_LOCAL_DIR}\n"
            "   Please run  1_download_cosmos3_nano.py  first, then update\n"
            "   MODEL_LOCAL_DIR in this file to match where you saved it."
        )


def load_pipeline():
    from diffusers import Cosmos3OmniPipeline

    print(f"\n[Loading] Reading model from  {MODEL_LOCAL_DIR} …")
    print("          (first load takes a minute — weights are being mapped to GPU)\n")

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        enable_safety_checker=False,   # set False only for local testing
    )

    # Enable layerwise CPU offload to fit on smaller GPUs
    if ENABLE_LAYERWISE_OFFLOAD:
        kwargs["enable_layerwise_offload"] = True
        print("  ⚡ Layerwise offload enabled (slower but uses less VRAM)\n")

    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_LOCAL_DIR, **kwargs)
    return pipe


def run_text_to_image(pipe):
    print(f"[MODE 1]  Text → Image\n  Prompt: {PROMPT[:80]}…\n")
    result = pipe(
        prompt=PROMPT,
        num_frames=1,
        height=HEIGHT,
        width=WIDTH,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE_FILE)
    result.video[0].save(out, format="JPEG", quality=90)
    print(f"\n✅ Image saved → {os.path.abspath(out)}\n")


def run_text_to_video(pipe):
    from diffusers.utils import export_to_video

    print(f"[MODE 2]  Text → Video  ({NUM_FRAMES} frames @ {FPS} fps)\n  Prompt: {PROMPT[:80]}…\n")
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT or None,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_FILE)
    export_to_video(result.video, out, fps=int(FPS), macro_block_size=1)
    print(f"\n✅ Video saved → {os.path.abspath(out)}\n")


def run_image_to_video(pipe):
    from diffusers.utils import export_to_video, load_image

    print(f"[MODE 3]  Image → Video  ({NUM_FRAMES} frames)\n  Image:  {CONDITIONING_IMAGE}\n  Prompt: {PROMPT[:80]}…\n")
    image = load_image(CONDITIONING_IMAGE)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT or None,
        image=image,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_FILE)
    export_to_video(result.video, out, fps=int(FPS), macro_block_size=1)
    print(f"\n✅ Video saved → {os.path.abspath(out)}\n")


def run_text_to_video_with_sound(pipe):
    from diffusers.utils import encode_video

    print(f"[MODE 4]  Text → Video + Sound  ({NUM_FRAMES} frames)\n  Prompt: {PROMPT[:80]}…\n")

    # Append an audio cue to the end of your prompt for best results.
    sound_prompt = PROMPT + (
        " Audio description: ambient environmental sounds matching the scene, "
        "natural atmosphere, high fidelity audio."
    )

    result = pipe(
        prompt=sound_prompt,
        negative_prompt=NEGATIVE_PROMPT or None,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        enable_sound=True,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, OUTPUT_SOUND_FILE)
    encode_video(
        result.video,
        fps=int(FPS),
        audio=result.sound,
        audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
        output_path=out,
    )
    print(f"\n✅ Video with sound saved → {os.path.abspath(out)}\n")


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NVIDIA Cosmos3-Nano — Inference Runner")
    print("=" * 60)

    # Validate setup
    check_model_dir()

    if not torch.cuda.is_available():
        raise EnvironmentError(
            "\n❌  No CUDA GPU detected.  Cosmos3-Nano requires an NVIDIA GPU.\n"
        )

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {vram_gb:.1f} GB")
    if vram_gb < 20 and not ENABLE_LAYERWISE_OFFLOAD:
        print("\n  ⚠️  Your GPU has less than 20 GB VRAM.")
        print("      Consider setting  ENABLE_LAYERWISE_OFFLOAD = True  above.\n")

    pipe = load_pipeline()

    mode_map = {
        1: run_text_to_image,
        2: run_text_to_video,
        3: run_image_to_video,
        4: run_text_to_video_with_sound,
    }

    if MODE not in mode_map:
        raise ValueError(f"MODE must be 1, 2, 3, or 4.  Got: {MODE}")

    mode_map[MODE](pipe)


if __name__ == "__main__":
    main()