# ReelFrame 🎬⚡

### Local 4K AI Video & Image Upscaler
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Author](https://img.shields.io/badge/Author-Raliq%20Hidayat%20BM3-purple.svg)](https://github.com/kouji999)
[![GPU](https://img.shields.io/badge/GPU-RTX%203050%2F4060%2F4090-orange.svg)](#hardware-requirements)

**ReelFrame** is an ultra-fast, local AI super-resolution workstation designed for creators, video editors, and social media producers. Upscale vertical reels, shorts, TikToks, and cinematic footage to pristine 4K using **Real-ESRGAN** and **NVIDIA NVENC** hardware acceleration.

Built from the ground up with a **zero-disk-scratch streaming pipeline**, **4GB VRAM safe tiling (RTX 3050 Laptop friendly)**, **batch queueing**, and a modern **interactive Web GUI**.

---

```
                       ┌──────────────────────────────┐
                       │   Input Video / Images       │
                       │ (Reels, Shorts, 1080p, 720p) │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │ Threaded Pipe Decoder (ffmpeg) │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ GPU AI Super-Resolution (Real-ESRGAN fp16)        │
             │   • Smart Tiling: 0 OOM on 4GB VRAM Cards        │
             │   • Social Presets: 9:16 Reels 4K / 16:9 Cinema  │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │ Hardware NVENC Encoder (HEVC)  │
                      │   • Bit-perfect copied audio   │
                      │   • FastStart MP4 container    │
                      └───────────────┬────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   Pristine 4K UHD Master     │
                       └──────────────────────────────┘
```

---

## 🌟 Highlights & Features

- **🎨 Modern Dark Web GUI**: Interactive web dashboard at `http://127.0.0.1:7860` with drag & drop uploads, model picker, live progress, FPS counter, and download manager.
- **🛡️ 4GB VRAM Safe Mode**: Custom overlapping tile algorithm (`--tile 512`) lets you upscale 1080p to 4K even on budget laptop GPUs like the RTX 3050 without CUDA OOM crashes.
- **📱 Creator & Social Presets**:
  - `reels-4k`: 9:16 vertical 4K master (2160×3840) optimized for Instagram Reels & TikTok.
  - `cinema-4k`: 16:9 standard 4K (3840×2160) for YouTube and production masters.
  - `photo-sharp`: Ultra-clean single-image enhancement for portraits and thumbnails.
- **🖼️ Video & Photo Support**: Upscales videos (`.mp4`, `.mov`, `.mkv`, `.webm`) and images (`.png`, `.jpg`, `.webp`) in one unified engine.
- **⚡ In-Memory Pipe Streaming**: Frames stream directly between decoder, GPU, and NVENC in RAM — no gigabytes of temporary PNGs filling up your hard drive.
- **🔊 100% Audio Fidelity**: Original audio streams and sync are preserved byte-for-byte.
- **🪟 1-Click Windows Launchers**: Double-click `run_gui.bat` or `setup.bat` to run instantly.

---

## 💻 Hardware Requirements

| Specification | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GPU 4GB VRAM (GeForce RTX 3050 Laptop+), CUDA 11.8+ | RTX 3060 / 4060 / 4080 / 4090 (8GB+ VRAM) |
| **System RAM** | 8 GB | 16 GB+ |
| **OS** | Windows 10 / Windows 11 (64-bit), Linux | Windows 11 64-bit |
| **Encoders** | NVIDIA NVENC (`hevc_nvenc` / `h264_nvenc` / `av1_nvenc`) | Automatic CPU fallback supported |

---

## 🚀 Quick Start

### Option 1: One-Click Web GUI (Easiest)
1. Run `setup.bat` (first time only)
2. Double-click `run_gui.bat`
3. Your browser will automatically open `http://127.0.0.1:7860`!

### Option 2: CLI Usage

```powershell
# Upscale a vertical reel to 4K (2x upscale, HEVC NVENC, CQ 20)
python upscale.py reel.mp4

# Upscale a photo / thumbnail
python upscale.py cover.png -o cover_4k.png

# Batch upscale an entire folder of clips
python upscale.py "C:\Footage\Reels" -o "C:\Footage\Upscaled"

# High-quality photo model (RRDBNet)
python upscale.py clip.mp4 --model RealESRGAN_x4plus.pth --cq 18

# Native 4x scale (e.g. 720p -> 4K or 1080p -> 8K)
python upscale.py clip.mp4 --downscale 1.0
```

---

## 🎛️ CLI Options

| Option | Default | Description |
|---|---|---|
| `input` | *required* | Video file, image file, or directory |
| `-o`, `--output` | `[stem]_4K.[ext]` | Output file or directory path |
| `--model` | `realesr-general-x4v3.pth` | Weights inside `./models` directory |
| `--preset-mode` | `reels-4k` | Preconfigured profile (`reels-4k`, `cinema-4k`, `photo-sharp`) |
| `--tile` | `512` | Overlapping tile size for 4GB VRAM safety (`0` = disabled) |
| `--cq` | `20` | NVENC Constant Quality target (lower = higher quality) |
| `--encoder` | `hevc_nvenc` | Hardware encoder (`hevc_nvenc`, `h264_nvenc`, `libx265`) |
| `--downscale` | `0.5` | Post-scale factor (`0.5` on 4x model = 2x clean output) |

---

## 🧠 Pretrained AI Models

Run `python download_models.py` or double-click `download_models.bat`:

| Model File | Architecture | Scale | Best Application |
|---|---|---|---|
| `realesr-general-x4v3.pth` | SRVGGNet | 4x | **Default video engine**: Fast, sharp, flicker-free |
| `RealESRGAN_x4plus.pth` | RRDBNet | 4x | High detail photography, textures & faces |
| `RealESRGAN_x4plus_anime_6B.pth`| Compact | 4x | Anime, illustrations, 2D art & line drawings |
| `RealESRGAN_x2plus.pth` | RRDBNet | 2x | Heavily compressed/grainy video restoration |
| `GFPGANv1.4.pth` | GFP-GAN | - | Facial enhancement & recovery |

---

## 👨‍💻 Creator & Author

**Raliq Hidayat BM3**
- GitHub: [@kouji999](https://github.com/kouji999)
- Project: [ReelFrame on GitHub](https://github.com/kouji999/ReelFrame)

---

## 📜 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
Pretrained models belong to their respective research teams (Real-ESRGAN / Tencent ARC).
