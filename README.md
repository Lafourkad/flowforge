# FlowForge 🎬

**Open-source, GPU-accelerated video frame interpolation.**  
Turn choppy 24fps video into buttery smooth 60fps+ playback.

Free alternative to SVP 4 Pro.

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20RTX-76B900.svg" alt="GPU">
</p>

---

## ✨ Features

- **Frame Interpolation** — Convert 24/30fps video to 60/120/144fps using RIFE neural networks
- **Real-Time Playback** — Watch any video with smooth interpolation via mpv + VapourSynth
- **GPU Accelerated** — Vulkan backend works with NVIDIA, AMD, and Intel GPUs
- **Scene Detection** — Smart scene change detection prevents artifacts at cuts
- **Presets** — Film, anime, sports, and custom profiles
- **Batch Processing** — Convert entire video files for later playback
- **Cross-Platform** — Linux, Windows, macOS

## 🚀 Quick Start

### Install

```bash
pip install flowforge
flowforge setup  # Downloads RIFE model (~430MB)
```

### Interpolate a Video File

```bash
# Double the frame rate (24fps → 48fps)
flowforge interpolate movie.mkv -o smooth.mkv --multiplier 2

# Target specific FPS
flowforge interpolate movie.mkv -o smooth.mkv --fps 60

# Use NVIDIA hardware encoding
flowforge interpolate movie.mkv -o smooth.mkv --fps 60 --nvenc
```

### Real-Time Playback

```bash
# Play with smooth interpolation
flowforge play movie.mkv --preset film

# Configure mpv integration
flowforge configure-mpv
```

### Video Info

```bash
flowforge info movie.mkv --estimate
```

## 📋 Requirements

- **Python** 3.11+
- **FFmpeg** (for video encoding/decoding)
- **GPU** with Vulkan support (NVIDIA RTX recommended)
- **mpv** + VapourSynth (for real-time playback)

## 🎮 Presets

| Preset | Source FPS | Target FPS | Best For |
|--------|-----------|------------|----------|
| `film` | 24fps | 60fps | Movies, TV shows |
| `anime` | 24fps | 60fps | Animation (strong artifact suppression) |
| `sports` | 30fps | 60fps | Fast motion, live action |
| `smooth` | any | 144fps | Maximum smoothness |
| `custom` | any | any | Your own settings |

## 🔧 How It Works

FlowForge uses [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) (Real-Time Intermediate Flow Estimation) to generate intermediate frames between existing ones. The ncnn-vulkan implementation runs on any GPU with Vulkan support — no CUDA toolkit needed.

```
Frame 1 ──┐
           ├── RIFE ──→ Interpolated Frame
Frame 2 ──┘
```

For a 2x multiplier, one new frame is generated between each pair of original frames, doubling the frame rate. 4x generates 3 intermediate frames, etc.

**Scene detection** analyzes frame similarity to find cuts. At scene boundaries, interpolation is skipped to prevent ghosting artifacts.

## 🏗️ Architecture

```
flowforge/
├── core/           # Interpolation engine
│   ├── rife.py          # RIFE model wrapper (ncnn-vulkan)
│   ├── interpolator.py  # Video interpolation pipeline
│   ├── ffmpeg.py        # FFmpeg utilities
│   └── scene_detect.py  # Scene change detection
├── playback/       # Real-time playback
│   ├── launcher.py      # mpv launcher
│   ├── presets.py       # Interpolation presets
│   └── vapoursynth_filter.py  # VapourSynth RIFE filter
├── gui/            # GUI application (Phase 3)
└── utils/          # Utilities
    └── download.py      # Model downloader
```

## 📊 Performance

Tested on NVIDIA RTX 4070 (12GB VRAM):

| Resolution | Multiplier | Processing Speed |
|-----------|------------|-----------------|
| 1080p | 2x (24→48fps) | ~45 fps |
| 1080p | 2x (24→60fps) | ~45 fps |
| 1080p | 4x (24→96fps) | ~15 fps |
| 4K | 2x (24→48fps) | ~12 fps |

*Real-time playback requires processing speed > target FPS.*

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License — see [LICENSE](LICENSE).

## 🙏 Credits

- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) — Frame interpolation model by Megvii Research
- [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) — Vulkan implementation by nihui
- [FFmpeg](https://ffmpeg.org/) — Video processing
- [mpv](https://mpv.io/) — Video player
- [VapourSynth](https://www.vapoursynth.com/) — Video processing framework
