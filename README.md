# FlowForge - Video Frame Interpolation GUI

A modern PyQt6 GUI for FlowForge, providing GPU-accelerated video frame interpolation using RIFE (Real-time Intermediate Flow Estimation).

## Features

### 🎬 **Modern Dark Interface**
- HandBrake/DaVinci Resolve inspired dark theme
- Intuitive drag-and-drop video loading
- Real-time video information display
- Responsive layout with status tracking

### ⚡ **Real-Time Playback** 
- Instant preview with VapourSynth + mpv integration
- Multiple presets: Film (24→60fps), Anime (24→60fps), Sports (30→60fps), Smooth (→144fps), Custom
- Configurable scene detection with threshold control
- GPU thread optimization (1-4 threads)
- RIFE model selection

### 🚀 **Batch Export**
- Full processing pipeline: extract → RIFE → encode
- Real-time progress tracking with detailed statistics
- Multiple encoding presets: Quality, Balanced, Fast
- NVENC hardware encoding support
- Cancellable operations with proper cleanup

### ⚙️ **Advanced Settings**
- Auto-detection of RIFE binary, models, and mpv
- Platform detection (Windows/WSL) with path conversion
- GPU information and selection
- Persistent configuration in `~/.flowforge/config.json`
- Comprehensive settings dialog

## Requirements

### Software Dependencies
- **Python 3.8+** with PyQt6
- **RIFE ncnn-vulkan** - GPU-accelerated RIFE implementation
- **mpv** - Video player with VapourSynth support
- **VapourSynth** - Video processing framework
- **RIFE VapourSynth plugin** - For real-time interpolation
- **FFmpeg** - Video encoding/decoding

### Hardware Requirements  
- **NVIDIA GPU** with Vulkan support (RTX series recommended)
- **8GB+ system RAM** (16GB+ recommended for 4K videos)
- **Fast storage** - SSD recommended for temp files

## Installation

### 1. Clone and Install Python Dependencies
```bash
git clone <repository-url>
cd flowforge
pip install -r requirements.txt
```

### 2. Setup RIFE Components
Place the following in your FlowForge directory structure:
```
FlowForge/
├── bin/rife-ncnn-vulkan.exe          # RIFE binary
├── models/rife-v4.6/                 # RIFE model files
├── vs-plugins/librife.dll             # VapourSynth plugin
└── flowforge_win.py                   # Existing batch script
```

### 3. Install Supporting Software
- **mpv**: Install with VapourSynth support (e.g., from SVP package)
- **FFmpeg**: Standard installation with libx264/NVENC support

## Usage

### Starting the GUI
```bash
# Method 1: Module execution
python -m flowforge.gui

# Method 2: Direct launcher
python run_gui.py

# Method 3: From package
cd flowforge && python -m gui
```

### Basic Workflow
1. **Load Video**: Drag & drop video file or use "Browse Files"
2. **Configure Playback**: 
   - Select preset (Film/Anime/Sports/Smooth/Custom)
   - Adjust scene detection threshold
   - Set GPU thread count
3. **Preview**: Click "▶️ Play Real-Time" for mpv preview
4. **Setup Export**:
   - Choose output path
   - Set FPS multiplier or target FPS
   - Select encoding preset
   - Enable/disable NVENC
5. **Process**: Click "🎬 Export Video" and monitor progress

### Settings Configuration
Access via **Settings → Preferences**:

- **Paths Tab**: Configure RIFE binary, model directory, mpv, VapourSynth plugin
- **Processing Tab**: Set default GPU, threads, scene detection, presets
- **Export Tab**: Configure encoding presets, quality, NVENC defaults
- **Advanced Tab**: View platform info, GPU detection, version information

## Technical Details

### Architecture
- **PyQt6** - Modern cross-platform GUI framework
- **QThread Workers** - Non-blocking background processing
- **Settings Persistence** - JSON configuration management
- **Platform Detection** - Automatic WSL↔Windows path conversion

### Processing Pipeline
1. **Analysis**: FFprobe metadata extraction
2. **Frame Extraction**: FFmpeg PNG sequence generation
3. **RIFE Interpolation**: GPU-accelerated frame generation
4. **Encoding**: FFmpeg with x264/NVENC to final video
5. **Cleanup**: Automatic temporary file removal

### Real-Time Playback
- Generates VapourSynth script dynamically
- Launches mpv with RIFE filter applied
- No intermediate files - direct GPU processing
- Configurable quality vs. speed tradeoffs

## File Structure

```
flowforge/
├── flowforge/
│   ├── __init__.py
│   └── gui/
│       ├── __init__.py
│       ├── __main__.py              # python -m flowforge.gui
│       ├── main.py                  # Entry point and main()
│       ├── app.py                   # QApplication with dark theme
│       ├── main_window.py           # Main GUI window
│       ├── settings.py              # Configuration management
│       ├── worker.py                # Background processing threads
│       └── widgets/
│           ├── __init__.py
│           ├── dragdrop.py          # Video drag & drop widget
│           ├── progress.py          # Progress tracking widgets
│           └── settings.py          # Settings dialog
├── requirements.txt                 # PyQt6 dependencies
├── run_gui.py                      # Simple launcher script
└── README.md                       # This file
```

## Integration with Existing Code

The GUI seamlessly integrates with existing FlowForge components:

- **`flowforge_win.py`** - Reuses batch processing logic and functions
- **`rife.vpy`** - Uses as template for dynamic VapourSynth script generation
- **Platform Detection** - Handles WSL/Windows path conversion automatically
- **Settings** - Inherits and extends existing configuration approach

## Troubleshooting

### Common Issues

**"RIFE binary not found"**
- Ensure `rife-ncnn-vulkan.exe` exists and is executable
- Check Settings → Paths → RIFE Binary path
- Verify NVIDIA drivers and Vulkan support

**"mpv failed to start"**
- Verify mpv installation with VapourSynth support
- Check Settings → Paths → mpv Path
- Ensure VapourSynth plugin path is correct

**"No GPU detected"**
- Install NVIDIA drivers with CUDA support
- Run `nvidia-smi` to verify GPU visibility
- Check Windows/WSL GPU passthrough if applicable

**Processing stuck or slow**
- Check available disk space for temp files
- Verify GPU isn't being used by other applications
- Try reducing thread count or GPU count

### Log Files
- Application logs: Check console output
- Settings file: `~/.flowforge/config.json`
- Temporary processing: System temp directory (auto-cleaned)

## Performance Tips

- **SSD Storage**: Use fast storage for temporary frame files
- **GPU Memory**: Higher VRAM allows larger frame batches
- **System RAM**: 16GB+ recommended for 4K processing
- **Thread Tuning**: Start with 1:2:2, adjust based on system
- **Scene Detection**: Can improve quality but adds processing time

## License

This project integrates with RIFE and other open-source components. Please respect their respective licenses.

---

**FlowForge GUI v1.0** - Modern video frame interpolation made easy.