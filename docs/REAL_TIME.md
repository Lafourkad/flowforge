# FlowForge Real-Time Pipeline

FlowForge Phase 2 introduces real-time video interpolation for smooth playback. This document explains how the real-time pipeline works and how to use it effectively.

## Overview

The real-time pipeline enables watching videos with RIFE frame interpolation applied on-the-fly, providing smooth high-fps playback without pre-processing. This is similar to SVP 4 Pro but uses the open-source RIFE algorithm.

## Architecture

The real-time system consists of several components working together:

```
Video Input → FFmpeg Decode → Staging Buffer → RIFE Interpolation → Output Buffer → FFmpeg Encode → mpv
```

### Core Components

1. **Stream Processor** (`stream_processor.py`)
   - Main orchestrator for the real-time pipeline
   - Manages sliding window buffering
   - Coordinates decode/interpolate/encode threads

2. **MPV Launcher** (`launcher.py`)
   - Detects and configures mpv
   - Handles VapourSynth integration
   - Manages system compatibility

3. **VapourSynth Filter** (`vapoursynth_filter.py`)
   - Provides mpv integration via VapourSynth
   - Falls back to binary mode if vs-rife plugin unavailable

4. **Adaptive Quality Manager** (`realtime_engine.py`)
   - Monitors performance metrics
   - Automatically adjusts quality to maintain real-time performance

## Pipeline Details

### 1. Video Decoding
- **FFmpeg** reads the input video and extracts frames as PNG files
- Frames are written to a **staging directory** for RIFE processing
- Operates in a separate thread for concurrency

### 2. Frame Buffering
- **Sliding Window Buffer** maintains a window of recent frames
- Default window size: 8 frames (configurable via preset)
- Old frames are automatically cleaned up to manage memory

### 3. RIFE Interpolation
- **RIFE Binary** processes frame pairs from the staging directory
- Interpolated frames written to **output directory**
- Scene detection can skip interpolation for cuts/transitions
- Multiple interpolation threads for performance

### 4. Video Encoding
- **FFmpeg** reads interpolated frames and creates video stream
- Outputs directly to mpv via pipe for real-time playback
- Frame timing synchronized to target FPS

### 5. Quality Adaptation
- **Adaptive Quality Manager** monitors performance metrics:
  - Frame processing rate
  - Buffer overruns  
  - CPU/GPU utilization
  - Average latency

- Automatically adjusts quality settings:
  - **Emergency**: 768px tiles, no TTA, minimal buffer
  - **Fast**: 640px tiles, no TTA, small buffer  
  - **Balanced**: 512px tiles, preset TTA, normal buffer
  - **Quality**: 384px tiles, TTA enabled, large buffer

## Integration Methods

FlowForge supports two integration approaches:

### Method 1: VapourSynth Plugin (Preferred)
- Uses `vs-rife-ncnn-vulkan` plugin if available
- Integrated directly into mpv's filter chain
- Lowest latency and best performance
- Automatic frame rate conversion

### Method 2: Binary Subprocess (Fallback)
- Uses `rife-ncnn-vulkan` binary via subprocess
- Works when VapourSynth plugin unavailable
- Slightly higher latency due to file I/O
- More compatible across systems

## Performance Characteristics

### Resource Usage
- **CPU**: 10-30% on modern systems (decode/encode + coordination)
- **GPU**: 70-90% utilization during interpolation (RIFE processing)  
- **Memory**: 200-500MB RAM (frame buffers + FFmpeg)
- **Storage**: 50-200MB temp space (sliding window frames)

### Latency Factors
- **Base Latency**: ~50-100ms (2-4 frame buffer)
- **RIFE Processing**: 20-80ms per frame pair (depends on preset)
- **Scene Detection**: +10ms when enabled
- **Quality Adaptation**: Can add 1-2 frame delays during transitions

### Throughput
- **Input Handling**: Up to 60fps input (standard videos)
- **Output Generation**: 60-144fps (depends on preset and hardware)
- **Maximum Multiplier**: 8x (limited by RIFE and hardware)

## Usage Examples

### Basic Real-Time Playback
```bash
# Play with default film preset (24→60fps)
flowforge play movie.mp4

# Use anime preset with strong artifact suppression  
flowforge play anime.mkv --preset anime

# Sports content with fast processing
flowforge play sports.mp4 --preset sports

# Maximum smoothness
flowforge play video.mp4 --preset smooth
```

### Custom Settings
```bash
# Custom target FPS
flowforge play video.mp4 --preset film --fps 120

# Additional mpv arguments
flowforge play movie.mp4 --mpv-args "--fullscreen --volume=75"

# Portable configuration
flowforge configure-mpv --preset anime --portable ./my-config
```

### System Status and Debugging
```bash
# Check system components
flowforge system-status

# Test installation
flowforge test

# List available presets
flowforge presets list --verbose
```

## Preset Configuration

### Built-in Presets

#### Film Preset
- **Target**: 24→60fps (2.5x multiplier)
- **Quality**: Balanced processing
- **Scene Detection**: Enabled (threshold: 0.25)
- **Use Case**: Movies, TV shows, narrative content

#### Anime Preset  
- **Target**: 24→60fps (2.5x multiplier)
- **Quality**: High quality with TTA enabled
- **Scene Detection**: Enabled (threshold: 0.35)
- **Special**: Strong artifact suppression for hard edges

#### Sports Preset
- **Target**: 30→60fps (2.0x multiplier)  
- **Quality**: Fast processing with minimal latency
- **Scene Detection**: Disabled (continuous motion)
- **Use Case**: Live sports, gaming content

#### Smooth Preset
- **Target**: Any→144fps (up to 6x multiplier)
- **Quality**: High quality with UHD mode
- **Scene Detection**: Enabled (threshold: 0.2)
- **Use Case**: Maximum smoothness, high-end systems

### Custom Presets
```bash
# Create custom preset
flowforge presets create gaming --base-preset sports --fps 144 --no-scene-detection

# Show preset details
flowforge presets show gaming

# Delete custom preset  
flowforge presets delete old-preset --yes
```

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **CPU**: 4+ cores, 2.5GHz+
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: DirectX 11 / Vulkan compatible
- **Storage**: 1GB free space for temp files

### Recommended Requirements  
- **OS**: Windows 11, Ubuntu 20.04+
- **CPU**: 8+ cores, 3.0GHz+ (Intel i5-10400 / AMD Ryzen 5 3600)
- **RAM**: 16GB+ 
- **GPU**: NVIDIA GTX 1060 / AMD RX 580 / Intel Arc A380
- **VRAM**: 4GB+ (8GB+ for 4K content)
- **Storage**: NVMe SSD recommended

### Dependencies
- **mpv**: Media player (0.33.0+)
- **FFmpeg**: Video processing (4.3.0+) 
- **VapourSynth**: Optional, for best performance (R55+)
- **vs-rife plugin**: Optional, for lowest latency
- **Python**: 3.8+ with required packages

## Performance Tuning

### GPU Optimization
```bash
# Force specific GPU (multi-GPU systems)
flowforge play video.mp4 --preset film --gpu 1

# Monitor GPU usage
nvidia-smi -l 1  # NVIDIA
radeontop        # AMD
```

### Quality vs Performance Trade-offs
- **Emergency Mode**: Use when struggling to maintain real-time
- **Fast Mode**: Good balance for most content  
- **Balanced Mode**: Default, works well on recommended hardware
- **Quality Mode**: Use with high-end GPUs and for critical viewing

### Buffer Tuning
- **Small Buffers** (2-4 frames): Lower latency, may drop frames
- **Medium Buffers** (6-8 frames): Good balance (default)
- **Large Buffers** (10-16 frames): Smoother playback, higher latency

### Scene Detection Tuning
- **Threshold 0.1-0.2**: Very sensitive, may skip valid interpolations
- **Threshold 0.2-0.4**: Good balance (default range)
- **Threshold 0.4+**: Less sensitive, may interpolate across cuts
- **Disabled**: Maximum smoothness but may artifact on cuts

## Troubleshooting

### Common Issues

#### "mpv not found"
- Install mpv from https://mpv.io
- Or install SVP 4 which includes mpv
- Check PATH environment variable

#### "RIFE binary not found"  
```bash
flowforge setup  # Download dependencies
```

#### "VapourSynth not found"
- Install VapourSynth for your platform
- Or use binary fallback mode (automatic)

#### Poor Performance
1. Check system requirements
2. Try faster preset: `--preset sports`
3. Lower target FPS: `--fps 30` 
4. Disable scene detection: `--no-scene-detection`
5. Monitor with: `flowforge system-status`

#### Audio/Video Sync Issues
- Usually resolved automatically
- Try different mpv audio driver: `--mpv-args "--ao=wasapi"`
- Check system audio latency

#### High Latency
- Reduce buffer size in preset
- Use VapourSynth plugin instead of binary mode
- Ensure GPU drivers are updated
- Close other GPU-intensive applications

### Debug Information
```bash
# Verbose output
flowforge play video.mp4 --verbose

# System status  
flowforge system-status --json-output

# Test performance
flowforge test --verbose
```

### Performance Monitoring

The real-time engine provides detailed statistics:
- **Input/Output FPS**: Decode and encode rates
- **Processing FPS**: RIFE interpolation rate  
- **Buffer Status**: Frame queues and overruns
- **Resource Usage**: CPU, memory, GPU utilization
- **Quality Adaptations**: Automatic quality changes

Statistics are available via:
- Console output (verbose mode)
- JSON export for external monitoring
- Optional callback integration for custom UIs

## Advanced Usage

### Integration with Custom Players
FlowForge can integrate with any media player supporting:
- VapourSynth filters (mpv, MPC-HC)
- External preprocessing (any player via named pipes)
- Direct frame serving (custom integration)

### Batch Processing Integration
While designed for real-time, the pipeline can also:
- Pre-process video segments
- Generate cached interpolations
- Hybrid real-time/cached playback

### API Integration
The stream processor can be used programmatically:
```python
from flowforge.playback import create_stream_processor, PresetManager

preset = PresetManager().get_preset("film")
processor = create_stream_processor(preset, "input.mp4")
processor.start()
# ... handle output stream
processor.stop()
```

## Future Improvements

- **Hardware Encode/Decode**: NVENC/QSV integration
- **Multi-GPU Support**: Distribute processing across GPUs
- **Advanced Scene Detection**: ML-based scene change detection
- **Streaming Input**: Live stream interpolation (RTSP, etc.)
- **HDR Support**: High dynamic range content
- **Audio Processing**: Advanced audio sync and enhancement