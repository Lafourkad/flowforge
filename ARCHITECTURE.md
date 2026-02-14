# FlowForge Architecture

FlowForge is a comprehensive video frame interpolation suite that provides both offline processing and real-time playback capabilities. This document outlines the overall system architecture, design decisions, and component interactions.

## System Overview

FlowForge is built as a modular Python package with a clean separation between core processing, real-time playback, and user interfaces. The architecture supports multiple interpolation backends, adaptive quality management, and extensive customization.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface   │    │   Core Engine    │    │  Media Integration │
│                 │    │                 │    │                 │
│  • CLI Commands │    │  • RIFE Backend │    │  • mpv Launcher │
│  • Configuration│◄──►│  • FFmpeg Utils │◄──►│  • VapourSynth  │
│  • Presets Mgmt │    │  • Scene Detect │    │  • Stream Proc. │
│                 │    │  • Interpolator │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Utilities      │    │  Real-time Sys  │    │   External Deps │
│                 │    │                 │    │                 │
│  • Model Downld │    │  • Adaptive Qty │    │  • RIFE Binary  │
│  • System Check │    │  • Buffer Mgmt  │    │  • FFmpeg       │
│  • Path Detection│    │  • Performance  │    │  • mpv          │
│                 │    │  • Statistics   │    │  • VapourSynth  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Package Structure

```
flowforge/
├── __init__.py                 # Package initialization
├── cli.py                     # Command-line interface
├── core/                      # Core processing components
│   ├── __init__.py
│   ├── interpolator.py        # Main interpolation orchestrator
│   ├── rife.py               # RIFE backend wrapper
│   ├── ffmpeg.py            # FFmpeg utilities
│   └── scene_detect.py      # Scene change detection
├── playback/                 # Real-time playback system
│   ├── __init__.py
│   ├── presets.py           # Interpolation presets
│   ├── launcher.py          # mpv integration
│   ├── mpv_config.py       # mpv configuration
│   ├── stream_processor.py # Streaming interpolation
│   ├── realtime_engine.py  # Real-time engine
│   └── vapoursynth_filter.py # VapourSynth integration
└── utils/                   # Utilities and helpers
    ├── __init__.py
    └── download.py         # Model/binary downloader
```

## Phase 1: Offline Processing

Phase 1 provides high-quality offline video interpolation similar to professional tools.

### Core Components

#### VideoInterpolator (`core/interpolator.py`)
The main orchestrator for offline processing:
- **Input Validation**: Checks video files and parameters
- **Workflow Management**: Coordinates extraction, interpolation, encoding
- **Progress Tracking**: Provides detailed progress callbacks
- **Quality Control**: Ensures consistent output quality
- **Resource Management**: Handles temporary files and memory

Key Features:
- Support for multiple RIFE models
- Scene-aware interpolation (skips cuts/transitions)
- Custom encoding parameters (codec, CRF, preset)
- Audio and subtitle preservation
- Batch processing capabilities

#### RIFE Backend (`core/rife.py`)
Wrapper around the rife-ncnn-vulkan binary:
- **Binary Management**: Locates and validates RIFE executable
- **Parameter Handling**: Converts presets to RIFE arguments
- **Process Management**: Spawns and monitors RIFE processes
- **Error Handling**: Graceful fallbacks and error reporting
- **Performance Optimization**: Multi-threading and GPU utilization

#### FFmpeg Integration (`core/ffmpeg.py`)
Comprehensive FFmpeg wrapper:
- **Frame Extraction**: Video to image sequences
- **Video Assembly**: Image sequences to video
- **Format Support**: Wide range of input/output formats
- **Metadata Preservation**: Maintains video properties
- **Stream Handling**: Audio, video, subtitle streams

#### Scene Detection (`core/scene_detect.py`)
Advanced scene change detection:
- **Multiple Algorithms**: Histogram, SSIM, MSE-based detection
- **Adaptive Thresholds**: Content-aware threshold adjustment
- **Performance Optimization**: Fast approximation algorithms
- **Integration**: Seamless integration with interpolation workflow

## Phase 2: Real-Time System

Phase 2 enables real-time interpolation for smooth playback, similar to SVP 4 Pro.

### Architecture Principles

1. **Concurrency**: Multiple threads handle decode, interpolate, encode
2. **Buffering**: Sliding window buffers manage frame flow
3. **Adaptivity**: Quality automatically adjusts to maintain real-time
4. **Modularity**: Components can be used independently
5. **Robustness**: Graceful handling of errors and edge cases

### Key Components

#### Stream Processor (`playback/stream_processor.py`)
The heart of real-time interpolation:

```
Input Video → Decode Thread → Staging Buffer → Interpolation Thread → Output Buffer → Encode Thread → mpv
```

**Threading Model**:
- **Decode Thread**: FFmpeg extracts frames to staging directory
- **Interpolation Thread**: RIFE processes frame pairs
- **Encode Thread**: FFmpeg assembles output stream for mpv
- **Stats Thread**: Monitors performance and triggers adaptations

**Buffer Management**:
- **Sliding Window**: Maintains N recent frames in memory/disk
- **Overflow Handling**: Drops frames when buffers full
- **Memory Optimization**: Automatic cleanup of old frames
- **Performance Monitoring**: Tracks buffer utilization

#### Real-Time Engine (`playback/realtime_engine.py`)
Advanced real-time processing with adaptive quality:

**Quality Management**:
- **Performance Monitoring**: Tracks FPS, latency, resource usage
- **Adaptive Scaling**: Automatically adjusts quality settings
- **Quality Levels**: Emergency → Fast → Balanced → Quality
- **Hysteresis**: Prevents rapid quality oscillation

**Resource Monitoring**:
- CPU/GPU utilization tracking
- Memory usage monitoring
- Thermal management (future)
- Power consumption awareness (future)

#### mpv Integration (`playback/launcher.py`)
Comprehensive mpv integration:
- **System Detection**: Locates mpv across platforms
- **Configuration Generation**: Creates optimal mpv configs
- **VapourSynth Integration**: Seamless filter chain setup
- **Process Management**: Launches and monitors mpv
- **IPC Communication**: Runtime control via mpv IPC

#### VapourSynth Filter (`playback/vapoursynth_filter.py`)
Two-mode VapourSynth integration:

**Plugin Mode** (Preferred):
- Uses vs-rife-ncnn-vulkan plugin
- Direct integration into mpv filter chain
- Lowest latency and highest performance
- Automatic frame rate conversion

**Binary Mode** (Fallback):
- Uses rife-ncnn-vulkan via subprocess
- Compatible with any system
- Slightly higher latency
- More complex frame management

### Preset System (`playback/presets.py`)

Comprehensive preset management for different content types:

#### Built-in Presets
- **Film**: 24→60fps, balanced quality, scene detection
- **Anime**: 24→60fps, high quality, strong artifact suppression  
- **Sports**: 30→60fps, fast processing, no scene detection
- **Smooth**: Any→144fps, maximum quality, high multiplier

#### Custom Presets
- User-defined configurations
- JSON serialization for sharing
- Validation and error checking
- Migration support for updates

#### Adaptive Presets
- Runtime creation based on input video
- Content-type detection
- Hardware capability awareness
- Performance optimization

## Design Patterns

### Component Communication
FlowForge uses several communication patterns:

1. **Observer Pattern**: Statistics callbacks for monitoring
2. **Producer-Consumer**: Queue-based thread communication
3. **Command Pattern**: CLI command dispatch
4. **Factory Pattern**: Preset and component creation
5. **Strategy Pattern**: Multiple interpolation backends

### Error Handling
Robust error handling throughout:

1. **Graceful Degradation**: Fallback to lower quality when needed
2. **Resource Cleanup**: Automatic temp file and process cleanup
3. **User Feedback**: Clear error messages and suggestions
4. **Recovery Mechanisms**: Automatic retry with adjusted parameters
5. **Logging**: Comprehensive logging for debugging

### Performance Optimization

#### Memory Management
- **Sliding Buffers**: Limit memory usage with bounded queues
- **Lazy Loading**: Load frames only when needed
- **Garbage Collection**: Explicit cleanup of large objects
- **Memory Mapping**: Use memory-mapped files for large datasets

#### CPU Optimization
- **Multi-threading**: Parallel processing where possible
- **Thread Pools**: Reuse threads to avoid creation overhead
- **Async I/O**: Non-blocking file operations
- **CPU Affinity**: Pin threads to specific cores (future)

#### GPU Optimization  
- **Batch Processing**: Process multiple frames together
- **Memory Pooling**: Reuse GPU memory allocations
- **Pipeline Overlap**: Overlap GPU compute and memory transfers
- **Multi-GPU**: Distribute work across multiple GPUs (future)

## Platform Compatibility

### Supported Platforms
- **Windows**: 10/11 (x64), WSL2 support
- **Linux**: Ubuntu 18.04+, other distributions
- **macOS**: 10.15+ (Intel and Apple Silicon)

### Platform-Specific Optimizations
- **Windows**: NVENC/DXVA hardware acceleration
- **Linux**: VAAPI/VDPAU hardware acceleration
- **macOS**: VideoToolbox integration
- **WSL2**: Cross-filesystem path translation

### Binary Distribution
- **Self-contained**: Include all dependencies
- **Platform Detection**: Automatic binary selection
- **Version Management**: Update mechanism for binaries
- **Fallback Modes**: CPU-only mode when GPU unavailable

## External Dependencies

### Core Dependencies
- **Python 3.8+**: Core runtime
- **NumPy**: Numerical computations
- **OpenCV**: Image processing (optional)
- **Pillow**: Image file handling
- **psutil**: System monitoring

### Media Dependencies
- **FFmpeg 4.3+**: Video processing
- **mpv 0.33+**: Media playback
- **VapourSynth R55+**: Filter framework (optional)

### RIFE Dependencies
- **rife-ncnn-vulkan**: Core interpolation binary
- **Vulkan Runtime**: GPU acceleration
- **RIFE Models**: Neural network weights

## Configuration System

### Hierarchical Configuration
1. **Built-in Defaults**: Sensible defaults for all settings
2. **User Config**: ~/.flowforge/config.json
3. **Project Config**: ./flowforge.json  
4. **Command Line**: Override any setting
5. **Environment**: Environment variable overrides

### Configuration Categories
- **Paths**: Binary locations, temp directories
- **Performance**: Thread counts, memory limits
- **Quality**: Default presets, quality thresholds
- **Integration**: mpv settings, VapourSynth options
- **Logging**: Log levels, output destinations

## Monitoring and Telemetry

### Performance Metrics
- **Frame Rates**: Input, processing, output FPS
- **Latency**: End-to-end processing latency
- **Resource Usage**: CPU, memory, GPU utilization
- **Quality Metrics**: PSNR, SSIM for interpolated frames
- **Error Rates**: Failed interpolations, dropped frames

### Statistics Collection
- **Real-time**: Live performance monitoring
- **Historical**: Long-term performance trends  
- **Comparative**: Before/after quality comparisons
- **Export**: JSON/CSV export for analysis

## Security Considerations

### Input Validation
- **File Path Sanitization**: Prevent directory traversal
- **Video Format Validation**: Reject malicious files
- **Parameter Bounds Checking**: Validate all numeric inputs
- **Resource Limits**: Prevent resource exhaustion attacks

### Process Isolation
- **Subprocess Sandboxing**: Isolate external processes
- **Temporary File Security**: Secure temp file creation
- **Network Isolation**: No unnecessary network access
- **Privilege Dropping**: Run with minimal privileges

## Future Architecture

### Planned Enhancements

#### Distributed Processing
- **Cluster Support**: Distribute processing across machines
- **Load Balancing**: Automatic work distribution
- **Fault Tolerance**: Handle node failures gracefully
- **Monitoring**: Cluster-wide performance monitoring

#### Advanced Algorithms
- **Multi-Model Ensemble**: Combine multiple RIFE versions
- **Content-Aware Processing**: Specialized algorithms for content types
- **Perceptual Quality**: Optimize for human visual perception
- **Temporal Consistency**: Reduce flickering artifacts

#### Cloud Integration
- **GPU Cloud**: Utilize cloud GPU resources
- **Serverless Processing**: Function-as-a-service interpolation
- **Storage Integration**: Direct cloud storage access
- **Cost Optimization**: Automatic resource scaling

### Extensibility Framework

#### Plugin System
- **Algorithm Plugins**: Support for new interpolation methods
- **Format Plugins**: Additional input/output formats
- **Integration Plugins**: New media player support
- **UI Plugins**: Custom user interfaces

#### API Framework
- **REST API**: Web service interface
- **gRPC API**: High-performance RPC interface
- **WebSocket API**: Real-time streaming interface
- **SDK**: Libraries for multiple languages

## Conclusion

FlowForge's architecture balances performance, quality, and usability through careful design decisions:

1. **Modularity**: Clean separation of concerns enables independent development and testing
2. **Performance**: Multi-threaded design and adaptive quality ensure real-time processing
3. **Compatibility**: Extensive platform and format support maximizes accessibility
4. **Extensibility**: Plugin architecture and configuration system enable customization
5. **Robustness**: Comprehensive error handling and fallback mechanisms ensure reliability

The architecture is designed to scale from simple offline processing to high-performance real-time systems, while maintaining ease of use and deployment flexibility.