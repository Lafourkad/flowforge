#!/bin/bash
# FlowForge VapourSynth Installation Script for Linux
# Installs VapourSynth and vs-rife-ncnn-vulkan plugin

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This script should typically be run as a regular user."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Detect Linux distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
    else
        log_error "Cannot detect Linux distribution"
        exit 1
    fi
    
    log_info "Detected distribution: $DISTRO $DISTRO_VERSION"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        log_error "Unsupported architecture: $ARCH (x86_64 required)"
        exit 1
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        log_info "Python version: $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
            log_success "Python version is compatible"
        else
            log_error "Python 3.8 or higher required (found $PYTHON_VERSION)"
            exit 1
        fi
    else
        log_error "Python 3 not found"
        exit 1
    fi
    
    # Check available disk space (need at least 1GB)
    AVAILABLE_SPACE=$(df /tmp --output=avail | tail -n1)
    if (( AVAILABLE_SPACE < 1048576 )); then  # 1GB in KB
        log_warning "Low disk space available ($(( AVAILABLE_SPACE / 1024 ))MB). Installation may fail."
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case "$DISTRO" in
        ubuntu|debian)
            # Update package list
            sudo apt update
            
            # Install build dependencies
            sudo apt install -y \
                build-essential \
                cmake \
                git \
                python3-dev \
                python3-pip \
                python3-setuptools \
                python3-wheel \
                cython3 \
                pkg-config \
                yasm \
                libtool \
                autoconf \
                automake \
                libass-dev \
                libfreetype6-dev \
                libgnutls28-dev \
                libmp3lame-dev \
                libtool \
                libva-dev \
                libvdpau-dev \
                libvorbis-dev \
                libxcb1-dev \
                libxcb-shm0-dev \
                libxcb-xfixes0-dev \
                meson \
                ninja-build \
                texinfo \
                wget \
                zlib1g-dev \
                nasm \
                libzimg-dev
            ;;
        fedora|centos|rhel)
            # Install using dnf/yum
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
            
            sudo $PKG_MANAGER install -y \
                gcc \
                gcc-c++ \
                cmake \
                git \
                python3-devel \
                python3-pip \
                python3-setuptools \
                python3-wheel \
                python3-Cython \
                pkgconfig \
                yasm \
                nasm \
                meson \
                ninja-build \
                wget \
                zlib-devel
            ;;
        arch|manjaro)
            # Install using pacman
            sudo pacman -Syu --noconfirm \
                base-devel \
                cmake \
                git \
                python \
                python-pip \
                python-setuptools \
                python-wheel \
                cython \
                pkgconfig \
                yasm \
                nasm \
                meson \
                ninja \
                wget \
                zlib
            ;;
        *)
            log_warning "Unsupported distribution: $DISTRO"
            log_info "Please install the following dependencies manually:"
            log_info "- Build tools (gcc, cmake, make)"
            log_info "- Python development headers"
            log_info "- YASM/NASM assemblers"
            log_info "- Meson build system"
            log_info "Press Enter to continue or Ctrl+C to abort..."
            read
            ;;
    esac
    
    log_success "System dependencies installed"
}

# Create build directory
setup_build_env() {
    BUILD_DIR="$HOME/.flowforge/build"
    INSTALL_DIR="$HOME/.flowforge/vapoursynth"
    
    log_info "Setting up build environment in $BUILD_DIR"
    
    mkdir -p "$BUILD_DIR"
    mkdir -p "$INSTALL_DIR"
    
    cd "$BUILD_DIR"
    
    # Set environment variables
    export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
    export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"
    export PATH="$INSTALL_DIR/bin:$PATH"
    export PYTHONPATH="$INSTALL_DIR/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"
}

# Install VapourSynth
install_vapoursynth() {
    log_info "Installing VapourSynth..."
    
    cd "$BUILD_DIR"
    
    # Check if VapourSynth is already installed
    if python3 -c "import vapoursynth" &> /dev/null; then
        log_info "VapourSynth already installed, checking version..."
        VS_VERSION=$(python3 -c "import vapoursynth; print(vapoursynth.core.version())")
        log_info "Current VapourSynth version: $VS_VERSION"
        
        read -p "Reinstall VapourSynth? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping VapourSynth installation"
            return 0
        fi
    fi
    
    # Download VapourSynth source
    if [[ ! -d "vapoursynth" ]]; then
        log_info "Downloading VapourSynth source..."
        git clone https://github.com/vapoursynth/vapoursynth.git
    else
        log_info "Updating VapourSynth source..."
        cd vapoursynth && git pull && cd ..
    fi
    
    cd vapoursynth
    
    # Configure with meson
    if [[ ! -d "build" ]]; then
        log_info "Configuring VapourSynth build..."
        meson build --prefix="$INSTALL_DIR" --buildtype=release
    fi
    
    # Build and install
    log_info "Building VapourSynth... (this may take a while)"
    cd build
    ninja
    
    log_info "Installing VapourSynth..."
    ninja install
    
    cd "$BUILD_DIR"
    
    # Install Python bindings
    log_info "Installing VapourSynth Python bindings..."
    pip3 install --user cython
    
    # Test installation
    if python3 -c "import vapoursynth; print('VapourSynth', vapoursynth.core.version())" &> /dev/null; then
        VS_VERSION=$(python3 -c "import vapoursynth; print(vapoursynth.core.version())")
        log_success "VapourSynth installed successfully: $VS_VERSION"
    else
        log_error "VapourSynth installation failed"
        exit 1
    fi
}

# Install vs-rife-ncnn-vulkan plugin
install_vs_rife_plugin() {
    log_info "Installing vs-rife-ncnn-vulkan plugin..."
    
    cd "$BUILD_DIR"
    
    # Check if plugin is already available
    if python3 -c "import vapoursynth; vs = vapoursynth.core; vs.rife" &> /dev/null; then
        log_info "vs-rife plugin already available"
        return 0
    fi
    
    # Download ncnn
    if [[ ! -d "ncnn" ]]; then
        log_info "Downloading ncnn..."
        git clone https://github.com/Tencent/ncnn.git
    else
        log_info "Updating ncnn..."
        cd ncnn && git pull && cd ..
    fi
    
    cd ncnn
    mkdir -p build && cd build
    
    log_info "Building ncnn..."
    cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
          -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_VULKAN=ON \
          -DNCNN_SYSTEM_GLSLANG=OFF \
          -DNCNN_BUILD_EXAMPLES=OFF \
          ..
    make -j$(nproc)
    make install
    
    cd "$BUILD_DIR"
    
    # Download vs-rife-ncnn-vulkan
    if [[ ! -d "vs-rife-ncnn-vulkan" ]]; then
        log_info "Downloading vs-rife-ncnn-vulkan..."
        git clone https://github.com/HomeOfVapourSynthEvolution/vs-rife-ncnn-vulkan.git
    else
        log_info "Updating vs-rife-ncnn-vulkan..."
        cd vs-rife-ncnn-vulkan && git pull && cd ..
    fi
    
    cd vs-rife-ncnn-vulkan
    
    # Build plugin
    mkdir -p build && cd build
    
    log_info "Building vs-rife-ncnn-vulkan plugin..."
    cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
          -DCMAKE_BUILD_TYPE=Release \
          ..
    make -j$(nproc)
    make install
    
    cd "$BUILD_DIR"
    
    # Test plugin
    if python3 -c "import vapoursynth; vs = vapoursynth.core; vs.rife" &> /dev/null 2>&1; then
        log_success "vs-rife-ncnn-vulkan plugin installed successfully"
    else
        log_warning "vs-rife plugin installation may have failed (this is optional)"
        log_info "FlowForge will fall back to using the RIFE binary"
    fi
}

# Create activation script
create_activation_script() {
    log_info "Creating activation script..."
    
    ACTIVATION_SCRIPT="$HOME/.flowforge/activate_vapoursynth.sh"
    
    cat > "$ACTIVATION_SCRIPT" << EOF
#!/bin/bash
# FlowForge VapourSynth Environment Activation Script

export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:\$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PATH="$INSTALL_DIR/bin:\$PATH"
export PYTHONPATH="$INSTALL_DIR/lib/python\$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:\$PYTHONPATH"

echo "FlowForge VapourSynth environment activated"
echo "VapourSynth path: $INSTALL_DIR"

# Test VapourSynth
if python3 -c "import vapoursynth" &> /dev/null; then
    VS_VERSION=\$(python3 -c "import vapoursynth; print(vapoursynth.core.version())")
    echo "VapourSynth version: \$VS_VERSION"
    
    # Test vs-rife plugin
    if python3 -c "import vapoursynth; vs = vapoursynth.core; vs.rife" &> /dev/null 2>&1; then
        echo "vs-rife plugin: Available"
    else
        echo "vs-rife plugin: Not available (will use binary fallback)"
    fi
else
    echo "ERROR: VapourSynth not found"
fi
EOF
    
    chmod +x "$ACTIVATION_SCRIPT"
    
    # Add to ~/.bashrc if not already present
    BASHRC="$HOME/.bashrc"
    if [[ -f "$BASHRC" ]] && ! grep -q "flowforge.*vapoursynth" "$BASHRC"; then
        log_info "Adding VapourSynth activation to ~/.bashrc"
        echo "" >> "$BASHRC"
        echo "# FlowForge VapourSynth environment" >> "$BASHRC"
        echo "source \"$ACTIVATION_SCRIPT\"" >> "$BASHRC"
    fi
    
    log_success "Activation script created: $ACTIVATION_SCRIPT"
}

# Cleanup build files
cleanup_build() {
    log_info "Cleaning up build files..."
    
    read -p "Remove build directory ($BUILD_DIR)? This will save disk space. (Y/n): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        rm -rf "$BUILD_DIR"
        log_success "Build directory cleaned up"
    else
        log_info "Build directory preserved: $BUILD_DIR"
    fi
}

# Test installation
test_installation() {
    log_info "Testing installation..."
    
    # Source the activation script
    source "$HOME/.flowforge/activate_vapoursynth.sh"
    
    # Test VapourSynth
    if python3 -c "import vapoursynth; print('VapourSynth version:', vapoursynth.core.version())" 2>/dev/null; then
        log_success "VapourSynth test passed"
    else
        log_error "VapourSynth test failed"
        return 1
    fi
    
    # Test vs-rife plugin (optional)
    if python3 -c "import vapoursynth; vs = vapoursynth.core; vs.rife" &> /dev/null 2>&1; then
        log_success "vs-rife plugin test passed"
    else
        log_warning "vs-rife plugin test failed (optional - will use binary fallback)"
    fi
    
    log_success "Installation test completed"
}

# Print installation summary
print_summary() {
    log_success "FlowForge VapourSynth Installation Complete!"
    echo ""
    echo "Installation Summary:"
    echo "- VapourSynth installed in: $INSTALL_DIR"
    echo "- Activation script: $HOME/.flowforge/activate_vapoursynth.sh"
    echo "- Environment variables added to ~/.bashrc"
    echo ""
    echo "Next steps:"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Test FlowForge: flowforge system-status"
    echo "3. Configure mpv: flowforge configure-mpv"
    echo "4. Play a video: flowforge play your_video.mp4"
    echo ""
    echo "If you encounter issues:"
    echo "- Check system status: flowforge system-status"
    echo "- View logs in: /tmp/flowforge_install.log"
    echo "- Report issues at: https://github.com/your-repo/FlowForge/issues"
}

# Main installation function
main() {
    log_info "FlowForge VapourSynth Installation Script"
    log_info "========================================"
    
    # Redirect all output to log file
    exec > >(tee -a /tmp/flowforge_install.log)
    exec 2>&1
    
    check_root
    detect_distro
    check_requirements
    
    log_info "Starting installation process..."
    
    install_system_deps
    setup_build_env
    install_vapoursynth
    install_vs_rife_plugin
    create_activation_script
    cleanup_build
    test_installation
    print_summary
    
    log_success "Installation completed successfully!"
}

# Handle script interruption
trap 'log_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"