#!/bin/bash

# FlowForge Dependencies Installation Script for Linux
# This script installs rife-ncnn-vulkan binary and dependencies

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$HOME/.flowforge"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_system() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "This script is for Linux systems only"
        exit 1
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        log_warning "Detected architecture: $ARCH (x86_64 recommended)"
    fi
    
    # Check required commands
    MISSING_DEPS=()
    
    if ! check_command "wget"; then
        MISSING_DEPS+=("wget")
    fi
    
    if ! check_command "unzip"; then
        MISSING_DEPS+=("unzip")
    fi
    
    if ! check_command "python3"; then
        MISSING_DEPS+=("python3")
    fi
    
    if ! check_command "pip3"; then
        MISSING_DEPS+=("python3-pip")
    fi
    
    if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${MISSING_DEPS[*]}"
        log_info "Install them with:"
        log_info "sudo apt-get update && sudo apt-get install ${MISSING_DEPS[*]}"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

check_ffmpeg() {
    log_info "Checking FFmpeg installation..."
    
    if check_command "ffmpeg" && check_command "ffprobe"; then
        FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
        log_success "FFmpeg found: $FFMPEG_VERSION"
    else
        log_warning "FFmpeg not found"
        log_info "Installing FFmpeg..."
        
        if check_command "apt-get"; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif check_command "yum"; then
            sudo yum install -y ffmpeg
        elif check_command "dnf"; then
            sudo dnf install -y ffmpeg
        elif check_command "pacman"; then
            sudo pacman -S ffmpeg
        else
            log_error "Cannot install FFmpeg automatically. Please install manually."
            exit 1
        fi
        
        if check_command "ffmpeg"; then
            log_success "FFmpeg installed successfully"
        else
            log_error "FFmpeg installation failed"
            exit 1
        fi
    fi
}

check_vulkan() {
    log_info "Checking Vulkan support..."
    
    # Check for Vulkan loader
    if ldconfig -p | grep -q libvulkan; then
        log_success "Vulkan loader found"
    else
        log_warning "Vulkan loader not found"
        log_info "Installing Vulkan support..."
        
        if check_command "apt-get"; then
            sudo apt-get install -y libvulkan1 vulkan-utils
        elif check_command "yum"; then
            sudo yum install -y vulkan-loader vulkan-tools
        elif check_command "dnf"; then
            sudo dnf install -y vulkan-loader vulkan-tools
        elif check_command "pacman"; then
            sudo pacman -S vulkan-icd-loader vulkan-tools
        else
            log_warning "Cannot install Vulkan automatically"
        fi
    fi
    
    # Check for GPU drivers
    if lspci | grep -i nvidia >/dev/null 2>&1; then
        log_info "NVIDIA GPU detected"
        if ! ldconfig -p | grep -q libnvidia-ml; then
            log_warning "NVIDIA drivers may not be properly installed"
            log_info "For optimal performance, ensure NVIDIA drivers are installed:"
            log_info "https://developer.nvidia.com/cuda-downloads"
        fi
    elif lspci | grep -i amd >/dev/null 2>&1; then
        log_info "AMD GPU detected"
        if ! ldconfig -p | grep -q libdrm_amdgpu; then
            log_warning "AMD drivers may not be properly installed"
        fi
    else
        log_info "No dedicated GPU detected (CPU-only mode will be used)"
    fi
}

install_rife_binary() {
    log_info "Installing RIFE-NCNN-Vulkan binary..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR/bin"
    
    RIFE_URL="https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip"
    RIFE_ARCHIVE="rife-ncnn-vulkan-ubuntu.zip"
    RIFE_DIR="rife-ncnn-vulkan-20221029-ubuntu"
    
    cd "$INSTALL_DIR"
    
    # Download RIFE binary
    if [[ -f "$RIFE_ARCHIVE" ]]; then
        log_info "RIFE archive already exists, skipping download"
    else
        log_info "Downloading RIFE binary..."
        wget -q --show-progress "$RIFE_URL" -O "$RIFE_ARCHIVE"
    fi
    
    # Extract binary
    if [[ -d "$RIFE_DIR" ]]; then
        rm -rf "$RIFE_DIR"
    fi
    
    log_info "Extracting RIFE binary..."
    unzip -q "$RIFE_ARCHIVE"
    
    # Move binary to bin directory
    if [[ -f "$RIFE_DIR/rife-ncnn-vulkan" ]]; then
        mv "$RIFE_DIR/rife-ncnn-vulkan" "$INSTALL_DIR/bin/"
        chmod +x "$INSTALL_DIR/bin/rife-ncnn-vulkan"
        log_success "RIFE binary installed: $INSTALL_DIR/bin/rife-ncnn-vulkan"
    else
        log_error "RIFE binary not found in archive"
        exit 1
    fi
    
    # Clean up
    rm -rf "$RIFE_DIR" "$RIFE_ARCHIVE"
}

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    cd "$PROJECT_DIR"
    
    # Install FlowForge in development mode
    pip3 install -e .
    
    log_success "Python dependencies installed"
}

test_installation() {
    log_info "Testing installation..."
    
    # Test RIFE binary
    if [[ -x "$INSTALL_DIR/bin/rife-ncnn-vulkan" ]]; then
        log_info "Testing RIFE binary..."
        if "$INSTALL_DIR/bin/rife-ncnn-vulkan" -h >/dev/null 2>&1; then
            log_success "RIFE binary test passed"
        else
            log_warning "RIFE binary test failed (may need GPU drivers)"
        fi
    else
        log_error "RIFE binary not found or not executable"
    fi
    
    # Test FlowForge CLI
    if check_command "flowforge"; then
        log_info "Testing FlowForge CLI..."
        if flowforge --version >/dev/null 2>&1; then
            log_success "FlowForge CLI test passed"
        else
            log_warning "FlowForge CLI test failed"
        fi
    else
        log_warning "FlowForge CLI not in PATH"
        log_info "You may need to add ~/.local/bin to your PATH"
    fi
}

print_next_steps() {
    log_info "Installation complete!"
    echo
    echo "📁 Installation directory: $INSTALL_DIR"
    echo "🔧 RIFE binary: $INSTALL_DIR/bin/rife-ncnn-vulkan"
    echo
    echo "🚀 Next steps:"
    echo "1. Download RIFE models: flowforge setup"
    echo "2. Test the installation: flowforge test"
    echo "3. Interpolate your first video:"
    echo "   flowforge interpolate input.mp4 -o output.mp4 --multiplier 2"
    echo
    echo "💡 If 'flowforge' command is not found, add this to your ~/.bashrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo
}

main() {
    echo "🔧 FlowForge Dependencies Installation"
    echo "====================================="
    echo
    
    check_system
    check_ffmpeg
    check_vulkan
    install_rife_binary
    install_python_deps
    test_installation
    
    echo
    print_next_steps
}

# Handle script interruption
trap 'log_error "Installation interrupted"; exit 1' INT TERM

# Check if running with bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi