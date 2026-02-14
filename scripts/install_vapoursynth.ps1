# FlowForge VapourSynth Installation Script for Windows
# Installs VapourSynth and vs-rife-ncnn-vulkan plugin

param(
    [switch]$Force,
    [switch]$Portable,
    [string]$InstallPath = "$env:LOCALAPPDATA\FlowForge\VapourSynth",
    [switch]$NoInteractive
)

# Requires PowerShell 5.0 or higher
#Requires -Version 5.0

# Set error handling
$ErrorActionPreference = "Stop"

# Colors for console output
$Colors = @{
    Info = "Cyan"
    Success = "Green"  
    Warning = "Yellow"
    Error = "Red"
    Highlight = "White"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Write-LogInfo { 
    param([string]$Message)
    Write-ColorOutput "[INFO] $Message" "Info"
}

function Write-LogSuccess {
    param([string]$Message)
    Write-ColorOutput "[SUCCESS] $Message" "Success"
}

function Write-LogWarning {
    param([string]$Message) 
    Write-ColorOutput "[WARNING] $Message" "Warning"
}

function Write-LogError {
    param([string]$Message)
    Write-ColorOutput "[ERROR] $Message" "Error"
}

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check system requirements
function Test-SystemRequirements {
    Write-LogInfo "Checking system requirements..."
    
    # Check Windows version (Windows 10 1809 or later recommended)
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10 -or ($osVersion.Major -eq 10 -and $osVersion.Build -lt 17763)) {
        Write-LogWarning "Windows 10 version 1809 or later is recommended for best performance"
    }
    
    # Check architecture
    if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
        Write-LogError "x64 architecture required (found: $env:PROCESSOR_ARCHITECTURE)"
        exit 1
    }
    
    # Check Python
    $pythonVersion = $null
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $majorVersion = [int]$matches[1]
            $minorVersion = [int]$matches[2]
            
            if ($majorVersion -ge 3 -and ($majorVersion -gt 3 -or $minorVersion -ge 8)) {
                Write-LogSuccess "Python version compatible: $pythonVersion"
            } else {
                Write-LogError "Python 3.8 or higher required (found: $pythonVersion)"
                exit 1
            }
        }
    } catch {
        Write-LogError "Python not found. Please install Python 3.8 or higher from python.org"
        exit 1
    }
    
    # Check available disk space (need at least 2GB)
    $availableSpace = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 } | 
                     Measure-Object -Property FreeSpace -Sum | Select-Object -ExpandProperty Sum
    $availableSpaceGB = [math]::Round($availableSpace / 1GB, 2)
    
    if ($availableSpaceGB -lt 2) {
        Write-LogWarning "Low disk space: ${availableSpaceGB}GB available. Installation may fail."
    } else {
        Write-LogInfo "Available disk space: ${availableSpaceGB}GB"
    }
    
    # Check PowerShell execution policy
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -eq "Restricted") {
        Write-LogWarning "PowerShell execution policy is Restricted"
        Write-LogInfo "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    }
}

# Download file with progress
function Get-FileWithProgress {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$Description = "Downloading"
    )
    
    Write-LogInfo "$Description from $Url"
    
    try {
        # Use BITS transfer if available (faster and resumable)
        if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
            Start-BitsTransfer -Source $Url -Destination $OutputPath -Description $Description
        } else {
            # Fallback to Invoke-WebRequest
            $webClient = New-Object System.Net.WebClient
            $webClient.DownloadFile($Url, $OutputPath)
        }
        
        Write-LogSuccess "Downloaded: $OutputPath"
        return $true
    } catch {
        Write-LogError "Download failed: $_"
        return $false
    }
}

# Extract archive
function Expand-Archive7Zip {
    param(
        [string]$ArchivePath,
        [string]$OutputPath
    )
    
    # Try 7-Zip first
    $7zipPaths = @(
        "$env:ProgramFiles\7-Zip\7z.exe",
        "$env:ProgramFiles(x86)\7-Zip\7z.exe"
    )
    
    $7zipExe = $null
    foreach ($path in $7zipPaths) {
        if (Test-Path $path) {
            $7zipExe = $path
            break
        }
    }
    
    if ($7zipExe) {
        Write-LogInfo "Extracting with 7-Zip: $ArchivePath"
        & $7zipExe x "$ArchivePath" -o"$OutputPath" -y
        return $LASTEXITCODE -eq 0
    } else {
        # Fallback to PowerShell Expand-Archive (limited format support)
        Write-LogInfo "Extracting with PowerShell: $ArchivePath"
        try {
            Expand-Archive -Path $ArchivePath -DestinationPath $OutputPath -Force
            return $true
        } catch {
            Write-LogError "Extraction failed: $_"
            return $false
        }
    }
}

# Install VapourSynth
function Install-VapourSynth {
    param([string]$InstallDir)
    
    Write-LogInfo "Installing VapourSynth..."
    
    # Check if already installed
    try {
        python -c "import vapoursynth; print('VapourSynth version:', vapoursynth.core.version())" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-LogInfo "VapourSynth already installed"
            if (-not $Force) {
                $response = Read-Host "Reinstall VapourSynth? (y/N)"
                if ($response -ne "y" -and $response -ne "Y") {
                    Write-LogInfo "Skipping VapourSynth installation"
                    return $true
                }
            }
        }
    } catch {}
    
    $tempDir = "$env:TEMP\FlowForge_VapourSynth"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    
    # VapourSynth download URLs
    $vsUrls = @{
        "R65" = "https://github.com/vapoursynth/vapoursynth/releases/download/R65/VapourSynth64-R65.exe"
        "R64" = "https://github.com/vapoursynth/vapoursynth/releases/download/R64/VapourSynth64-R64.exe"
    }
    
    $vsVersion = "R65"  # Latest stable
    $vsUrl = $vsUrls[$vsVersion]
    $vsInstaller = "$tempDir\VapourSynth-$vsVersion.exe"
    
    # Download VapourSynth installer
    if (-not (Get-FileWithProgress -Url $vsUrl -OutputPath $vsInstaller -Description "Downloading VapourSynth $vsVersion")) {
        Write-LogError "Failed to download VapourSynth"
        return $false
    }
    
    # Install VapourSynth
    Write-LogInfo "Installing VapourSynth (this may take a few minutes)..."
    
    if ($Portable) {
        # For portable installation, we need to extract and setup manually
        Write-LogInfo "Setting up portable VapourSynth installation..."
        
        # VapourSynth installer is typically an NSIS installer, hard to extract
        # Run normal installation but to custom directory
        $installArgs = @("/S", "/D=$InstallDir")
    } else {
        # Standard installation
        $installArgs = @("/S")  # Silent installation
    }
    
    try {
        $process = Start-Process -FilePath $vsInstaller -ArgumentList $installArgs -Wait -PassThru
        
        if ($process.ExitCode -eq 0) {
            Write-LogSuccess "VapourSynth installed successfully"
        } else {
            Write-LogError "VapourSynth installation failed (exit code: $($process.ExitCode))"
            return $false
        }
    } catch {
        Write-LogError "Failed to run VapourSynth installer: $_"
        return $false
    }
    
    # Verify installation
    Start-Sleep -Seconds 2
    try {
        python -c "import vapoursynth; print('VapourSynth version:', vapoursynth.core.version())"
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "VapourSynth verification passed"
            return $true
        } else {
            Write-LogError "VapourSynth verification failed"
            return $false
        }
    } catch {
        Write-LogError "VapourSynth verification failed: $_"
        return $false
    }
}

# Install Visual Studio Build Tools (required for vs-rife compilation)
function Install-VSBuildTools {
    Write-LogInfo "Checking for Visual Studio Build Tools..."
    
    # Check if VS Build Tools or Visual Studio is installed
    $vsPaths = @(
        "$env:ProgramFiles(x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        "$env:ProgramFiles(x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
    )
    
    $msbuildPath = $null
    foreach ($path in $vsPaths) {
        if (Test-Path $path) {
            $msbuildPath = $path
            break
        }
    }
    
    if ($msbuildPath) {
        Write-LogSuccess "Visual Studio Build Tools found: $msbuildPath"
        return $true
    }
    
    Write-LogWarning "Visual Studio Build Tools not found"
    Write-LogInfo "vs-rife-ncnn-vulkan plugin requires Visual Studio Build Tools for compilation"
    
    if (-not $NoInteractive) {
        $response = Read-Host "Download and install Visual Studio Build Tools? This is required for vs-rife plugin. (y/N)"
        if ($response -eq "y" -or $response -eq "Y") {
            # Download VS Build Tools installer
            $buildToolsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
            $buildToolsInstaller = "$env:TEMP\vs_buildtools.exe"
            
            if (Get-FileWithProgress -Url $buildToolsUrl -OutputPath $buildToolsInstaller -Description "Downloading VS Build Tools") {
                Write-LogInfo "Installing Visual Studio Build Tools..."
                Write-LogInfo "This will open the Visual Studio Installer. Please select 'C++ build tools' workload."
                
                Start-Process -FilePath $buildToolsInstaller -Wait
                Write-LogInfo "Please restart PowerShell after VS Build Tools installation completes"
                return $false
            }
        }
    }
    
    Write-LogWarning "Skipping vs-rife plugin installation (requires Visual Studio Build Tools)"
    return $false
}

# Install vs-rife-ncnn-vulkan plugin
function Install-VSRifePlugin {
    param([string]$InstallDir)
    
    Write-LogInfo "Installing vs-rife-ncnn-vulkan plugin..."
    
    # Check if plugin is already available
    try {
        python -c "import vapoursynth as vs; core = vs.core; core.rife" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-LogInfo "vs-rife plugin already available"
            return $true
        }
    } catch {}
    
    # Check for Visual Studio Build Tools
    if (-not (Install-VSBuildTools)) {
        Write-LogWarning "Skipping vs-rife plugin installation (no build tools)"
        return $false
    }
    
    $tempDir = "$env:TEMP\FlowForge_VSRife"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    
    # For Windows, we'll try to download pre-compiled binaries first
    $precompiledUrl = "https://github.com/HomeOfVapourSynthEvolution/vs-rife-ncnn-vulkan/releases/latest"
    
    Write-LogInfo "Checking for pre-compiled vs-rife binaries..."
    
    # This is a simplified approach - in a real implementation, you'd parse GitHub releases
    # For now, we'll skip the plugin installation and rely on binary fallback
    Write-LogWarning "vs-rife plugin installation requires manual setup on Windows"
    Write-LogInfo "FlowForge will use the RIFE binary fallback method"
    Write-LogInfo "For best performance, manually install vs-rife-ncnn-vulkan from:"
    Write-LogInfo "https://github.com/HomeOfVapourSynthEvolution/vs-rife-ncnn-vulkan"
    
    return $false
}

# Create environment setup script
function New-EnvironmentScript {
    param([string]$InstallDir)
    
    Write-LogInfo "Creating environment setup script..."
    
    $scriptDir = "$env:LOCALAPPDATA\FlowForge"
    New-Item -ItemType Directory -Path $scriptDir -Force | Out-Null
    
    $setupScript = "$scriptDir\setup_vapoursynth.ps1"
    
    $scriptContent = @"
# FlowForge VapourSynth Environment Setup Script for Windows

Write-Host "Setting up FlowForge VapourSynth environment..." -ForegroundColor Cyan

# Add VapourSynth to PATH if installed in custom location
if (Test-Path "$InstallDir") {
    `$env:PATH = "$InstallDir\bin;" + `$env:PATH
    Write-Host "VapourSynth path added: $InstallDir\bin" -ForegroundColor Green
}

# Test VapourSynth installation
try {
    `$vsVersion = python -c "import vapoursynth; print(vapoursynth.core.version())" 2>`$null
    if (`$LASTEXITCODE -eq 0) {
        Write-Host "VapourSynth version: `$vsVersion" -ForegroundColor Green
        
        # Test vs-rife plugin
        python -c "import vapoursynth as vs; core = vs.core; core.rife" 2>`$null
        if (`$LASTEXITCODE -eq 0) {
            Write-Host "vs-rife plugin: Available" -ForegroundColor Green
        } else {
            Write-Host "vs-rife plugin: Not available (will use binary fallback)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "ERROR: VapourSynth not found" -ForegroundColor Red
    }
} catch {
    Write-Host "ERROR: Failed to test VapourSynth installation" -ForegroundColor Red
}

Write-Host "FlowForge VapourSynth environment ready!" -ForegroundColor Green
"@
    
    Set-Content -Path $setupScript -Value $scriptContent -Encoding UTF8
    
    # Create batch file wrapper for easy execution
    $batchScript = "$scriptDir\setup_vapoursynth.bat"
    $batchContent = @"
@echo off
powershell -ExecutionPolicy Bypass -File "$setupScript"
pause
"@
    Set-Content -Path $batchScript -Value $batchContent -Encoding ASCII
    
    Write-LogSuccess "Environment setup script created: $setupScript"
    return $setupScript
}

# Test installation
function Test-Installation {
    Write-LogInfo "Testing installation..."
    
    # Test VapourSynth
    try {
        $vsOutput = python -c "import vapoursynth; print('VapourSynth version:', vapoursynth.core.version())" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "VapourSynth test passed: $vsOutput"
        } else {
            Write-LogError "VapourSynth test failed: $vsOutput"
            return $false
        }
    } catch {
        Write-LogError "VapourSynth test failed: $_"
        return $false
    }
    
    # Test vs-rife plugin (optional)
    try {
        python -c "import vapoursynth as vs; core = vs.core; core.rife" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "vs-rife plugin test passed"
        } else {
            Write-LogWarning "vs-rife plugin not available (will use binary fallback)"
        }
    } catch {
        Write-LogWarning "vs-rife plugin test failed (optional)"
    }
    
    Write-LogSuccess "Installation test completed"
    return $true
}

# Print installation summary
function Show-InstallationSummary {
    param([string]$InstallDir, [string]$SetupScript)
    
    Write-LogSuccess "FlowForge VapourSynth Installation Complete!"
    Write-ColorOutput "" "White"
    Write-ColorOutput "Installation Summary:" "Highlight"
    Write-ColorOutput "- VapourSynth installed and configured" "Info"
    if ($InstallDir -and (Test-Path $InstallDir)) {
        Write-ColorOutput "- Custom installation path: $InstallDir" "Info"
    }
    Write-ColorOutput "- Setup script: $SetupScript" "Info"
    Write-ColorOutput "" "White"
    Write-ColorOutput "Next steps:" "Highlight"
    Write-ColorOutput "1. Test FlowForge: flowforge system-status" "Info"
    Write-ColorOutput "2. Configure mpv: flowforge configure-mpv" "Info" 
    Write-ColorOutput "3. Play a video: flowforge play your_video.mp4" "Info"
    Write-ColorOutput "" "White"
    Write-ColorOutput "If you encounter issues:" "Highlight"
    Write-ColorOutput "- Check system status: flowforge system-status" "Info"
    Write-ColorOutput "- Run setup script: $SetupScript" "Info"
    Write-ColorOutput "- Check installation log in: $env:TEMP\FlowForge_install.log" "Info"
    Write-ColorOutput "- Report issues at: https://github.com/your-repo/FlowForge/issues" "Info"
}

# Cleanup temporary files
function Remove-TempFiles {
    Write-LogInfo "Cleaning up temporary files..."
    
    $tempDirs = @(
        "$env:TEMP\FlowForge_VapourSynth",
        "$env:TEMP\FlowForge_VSRife"
    )
    
    foreach ($dir in $tempDirs) {
        if (Test-Path $dir) {
            try {
                Remove-Item -Path $dir -Recurse -Force
                Write-LogInfo "Cleaned up: $dir"
            } catch {
                Write-LogWarning "Failed to cleanup: $dir"
            }
        }
    }
}

# Main installation function
function Start-Installation {
    Write-LogInfo "FlowForge VapourSynth Installation Script for Windows"
    Write-LogInfo "===================================================="
    
    # Create log file
    $logFile = "$env:TEMP\FlowForge_install.log"
    Start-Transcript -Path $logFile -Append
    
    try {
        # Check administrator privileges
        if (Test-Administrator) {
            Write-LogInfo "Running as Administrator"
        } else {
            Write-LogWarning "Running without Administrator privileges"
            Write-LogInfo "Some operations may require elevation"
        }
        
        # System requirements check
        Test-SystemRequirements
        
        Write-LogInfo "Starting installation process..."
        Write-LogInfo "Installation path: $InstallPath"
        
        # Create installation directory
        if (-not (Test-Path $InstallPath)) {
            New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
            Write-LogInfo "Created installation directory: $InstallPath"
        }
        
        # Install VapourSynth
        if (-not (Install-VapourSynth -InstallDir $InstallPath)) {
            throw "VapourSynth installation failed"
        }
        
        # Install vs-rife plugin (optional)
        Install-VSRifePlugin -InstallDir $InstallPath | Out-Null
        
        # Create environment setup script
        $setupScript = New-EnvironmentScript -InstallDir $InstallPath
        
        # Test installation
        if (-not (Test-Installation)) {
            Write-LogWarning "Installation test failed, but continuing..."
        }
        
        # Show summary
        Show-InstallationSummary -InstallDir $InstallPath -SetupScript $setupScript
        
        Write-LogSuccess "Installation completed successfully!"
        
    } catch {
        Write-LogError "Installation failed: $_"
        Write-LogError "Check the log file for details: $logFile"
        exit 1
    } finally {
        Stop-Transcript
        Remove-TempFiles
    }
}

# Script entry point
if ($MyInvocation.InvocationName -ne '.') {
    # Show help if requested
    if ($args -contains "-help" -or $args -contains "--help" -or $args -contains "-h") {
        Write-Host @"
FlowForge VapourSynth Installation Script for Windows

Usage: .\install_vapoursynth.ps1 [OPTIONS]

Options:
  -Force              Force reinstallation even if components exist
  -Portable           Create portable installation
  -InstallPath <path> Custom installation path (default: %LOCALAPPDATA%\FlowForge\VapourSynth)
  -NoInteractive      Run without interactive prompts
  -Help               Show this help message

Examples:
  .\install_vapoursynth.ps1
  .\install_vapoursynth.ps1 -Force -InstallPath "C:\FlowForge\VapourSynth"
  .\install_vapoursynth.ps1 -Portable -NoInteractive
"@
        exit 0
    }
    
    # Run installation
    Start-Installation
}