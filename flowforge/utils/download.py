"""Download utilities for RIFE models and binaries."""

import hashlib
import logging
import os
import platform
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlopen, urlretrieve

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Exception raised when downloads fail."""
    pass


class ModelDownloader:
    """Download and manage RIFE models and binaries."""
    
    # RIFE-NCNN-Vulkan releases
    RIFE_RELEASES = {
        "linux": {
            "url": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip",
            "filename": "rife-ncnn-vulkan-20221029-ubuntu.zip",
            "binary_name": "rife-ncnn-vulkan",
            "extract_dir": "rife-ncnn-vulkan-20221029-ubuntu"
        },
        "windows": {
            "url": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip",
            "filename": "rife-ncnn-vulkan-20221029-windows.zip",
            "binary_name": "rife-ncnn-vulkan.exe",
            "extract_dir": "rife-ncnn-vulkan-20221029-windows"
        }
    }
    
    # RIFE model files
    RIFE_MODELS = {
        "rife-v4.6": {
            "flownet.bin": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-v4.6-models.zip",
            "flownet.param": None  # Included in the same zip
        },
        "rife-v4.15-lite": {
            "flownet.bin": "https://github.com/hzwer/Practical-RIFE/releases/download/v4.15.0/rife-v4.15-lite.zip",
            "flownet.param": None  # Included in the same zip
        }
    }
    
    def __init__(self, install_dir: Optional[Path] = None):
        """Initialize model downloader.
        
        Args:
            install_dir: Directory to install models and binaries
        """
        if install_dir is None:
            install_dir = Path.home() / ".flowforge"
        
        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        self.binary_dir = self.install_dir / "bin"
        self.models_dir = self.install_dir / "models"
        
        self.binary_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Model downloader initialized: {self.install_dir}")
    
    @property
    def platform(self) -> str:
        """Get current platform identifier."""
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        elif system == "windows":
            return "windows"
        else:
            raise DownloadError(f"Unsupported platform: {system}")
    
    def get_rife_binary_path(self) -> Path:
        """Get path to RIFE binary."""
        platform_info = self.RIFE_RELEASES[self.platform]
        binary_path = self.binary_dir / platform_info["binary_name"]
        return binary_path
    
    def is_rife_installed(self) -> bool:
        """Check if RIFE binary is installed."""
        binary_path = self.get_rife_binary_path()
        return binary_path.exists() and binary_path.is_file()
    
    def download_rife_binary(self, force: bool = False) -> Path:
        """Download RIFE-NCNN-Vulkan binary.
        
        Args:
            force: Force re-download even if already exists
            
        Returns:
            Path to downloaded binary
            
        Raises:
            DownloadError: If download fails
        """
        binary_path = self.get_rife_binary_path()
        
        if binary_path.exists() and not force:
            logger.info(f"RIFE binary already exists: {binary_path}")
            return binary_path
        
        platform_info = self.RIFE_RELEASES[self.platform]
        url = platform_info["url"]
        filename = platform_info["filename"]
        extract_dir = platform_info["extract_dir"]
        
        logger.info(f"Downloading RIFE binary for {self.platform}")
        
        # Download archive
        archive_path = self.install_dir / filename
        self._download_file(url, archive_path)
        
        try:
            # Extract archive
            logger.info("Extracting RIFE binary...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(self.install_dir)
            
            # Move binary to bin directory
            extracted_dir = self.install_dir / extract_dir
            extracted_binary = extracted_dir / platform_info["binary_name"]
            if not extracted_binary.exists():
                raise DownloadError(f"Binary not found in archive: {extracted_binary}")
            
            shutil.move(str(extracted_binary), str(binary_path))
            
            # Make executable on Unix systems
            if self.platform == "linux":
                binary_path.chmod(0o755)
            
            # Extract bundled models before cleaning up
            for item in extracted_dir.iterdir():
                if item.is_dir() and item.name.startswith("rife-"):
                    dest_model = self.models_dir / item.name
                    if not dest_model.exists():
                        shutil.copytree(str(item), str(dest_model))
                        logger.info(f"Extracted bundled model: {item.name}")
            
            # Clean up
            archive_path.unlink()
            shutil.rmtree(extracted_dir)
            
            logger.info(f"RIFE binary installed: {binary_path}")
            return binary_path
            
        except Exception as e:
            # Clean up on failure
            if archive_path.exists():
                archive_path.unlink()
            if binary_path.exists():
                binary_path.unlink()
            
            raise DownloadError(f"Failed to extract RIFE binary: {e}")
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to model directory."""
        return self.models_dir / model_name
    
    def is_model_installed(self, model_name: str) -> bool:
        """Check if a model is installed.
        
        Args:
            model_name: Name of the model (e.g., 'rife-v4.6')
            
        Returns:
            True if model is installed
        """
        if model_name not in self.RIFE_MODELS:
            return False
        
        model_dir = self.get_model_path(model_name)
        if not model_dir.exists():
            return False
        
        # Check if required files exist
        required_files = ["flownet.bin", "flownet.param"]
        for filename in required_files:
            if not (model_dir / filename).exists():
                return False
        
        return True
    
    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download a RIFE model.
        
        Args:
            model_name: Name of the model to download
            force: Force re-download even if already exists
            
        Returns:
            Path to model directory
            
        Raises:
            DownloadError: If download fails
        """
        if model_name not in self.RIFE_MODELS:
            raise DownloadError(f"Unknown model: {model_name}")
        
        model_dir = self.get_model_path(model_name)
        
        if self.is_model_installed(model_name) and not force:
            logger.info(f"Model already exists: {model_name}")
            return model_dir
        
        model_info = self.RIFE_MODELS[model_name]
        url = model_info["flownet.bin"]  # URL to model archive
        
        logger.info(f"Downloading model: {model_name}")
        
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model archive
        archive_path = self.install_dir / f"{model_name}.zip"
        self._download_file(url, archive_path)
        
        try:
            # Extract model files
            logger.info("Extracting model files...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Extract all files to model directory
                zip_ref.extractall(model_dir)
            
            # Clean up archive
            archive_path.unlink()
            
            # Verify required files
            if not self.is_model_installed(model_name):
                raise DownloadError(f"Model files incomplete after extraction: {model_name}")
            
            logger.info(f"Model installed: {model_name} -> {model_dir}")
            return model_dir
            
        except Exception as e:
            # Clean up on failure
            if archive_path.exists():
                archive_path.unlink()
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            raise DownloadError(f"Failed to extract model {model_name}: {e}")
    
    def _download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> None:
        """Download a file with progress bar.
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            chunk_size: Download chunk size in bytes
            
        Raises:
            DownloadError: If download fails
        """
        try:
            logger.debug(f"Downloading {url} -> {dest_path}")
            
            # Get file size for progress bar
            response = requests.head(url, allow_redirects=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.debug(f"Downloaded: {dest_path}")
            
        except requests.RequestException as e:
            if dest_path.exists():
                dest_path.unlink()
            raise DownloadError(f"Failed to download {url}: {e}")
        except Exception as e:
            if dest_path.exists():
                dest_path.unlink()
            raise DownloadError(f"Download error: {e}")
    
    def list_available_models(self) -> list:
        """List all available RIFE models."""
        return list(self.RIFE_MODELS.keys())
    
    def list_installed_models(self) -> list:
        """List installed RIFE models."""
        installed = []
        for model_name in self.RIFE_MODELS:
            if self.is_model_installed(model_name):
                installed.append(model_name)
        return installed
    
    def get_install_info(self) -> Dict:
        """Get installation information.
        
        Returns:
            Dictionary with installation details
        """
        info = {
            "install_dir": str(self.install_dir),
            "binary_dir": str(self.binary_dir),
            "models_dir": str(self.models_dir),
            "platform": self.platform,
            "rife_binary_installed": self.is_rife_installed(),
            "rife_binary_path": str(self.get_rife_binary_path()),
            "available_models": self.list_available_models(),
            "installed_models": self.list_installed_models(),
        }
        
        if self.is_rife_installed():
            binary_path = self.get_rife_binary_path()
            info["rife_binary_size"] = binary_path.stat().st_size
        
        return info
    
    def setup_all(self, models: Optional[list] = None) -> Dict:
        """Download and setup everything.
        
        Args:
            models: List of models to download (default: ['rife-v4.6'])
            
        Returns:
            Setup results dictionary
        """
        if models is None:
            models = ["rife-v4.6"]
        
        results = {
            "binary": False,
            "models": {},
            "errors": []
        }
        
        # Download binary
        try:
            self.download_rife_binary()
            results["binary"] = True
            logger.info("RIFE binary setup complete")
        except Exception as e:
            error_msg = f"Failed to setup RIFE binary: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Download models
        for model_name in models:
            try:
                self.download_model(model_name)
                results["models"][model_name] = True
                logger.info(f"Model {model_name} setup complete")
            except Exception as e:
                error_msg = f"Failed to setup model {model_name}: {e}"
                results["errors"].append(error_msg)
                results["models"][model_name] = False
                logger.error(error_msg)
        
        return results