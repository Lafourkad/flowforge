"""Setup script for FlowForge."""

from setuptools import setup, find_packages
import os

# Read the version from __init__.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "flowforge", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

# Read the README file
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="flowforge",
    version=version,
    author="FlowForge Team",
    author_email="contact@flowforge.dev",
    description="High-quality video frame interpolation using RIFE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flowforge/flowforge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "tqdm>=4.64.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.6.0",
        "requests>=2.28.0",
        "scikit-image>=0.19.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "flowforge=flowforge.cli:main",
        ],
    },
)