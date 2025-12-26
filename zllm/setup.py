"""
Mini-vLLM: Setup Script
=======================

This script configures the Python package installation, including
building the C++/CUDA extension module.

Usage:
    pip install -e .           # Development install
    pip install .              # Standard install
    python setup.py build_ext  # Build extension only
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """
    Custom build extension that uses CMake to build C++/CUDA code.

    This allows us to use CMake for the complex CUDA compilation while
    still integrating with Python's packaging tools.
    """

    def build_extensions(self):
        # Check for CMake
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build this package")

        # Create build directory
        build_dir = Path(self.build_temp).absolute()
        build_dir.mkdir(parents=True, exist_ok=True)

        # Source directory
        source_dir = Path(__file__).parent.absolute()

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={self.build_lib}/mini_vllm",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # Build arguments
        build_args = [
            "--config",
            "Release",
            "--parallel",
            str(os.cpu_count() or 1),
        ]

        # Run CMake configure
        print(f"[CMake] Configuring in {build_dir}")
        subprocess.check_call(["cmake", str(source_dir)] + cmake_args, cwd=build_dir)

        # Run CMake build
        print(f"[CMake] Building...")
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)


# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


setup(
    name="mini_vllm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Mini vLLM - Educational LLM Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mini_vllm",
    # Package configuration
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    # Include CUDA extension
    ext_modules=[],  # Handled by CMakeBuildExt
    cmdclass={"build_ext": CMakeBuildExt},
    # Python version requirement
    python_requires=">=3.10",
    # Dependencies
    install_requires=[
        "torch>=2.1.0",
        "tiktoken>=0.5.1",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
    ],
    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    # Entry points
    entry_points={
        "console_scripts": [
            "mini-vllm-server=mini_vllm.server:main",
        ],
    },
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
