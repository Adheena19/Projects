#!/usr/bin/env python3
"""
Setup script for Enhanced Autoencoder Anomaly Detection
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="enhanced-autoencoder-anomaly-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enhanced autoencoder-based anomaly detection for network intrusion detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-autoencoder-nslkdd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pytest-cov>=3.0.0",
            "pre-commit>=2.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "shap>=0.40.0",
        ],
        "optimization": [
            "optuna>=3.0.0",
            "hyperopt>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "enhanced-autoencoder=enhanced_autoencoder:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="anomaly-detection autoencoder machine-learning cybersecurity intrusion-detection",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/enhanced-autoencoder-nslkdd/issues",
        "Source": "https://github.com/yourusername/enhanced-autoencoder-nslkdd",
        "Documentation": "https://github.com/yourusername/enhanced-autoencoder-nslkdd/wiki",
    },
)
