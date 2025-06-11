# Makefile for Enhanced Autoencoder Anomaly Detection

.PHONY: help install install-dev test test-cov lint format clean train evaluate docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install the package and dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black)"
	@echo "  clean        - Clean build artifacts"
	@echo "  train        - Train the model"
	@echo "  evaluate     - Evaluate trained model"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=enhanced_autoencoder --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 enhanced_autoencoder.py tests/ --max-line-length=100
	pylint enhanced_autoencoder.py --disable=C0103,R0903,R0913

format:
	black enhanced_autoencoder.py tests/ --line-length=100
	isort enhanced_autoencoder.py tests/

format-check:
	black --check enhanced_autoencoder.py tests/ --line-length=100
	isort --check-only enhanced_autoencoder.py tests/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage