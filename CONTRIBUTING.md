# Contributing to FlowForge

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/flowforge.git
cd flowforge
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
flowforge setup
```

## Running Tests

```bash
pytest tests/
ruff check flowforge/
```

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a PR with a clear description

## Code Style

- Python 3.11+ with type hints
- Docstrings on all public functions/classes
- Use `ruff` for linting
- Keep functions focused and small

## Architecture

- `flowforge/core/` — Interpolation engine (offline processing)
- `flowforge/playback/` — Real-time playback (mpv/VapourSynth)
- `flowforge/gui/` — GUI application
- `flowforge/utils/` — Shared utilities

## Reporting Issues

- Include your OS, GPU, Python version
- Include the full error traceback
- For video issues, include `flowforge info <video>` output
