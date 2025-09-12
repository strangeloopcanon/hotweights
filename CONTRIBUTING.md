# Contributing

Thanks for your interest in Hotweights! This document outlines how to develop and contribute effectively.

## Getting Started

- Python 3.10+
- Install dev deps:
  ```
  python -m venv .venv && source .venv/bin/activate
  python -m pip install -U pip
  pip install -e .[dev]
  ```
- Optional extras: `pyzmq`, `mpi4py`, `ucx-py`, `prometheus-client`, `redis`, `cupy`, `kvikio` depending on features you work on.

## Code Style & Lint

- 4‑space indents, 88‑char soft wrap. Type hints everywhere.
- Tools:
  - Ruff: `ruff --fix .`
  - Black: `black .`
  - Mypy (optional where applicable): `mypy hotweights`

## Tests

- Run all tests with `pytest -q`.
- Add unit tests for new functionality; avoid flaky tests (mark gpu/mpi if needed).

## Structure

- See `docs/ARCHITECTURE.md` for an overview and `docs/TUNING.md` for knobs.
- Key modules:
  - Planning: `hotweights/planner_bodo.py`
  - Transports: `hotweights/transport/*`
  - Coordinator: `hotweights/coordinator/*`
  - Adapters: `hotweights/adapters/*`
  - CLI: `hotweights/cli.py`

## PRs

- Use descriptive titles and a brief summary of changes and rationale.
- Link to issues/milestones if applicable.
- Keep changes focused; update docs if behavior changes.

## Security

- Don’t commit secrets; prefer environment variables.
- If handling authentication (e.g., handle tokens), use HMAC signing as documented.

## CI

- CI runs tests and a small fallback bench on PRs. For now, push builds are disabled to reduce noise.

Thanks again for contributing!

