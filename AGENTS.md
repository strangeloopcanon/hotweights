# Repository Guidelines

## Project Structure & Module Organization
- Package root: `hotweights/` (planned). Key modules: `planner_bodo.py`, `transport/mpi_stream.py`, `coordinator/`, `staging/`, `adapters/` (e.g., `vllm_ext.py`), `telemetry/`, and `cli.py`.
- Tests live in `tests/` (e.g., `tests/test_planner.py`, `tests/test_stream_small.py`).
- Examples in `examples/` (CLI and integration demos).
- Project config in `pyproject.toml`; repo spec is in `idea.md`.

## Build, Test, and Development Commands
- Create env + install (editable):
  - `python -m venv .venv && source .venv/bin/activate`
  - `python -m pip install -U pip && pip install -e .`
- Run tests: `pytest -q` (single file: `pytest tests/test_planner.py -q`).
- CLI (when package is present): `python -m hotweights.cli status`.
- MPI examples (if mpi4py installed): `mpirun -n 2 pytest -q -k stream_small`.

## Coding Style & Naming Conventions
- Python 3.10+; 4‑space indent; 88‑char soft wrap.
- Use type hints everywhere; module docstrings for public modules.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants; files `snake_case.py`; tests `test_*.py`.
- Tools: Black (format), Ruff (lint), isort (imports), mypy (types). Example: `ruff --fix . && black . && isort .`.

## Testing Guidelines
- Framework: Pytest. Unit tests for manifest, planner, transport; integration tests for vLLM/trainer adapters.
- Markers: `@pytest.mark.slow`, `gpu`, `mpi`. Default CI skips slow/GPU/MPI: `pytest -m "not slow and not gpu and not mpi"`.
- Aim ≥80% coverage for pure‑Python modules; validate hashing and delta determinism with fixed fixtures.

## Commit & Pull Request Guidelines
- Follow Conventional Commits: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `chore:`. Example: `perf(transport): double-buffer MPI Ibcast`.
- PRs must include: scope/intent, linked issue, testing notes (commands + results), and if touching `planner_bodo.py` or `transport/mpi_stream.py`, a before/after timing snippet.
- Keep PRs small and focused; update `examples/` or docs when behavior changes.

## Security & Configuration Tips
- Never commit weights/checkpoints; store outside repo and reference via paths/URIs.
- Verify content hashes before commit; treat `.env`/secrets carefully (use environment variables, not code).
- Large I/O: prefer `mmap` and pinned‑host buffers; avoid unbounded temp files.

## Agent‑Specific Instructions
- Do not rename files or public APIs without consensus.
- Prefer minimal, surgical patches; align with `idea.md` and this guide.
- When uncertain about behavior, add a failing test first, then implement.
