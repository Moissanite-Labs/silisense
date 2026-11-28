PYTHON ?= python3

.PHONY: help test test-3mode bench-cpu audit-hygiene clean

help:
	@echo "SiliSense targets:"
	@echo "  make test          Run pure-stdlib CPU unit test suite (0 sensor opens, 0 GPU touch)"
	@echo "  make test-3mode    Run 3-mode isolated -I parity (-I -B, -I -O -B, -I -B -W error)"
	@echo "  make bench-cpu     Run PREREG-S115 deterministic synthetic CPU preflight selftest"
	@echo "  make audit-hygiene Verify AST, RNG seeding, path/slug hygiene, and 0 binary/bytecode artifacts"
	@echo "  make clean         Remove build artifacts and Python bytecode inside repository"

test:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B -m unittest discover -s tests -p "test_*.py" -v

test-3mode:
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		echo "=== [$$flags] ==="; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags tests/run_all.py; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags silisense/nvml_profiler.py --self-test --json; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags silisense/resource_guard.py --self-test --json; \
	done

bench-cpu:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B silisense/benchmarks/prereg_s115_suite.py --cpu-preflight --json

audit-hygiene:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B tests/test_repo_hygiene_r5.py -v

clean:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -c 'import pathlib, shutil; r = pathlib.Path("."); [shutil.rmtree(p, ignore_errors=True) for p in list(r.glob("*.egg-info")) + [r / "build", r / "dist", r / ".pytest_cache"] if p.exists()]; [shutil.rmtree(p, ignore_errors=True) for p in r.rglob("__pycache__") if ".git" not in p.parts and ".venv" not in p.parts]; [p.unlink(missing_ok=True) for ext in ("*.pyc", "*.pyo") for p in r.rglob(ext) if ".git" not in p.parts and ".venv" not in p.parts]'
