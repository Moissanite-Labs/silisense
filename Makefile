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
	done

bench-cpu:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B silisense/benchmarks/prereg_s115_suite.py --cpu-preflight --json

audit-hygiene:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B tests/test_repo_hygiene_r5.py -v

clean:
	rm -rf build dist *.egg-info
	find . \( -path ./.git -o -path ./.venv \) -prune -o -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . \( -path ./.git -o -path ./.venv \) -prune -o -name "*.pyc" -type f -exec rm -f {} +
