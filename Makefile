.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src/__main__.py

clean:
	rm -rf .venv __pycache__ src/__pycache__ llm_sdk/__pycache__ .mypy_cache

lint:
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
	uv run flake8 src
	uv run mypy --strict src