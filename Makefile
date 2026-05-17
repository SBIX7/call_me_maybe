.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .venv __pycache__ src/__pycache__ llm_sdk/__pycache__ .mypy_cache

lint:
	uv run flake8 . --exclude=llm_sdk,.venv
	uv run mypy . --exclude=llm_sdk --no-sqlite-cache --follow-imports=silent --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=llm_sdk,.venv
	uv run mypy . --exclude=llm_sdk --no-sqlite-cache --follow-imports=silent --strict