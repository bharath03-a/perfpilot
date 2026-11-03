.PHONY: test lint sync cli-help bench bench-record
sync:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .

cli-help:
	uv run perfpilot --help

bench:
	uv run perfpilot bench --suite all --seeds 20

bench-record:
	uv run perfpilot bench --suite all --seeds 20 --record
