.PHONY: test lint sync cli-help
sync:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .

cli-help:
	uv run perfpilot --help
