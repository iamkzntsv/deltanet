.PHONY: test lint train

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check

train:
	uv run python train.py $(ARGS)
