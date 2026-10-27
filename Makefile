.PHONY: install test demo

install:
	python -m pip install -e ".[dev]"

test:
	pytest

demo:
	agentic-workflow demo
