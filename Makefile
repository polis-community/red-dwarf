TEST_FILTER ?= .

install: ## Install production dependencies
	uv pip install --editable .

install-dev: ## Install development dependencies
	uv sync --extra all --dev

docs-build: ## Build the static docs ./site directory
	uv run mkdocs build

docs-serve: ## Serve the documentation dev site
	uv run mkdocs serve

test: ## Run tests via pytest, optionally filtering (Ex.: `make test TEST_FILTER=map-xids`)
	uv run pytest -p no:nbmake -k '$(TEST_FILTER)'

test-nb: install ## Test all notebooks, optionally specifying file (Ex: `make test-nb NB_FILTER=map-xids`)
	uv run pytest -p no:cov --nbmake docs/notebooks/*$(NB_FILTER)*.ipynb

test-cov: ## Run tests via pytest (with coverage report)
	uv run pytest --cov=reddwarf --cov-report term-missing:skip-covered

cov-report-html: ## Build and open html coverage report
	uv run pytest --cov=reddwarf --cov-report html
	open htmlcov/index.html

test-debug: ## Run tests via pytest, optionally filtering (with verbose debugging)
	# Make sure stdout is rendered to screen.
	# Show full diffs on failure.
	uv run pytest -p no:nbmake --capture=no -vv -k '$(TEST_FILTER)'

test-all: test test-nb docs-build ## Run unit and notebook tests

clear-test-cache: ## Cleak the SQLite database of cached HTTP requests
	rm -f test_cache.sqlite

download: install ## Download Polis data into fixtures dir (Ex: `make download CONVO_ID=7vampckwrh DIR=example`)
	uv run scripts/download_sample_data.py $(CONVO_ID) $(DIR)

release: ## Print no-op documentation to guide the release process
	uv run scripts/release.py

# These make tasks allow the default help text to work properly.
%:
	@true

.PHONY: help

help:
	@echo 'Usage: make <command>'
	@echo
	@echo 'where <command> is one of the following:'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
