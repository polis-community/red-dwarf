TEST_FILTER ?= .
# Make it simpler to change args during `uv run` calls.
UV_RUN ?= uv run

venv: ## Make a virtualenv
	uv venv

install: ## Install only production dependencies
	uv pip install --editable .[all]

install-dev: ## Install production with development dependencies
	uv pip install --editable .[all,dev]

docs-build: ## Build the static docs ./site directory
	$(UV_RUN) mkdocs build

docs-serve: ## Serve the documentation dev site
	$(UV_RUN) mkdocs serve

test: ## Run tests via pytest, optionally filtering (Ex.: `make test TEST_FILTER=map-xids`)
	$(UV_RUN) pytest -p no:nbmake -k '$(TEST_FILTER)'

test-nb: install ## Test all notebooks, optionally specifying file (Ex: `make test-nb NB_FILTER=map-xids`)
	$(UV_RUN) pytest -p no:cov --nbmake docs/notebooks/*$(NB_FILTER)*.ipynb

test-cov: ## Run tests via pytest (with coverage report)
	$(UV_RUN) pytest --cov=reddwarf --cov-report term-missing:skip-covered

cov-report-html: ## Build and open html coverage report
	$(UV_RUN) pytest --cov=reddwarf --cov-report html
	open htmlcov/index.html

test-debug: ## Run tests via pytest, optionally filtering (with verbose debugging)
	# Make sure stdout is rendered to screen.
	# Show full diffs on failure.
	$(UV_RUN) pytest -p no:nbmake --capture=no -vv -k '$(TEST_FILTER)'

test-all: test test-nb docs-build ## Run unit and notebook tests

# Small hint to remove gitignore'd python version file, which can confuse usage of uv.
clean:
	rm .python-version

clear-test-cache: ## Cleak the SQLite database of cached HTTP requests
	rm -f test_cache.sqlite

download: install ## Download Polis data into fixtures dir (Ex: `make download CONVO_ID=7vampckwrh DIR=example`)
	$(UV_RUN) scripts/download_sample_data.py $(CONVO_ID) $(DIR)

release: ## Print no-op documentation to guide the release process
	$(UV_RUN) scripts/release.py

build: ## Build a wheel for upload to PyPI
	rm -r dist/
	$(UV_RUN) build

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
