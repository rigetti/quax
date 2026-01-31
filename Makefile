# Copyright 2026 Rigetti & Co, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GLOBALS

ifeq ($(origin .FEATURES), undefined)
$(error This Makefile requires GNU Make for the purpose of using the .ONESHELL directive.)
endif

# ... we use a single bash shell instance to run all commands in a rule, much like a script
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# ... the following are evaluated lazily, and should only be used in rules that check-poetry
PACKAGE = $(shell poetry version | awk '{print $$1;}')
VERSION = $(shell poetry version | awk '{print $$2;}')
INCLUDE = ${PACKAGE//-/_}

# ... the following are evaluated greedily, and form the targets for some rules related to testing
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
EXAMPLES_DIR := $(PROJECT_DIR)/examples
EXAMPLE_NOTEBOOKS := $(wildcard $(EXAMPLES_DIR)/*.ipynb)

.PHONY: FORCE  # special target you can use to force rebuild of any file; e.g. for testing the examples

# COMMANDS

.DEFAULT := help

.PHONY: help
help:
	@awk 'BEGIN { FS=":.*##"; print "Supported Makefile commands:\n" } \
          /^[a-zA-Z0-9_-]+:.*##/ { cmd=$$1; desc=$$2; printf "    \033[36m%-20s\033[0m %s\n", cmd, desc } \
          END { print "" }' $(MAKEFILE_LIST)

.PHONY: build-docs
build-docs: ## Build the project documentation.
	cd ${PROJECT_DIR}
	LC_ALL=C.UTF-8 poetry run sphinx-build -b html -D project=${PACKAGE} -D version=${VERSION} ./docs ./docs/_build

.PHONY: check-all
check-all: check-format check-types  ## Check conformance to code format and typing rules.

.PHONY: check-format
check-format:  ## Check conformance to code format rules.
	cd ${PROJECT_DIR}
	poetry run ruff format src tests examples --diff
	poetry run ruff check src tests examples --no-fix

.PHONY: check-types
check-types: ## Check conformance to code typing rules.
	cd ${PROJECT_DIR}
	poetry run pyright src tests examples

.PHONY: format
format: ## Make automatic updates to code format and style.
	cd ${PROJECT_DIR}
	poetry run ruff format src tests examples
	poetry run ruff check src tests examples --fix-only

.PHONY: test-examples
test-examples: ## Test all Jupyter notebooks in "examples" run via papermill.
	cd ${PROJECT_DIR}
	@for notebook in $(EXAMPLE_NOTEBOOKS); do \
		echo "Running $$notebook..."; \
		poetry run papermill "$$notebook" /dev/null --cwd $(EXAMPLES_DIR) || exit 1; \
	done
	@echo "✅ All example notebooks ran successfully."

.PHONY: test-package
test-package: ## Run all unit tests for the package, and report coverage.
	cd ${PROJECT_DIR}
	poetry run pytest -vv tests/

# go install github.com/google/addlicense@latest
.PHONY: add-license
add-license: ## Add license headers to all source files.
	cd ${PROJECT_DIR}
	addlicense -c "Rigetti & Co, LLC." -l "apache" -v .
