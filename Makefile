SHELL:=$(shell /usr/bin/env which bash)

DEV_DOCKER_IMG:=concrete-ml-dev
DEV_DOCKERFILE:=docker/Dockerfile.dev
DEV_CONTAINER_VENV_VOLUME:=concrete-ml-internal-venv
DEV_CONTAINER_CACHE_VOLUME:=concrete-ml-internal-cache
DOCKER_VENV_PATH:="$${HOME}"/dev_venv/
SRC_DIR:=src
CONCRETE_PACKAGE_PATH=$(SRC_DIR)/concrete
COUNT?=1
RANDOMLY_SEED?=$$RANDOM
PYTEST_OPTIONS:=
POETRY_VERSION:=1.2.2
APIDOCS_OUTPUT?="./docs/developer-guide/api"

# If one wants to force the installation of a given rc version
# /!\ WARNING /!\: This version should NEVER be a wildcard as it might create some
# issues when trying to run it in the future.
CP_VERSION_SPEC_FOR_RC="concrete-python==1.0.0"

# If one wants to use the last RC version
# CP_VERSION_SPEC_FOR_RC="$$(poetry run python \
# ./script/make_utils/pyproject_version_parser_helper.py \
# --pyproject-toml-file pyproject.toml \
# --get-pip-install-spec-for-dependency concrete-python)"

.PHONY: setup_env # Set up the environment
setup_env:
	@# The keyring install is to allow pip to fetch credentials for our internal repo if needed
	PIP_INDEX_URL=https://pypi.org/simple \
	PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
	poetry run python --version
	poetry run python -m pip install keyring
	poetry run python -m pip install -U pip wheel

	@# Only for linux and docker, reinstall setuptools. On macOS, it creates warnings, see 169
	if [[ $$(uname) != "Darwin" ]]; then \
		poetry run python -m pip install -U --force-reinstall setuptools; \
	fi
	if [[ $$(uname) != "Linux" ]] && [[ $$(uname) != "Darwin" ]]; then \
		poetry install --only dev; \
	else \
		poetry install; \
	fi

	echo "Installing $(CP_VERSION_SPEC_FOR_RC)" && \
	poetry run python -m pip install -U --pre "$(CP_VERSION_SPEC_FOR_RC)"

.PHONY: sync_env # Synchronise the environment
sync_env: check_poetry_version
	if [[ $$(uname) != "Linux" ]] && [[ $$(uname) != "Darwin" ]]; then \
		poetry install --remove-untracked --only dev; \
	else \
		poetry install --remove-untracked; \
	fi
	"$(MAKE)" setup_env

.PHONY: update_env # Same as sync_env, sets the venv state to be synced with the rpo
update_env: sync_env

.PHONY: reinstall_env # Remove venv and reinstall
reinstall_env:
	@VENV_SCRIPT_PATH=bin && \
	if [[ $$(uname) != "Linux" && $$(uname) != "Darwin" ]]; then \
		VENV_SCRIPT_PATH="Scripts"; \
	fi && \
	if [[ "$${VIRTUAL_ENV}" != "" ]]; then \
		echo "Please deactivate the current venv first."; \
	else \
		VENV_PATH=$$(./script/make_utils/remove_venv.sh); \
		SOURCE_VENV_PATH="$${VENV_PATH}$${VENV_SCRIPT_PATH}/activate"; \
		python3 -m venv "$${VENV_PATH}"; \
		source "$${SOURCE_VENV_PATH}"; \
		"$(MAKE)" setup_env; \
		echo "Source venv with:"; \
		echo "source $${SOURCE_VENV_PATH}"; \
	fi

.PHONY: check_poetry_version # Check poetry's version
check_poetry_version:
	if [[ $$(poetry --version) == "Poetry (version $(POETRY_VERSION))" ]];then \
		echo "Poetry version is ok";\
	else\
		echo "Expected poetry version is not the expected one: $(POETRY_VERSION)"\
		exit 1;\
	fi

.PHONY: python_format # Apply python formatting
python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir benchmarks --dir script --dir docker/release_resources \
	--dir use_case_examples --file conftest.py

.PHONY: check_python_format # Check python format
check_python_format:
	poetry run env bash ./script/source_format/format_python.sh \
	--dir $(SRC_DIR) --dir tests --dir benchmarks --dir script --dir docker/release_resources \
	--dir use_case_examples --file conftest.py \
	--check

.PHONY: check_finalize_nb # Check sanitization of notebooks
check_finalize_nb:
	poetry run python ./script/nbmake_utils/notebook_finalize.py docs --check &
	poetry run python ./script/nbmake_utils/notebook_finalize.py use_case_examples --check

.PHONY: pylint # Run pylint
pylint:
	"$(MAKE)" --keep-going pylint_src pylint_tests pylint_script pylint_benchmarks

.PHONY: pylint_src # Run pylint on sources
pylint_src:
	poetry run pylint --rcfile=pylintrc $(SRC_DIR)

.PHONY: pylint_tests # Run pylint on tests
pylint_tests:
	@# Disable duplicate code detection (R0801) in tests
	@# Disable unnecessary lambda (W0108) for tests
	@# Disable ungrouped-imports (C0412) because pylint does mistakes between our package and CP
	find ./tests/ -type f -name "*.py" | xargs poetry run pylint --disable=R0801,W0108,C0412 \
		--rcfile=pylintrc

	poetry run pylint --disable=R0801,W0108,C0412 conftest.py \
		--rcfile=pylintrc

.PHONY: pylint_benchmarks # Run pylint on benchmarks
pylint_benchmarks:
	@# Disable duplicate code detection (R0801) in benchmarks
	@# Disable duplicate code detection, docstring requirement, too many locals/statements
	@# Disable ungrouped-imports (C0412) because pylint does mistakes between our package and CP
	find ./benchmarks/ -type f -name "*.py" | xargs poetry run pylint \
	--disable=R0801,R0914,R0915,C0103,C0114,C0115,C0116,C0302,W0108,C0412 --rcfile=pylintrc

.PHONY: pylint_script # Run pylint on scripts
pylint_script:
	find ./script/ -type f -name "*.py" | xargs poetry run pylint --rcfile=pylintrc
	find docker/release_resources -type f -name "*.py" | xargs poetry run pylint --rcfile=pylintrc

.PHONY: flake8 # Run flake8 (including darglint)
flake8:
	poetry run flake8 --config flake8_src.cfg $(SRC_DIR) script/

	@# --extend-ignore=DAR is because we don't want to run darglint on tests/ script/ benchmarks/
	poetry run flake8 --config flake8_others.cfg tests/ script/ benchmarks/ \
		docker/release_resources/ conftest.py

.PHONY: ruff # Run ruff
ruff:
	poetry run ruff $(SRC_DIR) script tests conftest.py

.PHONY: ruff # Run ruff fix
fix_ruff:
	poetry run ruff $(SRC_DIR) tests script --fix

.PHONY: python_linting # Run python linters
python_linting: ruff pylint flake8

.PHONY: conformance # Run command to fix some conformance issues automatically
conformance: finalize_nb python_format licenses nbqa supported_ops mdformat

.PHONY: check_issues # Run command to check if all referenced issues are opened
check_issues:
	python ./script/make_utils/check_issues.py

.PHONY: pcc # Run pre-commit checks
pcc:
	@"$(MAKE)" --keep-going --jobs $$(./script/make_utils/ncpus.sh) --output-sync=recurse \
	--no-print-directory pcc_internal

.PHONY: spcc # Run selected pre-commit checks (those which break most often, for SW changes)
spcc:
	@"$(MAKE)" --keep-going --jobs $$(./script/make_utils/ncpus.sh) --output-sync=recurse \
	--no-print-directory spcc_internal

PCC_DEPS := check_python_format check_finalize_nb python_linting mypy_ci pydocstyle shell_lint
PCC_DEPS += check_version_coherence check_licenses check_nbqa check_supported_ops
PCC_DEPS += check_refresh_notebooks_list check_mdformat
PCC_DEPS += check_forbidden_words check_unused_images gitleaks

# Not commented on purpose for make help, since internal
.PHONY: pcc_internal
pcc_internal: $(PCC_DEPS)

# flake8 has been removed since it is too slow
SPCC_DEPS := check_python_format pylint_src pylint_tests mypy mypy_test pydocstyle ruff

# Not commented on purpose for make help, since internal
.PHONY: spcc_internal
spcc_internal: $(SPCC_DEPS)

# One can reproduce pytest thanks to the --randomly-seed which is given by
# pytest-randomly
# Replace --count=1 by a larger number to repeat _all_ tests
# To repeat a single test, apply something like @pytest.mark.repeat(3) on the test function
# --randomly-dont-reset-seed is used to make that, if we run the same test several times (with
# @pytest.mark.repeat(3)), not the same seed is used, even if things are still deterministic of the
# main seed
# --capture=tee-sys is to make that, in case of crash, we can search for "Forcing seed to" in stdout
# to try to reproduce
# --durations=10 is to show the 10 slowest tests
# -n Const because most tests include parallel execution using all CPUs, too many 
# parallel tests would lead to contention. Thus Const is set to something low and much lower than
# num_cpus
.PHONY: pytest # Run pytest
pytest:
	poetry run pytest --version
	poetry run pytest --durations=10 -svv \
	--capture=tee-sys \
	--global-coverage-infos-json=global-coverage-infos.json \
	-n 4 \
	--cov=$(SRC_DIR) --cov-fail-under=100 \
	--randomly-dont-reorganize \
	--cov-report=term-missing:skip-covered tests/ \
	--count=$(COUNT) \
	--randomly-dont-reset-seed \
	${PYTEST_OPTIONS}

.PHONY: pytest_one # Run pytest on a single file or directory (TEST) a certain number of times (COUNT)
pytest_one:
	poetry run pytest --durations=10 -svv \
	--capture=tee-sys \
	-n $$(./script/make_utils/ncpus.sh) \
	--randomly-dont-reorganize \
	--count=$(COUNT) \
	--randomly-dont-reset-seed \
	${PYTEST_OPTIONS} \
	"$${TEST}"

.PHONY: pytest_one_single_cpu # Run pytest on a single file or directory (TEST) with a single CPU with RANDOMLY_SEED seed
# Don't set --durations=10, because it is not reproducible and we use this target for determinism
# checks
pytest_one_single_cpu:
	poetry run pytest -svv \
	--capture=tee-sys \
	--randomly-dont-reorganize \
	--randomly-dont-reset-seed \
	${PYTEST_OPTIONS} \
	"$${TEST}" --randomly-seed=${RANDOMLY_SEED}

.PHONY: pytest_macOS_for_GitHub # Run pytest with some coverage options which are removed
# These options are removed since they look to fail on macOS for no obvious reason
# (see https://github.com/zama-ai/concrete-ml-internal/issues/1554)
pytest_macOS_for_GitHub:
	poetry run pytest --durations=10 -svv \
	--capture=tee-sys \
	-n 4 \
	--randomly-dont-reorganize \
	--count=$(COUNT) \
	--randomly-dont-reset-seed \
	${PYTEST_OPTIONS}


# Not a huge fan of ignoring missing imports, but some packages do not have typing stubs
.PHONY: mypy # Run mypy
mypy:
	poetry run mypy -p $(SRC_DIR) --ignore-missing-imports --implicit-optional --check-untyped-defs

# Friendly target to run mypy without ignoring missing stubs and still have errors messages
# Allows to see which stubs we are missing
.PHONY: mypy_ns # Run mypy (without ignoring missing stubs)
mypy_ns:
	poetry run mypy -p $(SRC_DIR)

.PHONY: mypy_test # Run mypy on test files
mypy_test:
	find ./tests/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports --implicit-optional --check-untyped-defs
	poetry run mypy conftest.py --ignore-missing-imports --check-untyped-defs

.PHONY: mypy_script # Run mypy on scripts
mypy_script:
	find ./script/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports --implicit-optional --check-untyped-defs
	find ./docker/release_resources -name "*.py" | xargs poetry run mypy --ignore-missing-imports --implicit-optional --check-untyped-defs

.PHONY: mypy_benchmark # Run mypy on benchmark files
mypy_benchmark:
	find ./benchmarks/ -name "*.py" | xargs poetry run mypy --ignore-missing-imports --implicit-optional --check-untyped-defs

# The plus indicates that make will be called by the command and allows to share the context with
# the parent make execution. We serialize calls to these targets as they may overwrite each others
# cache which can cause issues.
.PHONY: mypy_ci # Run all mypy checks for CI
mypy_ci:
	"$(MAKE)" --keep-going mypy mypy_test mypy_script mypy_benchmark

.PHONY: docker_build # Build dev docker
docker_build:
	BUILD_ARGS=; \
	if [[ $$(uname) == "Linux" ]]; then \
		BUILD_ARGS="--build-arg BUILD_UID=$$(id -u) --build-arg BUILD_GID=$$(id -g)"; \
	fi; \
	DOCKER_BUILDKIT=1 docker build $${BUILD_ARGS:+$$BUILD_ARGS} \
	--pull -t $(DEV_DOCKER_IMG) -f $(DEV_DOCKERFILE) .

.PHONY: docker_rebuild # Rebuild docker
docker_rebuild: docker_clean_volumes
	BUILD_ARGS=; \
	if [[ $$(uname) == "Linux" ]]; then \
		BUILD_ARGS="--build-arg BUILD_UID=$$(id -u) --build-arg BUILD_GID=$$(id -g)"; \
	fi; \
	DOCKER_BUILDKIT=1 docker build $${BUILD_ARGS:+$$BUILD_ARGS} \
	--pull --no-cache -t $(DEV_DOCKER_IMG) -f $(DEV_DOCKERFILE) .

.PHONY: docker_start # Launch docker
docker_start:
	EV_FILE="$$(mktemp tmp.docker.XXXX)" && \
	poetry run env bash ./script/make_utils/generate_authenticated_pip_urls.sh "$${EV_FILE}" && \
	echo "" >> "$${EV_FILE}" && \
	export $$(cat "$${EV_FILE}" | xargs) && rm -f "$${EV_FILE}" && \
	docker run --rm -it \
	-p 8888:8888 \
	--env DISPLAY=host.docker.internal:0 \
	$${PIP_INDEX_URL:+--env "PIP_INDEX_URL=$${PIP_INDEX_URL}"} \
	$${PIP_EXTRA_INDEX_URL:+--env "PIP_EXTRA_INDEX_URL=$${PIP_EXTRA_INDEX_URL}"} \
	--volume /"$$(pwd)":/src \
	--volume $(DEV_CONTAINER_VENV_VOLUME):/home/dev_user/dev_venv \
	--volume $(DEV_CONTAINER_CACHE_VOLUME):/home/dev_user/.cache \
	$(DEV_DOCKER_IMG) || rm -f "$${EV_FILE}"

.PHONY: docker_build_and_start # Docker build and start
docker_build_and_start: docker_build docker_start

.PHONY: docker_bas  # Docker build and start
docker_bas: docker_build_and_start

.PHONY: docker_clean_volumes  # Docker clean volumes
docker_clean_volumes:
	docker volume rm -f $(DEV_CONTAINER_VENV_VOLUME)
	docker volume rm -f $(DEV_CONTAINER_CACHE_VOLUME)

.PHONY: docker_cv # Docker clean volumes
docker_cv: docker_clean_volumes

.PHONY: docs # Build docs
docs: clean_docs check_docs_dollars
	@# Rebuild the index from README.md, to have in the home of Sphinx a copy of README.md
	echo "  .. Warning, auto-generated by \`make docs\`, don\'t edit" > docs/index.rst
	echo "" >> docs/index.rst
	pandoc --from markdown --to rst docs/README.md >> docs/index.rst
	@# To be sure there is a blank line at the end
	echo "" >> docs/index.rst
	cat docs/index.toc.txt >> docs/index.rst
	@# Create _static if nothing is commited in it
	mkdir -p docs/_static/
	@# Generate the auto summary of documentations
	@# Cannot do without specifying top module currently with sphinx-apidoc
	poetry run sphinx-apidoc --implicit-namespaces -o docs/_apidoc $(CONCRETE_PACKAGE_PATH)
	@# Doing a copy of docs, where we modify files
	rm -rf docs-copy
	cp -r docs docs-copy
	@# Admonitions
	./script/make_utils/sphinx_gitbook_admonitions.sh --gitbook_to_sphinx
	@# Check that there is no more GitBook hint
	! grep -r "hint style" docs-copy
	@# Replace $$, $/$ and /$$ by $
	./script/make_utils/fix_double_dollars_issues_with_mdformat.sh docs-copy --single_dollar
	@# Fix not-compatible paths
	./script/make_utils/fix_gitbook_paths.sh docs-copy
	@# Fixing cardboard
	poetry run python script/doc_utils/fix_gitbook_table.py --files docs-copy/getting-started/showcase.md
	@# Docs
	cd docs-copy && poetry run "$(MAKE)" html SPHINXOPTS='-W --keep-going'
	@# Copy images from GitBook
	cp docs/.gitbook/assets/*.png docs-copy/_build/html/_images
	cp -r docs-copy/_build docs/
	rm -rf docs-copy
	@# Check links
	"$(MAKE)" check_links

.PHONY: remove_5c_docs # Remove the ugly %5C that we have in the doc, due to management of _ with GitBook
remove_5c_docs:
	sed -i "" -e "s@%5C_@_@g" docs/*.md docs/*/*.md docs/*/*/*.md
	sed -i "" -e "s@%5C%5C@@g" docs/*.md docs/*/*.md docs/*/*/*.md

.PHONY: apidocs # Builds API docs
apidocs:
	@# At release time, one needs to change --src-base-url (to be a public release/A.B.x branch)
	./script/doc_utils/update_apidocs.sh "$(APIDOCS_OUTPUT)"

	# Update our summary
	./script/doc_utils/update_apidocs_files_in_SUMMARY.sh
	"$(MAKE)" mdformat

.PHONY: check_apidocs # Check that API docs are ok and
check_apidocs:
	@# Check nothing has changed
	./script/doc_utils/check_apidocs.sh

.PHONY: clean_docs # Clean docs build directory
clean_docs:
	rm -rf docs/_apidoc docs/_build

.PHONY: check_docs_dollars # Check that latex equations are enclosed by double dollar signs
check_docs_dollars:
# Find any isolated $ signs in the docs, except developer guide where 
# some codeblocks will show bash examples
# Only double dollar signs $$ should be used in order to 
# show properly in gitbook
	./script/make_utils/check_double_dollars_in_doc.sh

.PHONY: open_docs # Launch docs in a browser
open_docs:
	python3 -m webbrowser -t "file://${PWD}/docs/_build/html/index.html"

.PHONY: docs_and_open # Make docs and open them in a browser
docs_and_open: docs open_docs

.PHONY: pydocstyle # Launch syntax checker on source code documentation
pydocstyle:
	@# From http://www.pydocstyle.org/en/stable/error_codes.html
	poetry run pydocstyle $(SRC_DIR) --convention google --add-ignore=D1,D202 --add-select=D401
	poetry run pydocstyle tests --convention google --add-ignore=D1,D202 --add-select=D401
	poetry run pydocstyle conftest.py --convention google --add-ignore=D1,D202 --add-select=D401

.PHONY: finalize_nb # Sanitize notebooks
finalize_nb:
	poetry run python ./script/nbmake_utils/notebook_finalize.py docs use_case_examples

# A warning in a package unrelated to the project made pytest fail with notebooks
# Run notebook tests without warnings as sources are already tested with warnings treated as errors
# We need to disable xdist with -n0 to make sure to not have IPython port race conditions
.PHONY: pytest_nb # Launch notebook tests
pytest_nb:
	NOTEBOOKS=$$(find docs -name "*.ipynb" | grep -v _build | grep -v .ipynb_checkpoints || true) && \
	if [[ "$${NOTEBOOKS}" != "" ]]; then \
		echo "$${NOTEBOOKS}" | xargs poetry run pytest -svv \
		--capture=tee-sys \
		-n0 \
		--randomly-dont-reorganize \
		--count=$(COUNT) \
		--randomly-dont-reset-seed -Wignore --nbmake; \
	else \
		echo "No notebook found"; \
	fi

.PHONY: jupyter_open # Launch jupyter, to be able to choose notebook you want to run
jupyter_open:
	./script/make_utils/jupyter.sh --open

.PHONY: jupyter_execute # Execute all jupyter notebooks sequentially and sanitize
jupyter_execute:
	poetry run env ./script/make_utils/jupyter.sh --run_all_notebooks
	"$(MAKE)" finalize_nb

.PHONY: jupyter_execute_one # Execute one jupyter notebook and sanitize
jupyter_execute_one:
	poetry run env ./script/make_utils/jupyter.sh --run_notebook "$${NOTEBOOK}"
	"$(MAKE)" finalize_nb

.PHONY: jupyter_execute_parallel # Execute all jupyter notebooks in parallel (on the same machine) and sanitize
jupyter_execute_parallel:
	poetry run env ./script/make_utils/jupyter.sh --run_all_notebooks_parallel
	"$(MAKE)" finalize_nb

.PHONY: refresh_notebooks_list # Refresh the list of notebooks currently available 
refresh_notebooks_list:
	poetry run python script/actions_utils/refresh_notebooks_list.py .github/workflows/refresh-one-notebook.yaml

.PHONY: check_refresh_notebooks_list # Check if the list of notebooks currently available hasn't change
check_refresh_notebooks_list:
	poetry run python script/actions_utils/refresh_notebooks_list.py .github/workflows/refresh-one-notebook.yaml --check

.PHONY: release_docker # Build a docker release image
release_docker:
	EV_FILE="$$(mktemp tmp.docker.XXXX)" && \
	poetry run env bash ./script/make_utils/generate_authenticated_pip_urls.sh "$${EV_FILE}" && \
	PROJECT_VERSION="$$(poetry version)" && \
	PROJECT_VERSION="$$(echo "$${PROJECT_VERSION}" | cut -d ' ' -f 2)" && \
	IS_PRERELEASE="$$(poetry run python script/make_utils/version_utils.py \
	islatest \
	--new-version "$${PROJECT_VERSION}" \
	--existing-versions "$${PROJECT_VERSION}" | jq -rc '.is_prerelease')" && \
	echo "PRERELEASE=$${IS_PRERELEASE}" >> "$${EV_FILE}" && \
	echo "CP_VERSION='$(CP_VERSION_SPEC_FOR_RC)'" >> "$${EV_FILE}" && \
	echo "" >> "$${EV_FILE}" && \
	./docker/build_release_image.sh "$${EV_FILE}" && rm -f "$${EV_FILE}" || rm -f "$${EV_FILE}"

.PHONY: upgrade_py_deps # Upgrade python dependencies
upgrade_py_deps:
	./script/make_utils/upgrade_deps.sh

.PHONY: pytest_codeblocks # Test code blocks using pytest in the documentation
pytest_codeblocks:
	./script/make_utils/pytest_codeblocks.sh

.PHONY: pytest_codeblocks_pypi_wheel_cml # Test code blocks using the PyPI local wheel of Concrete ML
pytest_codeblocks_pypi_wheel_cml:
	./script/make_utils/pytest_pypi_cml.sh --wheel "$(CP_VERSION_SPEC_FOR_RC)" --codeblocks

.PHONY: pytest_codeblocks_pip_cml # Test code blocks using PyPI Concrete ML
pytest_codeblocks_pip_cml:
	./script/make_utils/pytest_pypi_cml.sh --codeblocks

.PHONY: pytest_codeblocks_one # Test code blocks using pytest in one file (TEST)
pytest_codeblocks_one:
	./script/make_utils/pytest_codeblocks.sh --file "$${TEST}"

# From https://stackoverflow.com/a/63523300 for the find command
.PHONY: shell_lint # Lint all bash scripts
shell_lint:
	@# grep -v "^\./\." is to avoid files in .hidden_directories
	find . -type f -name "*.sh" | grep -v "^\./\." | \
	xargs shellcheck

.PHONY: set_version_no_commit # Dry run for set_version
set_version_no_commit:
	@if [[ "$$VERSION" == "" ]]; then											\
		echo "VERSION env variable is empty. Please set to desired version.";	\
		exit 1;																	\
	fi && \
	poetry run python ./script/make_utils/version_utils.py set-version --version "$${VERSION}"

.PHONY: set_version # Generate a new version number and update all files with it accordingly
set_version:
	@if [[ "$$VERSION" == "" ]]; then											\
		echo "VERSION env variable is empty. Please set to desired version.";	\
		exit 1;																	\
	fi && \
	poetry run python ./script/make_utils/version_utils.py set-version --version "$${VERSION}" && \
	echo && \
	echo && \
	echo "Please do something like:" && \
	echo "git commit -m \"chore: bump version to $${VERSION}\"" && \
	echo

.PHONY: set_version_and_push # Generate a new version number, update all files with it accordingly and push them
set_version_and_push: set_version
	git push

.PHONY: check_version_coherence # Check that all files containing version have the same value
check_version_coherence:
	poetry run python ./script/make_utils/version_utils.py check-version

.PHONY: changelog # Generate a changelog
changelog: check_version_coherence
	PROJECT_VER=($$(poetry version)) && \
	PROJECT_VER="$${PROJECT_VER[1]}" && \
	poetry run python ./script/make_utils/changelog_helper.py > "CHANGELOG_$${PROJECT_VER}.md"

.PHONY: release # Create a new release
release: check_version_coherence check_apidocs
	@PROJECT_VER=($$(poetry version)) && \
	PROJECT_VER="$${PROJECT_VER[1]}" && \
	TAG_NAME="v$${PROJECT_VER}" && \
	git fetch --tags --force && \
	git tag -s -a -m "$${TAG_NAME} release" "$${TAG_NAME}" && \
	git push origin "refs/tags/$${TAG_NAME}"


.PHONY: show_scope # Show the accepted types and optional scopes (for git conventional commits)
show_scope:
	@echo "Accepted types and optional scopes:"
	@cat .github/workflows/continuous-integration.yaml | grep feat | grep pattern | cut -f 2- -d ":" | cut -f 2- -d " "

.PHONY: show_type # Show the accepted types and optional scopes (for git conventional commits)
show_type:show_scope

.PHONY: licenses # Generate the list of licenses of dependencies
licenses:
	./script/make_utils/licenses.sh --cp_version "$(CP_VERSION_SPEC_FOR_RC)"

.PHONY: force_licenses # Generate the list of licenses of dependencies (force the regeneration)
force_licenses:
	./script/make_utils/licenses.sh --cp_version "$(CP_VERSION_SPEC_FOR_RC)" --force_update

.PHONY: check_licenses # Check if the licenses of dependencies have changed
check_licenses:
	@TMP_OUT="$$(mktemp)" && \
	if ! poetry run env bash ./script/make_utils/licenses.sh --check \
		--cp_version "$(CP_VERSION_SPEC_FOR_RC)" > "$${TMP_OUT}"; then \
		cat "$${TMP_OUT}"; \
		rm -f "$${TMP_OUT}"; \
		echo "Error while checking licenses, see log above."; \
		echo "Consider re-running 'make licenses'"; \
		exit 1; \
	else \
		echo "Licenses check OK"; \
	fi
.PHONY: check_licenses

.PHONY: help # Generate list of targets with descriptions
help:
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1\t\2/' | expand -t30 | sort

.PHONY: pip_audit # Run pip-audit and check if there are known vulnerabilities in our dependencies
pip_audit:
	poetry run pip-audit

.PHONY: pip_audit_use_cases # Run pip-audit and check if there are known vulnerabilities in our use-cases
pip_audit_use_cases:
	@echo "Will study:"
	@find ./use_case_examples -type f -name "*requirements.txt" -exec echo -r "{}" \;
	@find ./use_case_examples -type f -name "*requirements.txt" -exec echo \; -exec echo "{}:" \; -exec poetry run pip-audit -r "{}" \;

.PHONY: clean_local_git # Tell the user how to delete local git branches, except main
clean_local_git:
	@git fetch --all --prune
	@echo "Consider doing: "
	@echo
	@# Don't consider deleting `main` or current branches
	@git branch | grep -v "^*" | grep -v main | xargs echo "git branch -D "
	@echo
	@read -p "Do it now (y/N)? " -r; \
	if [[ $$REPLY =~ ^[Yy] ]]; \
	then \
		git branch | grep -v "^*" | grep -v main | xargs git branch -D ; \
		echo; \
		git branch; \
	fi

.PHONY: mdformat # Apply markdown formatting
# Remark we need to remove .md's in venv. Remark that we fix $$ problems with mdformat at the end
mdformat:
	@# grep -v "^\./\." is to avoid files in .hidden_directories
	find . -type f -name "*.md" | grep -v "^\./\." | xargs poetry run mdformat

	@#Do ToC in the README
	poetry run mdformat README.md

	@# Because mdformat modifies $$, which GitBook really wants. Make $$ back
	./script/make_utils/fix_double_dollars_issues_with_mdformat.sh docs


.PHONY: check_mdformat # Check markdown format
# Remark we need to remove .md's in venv
check_mdformat:
	"$(MAKE)" mdformat
	find docs -name "*.md" | grep -v docs/developer-guide/tmp.api_for_check | xargs git diff --quiet

.PHONY: benchmark # Perform benchmarks
benchmark:
	rm -rf progress.json && \
	for script in benchmarks/*.py; do \
	  poetry run python $$script; \
	done

.PHONY: benchmark_one # Perform benchmark on one script file (SCRIPT)
benchmark_one:
	rm -rf progress.json && \
	poetry run python $${SCRIPT}; \

.PHONY: docker_publish_measurements # Run benchmarks in docker and publish results
docker_publish_measurements: docker_rebuild
	docker run --volume /"$$(pwd)":/src \
	--volume $(DEV_CONTAINER_VENV_VOLUME):/home/dev_user/dev_venv \
	--volume $(DEV_CONTAINER_CACHE_VOLUME):/home/dev_user/.cache \
	$(DEV_DOCKER_IMG) \
	/bin/bash ./script/progress_tracker_utils/benchmark_and_publish_findings_in_docker.sh \
	"$${LAUNCH_COMMAND}"

.PHONY: nbqa_one # Call nbqa on a single notebook
nbqa_one:
	./script/make_utils/nbqa.sh --notebook "$${NOTEBOOK}"
	"$(MAKE)" finalize_nb

.PHONY: nbqa # Call nbqa on all notebooks
nbqa:
	./script/make_utils/nbqa.sh --all_notebooks
	"$(MAKE)" finalize_nb

.PHONY: check_nbqa_one # Check with nbqa a single notebook
check_nbqa_one:
	./script/make_utils/nbqa.sh --notebook "$${NOTEBOOK}" --check

.PHONY: check_nbqa # Check with nbqa all notebooks
check_nbqa:
	./script/make_utils/nbqa.sh --all_notebooks --check

.PHONY: determinism # Check pytest determinism
determinism:
	./script/make_utils/check_pytest_determinism.sh

.PHONY: supported_ops # Update docs with supported ops
supported_ops:
	poetry run python script/doc_utils/gen_supported_ops.py docs/deep-learning/onnx_support.md

.PHONY: check_supported_ops # Check supported ops (for the doc)
check_supported_ops:
	poetry run python script/doc_utils/gen_supported_ops.py docs/deep-learning/onnx_support.md
	git diff docs/deep-learning/onnx_support.md
	git diff --quiet docs/deep-learning/onnx_support.md

.PHONY: gitleaks # Check for secrets in the repo using gitleaks
gitleaks:
	gitleaks --source "$${PWD}" detect --redact -v

.PHONY: sanity_check # Sanity checks, e.g. to check that a release is viable
sanity_check:
	poetry run python ./docker/release_resources/sanity_check.py

.PHONY: fast_sanity_check # Fast sanity checks, e.g. to check that a release is viable
fast_sanity_check:
	poetry run python ./docker/release_resources/sanity_check.py --fast

.PHONY: check_links # Check links in the documentation
check_links:
	@# Because of issues with priviledges and linkcheckmd
	find docs/ -name "*.md" -type f | xargs chmod +r

	@# Remark that this target is not in PCC, because it needs the doc to be built
	@# Mainly for web links and _api_doc (sphinx)
	poetry run python -m linkcheckmd docs -local

	poetry run python -m linkcheckmd README.md

	poetry run python ./script/make_utils/local_link_check.py
	poetry run python ./script/make_utils/check_headers.py

	@# For weblinks and internal references
	@# 	--ignore-url=_static/webpack-macros.html: useless file which contains wrong links
	@#  --ignore-url=https://www.conventionalcommits.org/en/v1.0.0/: because issues to connect to
	@#		the server from AWS
	@#  --ignore-url=https://www.openml.org: lot of time outs
	@#  --ignore-url=https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis: currently
	@#		private
	@#  --ignore-url=https://github.com/zama-ai/concrete-ml-internal: because some files are only
	@#		private at this time. We'll finally check files with check_links_after_release after
	@#		everything has been pushed to public repository
	@#	--ignore-url=.gitbook/assets : some gitbook functionalities use links to images to include
	@# 		them in the docs. But sphinx does not copy such as images to the _build dir since 
	@#		they are not included by image tags or sphinx image annotations. We ignore links 
	@#		to gitbook images in the HTML checker. But the images are actually checked by the 
	@#		markdown link checker, `local_link_check.sh`.
	poetry run linkchecker docs --check-extern \
		--ignore-url=_static/webpack-macros.html \
		--ignore-url=https://www.conventionalcommits.org/en/v1.0.0/ \
		--ignore-url=https://www.openml.org \
		--ignore-url=https://huggingface.co/spaces/zama-fhe/encrypted_sentiment_analysis \
		--ignore-url=https://github.com/zama-ai/concrete-ml-internal \
		--ignore-url=.gitbook/assets

	@# We don't want links to our internal GitBook. We may have to switch this test off for a
	@# moment if we link to links which are not made public in CP documentation, e.g. Worse case, it
	@# is tested in check_links_after_release
	./script/doc_utils/check_no_gitbook_links.sh


.PHONY: check_links_after_release # Check links in the documentation as if we were users
check_links_after_release: docs
	@# The difference between check_links_after_release and check_links is:
	@#	 - use check_links during dev time: we have --ignore to accept files which are not already
	@#	   in the public repo
	@#	 - at release time, use check_links_after_release, to check that the doc or public
	@#	   repository only use public links

	@# We don't want links to main branch
	grep -r "tree/main" docs | (! grep "\.md:" > /dev/null)

	@# We don't want links to internal repositories
	grep -r "concrete-ml-internal" docs | (! grep "\.md:" > /dev/null)
	grep -r "concrete-numpy-internal" docs | (! grep "\.md:" > /dev/null)

	@# We don't want links to our internal GitBook
	./script/doc_utils/check_no_gitbook_links.sh

	@# Because of issues with priviledges and linkcheckmd
	find docs/ -name "*.md" -type f | xargs chmod +r

	@# Remark that this target is not in PCC, because it needs the doc to be built
	@# Mainly for web links and _api_doc (sphinx)
	poetry run python -m linkcheckmd docs -local
	poetry run python -m linkcheckmd README.md
	poetry run python ./script/make_utils/local_link_check.py

	@# For weblinks and internal references
	@# 	--ignore-url=_static/webpack-macros.html: useless file which contains wrong links
	@#  --ignore-url=https://www.conventionalcommits.org/en/v1.0.0/: because issues to connect to
	@#		the server from AWS
	@#  --ignore-url=https://www.openml.org: lot of time outs
	@#  --ignore-url=.gitbook/assets : some gitbook functionalities use links to images to include
	@# 		them in the docs. But sphinx does not copy such as images to the _build dir since
	@#		they are not included by image tags or sphinx image annotations. We ignore links
	@#		to gitbook images in the HTML checker. But the images are actually checked by the
	@#		markdown link checker, `local_link_check.sh`.
	poetry run linkchecker docs --check-extern \
		--ignore-url=_static/webpack-macros.html \
		--ignore-url=https://www.conventionalcommits.org/en/v1.0.0/ \
		--ignore-url=https://www.openml.org \
		--ignore-url=.gitbook/assets

.PHONY: actionlint # Linter for our github actions
actionlint:
	./script/make_utils/actionlint.sh

.PHONY: check_forbidden_words # Check forbidden words
check_forbidden_words:
	./script/make_utils/check_forbidden_words.sh

.PHONY: check_forbidden_words_and_open # Check forbidden words and open bad files
check_forbidden_words_and_open:
	./script/make_utils/check_forbidden_words.sh --open

.PHONY: update_dependabot_prs # Update all dependabot PRs on origin
update_dependabot_prs:
	/bin/bash ./script/make_utils/update_dependabot_prs.sh

.PHONY: check_unused_images # Check for unused images in the doc
check_unused_images:
	./script/make_utils/check_all_images_are_used.sh

.PHONY: pytest_pypi_wheel_cml # Run tests using PyPI local wheel of Concrete ML
pytest_pypi_wheel_cml:
	./script/make_utils/pytest_pypi_cml.sh --wheel "$(CP_VERSION_SPEC_FOR_RC)"

.PHONY: pytest_pip_cml # Run tests using PyPI Concrete ML
pytest_pip_cml:
	./script/make_utils/pytest_pypi_cml.sh

.PHONY: clean_pycache # Clean __pycache__ directories
clean_pycache:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: clean_sklearn_cache # Clean sklearn cache directories, eg for benchmarks
clean_sklearn_cache:
	rm -rf ~/scikit_learn_data
