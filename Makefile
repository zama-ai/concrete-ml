SHELL:=$(shell /usr/bin/env which bash)

DEV_DOCKER_PYTHON?=py38
ifeq ($(DEV_DOCKER_PYTHON),py312)
UBUNTU_BASE=24.10
else 
UBUNTU_BASE=20.04
endif 

DEV_DOCKER_IMG:=concrete-ml-dev
DEV_DOCKERFILE:=docker/Dockerfile.dev
DEV_CONTAINER_VENV_VOLUME:=concrete-ml-venv-$(DEV_DOCKER_PYTHON)
DEV_CONTAINER_CACHE_VOLUME:=concrete-ml-cache-$(DEV_DOCKER_PYTHON)
DOCKER_VENV_PATH:="$${HOME}"/dev_venv/
SRC_DIR:=src
TEST?=tests
N_CPU?=4
CONCRETE_PACKAGE_PATH=$(SRC_DIR)/concrete
COUNT?=1
RANDOMLY_SEED?=$$RANDOM
PYTEST_OPTIONS:=
POETRY_VERSION:=1.8.4
APIDOCS_OUTPUT?="./docs/references/api"
OPEN_PR="true"

# Check the total number of CPU cores and use min(4, TOTAL_CPUS/4) for pytest
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin) # macOS
    TOTAL_CPUS := $(shell sysctl -n hw.ncpu)
else # Assume Linux
    TOTAL_CPUS := $(shell nproc)
endif
ifeq ($(shell test $(TOTAL_CPUS) -lt 4; echo $$?),0) # very few cores
	PYTEST_CORES := $(TOTAL_CPUS)
	FHE_NUMPY_CORES := 1
else
	PYTEST_CORES := $(shell if [ `expr $(TOTAL_CPUS) / 4` -lt 4 ]; then expr $(TOTAL_CPUS) / 4; else echo 4; fi)
	# Calculate cores per pytest worker: total_cores / pytest_cores
	FHE_NUMPY_CORES := $(shell expr $(TOTAL_CPUS) / $(PYTEST_CORES))
endif

# At the end of the command, we currently need to force an 'import skorch' in Python in order to 
# avoid an obscure bug that led to all pytest commands to fail when installing dependencies with 
# Poetry >= 1.3. It is however not very clear how this import fixes the issue, as the bug was 
# difficult to understand and reproduce, so the line might become obsolete in the future.
.PHONY: setup_env # Set up the environment
setup_env:
	@# The keyring install is to allow pip to fetch credentials for our internal repo if needed
	PIP_INDEX_URL=https://pypi.org/simple \
	poetry run python --version
	poetry run python -m pip install keyring
	poetry run python -m pip install -U pip wheel
	poetry run python -m pip install pip-system-certs

	@# Only for linux and docker, reinstall setuptools. On macOS, it creates warnings, see 169
	if [[ $$(uname) != "Darwin" ]]; then \
		poetry run python -m pip install -U --force-reinstall setuptools; \
	fi

	echo "Installing poetry lock ..."
	if [[ $$(uname) != "Linux" ]] && [[ $$(uname) != "Darwin" ]]; then \
		poetry install --only dev; \
	else \
		poetry install --with dev; \
	fi
	echo "Finished installing poetry lock."

	"$(MAKE)" fix_omp_issues_for_intel_mac
	poetry run python -c "import skorch" || true # Details above
	poetry run python -c "from brevitas.core.scaling import AccumulatorAwareParameterPreScaling" || true # Details above

.PHONY: sync_env # Synchronise the environment
sync_env: 
	if [[ $$(poetry --version) != "Poetry (version $(POETRY_VERSION))" ]];then \
		echo "Current Poetry version is different than $(POETRY_VERSION). Please update it.";\
	else \
		if [[ $$(uname) != "Linux" ]] && [[ $$(uname) != "Darwin" ]]; then \
			poetry install --remove-untracked --only dev; \
		else \
			poetry install --remove-untracked --with dev; \
		fi; \
		"$(MAKE)" setup_env; \
	fi

.PHONY: fix_omp_issues_for_intel_mac # Fix OMP issues for macOS Intel, https://github.com/zama-ai/concrete-ml-internal/issues/3951
fix_omp_issues_for_intel_mac:
	if [[ $$(uname) == "Darwin" ]]; then \
		./script/make_utils/fix_omp_issues_for_intel_mac.sh; \
	fi


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

# We need to launch forbidden words aftwerwards because of conflicts with the files created by nbqa
# https://nbqa.readthedocs.io/en/latest/known-limitations.html#known-limitations
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3516
.PHONY: pcc # Run pre-commit checks
pcc:
	@"$(MAKE)" --keep-going --jobs $$(./script/make_utils/ncpus.sh) --output-sync=recurse \
	--no-print-directory pcc_internal
	@"$(MAKE)" check_forbidden_words

.PHONY: spcc # Run selected pre-commit checks (those which break most often, for SW changes)
spcc:
	@"$(MAKE)" --keep-going --jobs $$(./script/make_utils/ncpus.sh) --output-sync=recurse \
	--no-print-directory spcc_internal

PCC_DEPS := check_python_format check_finalize_nb python_linting mypy_ci pydocstyle shell_lint
PCC_DEPS += check_version_coherence check_licenses check_nbqa check_supported_ops
PCC_DEPS += check_refresh_notebooks_list check_refresh_use_cases_list check_mdformat
PCC_DEPS += check_unused_images check_utils_use_case gitleaks check_symlinks

.PHONY: pcc_internal
pcc_internal: $(PCC_DEPS)

# flake8 has been removed since it is too slow
SPCC_DEPS := check_python_format pylint_src pylint_tests mypy mypy_test pydocstyle ruff

.PHONY: spcc_internal
spcc_internal: $(SPCC_DEPS)

# Provide TEST (directory or python file) to run pytest on themthese tests
# Provide PYTEST_OPTIONS for running pytest with additional options, such as --randomly-seed SEED
# for reproducing tests
# To repeat a single test, apply something like @pytest.mark.repeat(3) on the test function
# -svv disables capturing of stdout/stderr and enables verbose output
# --count N is to repeate all tests N times (with different seeds). Default is to COUNT=1.
# --randomly-dont-reorganize is to prevent Pytest from shuffling the tests' order
# --randomly-dont-reset-seed is important: if it was not there, the randomly package would reset
# seeds to the same value, for all tests, resulting in same random's being taken in the tests, which
# reduces a bit the impact / coverage of our tests
# --capture=tee-sys is to make sure that, in case of crash, we can search for "Forcing seed to" in 
# stdout in order to be able to reproduce the failed test using that seed
# --cache-clear is to clear all Pytest's cache at before running the tests. This is done in order to 
# prevent inconsistencies when running pytest several times in a row (for example, in the CI)
.PHONY: pytest_internal # Run pytest
pytest_internal:
	poetry run pytest --version
	MKL_NUM_THREADS=$(FHE_NUMPY_CORES) \
	OMP_NUM_THREADS=$(FHE_NUMPY_CORES) \
	OPENBLAS_NUM_THREADS=$(FHE_NUMPY_CORES) \
	VECLIB_MAXIMUM_THREADS=$(FHE_NUMPY_CORES) \
	NUMEXPR_NUM_THREADS=$(FHE_NUMPY_CORES) \
	poetry run pytest $(TEST) \
	-svv \
	--count=$(COUNT) \
	--randomly-dont-reorganize \
	--randomly-dont-reset-seed \
	--capture=tee-sys \
	--cache-clear \
	${PYTEST_OPTIONS}

# -n N is to set the number of CPUs to use for pytest. We set it to N_CPU=4 by default because most 
# tests include parallel execution using all CPUs and too many parallel tests would lead to 
# contention. Thus N is set to something lower than the overall number of CPUs
# --durations=10 is to show the 10 slowest tests
.PHONY: pytest_internal_parallel # Run pytest with multiple CPUs
pytest_internal_parallel:
	@echo "Total CPUs: $(TOTAL_CPUS)"
	@echo "Assigning $(PYTEST_CORES) cores to pytest"
	@echo "Leaving $(FHE_NUMPY_CORES) cores per pytest worker for FHE/Numpy/sklearn"
	MKL_NUM_THREADS=$(FHE_NUMPY_CORES) \
	OMP_NUM_THREADS=$(FHE_NUMPY_CORES) \
	OPENBLAS_NUM_THREADS=$(FHE_NUMPY_CORES) \
	VECLIB_MAXIMUM_THREADS=$(FHE_NUMPY_CORES) \
	NUMEXPR_NUM_THREADS=$(FHE_NUMPY_CORES) \
	"$(MAKE)" pytest_internal PYTEST_OPTIONS="-n $(PYTEST_CORES) --durations=10 ${PYTEST_OPTIONS}"

# --global-coverage-infos-json=global-coverage-infos.json is to dump the coverage report in the file 
# --cov PATH is the directory PATH to consider for coverage. Default to SRC_DIR=src
# --cov-fail-under=100 is to make the command fail if coverage does not reach a 100%
# --cov-report=term-missing:skip-covered is used to avoid printing covered lines for all files
.PHONY: pytest # Run pytest on all tests
pytest:
	"$(MAKE)" pytest_internal_parallel \
	PYTEST_OPTIONS=" \
	--global-coverage-infos-json=global-coverage-infos.json \
	--cov=$(SRC_DIR) \
	--cov-fail-under=100 \
	--cov-report=term-missing:skip-covered \
	${PYTEST_OPTIONS}"

# Coverage options are not included since they look to fail on macOS
# (see https://github.com/zama-ai/concrete-ml-internal/issues/4428)
.PHONY: pytest_macOS_for_GitHub # Run pytest without coverage options
pytest_macOS_for_GitHub:
	"$(MAKE)" pytest_internal_parallel \
	PYTEST_OPTIONS="\
	--json-report \
	--json-report-file='pytest_report.json' \
	--json-report-omit collectors log traceback streams warnings  \
	--json-report-indent=4 \
	${PYTEST_OPTIONS}"

.PHONY: pytest_and_report # Run pytest and output the report in a JSON file
pytest_and_report:
	"$(MAKE)" pytest \
	PYTEST_OPTIONS="\
	--json-report \
	--json-report-file='pytest_report.json' \
	--json-report-omit collectors log traceback streams warnings  \
	--json-report-indent=4 \
	${PYTEST_OPTIONS}"

# --no-flaky makes pytest skip tests that are makred as flaky
.PHONY: pytest_no_flaky # Run pytest but ignore known flaky issues (so no coverage as well)
pytest_no_flaky: check_current_flaky_tests
	echo "Warning: known flaky tests are skipped and coverage is disabled"
	"$(MAKE)" pytest_internal_parallel PYTEST_OPTIONS="--no-flaky ${PYTEST_OPTIONS}"

# Running latest failed tests works by accessing pytest's cache. It is therefore recommended to
# call '--cache-clear' when calling the previous pytest run. 
# --cov PATH is the directory PATH to consider for coverage. Default to SRC_DIR=src
# --cov-append is to make the coverage of the previous pytest run to also consider the tests that are
# going to be re-executed by 'pytest_run_last_failed'
# --cov-fail-under=100 is to make the command fail if coverage does not reach a 100%
# --cov-report=term-missing:skip-covered is used to avoid printing covered lines for all files
# --global-coverage-infos-json=global-coverage-infos.json is to dump the coverage report in the file 
# --last-failed runs all last failed tests
# --last-failed-no-failures none' indicates pytest not to run anything (instead of running 
# all tests over again) if no failed tests are found
.PHONY: pytest_run_last_failed # Run all failed tests from the previous pytest run
pytest_run_last_failed:
	poetry run pytest $(TEST) \
	--cov=$(SRC_DIR) \
	--cov-append \
	--cov-fail-under=100 \
	--cov-report=term-missing:skip-covered \
	--global-coverage-infos-json=global-coverage-infos.json \
	--last-failed \
	--last-failed-no-failures none

.PHONY: pytest_one # Run pytest on a single file or directory (TEST)
pytest_one:
	@if [[ "$$TEST" == "" ]]; then \
		echo "TEST env variable is empty. Please specify which tests to run or use 'make pytest' instead.";\
		exit 1; \
	fi

	"$(MAKE)" pytest_internal

# --randomly-seed=SEED is to reproduce tests using the given seed
.PHONY: pytest_one_single_cpu # Run pytest on a single file or directory (TEST) using a specific seed (RANDOMLY_SEED) with a single CPU
pytest_one_single_cpu:
	"$(MAKE)" pytest_one PYTEST_OPTIONS="--randomly-seed=${RANDOMLY_SEED} ${PYTEST_OPTIONS}" 

.PHONY: check_current_flaky_tests # Print the current list of known flaky tests
check_current_flaky_tests:
	echo "Skip the following known flaky tests (test file: number of skipped configs):"
	poetry run pytest $(TEST) --collect-only -m flaky -qq

# Printing latest failed tests works by accessing pytest's cache. It is therefore recommended to
# call '--cache-clear' when calling the previous pytest run. 
# --cache-show prints pytest's complete cache (cacehd tests, random_seed, ...). Last failed tests 
# are found in the 'cache/lastfailed' section, which can be used to filter the output 
.PHONY: pytest_get_last_failed # Get the list of last failed tests 
pytest_get_last_failed:
	poetry run pytest $(TEST) --cache-show cache/lastfailed

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
	BUILD_ARGS="--build-arg UBUNTU_BASE=$(UBUNTU_BASE) ";\
	if [[ $$(uname) == "Linux" ]]; then \
		BUILD_ARGS+="--build-arg BUILD_UID=$$(id -u) --build-arg BUILD_GID=$$(id -g)"; \
	fi; \
	DOCKER_BUILDKIT=1 docker build $${BUILD_ARGS:+$$BUILD_ARGS} \
	--pull -t $(DEV_DOCKER_IMG):$(DEV_DOCKER_PYTHON) -f $(DEV_DOCKERFILE) .

.PHONY: docker_rebuild # Rebuild docker
docker_rebuild: docker_clean_volumes
	BUILD_ARGS=; \
	if [[ $$(uname) == "Linux" ]]; then \
		BUILD_ARGS="--build-arg BUILD_UID=$$(id -u) --build-arg BUILD_GID=$$(id -g)"; \
	fi; \
	DOCKER_BUILDKIT=1 docker build $${BUILD_ARGS:+$$BUILD_ARGS} \
	--pull --no-cache -t $(DEV_DOCKER_IMG):$(DEV_DOCKER_PYTHON) -f $(DEV_DOCKERFILE) .

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
	--volume /"$$(pwd)":/src \
	--volume $(DEV_CONTAINER_VENV_VOLUME):/home/dev_user/dev_venv \
	--volume $(DEV_CONTAINER_CACHE_VOLUME):/home/dev_user/.cache \
	$(DEV_DOCKER_IMG):$(DEV_DOCKER_PYTHON) || rm -f "$${EV_FILE}"

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

.PHONY: docs_no_links # Build docs
docs_no_links: check_docs_dollars
	@# Fix not-compatible paths that are sometimes generated by GitBook edits
	./script/make_utils/fix_gitbook_paths.sh docs

.PHONY: docs # Build docs and check links
docs: docs_no_links
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
	"$(MAKE)" mdformat

.PHONY: check_apidocs # Check that API docs are ok and
check_apidocs:
	@# Check nothing has changed
	./script/doc_utils/check_apidocs.sh


.PHONY: check_docs_dollars # Check that latex equations are enclosed by double dollar signs
check_docs_dollars:
# Find any isolated $ signs in the docs, except developer guide where 
# some codeblocks will show bash examples
# Only double dollar signs $$ should be used in order to 
# show properly in gitbook
	./script/make_utils/check_double_dollars_in_doc.sh

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
# The deployment notebook is currently skipped until the AMI is fixed
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4064
.PHONY: pytest_nb # Launch notebook tests
pytest_nb:
	NOTEBOOKS=$$(find docs -name "*.ipynb" ! -name "*Deployment*" | grep -v _build | grep -v .ipynb_checkpoints || true) && \
	if [[ "$${NOTEBOOKS}" != "" ]]; then \
		echo "$${NOTEBOOKS}" | xargs poetry run pytest -svv \
		--capture=tee-sys \
		-n0 \
		--randomly-dont-reorganize \
		--count=$(COUNT) \
		--durations=0 \
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

.PHONY: jupyter_execute_gpu # Execute all GPU jupyter notebooks sequentially and sanitize
jupyter_execute_gpu:
	poetry run env ./script/make_utils/jupyter.sh --run_all_notebooks_gpu
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

.PHONY: refresh_use_cases_list # Refresh the list of use cases currently available 
refresh_use_cases_list:
	poetry run python script/actions_utils/refresh_use_cases_list.py .github/workflows/run_one_use_cases_example.yaml

.PHONY: check_refresh_notebooks_list # Check if the list of notebooks currently available hasn't change
check_refresh_notebooks_list:
	poetry run python script/actions_utils/refresh_notebooks_list.py .github/workflows/refresh-one-notebook.yaml --check

.PHONY: check_refresh_use_cases_list # Check if the list of use cases currently available hasn't change
check_refresh_use_cases_list:
	poetry run python script/actions_utils/refresh_use_cases_list.py .github/workflows/run_one_use_cases_example.yaml --check

.PHONY: release_docker # Build a docker release image
release_docker:
	EV_FILE="$$(mktemp tmp.docker.XXXX)" && \
	poetry run env bash ./script/make_utils/generate_authenticated_pip_urls.sh "$${EV_FILE}" && \
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
	./script/make_utils/pytest_pypi_cml.sh --wheel --codeblocks

.PHONY: pytest_codeblocks_pypi_cml # Test code blocks using PyPI Concrete ML
pytest_codeblocks_pypi_cml:
	./script/make_utils/pytest_pypi_cml.sh --codeblocks --version "$${VERSION}"

.PHONY: pytest_codeblocks_one # Test code blocks using pytest in one file (TEST)
pytest_codeblocks_one:
	./script/make_utils/pytest_codeblocks.sh --file "$${TEST}"

# From https://stackoverflow.com/a/63523300 for the find command
.PHONY: shell_lint # Lint all bash scripts
shell_lint:
	@# grep -v "^\./\." is to avoid files in .hidden_directories
	find . -type f -name "*.sh" | grep -v "^\./\." | \
	xargs shellcheck

.PHONY: set_version # Generate a new version number and update all files with it accordingly
set_version:
	@if [[ "$$VERSION" == "" ]]; then											\
		echo "VERSION env variable is empty. Please set to desired version.";	\
		exit 1;																	\
	fi && \
	poetry run python ./script/make_utils/version_utils.py set-version --version "$${VERSION}"

# By default, check that all files containing version have the same value. If a 'VERSION' value is 
# given, check that it matches the version found in the files.
.PHONY: check_version_coherence # Check version coherence
check_version_coherence:
	@if [[ "$$VERSION" == "" ]]; then \
		poetry run python ./script/make_utils/version_utils.py check-version; \
	else \
		poetry run python ./script/make_utils/version_utils.py check-version --version "$${VERSION}"; \
	fi

.PHONY: changelog # Generate a changelog
changelog: check_version_coherence
	PROJECT_VER="${poetry version --short}" && \
	poetry run python ./script/make_utils/changelog_helper.py > "CHANGELOG_$${PROJECT_VER}.md"

.PHONY: show_commit_rules # Show the accepted rules for git conventional commits
show_commit_rules:
	@echo "Accepted commit rules:"
	@cat .github/workflows/continuous-integration.yaml | grep feat | grep pattern | cut -f 2- -d ":" | cut -f 2- -d " "

.PHONY: licenses # Generate the list of licenses of dependencies
licenses:
	./script/make_utils/licenses.sh

.PHONY: force_licenses # Generate the list of licenses of dependencies (force the regeneration)
force_licenses:
	./script/make_utils/licenses.sh --force_update

.PHONY: check_licenses # Check if the licenses of dependencies have changed
check_licenses:
	@TMP_OUT="$$(mktemp)" && \
	if ! poetry run env bash ./script/make_utils/licenses.sh --check > "$${TMP_OUT}"; then \
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
	find docs -name "*.md" | grep -v docs/references/tmp.api_for_check | xargs git diff --quiet

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
	$(DEV_DOCKER_IMG):$(DEV_DOCKER_PYTHON) \
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

# The log-opts option checks the whole history between the first commit and the current commit
# The default of gitleaks it to check the whole history.
.PHONY: gitleaks # Check for secrets in the repo using gitleaks
gitleaks:
	gitleaks --source "$${PWD}" detect --redact -v --log-opts="$$(git rev-list HEAD | tail -n 1)..$$(git rev-parse HEAD)"

.PHONY: sanity_check # Sanity checks, e.g. to check that a release is viable
sanity_check:
	poetry run python ./docker/release_resources/sanity_check.py

.PHONY: fast_sanity_check # Fast sanity checks, e.g. to check that a release is viable
fast_sanity_check:
	poetry run python ./docker/release_resources/sanity_check.py --fast

.PHONY: check_links # Check links in the documentation
check_links:
	@# It is important to understand that 'make check_links' should only be used after updating the documentation.
	@# Since 'make docs' automatically calls 'check_links' at the end, there is no obvious reason to 
	@# manually call 'make check_links' instead of 'make docs' !
	
	@# Check that no links target the main branch, some internal repositories (Concrete ML or Concrete) or our internal GitBook
	./script/make_utils/check_internal_links.sh

	@# To avoid some issues with privileges and linkcheckmd
	find docs/ -name "*.md" -type f | xargs chmod +r

	@# Run linkcheck on markdown files to check only local files
	poetry run python -m linkcheckmd docs -local

	@# Check that web links are functional or not broken
	poetry run python ./script/make_utils/check_links_with_agent.py README.md --verbose

	@# Check that relative links in markdown files are targeting existing files 
	poetry run python ./script/make_utils/local_link_check.py

	@# Check that links to markdown headers in markdown files are targeting existing headers 
	poetry run python ./script/make_utils/check_headers.py

	@# For weblinks and internal references
	@# 	--ignore-url=_static/webpack-macros.html: useless file which contains wrong links
	@#  --ignore-url=https://www.conventionalcommits.org/en/v1.0.0/: because issues to connect to
	@#		the server from AWS
	@#  --ignore-url=https://www.openml.org: this website returns a lots of timeouts
	@#  --ignore-url=https://github.com/zama-ai/concrete-ml-internal/issues: because issues are
	@#		private
	@#  --ignore-url=https://arxiv.org: this website returns a lots of timeouts
	poetry run linkchecker docs --check-extern \
		--no-warnings \
		--ignore-url=https://www.conventionalcommits.org/en/v1.0.0/ \
		--ignore-url=https://www.openml.org \
		--ignore-url=https://github.com/zama-ai/concrete-ml-internal/issues \
		--ignore-url=https://arxiv.org

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
	./script/make_utils/pytest_pypi_cml.sh --wheel

.PHONY: pytest_pypi_wheel_cml_no_flaky # Run tests (except flaky ones) using PyPI local wheel of Concrete ML
pytest_pypi_wheel_cml_no_flaky:
	./script/make_utils/pytest_pypi_cml.sh --wheel --noflaky

.PHONY: pytest_pypi_cml # Run tests using PyPI Concrete ML
pytest_pypi_cml:
	./script/make_utils/pytest_pypi_cml.sh

.PHONY: pytest_pypi_cml_no_flaky # Run tests (except flaky ones) using PyPI Concrete ML
pytest_pypi_cml_no_flaky:
	./script/make_utils/pytest_pypi_cml.sh --noflaky --version "$${VERSION}"

.PHONY: clean_pycache # Clean __pycache__ directories
clean_pycache:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: clean_sklearn_cache # Clean sklearn cache directories, eg for benchmarks
clean_sklearn_cache:
	rm -rf ~/scikit_learn_data

.PHONY: run_one_use_case_example # Run one use-case example (USE_CASE, e.g. hybrid_model)
run_one_use_case_example:
	USE_CASE=$(USE_CASE) ./script/make_utils/run_use_case_examples.sh

.PHONY: run_all_use_case_examples # Run all use-case examples
run_all_use_case_examples:
	USE_CASE="" ./script/make_utils/run_use_case_examples.sh

.PHONY: check_utils_use_case # Check that no utils.py are found in use_case_examples
check_utils_use_case:
	./script/make_utils/check_utils_in_use_case.sh

# This command does not use a make script because of obscure import issues with Skops on macOS
.PHONY: update_encrypted_dataframe # Update encrypted data-frame's development files
update_encrypted_dataframe:
	poetry run python ./src/concrete/ml/pandas/_development.py

.PHONY: check_symlinks # Check that no utils.py are found in use_case_examples
check_symlinks:
	if [[ -z $$(find . -xtype l -name abc) ]]; then \
		echo "All symlinks point to exiting files"; \
	else \
		echo "Bad symlinks found: " && echo $$(find . -xtype l) && \
		exit 1; \
	fi

.PHONY: pytest_gpu  # Run sequentially tests that require a GPU
pytest_gpu:
	@echo "Collecting GPU tests..."

	@TESTS=$$(poetry run pytest -m "use_gpu" --collect-only -q); \
	if [ -z "$$TESTS" ]; then \
		echo "No tests marked @pytest.mark.use_gpu found."; \
		exit 1; \
	else \
		echo "Running GPU tests only..."; \
		env POETRY_RUN_GPU_TESTS=1 poetry run pytest -n0 --dist no -m "use_gpu" -v --color=yes --durations=0; \
	fi
