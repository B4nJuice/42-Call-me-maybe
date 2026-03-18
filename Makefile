PROJECT_NAME	= Call_me_maybe
MAIN_PROGRAM	= main.py
VENV			= .venv
PYTHON			= python3
V_PYTHON		= $(VENV)/bin/python3
V_FLAKE			= $(VENV)/bin/flake8
V_MYPY			= $(VENV)/bin/mypy
V_UV			= $(VENV)/bin/uv
V_PIP			= $(V_PYTHON) -m pip
LIBS			= ./libs

UV_CACHE_DIR	= /tmp/.uv-cache

LLM_DIR			= llm_sdk
LLM_LIB			= llm_sdk-0.1.0-py3-none-any.whl
LLM_VENV		= $(LLM_DIR)/.venv
LLM_DIST		= $(LLM_DIR)/dist
LLM_LIB_PATH	= $(LLM_DIST)/$(LLM_LIB)

SRCS			= $(MAIN_PROGRAM) ./src

DEPENDENCIES	= uv

ARGS			?=

run: install
	$(V_PYTHON) $(MAIN_PROGRAM) $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf $(LLM_VENV)
	rm -rf $(LLM_DIST)
	rm -rf src/$(PROJECT_NAME).egg-info

fclean: clean
	rm -rf $(VENV)
	# rm -rf $(LIBS)/llm_sdk-0.1.0-py3-none-any

remove-cache:
	rm -rf $(UV_CACHE_DIR)

install: $(VENV)
	$(V_PIP) install $(DEPENDENCIES)
	$(V_UV) sync --project $(LLM_DIR) --cache-dir $(UV_CACHE_DIR)
	$(V_UV) build --project $(LLM_DIR) --cache-dir $(UV_CACHE_DIR)
	mv $(LLM_LIB_PATH) $(LIBS)/
	$(V_UV) sync --cache-dir $(UV_CACHE_DIR)

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(V_PIP) install --upgrade pip

lint: install
	$(V_FLAKE) $(SRCS)
	$(V_MYPY) $(SRCS) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	$(V_FLAKE) $(SRCS)
	$(V_MYPY) $(SRCS) --strict

debug: install
	$(PYTHON) -m pdb $(MAIN_PROGRAM)

.PHONY: run clean fclean install lint lint-strict remove-cache