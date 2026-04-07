PROJECT_NAME	= Call_me_maybe
VENV			= .venv
PYTHON			= python3
V_PYTHON		= $(VENV)/bin/$(PYTHON)
V_FLAKE			= $(VENV)/bin/flake8
V_MYPY			= $(VENV)/bin/mypy
UV			= uv
LIBS			= ./libs

UV_CACHE_DIR	= /tmp/.uv-cache

LLM_DIR			= llm_sdk
LLM_LIB			= llm_sdk-0.1.0-py3-none-any.whl
LLM_VENV		= $(LLM_DIR)/.venv
LLM_DIST		= $(LLM_DIR)/dist
LLM_LIB_PATH	= $(LLM_DIST)/$(LLM_LIB)

SRCS			= ./src

ARGS			?=

run: install
	$(UV) run python -B -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf $(LLM_VENV)
	rm -rf $(LLM_DIST)
	rm -rf src/$(PROJECT_NAME).egg-info
	rm -rf data/output

fclean: clean
	rm -rf $(VENV)
	rm -rf $(LIBS)/llm_sdk-0.1.0-py3-none-any

remove-cache:
	rm -rf $(UV_CACHE_DIR)

install: $(VENV)
	$(UV) sync --project $(LLM_DIR) --cache-dir $(UV_CACHE_DIR)
	$(UV) build --project $(LLM_DIR) --cache-dir $(UV_CACHE_DIR)
	mv $(LLM_LIB_PATH) $(LIBS)/
	$(UV) sync --cache-dir $(UV_CACHE_DIR)

$(VENV):
	$(PYTHON) -m venv $(VENV)

lint: install
	$(V_FLAKE) $(SRCS)
	$(V_MYPY) $(SRCS) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	$(V_FLAKE) $(SRCS)
	$(V_MYPY) $(SRCS) --strict

debug: install
	$(PYTHON) -m pdb $(MAIN_PROGRAM)

.PHONY: run clean fclean install lint lint-strict remove-cache