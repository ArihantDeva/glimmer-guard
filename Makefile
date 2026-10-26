# Thin wrapper so `make check` works without mise. Python deps come from uv so
# a clean checkout needs no virtualenv management.
PYRUN := uv run --with numpy --with pytest

.PHONY: check test sync train
check:
	@$(PYRUN) bash scripts/check.sh
test:
	@$(PYRUN) python -m pytest tests/ -q
sync:
	@bash scripts/sync.sh
train:
	@bash scripts/train.sh
