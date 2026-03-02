.PHONY: notebook train-local train-remote validate-local validate-remote test check

COMP ?= titanic

notebook:
	jupyter lab

validate-local:
	ts validate-env --competition $(COMP) --profile local_arm64

validate-remote:
	ts validate-env --competition $(COMP) --profile remote_gpu

train-local:
	ts train --competition $(COMP) --profile local_arm64

train-remote:
	ts train --competition $(COMP) --profile remote_gpu

test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

check:
	PYTHONPATH=src .venv/bin/python -m tabular_shenanigans.cli.main check
