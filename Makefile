SHELL := /bin/bash
.DEFAULT_GOAL := help
PYTHON := .venv/bin/python
MODEL_ENV := PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=2 MPLCONFIGDIR=/tmp/chihuahua-muffin-matplotlib
ALLOW_CPU ?= 0
RUN_ID ?=

.PHONY: help data validate-data check smoke-train train evaluate

help:
	@printf '%s\n' \
		'Chihuahua vs. Muffin — interface do harness' \
		'' \
		'Alvo                 Estado' \
		'make help             operacional: exibe esta ajuda' \
		'make data             operacional: gera artefatos finais locais da SPEC 01' \
		'make validate-data    operacional: valida artefatos e originais sem escrita' \
		'make check            operacional: compilação e testes automatizados' \
		'make smoke-train      operacional: treino mínimo; CPU exige ALLOW_CPU=1' \
		'make train            operacional: treino completo; GPU obrigatória' \
		'make evaluate         operacional: avalia RUN_ID=<run> congelado' \
		'' \
		'Treino completo falha sem GPU TensorFlow utilizável; não há fallback.'

data:
	@test -x '$(PYTHON)' || { printf 'ERRO: interpretador não encontrado em %s.\n' '$(PYTHON)' >&2; exit 2; }
	@PYTHONPATH=src $(PYTHON) -m chihuahua_muffin.data.prepared_cli generate

validate-data:
	@test -x '$(PYTHON)' || { printf 'ERRO: interpretador não encontrado em %s.\n' '$(PYTHON)' >&2; exit 2; }
	@PYTHONPATH=src $(PYTHON) -m chihuahua_muffin.data.prepared_cli validate

check:
	@test -x '$(PYTHON)' || { printf 'ERRO: interpretador não encontrado em %s.\n' '$(PYTHON)' >&2; exit 2; }
	@printf 'Verificando compilação dos módulos e testes...\n'
	@$(PYTHON) -m compileall -q src tests
	@printf 'Executando a suíte pytest...\n'
	@$(MODEL_ENV) $(PYTHON) -m pytest

smoke-train:
	@test -x '$(PYTHON)' || { printf 'ERRO: interpretador não encontrado em %s.\n' '$(PYTHON)' >&2; exit 2; }
	@if [[ '$(ALLOW_CPU)' == '1' ]]; then \
		$(MODEL_ENV) $(PYTHON) -m chihuahua_muffin.model.cli smoke --allow-cpu; \
	else \
		$(MODEL_ENV) $(PYTHON) -m chihuahua_muffin.model.cli smoke; \
	fi

train:
	@test -x '$(PYTHON)' || { printf 'ERRO: interpretador não encontrado em %s.\n' '$(PYTHON)' >&2; exit 2; }
	@$(MODEL_ENV) $(PYTHON) -m chihuahua_muffin.model.cli train

evaluate:
	@test -x '$(PYTHON)' || { printf 'ERRO: interpretador não encontrado em %s.\n' '$(PYTHON)' >&2; exit 2; }
	@test -n '$(RUN_ID)' || { printf 'ERRO: informe RUN_ID=<run treinado e congelado>.\n' >&2; exit 2; }
	@$(MODEL_ENV) $(PYTHON) -m chihuahua_muffin.model.cli evaluate --run-id '$(RUN_ID)'
