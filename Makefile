SHELL := /bin/bash
.DEFAULT_GOAL := help
PYTHON := .venv/bin/python

.PHONY: help data validate-data check smoke-train train evaluate

help:
	@printf '%s\n' \
		'Chihuahua vs. Muffin — interface do harness' \
		'' \
		'Alvo                 Estado' \
		'make help             operacional: exibe esta ajuda' \
		'make data             NÃO IMPLEMENTADO: aquisição e preparação' \
		'make validate-data    NÃO IMPLEMENTADO: validação da SPEC 01' \
		'make check            operacional: compilação e testes automatizados' \
		'make smoke-train      NÃO IMPLEMENTADO: treino mínimo' \
		'make train            NÃO IMPLEMENTADO: treino completo com GPU' \
		'make evaluate         NÃO IMPLEMENTADO: avaliação final' \
		'' \
		'Alvos não implementados encerram com erro; não representam sucesso.'

define not_implemented
	@printf 'ERRO: make %s ainda não está implementado. Consulte README.md e as SPECs.\n' '$@' >&2
	@exit 2
endef

data:
	$(not_implemented)

validate-data:
	$(not_implemented)

check:
	@test -x '$(PYTHON)' || { printf 'ERRO: interpretador não encontrado em %s.\n' '$(PYTHON)' >&2; exit 2; }
	@printf 'Verificando compilação dos módulos e testes...\n'
	@$(PYTHON) -m compileall -q src tests
	@printf 'Executando a suíte pytest...\n'
	@$(PYTHON) -m pytest

smoke-train:
	$(not_implemented)

train:
	$(not_implemented)

evaluate:
	$(not_implemented)
