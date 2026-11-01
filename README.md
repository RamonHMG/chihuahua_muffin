# Chihuahua vs. Muffin

Projeto didático de visão computacional para classificação binária de imagens
de chihuahuas e muffins com TensorFlow/Keras.

## Estado atual

O repositório está na etapa de **bootstrap documental e estrutural**. Ainda não
existem código de aquisição, pipeline `tf.data`, modelo, testes funcionais ou
artefatos de treinamento.

O trabalho está dividido em duas especificações conectadas por um contrato de
dados:

- [SPEC 01 — Dataset e pré-processamento](docs/specs/01_dataset.md):
  proveniência, auditoria, qualidade, duplicatas, particionamento e manifesto.
- [SPEC 02 — CNN, treinamento e avaliação](docs/specs/02_modelo.md): consumo
  do manifesto, CNN, GPU, treinamento, seleção do checkpoint e avaliação.

A SPEC 02 não pode refazer o particionamento produzido pela SPEC 01.

## Convenções principais

- Python-alvo: 3.12.
- Framework planejado: TensorFlow/Keras, com versões ainda pendentes de
  validação de compatibilidade.
- Classes: `0 = chihuahua`, `1 = muffin`.
- Entrada: RGB `128 × 128 × 3`, com correção EXIF, redimensionamento com
  preenchimento branco, interpolação bilinear e normalização para `[0, 1]`.
- Split preferencial: 70% treino, 15% validação e 15% teste, seed 42, sujeito à
  preservação de partições oficiais válidas.
- Manifesto de dataset: `schema_version = "1.0"`, serialização canônica e
  digests do índice e do relatório de qualidade.
- CNN inicial: convoluções com 16/32/64 filtros, Global Average Pooling, Dense
  32 e saída sigmoide; total esperado de 25.697 parâmetros treináveis.
- Treinamento inicial: batch 32, Adam 0.001, Binary Crossentropy, até 15 épocas,
  sem data augmentation.
- Dataset candidato:
  `samuelcortinhas/muffin-vs-chihuahua-image-classification`.

A versão, a licença, a estrutura e a qualidade reais do dataset ainda não foram
auditadas. Nenhum dado foi baixado no bootstrap.

## Interface operacional

`make help` é o único alvo operacional útil neste momento. Os demais alvos
compõem a interface estável planejada, mas falham explicitamente até receberem
implementação.

| Comando | Responsabilidade futura | Estado |
|---|---|---|
| `make help` | Exibir a interface e seu estado | Operacional |
| `make data` | Adquirir e preparar os dados | Não implementado |
| `make validate-data` | Validar contrato e critérios da SPEC 01 | Não implementado |
| `make check` | Executar verificações automatizadas reais | Não implementado |
| `make smoke-train` | Exercitar treinamento mínimo | Não implementado |
| `make train` | Executar treinamento completo com GPU | Não implementado |
| `make evaluate` | Avaliar o checkpoint final no teste | Não implementado |

Um alvo que termina com a mensagem "não implementado" não constitui uma
verificação aprovada.

## Ambiente local auditado

Ambiente recriado e validado em 2026-11-01:

- Pacotes de sistema confirmados: `python3.12-venv 3.12.3-1ubuntu0.17`,
  `python3-pip-whl 24.0+dfsg-1ubuntu1.3` e
  `python3-setuptools-whl 68.1.2-2ubuntu1.2`.
- O ambiente antigo continha somente `pip==26.0.1`; foi preservado durante a
  troca e excluído apenas após a validação do novo ambiente.
- A nova `.venv` foi criada por `/usr/bin/python3.12 -m venv .venv`.
- `.venv/bin/python` executa CPython 3.12.3 com prefixo e executáveis apontando
  para a raiz atual do projeto.
- `python -m pip` e `.venv/bin/pip` funcionam com pip 24.0; `pip check` não
  encontrou requisitos quebrados e nenhum pacote de projeto foi instalado.
- A ativação Bash define `VIRTUAL_ENV`, `python` e `pip` para a `.venv` atual.
  Os scripts csh e fish também contêm o caminho correto, mas esses shells não
  estavam instalados para teste de execução.
- Não há referências a `CNN_Example` no novo ambiente e não restou
  `.venv_backup`.
- O `pip` ainda avisa que `/home/ramon_hmg/.cache/pip` não está gravável para o
  usuário atual e desabilita seu cache; isso não impede o funcionamento da
  `.venv`.
- Uma NVIDIA GeForce RTX 5070 aparece no barramento PCI e o módulo do kernel
  `nvidia` 595.99.02 está carregado.
- `nvidia-smi` não consegue comunicar-se com o driver e nenhum dispositivo
  `/dev/nvidia*` foi encontrado durante a auditoria.
- `nvcc`, CUDA de runtime completa e cuDNN não foram demonstrados como
  disponíveis. TensorFlow não está instalado.

A base Python está pronta para a implementação incremental da SPEC 01. A
declaração e instalação das dependências do projeto continuam sujeitas a etapa
e autorização próprias. Drivers e pilha de GPU não foram modificados.

## Configuração

`config/example.toml` registra somente decisões já aprovadas e campos de
compatibilidade pendentes. Ele não contém credenciais e não deve ser tratado
como configuração validada para execução.

Dados, credenciais e artefatos gerados ficarão fora do Git. O arquivo
`bootstrap_prompt` é preservado e permanece disponível para versionamento como
registro da origem do projeto.

## Próximas etapas

Após revisão deste bootstrap:

1. aprovar e instalar somente as dependências necessárias à SPEC 01;
2. inspecionar informações públicas sobre fonte, licença e versão do dataset;
3. implementar incrementalmente a SPEC 01 e seus testes de contrato;
4. após download autorizado, auditar amostras e propor o limiar perceptual;
5. estabelecer o baseline e uma meta preditiva antes da avaliação final;
6. diagnosticar a GPU e implementar a SPEC 02 sem alterar o particionamento.

## Referências

- [AGENTS.md](https://github.com/agentsmd/agents.md)
- [GitHub Spec Kit — template de especificação](https://github.com/github/spec-kit/blob/main/templates/spec-template.md)
- [KaggleHub](https://github.com/Kaggle/kagglehub)
