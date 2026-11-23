# Chihuahua vs. Muffin

Classificador binário didático de imagens de chihuahuas e muffins, implementado
com Python 3.12 e TensorFlow/Keras.

## Dataset em uso

O projeto usa exclusivamente o dataset Kaggle
[`samuelcortinhas/muffin-vs-chihuahua-image-classification`](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification),
ID `2733586`, na versão imutável `2`.

A Kaggle declara a licença `CC0: Public Domain`, mas a descrição informa que as
imagens vieram do Google Images sem demonstrar a cadeia de direitos. Por isso, o
dataset é usado somente de forma local, educacional e sem redistribuição, sob a
exceção registrada na [SPEC 01](docs/specs/01_dataset.md).

O conjunto preparado contém 5.917 imagens JPEG:

| Split | Chihuahua (`0`) | Muffin (`1`) | Total |
|---|---:|---:|---:|
| Treino | 2.175 | 1.848 | 4.023 |
| Validação | 384 | 326 | 710 |
| Teste | 640 | 544 | 1.184 |
| **Total** | **3.199** | **2.718** | **5.917** |

O `test` original foi preservado. A validação foi extraída somente do `train`
original, com 15% de cada classe, seed 42 e atribuição determinística. A SPEC 02
consome esses splits sem redistribuir amostras.

Cada imagem é lida com o seguinte contrato:

- correção da orientação EXIF;
- conversão para RGB, com grayscale replicado e RGBA composto sobre branco;
- redimensionamento proporcional para `128 × 128`, com padding branco e
  interpolação bilinear;
- normalização dos pixels para `[0, 1]`;
- rótulos imutáveis: `0 = chihuahua` e `1 = muffin`.

Os artefatos preparados ficam em `data/prepared/`:

- `dataset_manifest.json`: identidade, proveniência e contrato do dataset;
- `index.csv`: arquivos, rótulos e splits;
- `quality_report.json`: auditoria e limitações conhecidas.

O dataset não possui duplicatas byte a byte, mas mantém 719 pares candidatos por
pHash sem revisão manual. Versões da mesma fotografia ainda podem atravessar os
splits; essa limitação deve ser considerada ao interpretar a avaliação final.

## Arquitetura da rede

A rede `small_cnn_v1` é construída do zero, sem transfer learning e sem data
augmentation no baseline.

| Camada | Saída | Parâmetros treináveis |
|---|---:|---:|
| Entrada RGB | `128 × 128 × 3` | 0 |
| Conv2D, 16 filtros `3 × 3`, ReLU, padding `same` | `128 × 128 × 16` | 448 |
| MaxPooling `2 × 2` | `64 × 64 × 16` | 0 |
| Conv2D, 32 filtros `3 × 3`, ReLU, padding `same` | `64 × 64 × 32` | 4.640 |
| MaxPooling `2 × 2` | `32 × 32 × 32` | 0 |
| Conv2D, 64 filtros `3 × 3`, ReLU, padding `same` | `32 × 32 × 64` | 18.496 |
| MaxPooling `2 × 2` | `16 × 16 × 64` | 0 |
| GlobalAveragePooling2D | `64` | 0 |
| Dense, 32 unidades, ReLU | `32` | 2.080 |
| Dense, 1 unidade, sigmoid | `1` | 33 |
| **Total** | — | **25.697** |

A saída representa `P(muffin)`: valores menores que `0,5` são classificados
como chihuahua e valores maiores ou iguais a `0,5`, como muffin.

Configuração de treinamento atual:

- batch size 32;
- Adam com learning rate `0.001`;
- Binary Crossentropy;
- no máximo 15 épocas;
- early stopping em `val_loss`, com `patience = 3`, `min_delta = 0.001` e
  restauração dos melhores pesos;
- checkpoint escolhido pelo menor `val_loss` absoluto;
- seed 42 e operações determinísticas quando disponíveis.

## Como rodar

### 1. Preparar o ambiente

O projeto requer Python 3.12. As dependências devem ser instaladas somente na
`.venv` do repositório:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

O treinamento completo exige uma GPU NVIDIA reconhecida e efetivamente usada
pelo TensorFlow. O projeto não instala nem altera drivers do sistema.

### 2. Conferir a interface e os dados

```bash
make help
make check
make validate-data
```

`make validate-data` verifica os artefatos preparados e os arquivos originais
sem escrever neles. Para reconstruir `data/prepared/` a partir dos dados e das
auditorias locais já aprovadas:

```bash
make data
```

Esse alvo não baixa o dataset. Uma nova aquisição exige autorização explícita e
deve seguir a SPEC 01.

### 3. Diagnosticar a GPU

```bash
PYTHONPATH=src .venv/bin/python -m chihuahua_muffin.model.cli diagnose
```

O diagnóstico termina com status `gpu_usable` somente quando o TensorFlow lista
a GPU e executa nela a operação de prova.

### 4. Executar o smoke training

Com GPU:

```bash
make smoke-train
```

O smoke percorre leitura, forward pass, backpropagation, validação, checkpoint e
geração de artefatos com um subconjunto pequeno. Ele não mede a qualidade do
modelo. Para autorizar explicitamente um smoke em CPU:

```bash
make smoke-train ALLOW_CPU=1
```

### 5. Treinar e avaliar

```bash
make train
```

O comando imprime o `RUN_ID` e grava a execução em `artifacts/runs/<RUN_ID>/`.
O treinamento falha sem GPU utilizável e nunca faz fallback silencioso para CPU.

Depois que o treinamento terminar e congelar o melhor checkpoint, execute a
avaliação final uma única vez:

```bash
make evaluate RUN_ID=<RUN_ID>
```

A avaliação aceita somente um run completo e congelado. Smokes, runs parciais
ou artefatos divergentes são recusados antes da leitura do conjunto de teste.

Para gerar, em um run já avaliado, o painel local com 3 VP, 3 VN, 3 FP e 3 FN:

```bash
make evaluation-panel RUN_ID=<RUN_ID>
```

O arquivo `evaluation_panel.png` mostra as entradas pré-processadas exatamente
como o modelo as recebe. A seleção é determinística por `sample_id`, e o
manifesto registra os exemplos, as probabilidades e o digest do painel.

As decisões e a evolução do projeto estão em
[Histórico e decisões do projeto](docs/historico_decisoes.md). Os contratos
normativos permanecem na [SPEC 01](docs/specs/01_dataset.md) e na
[SPEC 02](docs/specs/02_modelo.md).
