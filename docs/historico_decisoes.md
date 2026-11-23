# Histórico e decisões do projeto

Este documento concentra a evolução do projeto e as decisões que atravessam
mais de uma etapa. O estado operacional fica no [README](../README.md), enquanto
os requisitos e critérios de aceitação normativos permanecem na
[SPEC 01](specs/01_dataset.md) e na [SPEC 02](specs/02_modelo.md).

## Situação atual

Atualizado em 2026-11-02:

- a SPEC 01 está concluída, validada e aprovada para consumo;
- o pipeline `tf.data`, a CNN, o ciclo de runs, o smoke training e a avaliação
  congelada da SPEC 02 estão implementados;
- o treinamento completo foi concluído em GPU e atendeu a meta na validação;
- a avaliação final foi executada no checkpoint congelado;
- um painel auditável com 3 VP, 3 VN, 3 FP e 3 FN foi gerado para o run final.

Os resultados de teste encerram a avaliação deste baseline e não podem ser
usados retroativamente para escolher uma nova configuração.

## Linha do tempo

### 2026-11-01 — Fundação e contrato do dataset

- O ambiente do projeto foi recriado com Python 3.12 em `.venv`, sem instalar
  dependências fora dela.
- O dataset Kaggle
  `samuelcortinhas/muffin-vs-chihuahua-image-classification`, ID `2733586`,
  versão 2, foi escolhido como única fonte do experimento.
- A Kaggle declara `CC0: Public Domain`, mas a origem descrita como Google
  Images não demonstra a cadeia de direitos. Foi aprovada uma exceção estrita
  para uso local, educacional, não comercial e sem redistribuição; o estado da
  licença permaneceu `ambiguous`.
- A aquisição autorizada usou KaggleHub 1.0.3 e preservou os originais em
  `data/raw/kaggle-2733586-v2`.
- A auditoria encontrou 5.917 imagens JPEG e um marcador não-imagem do
  KaggleHub. Não foram encontrados arquivos ilegíveis, animações nem hashes
  SHA-256 repetidos.

### 2026-11-01 — Qualidade, duplicatas e splits

- A deduplicação exata foi implementada de forma lógica e não destrutiva. Na
  versão real, ela encontrou zero grupos duplicados e zero conflitos.
- A auditoria perceptual comparou 17.502.486 pares com pHash. Foram mantidos 719
  pares candidatos até distância 12 para possível revisão; nenhum deles foi
  confirmado automaticamente como duplicata.
- Para o experimento didático, foi escolhido o Cenário A: preservar todo o
  `test` original e extrair a validação exclusivamente do `train` original.
- A validação recebeu 15% de cada classe, com seed 42 e o algoritmo
  determinístico `stratified_sha256_rank_v1`.
- A possibilidade de versões da mesma fotografia atravessarem splits foi aceita
  como limitação metodológica explícita.

Contagens publicadas:

| Split | Chihuahua | Muffin | Total |
|---|---:|---:|---:|
| Treino | 2.175 | 1.848 | 4.023 |
| Validação | 384 | 326 | 710 |
| Teste | 640 | 544 | 1.184 |
| **Total** | **3.199** | **2.718** | **5.917** |

### 2026-11-01 — Publicação da SPEC 01

- `make data` passou a integrar os artefatos auditados e publicar
  atomicamente `index.csv`, `quality_report.json` e `dataset_manifest.json`.
- `make validate-data` passou a validar, sem escrita, os esquemas, digests,
  originais, classes, splits, contagens e o contrato de pré-processamento.
- A SPEC 01 foi concluída e aprovada para consumo pela SPEC 02.

Digests da publicação validada:

| Artefato ou contrato | SHA-256 |
|---|---|
| `index.csv` | `79213ec1780749dbd264f4a5752f2dc268e28acfc40b48b47691d127609ef74a` |
| `quality_report.json` | `fc1a3fe67ea1e076b48d9851ec68ed40c6c416ec188f921a32eb10dc99e7632d` |
| Conteúdo do manifesto sem o próprio digest | `c9b3f6be3147d8b861eb35e763817155bc612697487289e15e3cfe36a0e15762` |
| Arquivo `dataset_manifest.json` | `8da1e331b5a44e84ca60dae0fb90f9fb26491acecf2e89cc58f74ce86e0e16aa` |

### 2026-11-01 — Implementação da SPEC 02

- Foram fixados TensorFlow 2.21.0, Keras 3.15.1 e Matplotlib 3.10.7 na `.venv`.
  A wheel do TensorFlow informa CUDA 12.5.1 e cuDNN 9.
- O pipeline `tf.data` foi implementado sobre a referência Pillow da SPEC 01 por
  meio de `tf.numpy_function`, evitando uma segunda definição de
  pré-processamento.
- Foi implementada a CNN `small_cnn_v1`, com 25.697 parâmetros treináveis,
  saída sigmoide representando `P(muffin)` e sem transfer learning.
- Foram implementados reprodutibilidade, diagnóstico de dispositivo, ciclo de
  vida dos runs, checkpoint pelo menor `val_loss`, métricas, curvas e o gate da
  avaliação final.
- Smokes autorizados em CPU exercitaram o fluxo funcional. Smokes continuaram
  inelegíveis para avaliação final e para qualquer alegação de qualidade.
- Naquele momento, `nvidia-smi` não comunicava com o driver, o TensorFlow não
  listava GPU e o treinamento completo permanecia bloqueado externamente.

### 2026-11-02 — Smoke training concluído na GPU

O run local `20261102T124140Z-smoke-3c7450d5` registrou:

- status `smoke_completed`;
- NVIDIA GeForce RTX 5070 com driver 595.99.02;
- dispositivo TensorFlow selecionado `/GPU:0`;
- operação de prova executada em
  `/job:localhost/replica:0/task:0/device:GPU:0`;
- uma época com 16 amostras de treino e 16 de validação;
- checkpoint, métricas, resumo do modelo e curvas gerados;
- `test_consulted = false`;
- `smoke_cpu_authorized = false`.

A evidência local está em
`artifacts/runs/20261102T124140Z-smoke-3c7450d5/run_manifest.json`.
As métricas desse subconjunto não são evidência de qualidade e não substituem o
treinamento completo.

### 2026-11-02 — Treinamento completo, avaliação e painel diagnóstico

O run `20261102T131032Z-train-5e1f0311` executou as 15 épocas em GPU e
selecionou o checkpoint da época 15 pelo menor `val_loss`, igual a `0.283202`.
Na validação, alcançou acurácia `0.901408` e macro-F1 `0.900266`, superando a
meta didática aprovada antes da consulta ao teste.

A avaliação final das 1.184 imagens de teste registrou:

- acurácia `0.880068`;
- precisão de muffin `0.897233`;
- recall de muffin `0.834559`;
- F1 de muffin `0.864762`;
- macro-F1 `0.878511`;
- matriz de confusão `[[588, 52], [90, 454]]`.

Foi criado `evaluation_panel.png` com exatamente 3 VP, 3 VN, 3 FP e 3 FN. Os
exemplos são escolhidos por ordem de `sample_id`, sem seleção manual, e mostram
a entrada RGB 128 × 128 processada, classe real, classe prevista e
`P(muffin)`. Como a avaliação já havia sido concluída, as probabilidades foram
recalculadas somente para produzir o painel; a matriz de confusão teve de
coincidir com a avaliação congelada antes da publicação do artefato.

## Decisões vigentes

### Dados e proveniência

- Usar somente o dataset Kaggle ID `2733586`, versão imutável 2.
- Manter a licença como ambígua, apesar da declaração CC0 da plataforma.
- Restringir o uso à finalidade educacional local autorizada e não redistribuir
  as imagens.
- Preservar todos os originais; exclusões e decisões de qualidade são lógicas e
  auditáveis.

### Contrato de entrada

- Manter `0 = chihuahua` e `1 = muffin` em todo o projeto.
- Usar entradas RGB `128 × 128 × 3`, corrigir EXIF, replicar grayscale, compor
  RGBA sobre branco, redimensionar proporcionalmente com padding branco e
  interpolação bilinear e normalizar para `[0, 1]`.
- Aplicar a mesma transformação determinística em treino, validação, teste e
  futura inferência.

### Particionamento e duplicatas

- A SPEC 01 é a única proprietária dos splits; a SPEC 02 não pode
  reparticionar.
- Preservar o `test` original e derivar validação somente do `train` original.
- Usar seed 42 e estratificação por classe.
- Usar SHA-256 para duplicatas exatas e pHash apenas como triagem. Distância
  perceptual isolada não autoriza exclusão, relabeling ou agrupamento.

### Modelo e treinamento

- Usar uma CNN do zero com convoluções 16/32/64, Global Average Pooling, Dense
  32 e saída sigmoid, totalizando 25.697 parâmetros treináveis.
- Interpretar a saída como `P(muffin)` e manter o limiar inicial em 0,5.
- Usar batch 32, Adam `0.001`, Binary Crossentropy, até 15 épocas e sem data
  augmentation no baseline.
- Selecionar o checkpoint pelo menor `val_loss` absoluto. O `min_delta = 0.001`
  pertence apenas ao early stopping.
- Exigir GPU no treinamento completo e nunca fazer fallback silencioso para
  CPU. CPU no smoke requer autorização explícita.
- Não usar o conjunto de teste para escolher arquitetura, hiperparâmetros,
  checkpoint ou limiar.

### Avaliação e evidências

- A meta didática de validação é superar estritamente a acurácia `0.540845` e o
  macro-F1 `0.351005` do classificador que sempre prevê chihuahua.
- Execução sem erro não é evidência de qualidade preditiva.
- Apenas um run `trained_frozen`, com configuração, dados e checkpoint íntegros,
  pode abrir o teste.
- Registrar versões, dispositivo, configuração, seeds, dados, checkpoint,
  métricas, limitações e digests em cada run.
- Permitir o painel pós-avaliação apenas como diagnóstico, com seleção
  determinística e `used_for_selection_or_tuning = false`.

## Situação após o baseline

Treino, verificação da meta, avaliação final e painel diagnóstico estão
concluídos. Qualquer experimento futuro deve ser tratado como uma nova etapa e
não pode usar os resultados deste teste para ajustar arquitetura,
hiperparâmetros ou limiar. As métricas permanecem condicionadas à limitação dos
719 pares perceptuais não revisados.

O histórico detalhado de requisitos, critérios e decisões locais de cada etapa
continua preservado nas seções “Registro de decisões” das respectivas SPECs.
