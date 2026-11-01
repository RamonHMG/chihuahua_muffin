# SPEC 02 — CNN, treinamento e avaliação

**Status:** Draft

**Criada em:** 2026-11-01

**Dependência:** contrato aprovado e validado da SPEC 01

## Objetivo

Treinar e avaliar uma CNN pequena e didática com TensorFlow/Keras, consumindo
sem alteração os dados e splits da SPEC 01 e produzindo evidências auditáveis do
ambiente, da configuração, do checkpoint e dos resultados.

## Cenários e testes independentes

### Cenário 1 — Consumir o contrato de dados (P1)

**Teste independente:** a execução rejeita esquema incompatível, digest
incorreto, classe desconhecida ou tentativa de reparticionamento.

**Aceitação:** dado um manifesto válido, quando o pipeline é construído, então
cada amostra usa o split e o rótulo definidos pela SPEC 01.

### Cenário 2 — Executar um smoke training (P1)

**Teste independente:** um subconjunto pequeno percorre leitura, forward pass,
backpropagation, métrica e salvamento em tempo limitado.

**Aceitação:** CPU só é aceita por opção explícita; o resultado não é apresentado
como evidência de qualidade ou treinamento completo.

### Cenário 3 — Treinar com GPU usando validação (P1)

**Teste independente:** sem GPU TensorFlow utilizável, o treinamento completo
falha; com GPU, registra o dispositivo e consome exatamente os conjuntos de
treino, validação e teste declarados pelo manifesto da SPEC 01.

### Cenário 4 — Avaliar no teste independente (P1)

**Teste independente:** o checkpoint já selecionado é carregado sem novo ajuste
e produz métricas e matriz de confusão no split de teste.

## Escopo

Inclui pipeline `tf.data`, CNN pequena, treinamento, callbacks, logs,
checkpoint, avaliação final, gráficos, métricas e manifesto de execução.

Não inclui reparticionamento, alteração de rótulos, ajuste de parâmetros com o
teste, download do dataset, instalação automática de drivers ou pretensão de
equivalência bit a bit entre ambientes distintos.

## Requisitos funcionais

### Contrato e pipeline de entrada

A SPEC 01 define e valida o contrato determinístico de pré-processamento. Esta
SPEC é responsável por aplicá-lo efetivamente com `tf.data` e por comprovar,
por testes compartilhados de conformidade, que não há divergência entre a
definição publicada e os pipelines de treino, validação, teste e inferência.

- **MODEL-001:** a execução deve validar versão do esquema, digests, classes,
  splits e parâmetros da SPEC 01 antes do treino.
- **MODEL-002:** nenhum comando da SPEC 02 pode criar ou modificar splits.
- **MODEL-003:** o pipeline deve consumir uma única implementação do contrato da
  SPEC 01: EXIF, decodificação compatível com os formatos aceitos, RGB, RGBA em
  fundo branco, grayscale replicado, resize proporcional com padding branco,
  bilinear e `[0, 1]`. O mesmo procedimento determinístico vale para treino,
  validação, teste e inferência futura e deve ser coberto por testes.
- **MODEL-004:** shuffling é permitido somente no treino. Data augmentation
  permanece desativado no baseline inicial.
- **MODEL-005:** o pipeline pode prever extensão futura para augmentation, mas
  ela não pode ser ativada sem autorização e deverá ocorrer somente no treino.

### Modelo

- **MODEL-006:** a arquitetura construída do zero deve ser exatamente:
  `Input(128,128,3)`; três blocos `Conv2D + ReLU + MaxPooling2D` com 16, 32 e 64
  filtros, kernel 3×3, padding `same`, stride 1, pooling 2×2 com stride 2 e viés;
  `GlobalAveragePooling2D`; `Dense(32, ReLU)` com viés; e `Dense(1, Sigmoid)` com
  viés. Transfer learning não é permitido nesta versão.
- **MODEL-007:** a saída deve ser uma unidade sigmoide que estima `P(muffin)`.
- **MODEL-008:** a perda deve ser Binary Crossentropy, sem inverter o mapeamento
  `0 = chihuahua`, `1 = muffin`.
- **MODEL-009:** a arquitetura deve totalizar 25.697 parâmetros treináveis. A
  implementação deve confirmar parâmetros e dimensões com `model.summary()` e
  testes, registrando qualquer divergência como falha.

### Dimensões e parâmetros esperados

| Etapa | Saída esperada | Parâmetros treináveis |
|---|---:|---:|
| Entrada | 128 × 128 × 3 | 0 |
| Conv2D 16 | 128 × 128 × 16 | 448 |
| MaxPool 1 | 64 × 64 × 16 | 0 |
| Conv2D 32 | 64 × 64 × 32 | 4.640 |
| MaxPool 2 | 32 × 32 × 32 | 0 |
| Conv2D 64 | 32 × 32 × 64 | 18.496 |
| MaxPool 3 | 16 × 16 × 64 | 0 |
| GlobalAveragePooling | 64 | 0 |
| Dense 32 | 32 | 2.080 |
| Dense sigmoid | 1 | 33 |
| **Total** | — | **25.697** |

### Reprodutibilidade

- **MODEL-010:** seed 42 deve ser aplicada a Python, bibliotecas numéricas
  utilizadas, TensorFlow/Keras, inicialização e embaralhamento controlável.
- **MODEL-011:** operações determinísticas devem ser solicitadas quando
  tecnicamente viáveis; falhas ou operações não determinísticas devem ser
  registradas.
- **MODEL-012:** não se exige identidade numérica bit a bit entre hardwares ou
  versões diferentes.

### GPU e modos de execução

- **MODEL-013:** treinamento completo exige GPU reconhecida pelo TensorFlow e
  efetivamente selecionada para as operações de treino.
- **MODEL-014:** ausência ou falha da GPU deve encerrar `make train` com erro
  explicativo, sem fallback silencioso para CPU.
- **MODEL-015:** `smoke-train` só pode usar CPU mediante opção explícita; testes
  sem dependência de GPU devem continuar executáveis em CPU.
- **MODEL-016:** não deve haver instalação ou alteração automática de driver,
  CUDA ou cuDNN.

### Treinamento e seleção

A SPEC 01 escolheu definitivamente o **Cenário A** para a versão 2: treino e
validação derivam do `train` original, enquanto o `test` original é preservado.
A SPEC 02 deve consumir os três conjuntos sem reparticionar, aplicar early
stopping e selecionar o checkpoint pelo menor `val_loss`.

O Cenário B permanece documentado apenas como contrato histórico/condicional e
não é aplicável ao manifesto desta versão. O teste não participa de escolha,
ajuste ou decisão de repetir experimentos. Seus resultados não podem ser usados
para tentar novas configurações até obter desempenho melhor.

- **MODEL-017:** máximo inicial de 15 épocas.
- **MODEL-018:** `EarlyStopping` monitora `val_loss`, com
  `patience = 3`, `min_delta = 0.001` e restauração dos melhores pesos.
- **MODEL-019:** um mecanismo explícito de checkpoint, monitorando `val_loss`
  em modo `min` e com salvamento somente do melhor, deve persistir o modelo do
  menor valor absoluto observado, independentemente do `min_delta` do
  `EarlyStopping`. Época e valor devem ser registrados.
- **MODEL-019A:** em um manifesto futuro que selecionasse o Cenário B, o
  treinamento executaria 15 épocas fixas, não usaria
  callbacks ou métricas de validação para parada/seleção e salvaria o modelo da
  última época como único modelo elegível para avaliação.
- **MODEL-020:** seleção de checkpoint e hiperparâmetros não pode consultar o
  teste.
- **MODEL-021:** término normal ou early stopping não prova convergência nem
  qualidade satisfatória.
- **MODEL-022:** valores iniciais aprovados: batch size 32, Adam, learning rate
  0.001 e Binary Crossentropy. São configurações iniciais, não resultados de
  otimização.
- **MODEL-022A:** a avaliação final deve carregar explicitamente o checkpoint
  salvo pelo menor `val_loss`; os pesos restaurados pelo `EarlyStopping` não o
  substituem como fonte da avaliação.
- **MODEL-022B:** em um manifesto futuro com Cenário B, a avaliação final deve
  carregar explicitamente o
  modelo da última época e o manifesto deve registrar que não houve validação,
  early stopping nem seleção de checkpoint.

### Métricas e avaliação

- **MODEL-023:** registrar loss, acurácia, precisão e recall durante as fases em
  que forem aplicáveis.
- **MODEL-024:** calcular F1-score, macro-F1 e matriz de confusão de forma
  reproduzível na avaliação.
- **MODEL-024A:** quando precisão, recall ou F1 tiver denominador nulo, o valor
  adotado deve ser `0.0` (`zero_division = 0`), acompanhado dos suportes e da
  matriz de confusão para tornar a situação auditável. Macro-F1 usa a média
  aritmética do F1 das duas classes sob a mesma convenção.
- **MODEL-025:** o limiar inicial é 0.5: `< 0.5` chihuahua; `>= 0.5` muffin.
- **MODEL-026:** qualquer ajuste futuro do limiar deve usar somente validação e
  ser registrado; o teste não pode participar.
- **MODEL-027:** o teste deve permanecer reservado até a definição de modelo,
  hiperparâmetros, checkpoint e limiar.
- **MODEL-028:** curvas de loss e métricas devem distinguir treino e validação.
  Em um manifesto futuro com Cenário B, somente curvas de treino poderiam ser
  geradas e deveriam ser identificadas sem sugerir validação.

## Baseline e qualidade preditiva

Após a auditoria da SPEC 01, deve ser calculado sem usar o teste para seleção um
baseline que sempre prevê a classe majoritária. A meta preditiva será submetida
a aprovação humana antes da avaliação final.

Os relatórios devem separar explicitamente:

1. correção funcional da implementação;
2. funcionamento do pipeline de treinamento;
3. qualidade preditiva;
4. evidência de generalização em dados independentes.

Até a aprovação de uma meta, concluir o treinamento não permite declarar o
modelo satisfatório.

## Artefatos e manifesto de execução

Organização futura prevista, ainda inexistente:

```text
artifacts/runs/<run-id>/
├── run_manifest.json
├── config.toml
├── model.keras
├── metrics.json
├── confusion_matrix.png
└── training_curves.png
```

O manifesto deve registrar no mínimo:

- ID e horário da execução;
- revisão Git e indicação de worktree suja;
- digest e versão do manifesto da SPEC 01;
- configuração efetiva e seeds;
- versões de Python, TensorFlow, Keras e demais bibliotecas relevantes;
- sistema operacional, GPU, driver, dispositivos TensorFlow e dispositivo
  efetivamente usado;
- arquitetura e número de parâmetros;
- épocas solicitadas e executadas, motivo de parada e histórico;
- Cenário A recebido da SPEC 01 e evidência de que treino, validação e teste
  foram consumidos sem reparticionamento;
- checkpoint selecionado pelo menor `val_loss`, sempre com digest;
- limiar efetivo;
- métricas finais e caminhos/digests dos artefatos;
- limitações e fontes conhecidas de não determinismo.

## Critérios de aceitação verificáveis

- **AC-MODEL-001:** um contrato inválido é rejeitado antes do treino.
- **AC-MODEL-002:** os IDs e splits consumidos são idênticos aos da SPEC 01.
- **AC-MODEL-003:** a saída possui uma unidade sigmoide e representa muffin.
- **AC-MODEL-003A:** `model.summary()` e testes confirmam as dimensões esperadas
  e exatamente 25.697 parâmetros treináveis.
- **AC-MODEL-004:** `make train` falha sem GPU TensorFlow utilizável e não faz
  fallback silencioso.
- **AC-MODEL-005:** uso de CPU no smoke requer autorização explícita registrada.
- **AC-MODEL-006:** o checkpoint é selecionado pelo menor `val_loss`, sem teste.
- **AC-MODEL-006A:** a avaliação carrega o checkpoint persistido e registra sua
  época e `val_loss`, sem confundi-lo com a regra `min_delta` do early stopping.
- **AC-MODEL-006B:** não aplicável à versão 2. Em um futuro Cenário B, não
  existiria seleção por validação; seriam executadas 15 épocas fixas e a
  avaliação carregaria o modelo da última época.
- **AC-MODEL-007:** early stopping usa os valores aprovados e restaura os
  melhores pesos.
- **AC-MODEL-008:** teste é avaliado somente após congelar checkpoint,
  configuração e limiar.
- **AC-MODEL-009:** métricas exigidas, matriz de confusão e curvas são geradas
  com rótulos coerentes.
- **AC-MODEL-010:** o manifesto contém ambiente, GPU, configuração, dados,
  seeds, checkpoint, limiar, métricas e limitações.
- **AC-MODEL-011:** duas execuções preservam manifesto de dados e configuração;
  diferenças numéricas permitidas são reportadas, não ocultadas.
- **AC-MODEL-012:** nenhuma alegação de qualidade é feita antes da meta
  preditiva ser aprovada e verificada.

## Casos de borda

- GPU visível ao sistema, mas indisponível para TensorFlow;
- GPU listada, mas operações posicionadas em CPU;
- falta de memória da GPU;
- manifesto alterado entre treino e avaliação;
- split vazio ou com apenas uma classe;
- métricas indefinidas por ausência de positivos previstos ou reais;
- interrupção antes de produzir checkpoint válido;
- artefato parcial após falha;
- configuração incompatível com o contrato de pré-processamento;
- operação determinística indisponível para a combinação de hardware/software.

## Dependências e decisões pendentes

- versão compatível de TensorFlow/Keras e requisitos CUDA/cuDNN;
- disponibilidade da GPU RTX 5070 para o TensorFlow;
- formato/versionamento final do manifesto de execução;
- baseline medido e meta mínima de qualidade preditiva.

## Registro de decisões

- **2026-11-01:** treinamento completo requer GPU; CPU no smoke somente por
  opção explícita.
- **2026-11-01:** seleção do checkpoint pelo menor `val_loss`, sem usar teste.
- **2026-11-01:** registrar acurácia, precisão, recall, F1, macro-F1 e matriz de
  confusão.
- **2026-11-01:** limiar inicial 0.5, com muffin como classe positiva.
- **2026-11-01:** máximo de 15 épocas; early stopping em `val_loss`, patience 3,
  min_delta 0.001 e restauração dos melhores pesos.
- **2026-11-01:** seed 42 e determinismo quando tecnicamente viável, sem exigir
  equivalência bit a bit entre ambientes.
- **2026-11-01:** meta de qualidade será definida somente após auditoria e
  baseline da classe majoritária.
- **2026-11-01:** CNN do zero com blocos 16/32/64, GAP, Dense 32 e saída
  sigmoide, total esperado de 25.697 parâmetros; sem transfer learning.
- **2026-11-01:** batch 32, Adam 0.001 e Binary Crossentropy são valores iniciais
  aprovados, não hiperparâmetros otimizados.
- **2026-11-01:** augmentation permanece desativado no baseline inicial.
- **2026-11-01:** checkpoint explícito usa o menor `val_loss` absoluto; o
  `min_delta` 0.001 pertence somente ao critério do EarlyStopping.
- **2026-11-01:** métricas com denominador nulo adotam `zero_division = 0` e
  registram suporte e matriz de confusão.
- **2026-11-01:** a SPEC 01 escolheu definitivamente o Cenário A para a versão
  2. A SPEC 02 consumirá treino, validação e teste; usará no máximo 15 épocas,
  early stopping em `val_loss` e o checkpoint de menor `val_loss`.
- **2026-11-01:** o `test` original será avaliado somente depois de congelar
  modelo, hiperparâmetros, checkpoint e limiar; suspeitas perceptuais aceitas
  como limitação não autorizam uso do teste para seleção.
