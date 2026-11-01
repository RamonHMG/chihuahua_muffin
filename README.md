# Chihuahua vs. Muffin

Projeto didático de visão computacional para classificação binária de imagens
de chihuahuas e muffins com TensorFlow/Keras.

## Estado atual

Os **Incrementos 1 a 8 da SPEC 01** implementam a estrutura mínima do pacote, o
carregamento e a validação estrutural de `config/example.toml`, serialização
JSON/CSV canônica, digests SHA-256, `sample_id` determinístico, proveniência e
um adaptador KaggleHub testável sem rede. O Incremento 4 acrescenta inventário
determinístico, relatório inicial de qualidade e uma referência Pillow do
contrato de pré-processamento. O Incremento 5 acrescenta deduplicação exata não
destrutiva, decisões humanas canônicas e análise preliminar de validação. O
Incremento 6 acrescenta triagem perceptual pHash, distribuição completa de
distâncias e painel local de revisão. O Incremento 7 fixa o Cenário A e cria
atribuições estratificadas e determinísticas de treino, validação e teste. O
Incremento 8 integra essas evidências, publica atomicamente o índice, o
relatório de qualidade e o manifesto finais e os verifica com um validador
independente. Pipeline `tf.data`, modelo e treinamento continuam não
implementados e pertencem à SPEC 02.

O trabalho está dividido em duas especificações conectadas por um contrato de
dados:

- [SPEC 01 — Dataset e pré-processamento](docs/specs/01_dataset.md):
  proveniência, auditoria, qualidade, duplicatas, particionamento e manifesto.
- [SPEC 02 — CNN, treinamento e avaliação](docs/specs/02_modelo.md): consumo
  do manifesto, CNN, GPU, treinamento, seleção do checkpoint e avaliação.

A SPEC 02 não pode refazer o particionamento produzido pela SPEC 01.
A SPEC 01 também define e valida o contrato determinístico de pré-processamento;
a SPEC 02 será responsável por aplicá-lo com `tf.data` e comprovar por testes a
equivalência, sem manter uma definição divergente.

## Convenções principais

- Python-alvo: 3.12.
- Framework planejado: TensorFlow/Keras, com versões ainda pendentes de
  validação de compatibilidade.
- Classes: `0 = chihuahua`, `1 = muffin`.
- Entrada: RGB `128 × 128 × 3`, com correção EXIF, redimensionamento com
  preenchimento branco, interpolação bilinear e normalização para `[0, 1]`.
- As partições originais `train` e `test` da versão 2 são preservadas. A
  validação contém aproximadamente 15% somente de `train`, com seed 42 e
  estratificação por classe; o `test` original não é redistribuído.
- Manifesto de dataset: `schema_version = "1.0"`, serialização canônica e
  digests do índice e do relatório de qualidade.
- CNN inicial: convoluções com 16/32/64 filtros, Global Average Pooling, Dense
  32 e saída sigmoide; total esperado de 25.697 parâmetros treináveis.
- Treinamento inicial: batch 32, Adam 0.001, Binary Crossentropy, até 15 épocas,
  sem data augmentation.
- Dataset candidato:
  `samuelcortinhas/muffin-vs-chihuahua-image-classification`.

A API pública da Kaggle informa ID `2733586`, versão 2 e licença declarada
`CC0: Public Domain`. Como a descrição afirma que as imagens foram obtidas do
Google Images sem demonstrar sua cadeia de direitos, a licença permanece
ambígua. Em 2026-11-01, o responsável pelo projeto autorizou excepcionalmente o
download e uso educacional local apenas desse dataset e versão, reconhecendo a
ressalva, proibindo redistribuição e sem transformar o estado da licença em
`confirmed`. A implementação exige que essa exceção corresponda exatamente à
fonte, versão e finalidade registradas.

## Interface operacional

`make help`, `make data`, `make validate-data` e `make check` estão
operacionais. Os alvos da SPEC 02 continuam falhando explicitamente até
receberem implementação.

| Comando | Responsabilidade | Estado |
|---|---|---|
| `make help` | Exibir a interface e seu estado | Operacional |
| `make data` | Gerar artefatos finais a partir dos dados e decisões locais aprovados | Operacional |
| `make validate-data` | Validar artefatos, originais e contrato sem escrita | Operacional |
| `make check` | Compilar módulos e executar a suíte sintética/offline | Operacional nos Incrementos 1 a 8 |
| `make smoke-train` | Exercitar treinamento mínimo | Não implementado |
| `make train` | Executar treinamento completo com GPU | Não implementado |
| `make evaluate` | Avaliar o checkpoint final no teste | Não implementado |

Um alvo que termina com a mensagem "não implementado" não constitui uma
verificação aprovada. O sucesso de `make check` cobre os componentes
implementados usando dados sintéticos e temporários; a árvore real e seus
artefatos são verificados separadamente por `make data` e
`make validate-data`. Nenhum desses comandos realiza download, treinamento ou
alteração das imagens originais.

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
  encontrou requisitos quebrados. O Incremento 1 instalou `pytest==9.2.0` e
  somente suas dependências transitivas necessárias.
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
declaração e instalação das demais dependências continuam sujeitas a etapa e
autorização próprias. Drivers e pilha de GPU não foram modificados.

## Configuração

`config/example.toml` é a configuração padrão para validação estrutural e
registra decisões aprovadas e campos pendentes. O modo estrutural permite
testar o contrato sem executar aquisição. O modo de aquisição exige versão
confirmada, licença identificada e confirmada ou uma exceção humana exata; a
configuração atual passa somente para o dataset/version/finalidade autorizados.
A chamada do adaptador ainda exige `download_authorized=True`, portanto validar
a configuração isoladamente não inicia nem autoriza uma operação de rede.

JSON canônico usa UTF-8, chaves ordenadas, representação compacta, floats
finitos e nenhum newline final. O CSV do índice tem cabeçalho fixo, quoting
mínimo explícito, LF e linhas ordenadas por `sample_id`. O identificador combina
versão imutável e caminho POSIX normalizado com codificação prefixada por
comprimento; o hash do conteúdo continua separado.

A única dependência direta de desenvolvimento instalada no Incremento 1 é
`pytest==9.2.0`, registrada no extra `dev` do `pyproject.toml`; também foram
instaladas suas dependências transitivas necessárias.

O Incremento 3 adiciona `kagglehub==1.0.3` como dependência direta. A versão é
compatível com Python 3.12 e instalou somente suas dependências transitivas. O
adaptador não é chamado por `make data`: uma aquisição exige versão confirmada,
autorização operacional explícita e licença confirmada ou a exceção humana
restrita registrada.

O Incremento 4 adiciona `Pillow==13.0.0`. O pacote oficial requer Python 3.11
ou superior, disponibiliza wheel CPython 3.12 para Linux x86-64 e não introduziu
dependências transitivas. O inventário calcula SHA-256 e registra todos os
arquivos como aceitos, excluídos ou em revisão; duplicatas exatas são somente
candidatas nesta etapa. A referência de pré-processamento aplica EXIF, conversão
RGB, padding branco, bilinear e normalização sem modificar nem materializar os
originais.

O Incremento 6 adiciona `ImageHash==4.3.2` para pHash. A versão possui wheel
universal e foi validada localmente com Python 3.12.3, Pillow 13.0.0, NumPy
2.5.4, SciPy 1.18.1 e PyWavelets 1.10.0; `pip check` não encontrou requisitos
quebrados. O pHash é usado somente para formar uma fila de revisão. Nenhuma
distância confirma duplicação ou autoriza exclusão automaticamente.

Dados, credenciais e artefatos gerados ficarão fora do Git. O arquivo
`bootstrap_prompt` é preservado e permanece disponível para versionamento como
registro da origem do projeto.

## Auditoria inicial do dataset real

Aquisição local realizada em 2026-11-01 com KaggleHub 1.0.3 e handle imutável
`samuelcortinhas/muffin-vs-chihuahua-image-classification/versions/2`. O destino
`data/raw/kaggle-2733586-v2` ocupa aproximadamente 499 MB e permanece ignorado
pelo Git.

- 5.917 imagens JPEG `.jpg`: 3.199 chihuahuas e 2.718 muffins;
- partições de origem `train` (4.733: 2.559/2.174) e `test` (1.184:
  640/544), sem validação oficial;
- 5.913 RGB e 4 grayscale L; nenhuma RGBA, animação, imagem ilegível ou EXIF de
  orientação foi encontrada;
- 2.171 dimensões distintas, com larguras de 102 a 4.603 e alturas de 129 a
  4.096 pixels;
- nenhum SHA-256 repetido entre as imagens;
- um arquivo interno `bundle.complete` do KaggleHub foi inventariado e excluído
  explicitamente como não-imagem.

Os artefatos locais não finais estão em
`data/audit/kaggle-2733586-v2-initial/`: `inventory.json` e
`quality_report.json`. Esta é uma auditoria inicial, não valida ausência de
duplicatas perceptuais ou vazamento, não aprova as partições para treinamento e
não constitui manifesto final.

## Deduplicação exata e validação candidata

O Incremento 5 agrupa SHA-256 repetidos deterministicamente, preserva `test`
preferencialmente em grupos de mesmo rótulo entre partições e registra apenas
exclusões lógicas. Conflitos de rótulo exigem decisão humana; confirmar a
duplicação não resolve automaticamente quais amostras serão preservadas.

No dataset real foram encontrados zero grupos exatos, zero conflitos, zero
ocorrências entre `train` e `test`, zero exclusões e zero bloqueios. Permanecem
elegíveis 2.559/2.174 imagens de chihuahua/muffin em `train` e 640/544 em `test`.
Uma proporção de 15% por classe de `train` corresponde a 384 chihuahuas e 326
muffins para validação, deixando 2.175/1.848 para treino. Esses valores foram
materializados logicamente no Incremento 7 após a escolha humana do Cenário A.

Os relatórios locais não finais estão em
`data/audit/kaggle-2733586-v2-exact/`: `exact_deduplication.json`,
`human_decisions.json` e `validation_feasibility.json`.

## Auditoria perceptual e revisão visual

O Incremento 6 calculou pHash de 64 bits com `hash_size = 8` e
`highfreq_factor = 4` para as 5.917 imagens elegíveis e comparou todos os
17.502.486 pares únicos em blocos vetorizados. Os originais e as partições não
foram modificados. A cauda inferior observada foi:

| Distância | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pares | 300 | 101 | 8 | 11 | 17 | 69 | 213 | 1.104 |

A triagem local inclui os 719 pares com distância até 12 para permitir revisão
da fronteira: 441 estão dentro de `train`, 250 cruzam `train`/`test`, 28 estão
dentro de `test` e 31 envolvem rótulos conflitantes. Isso não significa que
sejam 719 duplicatas. A inspeção amostral encontrou versões reais da mesma
imagem entre 0 e 10, enquanto a distância 12 já mistura duplicatas com falsos
positivos; em 14 ocorre forte crescimento de falsos candidatos.

Foram propostas duas faixas: `0–10` como fila conservadora principal (506
pares, sem conflitos de rótulo) e `12` como faixa de fronteira/calibração (213
pares, incluindo 31 conflitos). Distância 14 não é recomendada como teto, pois
adiciona 1.104 pares e forma um componente espúrio de 133 imagens. Nenhuma
decisão humana foi registrada.

Os artefatos locais estão em
`data/audit/kaggle-2733586-v2-perceptual/`: relatório JSON, registro canônico de
decisões, avaliação de viabilidade e `review_panel.html`. O painel mostra as
imagens lado a lado e exporta decisões compatíveis com o registro do Incremento
5. Para este exercício didático, foi decidido não revisar manualmente os 719
pares nem formar grupos a partir de suspeitas não confirmadas. O Cenário A foi
escolhido conscientemente apesar do risco residual de sobreposição.

## Particionamento aprovado

O Incremento 7 preserva todos os 1.184 IDs do `test` original e seleciona a
validação exclusivamente entre os 4.733 exemplos do `train` original. Para
isso, carrega e valida o `inventory.json` anterior. Para cada classe, as amostras
são ordenadas por uma chave SHA-256 derivada de domínio versionado, seed 42 e
`sample_id`; a seleção independe da ordem de leitura.

| Split lógico | Chihuahua | Muffin | Total |
|---|---:|---:|---:|
| `train` | 2.175 | 1.848 | 4.023 |
| `validation` | 384 | 326 | 710 |
| `test` | 640 | 544 | 1.184 |
| **total** | **3.199** | **2.718** | **5.917** |

Os registros locais que alimentam os artefatos finais estão em
`data/audit/kaggle-2733586-v2-splits/`: `split_assignments.json` e
`split_report.json`. Eles preservam ID, caminho, rótulo, partição original e
SHA-256 e são consumidos diretamente, sem novo sorteio, pelo Incremento 8.

Os 719 pares de pHash permanecem candidatos não revisados: não causaram
exclusão, alteração de rótulo, agrupamento ou escolha de split. Assim, versões
da mesma fotografia ainda podem atravessar treino, validação e teste, o que
limita a interpretação futura das métricas como evidência de generalização.

## Artefatos finais da SPEC 01

O Incremento 8 publicou localmente, sob `data/prepared/`, exatamente:

- `index.csv`, com 5.918 linhas de dados: 5.917 imagens aceitas e o marcador
  KaggleHub excluído com motivos documentados;
- `quality_report.json`, que reconcilia estados, classes, partições, formatos,
  deduplicação exata e a limitação perceptual aceita;
- `dataset_manifest.json`, com proveniência, licença ainda ambígua, autorização
  local educacional, contrato de pré-processamento, split e digests.

O gerador valida os digests dos 11 artefatos de auditoria aprovados, compara
cada arquivo original ao SHA-256 inventariado e consome as atribuições do
Incremento 7 sem chamar novamente o algoritmo de split. A publicação ocorre
por diretório temporário e renomeação atômica. Reexecuções idênticas retornam
`up_to_date`; conteúdo final ausente, extra ou divergente é recusado sem
sobrescrita.

Digests da execução real validada em 2026-11-01:

| Artefato/contrato | SHA-256 |
|---|---|
| `index.csv` | `79213ec1780749dbd264f4a5752f2dc268e28acfc40b48b47691d127609ef74a` |
| `quality_report.json` | `fc1a3fe67ea1e076b48d9851ec68ed40c6c416ec188f921a32eb10dc99e7632d` |
| conteúdo do manifesto sem o próprio digest | `c9b3f6be3147d8b861eb35e763817155bc612697487289e15e3cfe36a0e15762` |
| arquivo `dataset_manifest.json` | `8da1e331b5a44e84ca60dae0fb90f9fb26491acecf2e89cc58f74ce86e0e16aa` |

## Próximas etapas

Próximas etapas, somente após revisão do Incremento 8:

1. estabelecer o baseline e uma meta preditiva antes da avaliação final;
2. diagnosticar a GPU e implementar a SPEC 02 sem alterar o manifesto nem o
   particionamento;
3. implementar a equivalência do pré-processamento com `tf.data` e somente
   então iniciar treinamento, mediante autorização própria.

## Referências

- [AGENTS.md](https://github.com/agentsmd/agents.md)
- [GitHub Spec Kit — template de especificação](https://github.com/github/spec-kit/blob/main/templates/spec-template.md)
- [KaggleHub](https://github.com/Kaggle/kagglehub)
