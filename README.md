# Chihuahua vs. Muffin

Projeto didático de visão computacional para classificação binária de imagens
de chihuahuas e muffins com TensorFlow/Keras.

## Estado atual

Os **Incrementos 1 a 5 da SPEC 01** implementam a estrutura mínima do pacote, o
carregamento e a validação estrutural de `config/example.toml`, serialização
JSON/CSV canônica, digests SHA-256, `sample_id` determinístico, proveniência e
um adaptador KaggleHub testável sem rede. O Incremento 4 acrescenta inventário
determinístico, relatório inicial de qualidade e uma referência Pillow do
contrato de pré-processamento. O Incremento 5 acrescenta deduplicação exata
não destrutiva, decisões humanas canônicas e análise preliminar de validação.
Deduplicação perceptual, particionamento, manifestos finais, pipeline `tf.data`,
modelo e treinamento continuam não implementados.

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
- As partições originais `train` e `test` da versão 2 serão preservadas. Uma
  eventual validação conterá aproximadamente 15% somente de `train`, seed 42,
  após aprovação baseada também na auditoria perceptual.
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

`make help` e `make check` estão operacionais. Os demais alvos compõem a
interface estável planejada, mas falham explicitamente até receberem
implementação.

| Comando | Responsabilidade futura | Estado |
|---|---|---|
| `make help` | Exibir a interface e seu estado | Operacional |
| `make data` | Adquirir e preparar os dados | Não implementado |
| `make validate-data` | Validar contrato e critérios da SPEC 01 | Não implementado |
| `make check` | Compilar módulos e executar a suíte pytest | Operacional nos Incrementos 1 a 5 |
| `make smoke-train` | Exercitar treinamento mínimo | Não implementado |
| `make train` | Executar treinamento completo com GPU | Não implementado |
| `make evaluate` | Avaliar o checkpoint final no teste | Não implementado |

Um alvo que termina com a mensagem "não implementado" não constitui uma
verificação aprovada. O sucesso de `make check` cobre somente os componentes já
implementados; não valida aquisição, dados reais ou treinamento.

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
Uma estimativa de 15% por classe de `train` corresponderia a 384 chihuahuas e
326 muffins para validação, deixando 2.175/1.848 para treino. Nenhuma amostra foi
atribuída: a decisão entre treinar com validação (Cenário A) ou por 15 épocas
fixas sem validação (Cenário B) depende do Incremento 6 e só poderá resultar em
split no Incremento 7.

Os relatórios locais não finais estão em
`data/audit/kaggle-2733586-v2-exact/`: `exact_deduplication.json`,
`human_decisions.json` e `validation_feasibility.json`.

## Próximas etapas

Após revisão deste bootstrap:

1. aprovar e instalar somente as dependências necessárias à SPEC 01;
2. inspecionar informações públicas sobre fonte, licença e versão do dataset;
3. implementar incrementalmente a SPEC 01 e seus testes de contrato;
4. no Incremento 6, auditar duplicatas perceptuais e propor método/limiar;
5. decidir entre os Cenários A e B e, somente no Incremento 7, materializar o
   contrato de splits aprovado;
6. estabelecer o baseline e uma meta preditiva antes da avaliação final;
7. diagnosticar a GPU e implementar a SPEC 02 sem alterar o particionamento.

## Referências

- [AGENTS.md](https://github.com/agentsmd/agents.md)
- [GitHub Spec Kit — template de especificação](https://github.com/github/spec-kit/blob/main/templates/spec-template.md)
- [KaggleHub](https://github.com/Kaggle/kagglehub)
