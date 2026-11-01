# SPEC 01 — Aquisição e preparação dos dados

**Status:** Implementada no Incremento 8; aguardando revisão

**Criada em:** 2026-11-01

**Entrada:** dataset candidato
`samuelcortinhas/muffin-vs-chihuahua-image-classification`

**Consumidor:** SPEC 02 — CNN, treinamento e avaliação

## Objetivo

Produzir um inventário auditável das imagens, um particionamento reproduzível e
um contrato de pré-processamento que possa ser consumido pela SPEC 02 sem
reparticionamento ou decisões implícitas.

Esta SPEC não implementa nem treina um classificador.

A SPEC 01 é proprietária da definição e da validação do contrato determinístico
de pré-processamento. A aplicação efetiva desse contrato durante a leitura com
`tf.data` pertence à SPEC 02 e deverá demonstrar equivalência por testes, sem
redefinir ou enfraquecer as regras desta SPEC.

## Cenários e testes independentes

### Cenário 1 — Fixar origem e proveniência (P1)

Um desenvolvedor precisa reproduzir exatamente a fonte utilizada.

**Teste independente:** o manifesto identifica o handle, a versão imutável, a
URL, a licença, a data de obtenção e os digests disponíveis.

**Aceitação:** dado um dataset candidato, quando a aquisição for concluída,
então a versão utilizada pode ser identificada sem depender de "latest".

### Cenário 2 — Auditar qualidade e duplicatas (P1)

Um desenvolvedor precisa conhecer tudo que foi aceito, excluído ou encaminhado
para revisão.

**Teste independente:** o relatório de qualidade reconcilia todos os arquivos
descobertos e contém motivos explícitos para cada exclusão ou pendência.

**Aceitação:** dado um arquivo inválido, animado ou duplicado, quando a auditoria
for executada, então ele não é descartado silenciosamente.

### Cenário 3 — Produzir splits reproduzíveis com limitações explícitas (P1)

O treinamento precisa de subconjuntos independentes e reproduzíveis.

**Teste independente:** amostras e grupos confirmados não atravessam splits;
uma nova execução com as mesmas entradas e seed produz as mesmas atribuições.
Suspeitas perceptuais não revisadas são registradas como limitação, não tratadas
como grupos confirmados.

**Aceitação:** dado um grupo confirmado de versões da mesma fotografia, quando
o split for criado, então todas as ocorrências aceitas do grupo pertencem ao
mesmo subconjunto. Para a execução didática da versão 2, os pares de pHash não
revisados não formam grupos e podem atravessar subconjuntos.

### Cenário 4 — Publicar o contrato para a SPEC 02 (P1)

A etapa de modelo precisa consumir caminhos, rótulos, splits e transformações
sem inferir convenções.

**Teste independente:** um validador consegue verificar o esquema, o digest, o
mapeamento de classes e os parâmetros de entrada sem abrir código de treino.

## Escopo

Inclui aquisição futura, inspeção, controle de qualidade, identificação de
duplicatas, particionamento, índices, contrato de pré-processamento, relatório
de qualidade e manifesto.

Não inclui aumento de dados aleatório, arquitetura da CNN, treinamento,
otimização de limiar ou avaliação preditiva.

## Requisitos funcionais

### Proveniência e aquisição

- **DATA-001:** a implementação deve usar preferencialmente KaggleHub e fixar a
  versão efetivamente obtida.
- **DATA-002:** antes do download, deve verificar as informações publicamente
  disponíveis sobre fonte, licença, versão, organização aparente e condições de
  uso. Ausência ou ambiguidade de licença bloqueia o uso até revisão humana.
- **DATA-002A:** somente após um download futuramente autorizado, deve verificar
  a estrutura real dos arquivos, partições existentes, integridade, classes e
  consistência; a inspeção pública não pode presumir todo o conteúdo interno.
- **DATA-003:** as imagens originais devem ser preservadas sem sobrescrita pelo
  pré-processamento.
- **DATA-004:** o manifesto deve registrar fonte, versão, URL, licença, horário
  de aquisição e versão da ferramenta de aquisição.

### Duas etapas de verificação da fonte

1. **Pré-download:** consultar somente informações públicas e registrar o que é
   conhecido, desconhecido ou ambíguo sobre proveniência, licença, versão,
   organização e condições de uso. Esta etapa pode bloquear o download, mas não
   valida a estrutura interna.
2. **Pós-download futuro:** quando houver autorização separada, inspecionar os
   arquivos efetivamente recebidos, suas partições, integridade, classes e
   consistência. Diferenças em relação à descrição pública devem ser registradas.

Nenhum dado pode ser utilizado se a licença estiver ausente ou ambígua até que
haja decisão humana explícita.

### Exceção humana aprovada para uso educacional local

Em 2026-11-01, o responsável pelo projeto autorizou explicitamente o download e
o uso local, não comercial e educacional do dataset Kaggle
`samuelcortinhas/muffin-vs-chihuahua-image-classification`, versão imutável 2.
A decisão reconhece que a página declara CC0, mas que a cadeia de direitos das
fotografias obtidas do Google Images não foi demonstrada. Ela não confirma a
licença, não autoriza redistribuição e não se aplica a outro dataset, versão ou
finalidade.

O gate implementado mantém a licença como `ambiguous` e somente libera essa
ocorrência quando uma autorização tipada coincide com fonte, ID, handle, versão,
finalidade `educational_local_use`, escopo apenas local, reconhecimento da
ambiguidade e proibição de redistribuição. Versão pendente, licença ausente ou
pendente, autorização divergente ou finalidade diferente continuam bloqueadas.

### Estado público verificado da fonte candidata

Consulta somente leitura realizada em 2026-11-01 no endpoint público da Kaggle:

- fonte: Kaggle;
- ID público: `2733586`;
- handle: `samuelcortinhas/muffin-vs-chihuahua-image-classification`;
- proprietário aparente: `samuelcortinhas` / Samuel Cortinhas;
- versão corrente informada: `2`, registrada como confirmada;
- licença declarada pela plataforma: `CC0: Public Domain`;
- descrição pública: aproximadamente 6.000 imagens obtidas do Google Images.

A declaração CC0 existe e sua evidência foi registrada, mas a descrição não
demonstra a cadeia de direitos das imagens coletadas. Por isso, o estado da
licença permanece `ambiguous` e bloqueia aquisição por padrão. Nenhum arquivo
foi baixado durante essa verificação pública; o download posterior ocorreu
somente após a exceção humana específica registrada abaixo.

O endpoint usado como evidência é
`https://www.kaggle.com/api/v1/datasets/view/samuelcortinhas/muffin-vs-chihuahua-image-classification`.

### Contrato inicial do adaptador de aquisição

- KaggleHub é fixado em `1.0.3` e recebe sempre um handle versionado no formato
  `owner/dataset/versions/N`, nunca `latest`.
- Uma chamada efetiva exige proveniência liberada e autorização humana
  explícita separada.
- O destino é criado de forma exclusiva. Arquivo, diretório ou link já
  existente causa falha sem invocar o backend e sem sobrescrita.
- O adaptador passa `force_download=False`, não transforma arquivos recebidos e
  registra horário, ferramenta, versão, destino e contagem de arquivos.
- Falhas produzem erro estruturado. Conteúdo parcial criado pela própria chamada
  é removido; falha de limpeza é informada explicitamente e nunca ocultada.
- A função de download pode ser injetada, permitindo testes completos sem rede.
- A exceção humana aprovada é passada separadamente ao adaptador e fica
  registrada no resultado; ela não desativa as verificações globais.

### Inventário e controle de qualidade

- **DATA-005:** todos os arquivos candidatos devem receber um estado: aceito,
  excluído ou revisão humana.
- **DATA-006:** arquivos inválidos ou ilegíveis devem ser excluídos com motivo.
- **DATA-007:** imagens animadas devem ser excluídas da versão inicial e
  registradas.
- **DATA-008:** o relatório deve conter contagens por classe, partição de origem,
  formato, estado, motivo de exclusão e grupo de duplicação.
- **DATA-009:** rótulos conflitantes, estrutura inesperada ou casos não
  resolvíveis com segurança devem bloquear a preparação final ou permanecer em
  revisão humana, nunca ser corrigidos silenciosamente.

#### Contrato implementado no Incremento 4

- o inventário percorre arquivos em ordem POSIX relativa e gera `sample_id`
  usando o identificador imutável da versão;
- cada arquivo regular recebe SHA-256, tamanho e um dos estados `accepted`,
  `excluded` ou `review`; links simbólicos não são seguidos e exigem revisão;
- classe e partição original são inferidas apenas de nomes explícitos de
  diretório; ausência de partição é registrada sem inventar um split;
- dimensões, formato, modo, canais, animação e orientação EXIF são lidos com
  Pillow; arquivo não decodificável e animação são excluídos com motivo;
- JPEG e PNG são formatos esperados inicialmente. Outros formatos decodificáveis,
  extensões incompatíveis, modos fora de RGB/RGBA/L, classe ambígua e EXIF
  inválido seguem para revisão em vez de conversão silenciosa;
- hashes repetidos são reportados somente como candidatos a duplicatas exatas.
  Nenhuma ocorrência é removida e nenhum grupo definitivo é criado neste
  incremento;
- `inventory.json` e `quality_report.json` da auditoria inicial usam o JSON
  canônico aprovado, permanecem locais e são explicitamente distintos do índice
  e manifesto finais.

### Resultado da auditoria inicial da versão 2

A aquisição autorizada foi concluída em `2026-11-01T18:31:19.646234Z` com
KaggleHub 1.0.3, handle imutável
`samuelcortinhas/muffin-vs-chihuahua-image-classification/versions/2` e destino
local `data/raw/kaggle-2733586-v2`. O adaptador contou 5.918 arquivos e preservou
os originais. A árvore contém partições `train` e `test`, ambas com diretórios
`chihuahua` e `muffin`; não existe partição de validação na fonte.

O inventário inicial reconciliou:

| Partição | Chihuahua | Muffin | Total |
|---|---:|---:|---:|
| treino da fonte | 2.559 | 2.174 | 4.733 |
| teste da fonte | 640 | 544 | 1.184 |
| **imagens** | **3.199** | **2.718** | **5.917** |

Todas as 5.917 imagens possuem extensão `.jpg` e formato decodificado JPEG.
São 5.913 imagens RGB e 4 grayscale L; não foram encontradas RGBA, animações,
arquivos de imagem ilegíveis, formatos inesperados, EXIF de orientação ou
SHA-256 repetidos. Há 2.171 dimensões distintas: larguras entre 102 e 4.603 e
alturas entre 129 e 4.096 pixels. Um sexto tipo de arquivo, o marcador interno
`.complete/.../bundle.complete` criado pelo KaggleHub, foi inventariado e
excluído explicitamente com os motivos `class_not_identified` e
`non_image_file`.

Os artefatos locais de auditoria inicial são:

- `inventory.json`, SHA-256
  `6994b3d3d4226d958f9cc34a3e54b076356d368a7ac9b3e76608200380dbf4f3`;
- `quality_report.json`, SHA-256
  `72b8d34d1f56b87ce8cf1320ade72abff74575b9af9fff43a0eddea24fa84aae`.

Esses resultados não constituem validação completa: pHash, agrupamento de
versões da mesma fotografia, análise de vazamento entre `train`/`test`, decisão
definitiva sobre duplicatas exatas, criação da validação, splits preparados e
manifesto final permanecem fora deste incremento.

### Classes

- **DATA-010:** o mapeamento imutável é `0 = chihuahua` e `1 = muffin`.
- **DATA-011:** nomes de diretórios ou metadados da fonte devem ser convertidos
  para esse mapeamento de maneira explícita e validada.

### Duplicatas

- **DATA-012:** duplicatas exatas devem ser identificadas por SHA-256 do conteúdo
  original.
- **DATA-013:** duplicatas exatas da mesma classe devem resultar em uma única
  ocorrência aceita; as demais permanecem registradas como excluídas.
- **DATA-014:** pHash é o método candidato inicial para localizar possíveis
  versões da mesma fotografia, mas algoritmo final, parâmetros e limiar devem
  permanecer pendentes até proposta, auditoria de amostras e aprovação.
- **DATA-015:** um grupo confirmado de versões da mesma fotografia não pode
  atravessar treino, validação e teste. Proximidade perceptual não revisada não
  cria um grupo automaticamente.
- **DATA-016:** semelhança semântica entre chihuahuas e muffins não caracteriza
  duplicação. O método não pode agrupar imagens apenas por aparência da classe.
- **DATA-017:** distância perceptual próxima não autoriza exclusão. Candidatos
  próximos ao futuro limiar, ambíguos ou com rótulos conflitantes devem ir para
  revisão humana; rótulos não podem ser alterados automaticamente.
- **DATA-018:** metodologia, algoritmo, parâmetros, critérios, grupos,
  exclusões e revisões devem constar no relatório.

#### Deduplicação exata implementada no Incremento 5

- somente imagens aceitas, com formato decodificado e SHA-256 não vazio,
  participam da análise exata;
- grupos usam o identificador determinístico `exact-sha256:<hash>` e são
  ordenados pelo hash; nomes diferentes não alteram a identidade do conteúdo;
- em grupos com o mesmo rótulo, uma ocorrência é preservada e as demais recebem
  exclusão lógica, sem remoção física. A escolha prioriza `test`, depois
  `train`, `validation`, partição ausente e, por fim, caminho relativo e
  `sample_id`;
- em duplicatas entre `train` e `test` com o mesmo rótulo, a ocorrência de
  `test` é preservada e a de `train` é excluída logicamente com motivo explícito;
- grupos com rótulos conflitantes não alteram nenhum rótulo automaticamente:
  todas as ocorrências ficam bloqueadas até uma decisão humana compatível;
- cada ocorrência recebe uma ação lógica `keep`, `exclude` ou `block`, com
  justificativa. Os arquivos, diretórios e partições originais não são
  modificados;
- uma decisão `not_duplicate` para arquivos com SHA-256 idêntico é registrada,
  mas permanece bloqueante por contradizer a evidência byte a byte.

#### Registro de revisão humana

O artefato local `human_decisions.json` usa JSON canônico e contém, para cada
decisão: `review_id`, IDs-alvo, tipo de ocorrência, estado, justificativa,
revisor, data e listas separadas de amostras preservadas e excluídas. Os estados
permitidos são `confirmed_duplicate`, `not_duplicate` e `uncertain`.

A confirmação da duplicação e o destino das amostras são decisões separadas.
Em especial, um conflito de rótulos marcado `confirmed_duplicate` sem uma
disposição completa continua bloqueante. Uma disposição resolvida deve cobrir
todos os IDs, preservar exatamente um e excluir logicamente os demais.
`uncertain` não pode conter disposição; `not_duplicate` deve preservar todos,
mas não libera um grupo de SHA-256 idêntico. Decisões com IDs, tipo, versão do
dataset ou esquema divergentes são rejeitadas.

### Protocolo proposto para duplicatas perceptuais

Esta proposta orientará a auditoria, mas **não está aprovada para deduplicação
automática**:

1. avaliar como candidato um pHash de 64 bits (`hash_size = 8` e
   `highfreq_factor = 4`), calculado após correção EXIF e conversões necessárias
   para uma representação decodificável consistente; biblioteca, versão e
   detalhes exatos da conversão devem ser fixados na proposta de implementação;
2. medir distância de Hamming somente como sinal de triagem, não como prova;
3. separar amostras de candidatos por faixas de distância, incluindo uma faixa
   próxima ao limiar que vier a ser proposto;
4. avaliar falsos positivos com atenção a imagens semanticamente parecidas de
   chihuahuas e muffins, fundos semelhantes e enquadramentos comuns;
5. avaliar falsos negativos usando transformações controladas da mesma imagem,
   como recompressão, redimensionamento e pequenas alterações de enquadramento;
6. apresentar pares ou grupos candidatos em relatório revisável, junto com
   dimensões, rótulos, origem e evidências adicionais;
7. submeter casos ambíguos e rótulos conflitantes à revisão humana;
8. avaliar esse candidato e propor método final, parâmetros e limiar somente com
   as evidências da auditoria, antes de processar o dataset completo.

Uma versão da mesma fotografia deve conservar o conteúdo e a composição
essenciais apesar de transformações técnicas. Compartilhar textura, cor, forma,
classe ou dificuldade visual não é suficiente. O objetivo é impedir vazamento
entre splits, não remover exemplos difíceis.

#### Auditoria perceptual implementada no Incremento 6

- o método candidato é `ImageHash==4.3.2`, `phash`, `hash_size = 8`,
  `highfreq_factor = 4`, hash de 64 bits e distância de Hamming;
- antes do pHash, a imagem é aberta somente para leitura, seu SHA-256 é
  comparado ao inventário, a orientação EXIF é corrigida, RGB é preservado,
  grayscale é replicado para RGB e RGBA é composto sobre branco;
- somente ocorrências elegíveis após a deduplicação exata participam. Nenhum
  arquivo, rótulo ou partição é alterado;
- para 5.917 imagens, todos os 17.502.486 pares únicos são comparados em blocos
  vetorizados. Isso mantém tempo `O(n²)`, necessário para a distribuição
  completa, e memória de trabalho `O(block_size²)`;
- a distribuição completa, distribuições por relação de partição/classe e a
  distância do vizinho mais próximo são registradas deterministicamente;
- pares de revisão usam IDs determinísticos, estado pendente por padrão e os
  mesmos tipos canônicos de decisão humana do Incremento 5, estendidos para
  ocorrências perceptuais com ou sem conflito de rótulo;
- decisões `confirmed_duplicate`, `not_duplicate` e `uncertain` continuam
  separadas de qualquer disposição futura. Nenhuma proximidade produz exclusão,
  relabeling ou agrupamento definitivo automaticamente;
- o painel HTML local referencia os originais sem copiá-los, mostra os pares
  lado a lado e exporta um documento de decisões compatível com o validador
  canônico. Revisor, data e justificativa são obrigatórios para exportar uma
  decisão.

#### Resultado real e faixas candidatas

A auditoria calculou os 5.917 hashes e reconciliou 17.502.486 comparações. A
cauda inferior da distribuição foi:

| Distância | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pares | 300 | 101 | 8 | 11 | 17 | 69 | 213 | 1.104 |

O relatório mantém 719 pares com distância até 12 para revisão e calibração:

| Relação | Pares |
|---|---:|
| dentro de `train` | 441 |
| entre `train` e `test` | 250 |
| dentro de `test` | 28 |
| **total** | **719** |

Dos 719 pares, 31 têm rótulos conflitantes; todos aparecem em distância 12.
Até distância 10 são 506 pares, sendo 324 dentro de `train`, 165 entre
`train`/`test` e 17 dentro de `test`, sem conflito de rótulo. Esses números são
pares de triagem, não contagens de duplicatas confirmadas.

A inspeção visual amostral encontrou versões da mesma imagem em distâncias 0,
2, 4, 6, 8 e 10, incluindo recompressão, redimensionamento, recorte e marcas
d'água. Em distância 12 foram observados tanto pares reais quanto falsos
positivos semanticamente ou composicionalmente semelhantes. Distância 14
adiciona 1.104 pares, 303 conflitos de rótulo e cria um componente espúrio de
133 imagens; por isso não é proposta como teto de revisão principal.

As faixas candidatas submetidas à decisão humana são:

1. `0–10`: fila conservadora principal, ainda sem confirmação automática;
2. `12`: faixa de fronteira para calibrar falsos positivos e possíveis falsos
   negativos da fila principal;
3. `14`: controle negativo amostral, não recomendado como limiar candidato.

O campo `perceptual_threshold` permanece vazio e a configuração registra
`audited_candidate_pending_human_approval`. O teto 12 do relatório serve para
montar a fila local, não é um limiar aprovado de deduplicação.

Artefatos locais não finais em
`data/audit/kaggle-2733586-v2-perceptual/`:

- `perceptual_audit.json`, SHA-256
  `78a5da381d6561cce8faf71bb3ed401bbeed8c999b35a1789454108be6a1ee21`;
- `perceptual_human_decisions.json`, ainda vazio, SHA-256
  `aaa627f05b4907a0679fa1cd5e09f79b986841060ce329fccf2eb9d813494dce`;
- `validation_feasibility.json`, SHA-256
  `3e9546d23845d092d49c27ea526b4798cd2e6349a3f345d489b41aa394989084`;
- `review_panel.html`, SHA-256
  `d44c6bacc33546c0fa474d1c5c462f941549c5fe9b5812289257465b032c0e98`.

Quantitativamente, as classes continuam com suporte amplo para uma validação
de aproximadamente 15% de `train`. Após revisar o Incremento 6, o responsável
escolheu o Cenário A para este exercício didático sem exigir a revisão manual
dos 719 pares. Nenhum desses pares foi convertido em duplicata confirmada,
grupo, exclusão ou alteração de rótulo. A decisão aceita conscientemente que
versões da mesma fotografia podem atravessar os conjuntos e limitar a
interpretação das métricas de generalização.

### Particionamento

- **DATA-019:** antes de dividir, a implementação deve inspecionar e registrar
  partições preexistentes.
- **DATA-020:** partições oficiais de treino, validação e teste devem ser
  preservadas. Limitações conhecidas de independência devem ser resolvidas ou
  aceitas por decisão humana explícita antes da liberação experimental.
- **DATA-021:** se houver somente treino e teste oficiais, o teste deve ser
  preservado e a validação deve ser derivada exclusivamente do treino.
- **DATA-022:** sem partições oficiais utilizáveis, deve ser aplicado split
  estratificado e orientado por grupos, preferencialmente 70%/15%/15%.
- **DATA-023:** a seed de particionamento é 42.
- **DATA-024:** proporções efetivas e justificativa para diferenças devem ser
  registradas.
- **DATA-025:** se uma duplicata do teste oficial estiver no treino, o teste é
  preservado e o conflito do treino é removido ou isolado, com registro.
- **DATA-026:** conflitos que não possam ser resolvidos com segurança exigem
  revisão humana.

Para a versão 2 deste dataset foi escolhido definitivamente o **Cenário A**. As
partições originais `train` e `test` são preservadas e não são reconstruídas.
A validação é extraída exclusivamente de `train`, com 15% por classe, seed 42
e estratificação. O `test` original permanece integral e reservado.

A escolha é uma simplificação didática consciente. Os 719 pares candidatos de
pHash não foram revisados e não formaram grupos; portanto, não são usados para
excluir, relabelar ou orientar atribuições. A possível sobreposição entre os
conjuntos não bloqueia este experimento, mas limita futuras alegações de
generalização.

### Resultado da deduplicação exata e viabilidade preliminar

A análise SHA-256 do Incremento 5 produziu zero grupos duplicados, zero conflitos
de rótulo, zero ocorrências entre `train` e `test`, zero exclusões lógicas e zero
bloqueios. Permanecem elegíveis nesta etapa:

| Partição original | Chihuahua | Muffin | Total |
|---|---:|---:|---:|
| `train` | 2.559 | 2.174 | 4.733 |
| `test` | 640 | 544 | 1.184 |

Uma proporção de 15% de cada classe de `train` corresponde a 384 chihuahuas e
326 muffins, total de 710 imagens, deixando 2.175 e 1.848 para treinamento. A
análise exata não identificou impedimento quantitativo, mas não demonstra
independência entre as partições: versões comprimidas, redimensionadas ou
alteradas da mesma fotografia permanecem uma limitação perceptual.

Artefatos locais não finais em `data/audit/kaggle-2733586-v2-exact/`:

- `exact_deduplication.json`, SHA-256
  `cc9eebf742ad0d7069a24d1a7eeaef6f5610b1e01516cee7756270553bc62ad6`;
- `human_decisions.json`, atualmente sem decisões, SHA-256
  `aaa627f05b4907a0679fa1cd5e09f79b986841060ce329fccf2eb9d813494dce`;
- `validation_feasibility.json`, SHA-256
  `e53bcedb6cc313f4b33f9c912788f770b34ffe56149513267c2f09f91a4593e3`.

### Particionamento implementado no Incremento 7

O algoritmo `stratified_sha256_rank_v1` considera somente amostras elegíveis
após a deduplicação exata. O `inventory.json` anterior é carregado com validação
estrita de esquema, versão, IDs, caminhos e metadados, sem depender de nova
varredura dos originais. Em cada classe do `train` original, ordena os
`sample_id` por uma chave SHA-256 formada por domínio versionado, seed 42 e o
identificador com comprimento explícito. Os primeiros 15%, arredondados ao
inteiro mais próximo e sem esvaziar nenhum lado, formam a validação. Isso torna
a escolha independente da ordem de leitura e não requer nova dependência.

Contagens efetivas:

| Split lógico | Chihuahua | Muffin | Total |
|---|---:|---:|---:|
| `train` | 2.175 | 1.848 | 4.023 |
| `validation` | 384 | 326 | 710 |
| `test` | 640 | 544 | 1.184 |
| **total** | **3.199** | **2.718** | **5.917** |

Os 1.184 IDs de `test` são exatamente os IDs aceitos da partição original; a
validação contém somente IDs do `train` original. ID, caminho, rótulo, partição
original e SHA-256 são preservados em cada atribuição. Nenhuma imagem foi
movida, excluída ou modificada.

Artefatos locais não finais em
`data/audit/kaggle-2733586-v2-splits/`:

- `split_assignments.json`, SHA-256
  `e37e955f8c3da8d74e56d4a26511de0411e0e9c6f3efffbfd549e5ab46ef727e`;
- `split_report.json`, SHA-256
  `70ef339bbbf77d552205e4a41f89d52a7c3c35f42d6dddb0454ec8b90a3352c3`.

O relatório registra a escolha A, configuração, contagens, preservação do teste
e a decisão de não usar os 719 candidatos perceptuais para agrupamento ou
atribuição. Ele é uma entrada do futuro manifesto, não o manifesto final.

### Pré-processamento

- **DATA-027:** a entrada lógica deve ser RGB `128 × 128 × 3`.
- **DATA-028:** orientação EXIF deve ser corrigida antes do redimensionamento.
- **DATA-029:** RGB preserva três canais; grayscale é convertido pela replicação
  da informação nos três canais; RGBA é composto sobre fundo branco e convertido
  para RGB.
- **DATA-030:** o redimensionamento deve preservar proporção, usar preenchimento
  branco `RGB = (255, 255, 255)` e interpolação bilinear. Após normalização, o
  preenchimento deve ser `(1.0, 1.0, 1.0)`.
- **DATA-031:** pixels devem ser convertidos para ponto flutuante e normalizados
  para `[0, 1]` pela divisão por 255.
- **DATA-032:** a SPEC 01 deve definir e validar as transformações
  determinísticas. A SPEC 02 deve aplicá-las durante a leitura com `tf.data`,
  usando comportamento equivalente no treino, validação, teste e inferência;
  não serão produzidas cópias redimensionadas nesta versão.
- **DATA-033:** parâmetros que precisem ser estimados devem usar apenas treino e
  depois ser aplicados sem reajuste a validação e teste.
- **DATA-034:** qualquer aumento de dados futuro deve ser aleatório apenas no
  treino e pertencer à configuração da SPEC 02, não ao split.

#### Referência verificável implementada no Incremento 4

A referência Pillow abre o original somente para leitura, rejeita animações e
modos não aprovados, aplica `ImageOps.exif_transpose`, preserva RGB, replica L
para RGB ou compõe RGBA sobre branco. Em seguida usa `ImageOps.pad` centralizado
para 128 × 128, `Image.Resampling.BILINEAR`, branco `(255, 255, 255)` e gera uma
representação HWC em ponto flutuante dividida por 255. Os testes verificam EXIF,
modos, interpolação, padding, faixa, dimensões e preservação dos bytes originais.
Esta é a referência da SPEC 01; sua implementação equivalente com `tf.data`
permanece responsabilidade futura da SPEC 02.

## Precedência para criação dos splits

1. Inspecionar estrutura e partições originais.
2. Inventariar arquivos, corrigir metadados para leitura e calcular hashes.
3. Formar candidatos a grupos de duplicatas e resolver revisões necessárias.
4. Verificar vazamento nas partições oficiais.
5. Preservar as partições originais `train` e `test` desta versão.
6. Registrar a decisão humana pelo Cenário A e a aceitação da limitação
   perceptual não revisada.
7. No Incremento 7, extrair 15% por classe de `train` para validação, com seed
   42, estratificação e indivisibilidade dos grupos confirmados.
8. Registrar proporções efetivas, atribuições, decisões e justificativa.

## Contrato de dados

Artefatos finais implementados:

- `data/prepared/index.csv`: uma linha por arquivo descoberto;
- `data/prepared/dataset_manifest.json`: identidade, esquema, parâmetros e
  resumo do dataset preparado;
- `data/prepared/quality_report.json`: ocorrências e decisões de qualidade.

Os três artefatos foram gerados e validados localmente no Incremento 8. Os
artefatos de auditoria dos Incrementos 4 a 7 continuam preservados sob `data/`
como entradas imutáveis do pipeline e não são confundidos com essa publicação
final.

### Integração e publicação implementadas no Incremento 8

`make data` valida os digests dos 11 artefatos aprovados dos incrementos
anteriores, carrega estritamente `split_assignments.json` e não invoca o
algoritmo de particionamento. O pipeline reconcilia a árvore original inteira,
compara os bytes de cada arquivo ao SHA-256 inventariado e constrói em memória
o índice, o relatório e o manifesto. Somente depois de verificar o conjunto
completo ele publica o diretório por renomeação atômica.

Se os três arquivos finais já existem e são byte a byte idênticos, a execução
retorna `up_to_date`. Arquivo ausente, extra ou divergente bloqueia a execução
sem sobrescrita. Falha durante a preparação remove somente o diretório
temporário próprio e nunca deixa uma publicação parcial.

`make validate-data` é independente do gerador e somente leitura. Ele valida
esquemas, tipos, serialização canônica, digest autorreferente do manifesto,
digests dos artefatos, IDs e caminhos únicos, metadados, originais
referenciados, SHA-256, classes, partições, contagens e o contrato completo de
pré-processamento. Também comprova que todos os IDs do `test` original foram
preservados e que `validation` contém apenas amostras do `train` original.

Publicação real validada:

| Artefato/contrato | SHA-256 |
|---|---|
| `index.csv` | `79213ec1780749dbd264f4a5752f2dc268e28acfc40b48b47691d127609ef74a` |
| `quality_report.json` | `fc1a3fe67ea1e076b48d9851ec68ed40c6c416ec188f921a32eb10dc99e7632d` |
| conteúdo do manifesto sem `manifest_sha256` | `c9b3f6be3147d8b861eb35e763817155bc612697487289e15e3cfe36a0e15762` |
| bytes de `dataset_manifest.json` | `8da1e331b5a44e84ca60dae0fb90f9fb26491acecf2e89cc58f74ce86e0e16aa` |

### Campos mínimos do índice

- `sample_id`: identificador estável;
- `relative_path`: caminho relativo sob a raiz original;
- `class_name` e `class_id`;
- `source_partition` e `split`;
- `status` e `reason`;
- `sha256` e `duplicate_group_id`;
- método e evidência usados para agrupamento perceptual, quando aplicável;
- formato, largura, altura, canais e indicação de animação;
- decisões de EXIF/conversão relevantes.

### Campos mínimos do manifesto

- `schema_version = "1.0"` e digest do conteúdo canônico do contrato;
- handle, versão, URL, licença e data de aquisição;
- digests e localização dos arquivos de índice e qualidade;
- mapeamento de classes;
- seed, algoritmo de split, partições originais e proporções efetivas;
- parâmetros completos de pré-processamento;
- contagens por classe, split e estado;
- método e parâmetros de duplicação;
- versões das ferramentas que produziram o contrato;
- revisões humanas pendentes e limitações conhecidas.

### Serialização e digests

- O índice deve ter colunas e ordenação de linhas determinísticas, codificação
  UTF-8 e finais de linha LF; seu SHA-256 deve constar no manifesto.
- O relatório de qualidade deve usar JSON canônico e seu SHA-256 deve constar no
  manifesto.
- O JSON canônico deve ordenar chaves recursivamente, usar UTF-8, representação
  determinística de valores e nenhum espaço insignificante.
- O digest do manifesto deve ser calculado sobre sua forma canônica excluindo
  somente o próprio campo de digest, evitando autorreferência.
- A implementação deverá documentar e testar o serializador exato utilizado,
  sem depender da ordem de inserção das chaves.
- Manifestos e artefatos de dados permanecem locais sob `data/`, inicialmente
  ignorados pelo Git. Publicação ou versionamento exigirá decisão posterior.

#### Convenção canônica aprovada

- JSON usa UTF-8, `sort_keys=True`, `ensure_ascii=False`, separadores
  `(",", ":")`, `allow_nan=False` e nenhum newline adicional. São aceitos
  somente `null`, booleanos, inteiros, floats finitos, strings UTF-8, listas e
  objetos com chaves string; demais tipos e ciclos são rejeitados.
- O digest JSON é SHA-256 dos bytes canônicos. Para um manifesto, somente o
  campo superior indicado como seu próprio digest é removido do cálculo; o
  objeto recebido não é alterado.
- O CSV do índice usa UTF-8 sem BOM, delimitador vírgula, aspas duplas,
  `csv.QUOTE_MINIMAL`, duplicação de aspas internas, nenhum caractere de escape
  e terminador LF. Linhas são ordenadas por `sample_id` e IDs repetidos são
  inválidos.
- A ordem fixa do CSV é: `sample_id`, `relative_path`, `class_name`, `class_id`,
  `source_partition`, `split`, `status`, `reason`, `sha256`,
  `duplicate_group_id`, `perceptual_method`, `perceptual_evidence`, `format`,
  `width`, `height`, `channels`, `is_animated`, `exif_orientation_applied` e
  `color_conversion`.
- Células CSV aceitam vazio, booleano, inteiro, float finito ou string UTF-8.
  `None` vira campo vazio, booleanos usam `true`/`false` e valores não escalares
  são rejeitados.

#### Identificadores de amostra

O `sample_id` é o SHA-256 hexadecimal minúsculo de um domínio versionado seguido
pelo identificador imutável da versão e pelo caminho relativo POSIX normalizado.
Cada componente usa UTF-8 e prefixo de comprimento de 64 bits big-endian, o que
torna os limites inequívocos. O domínio exato é a sequência de bytes ASCII
`chihuahua-muffin:sample-id:v1` seguida por um byte NUL (`0x00`). Caminhos
absolutos, qualificados por drive, vazios ou contendo `..` são rejeitados. O
conteúdo do arquivo não participa do `sample_id`; seu SHA-256 permanece uma
evidência independente de duplicação.

## Critérios de aceitação verificáveis

- **AC-DATA-001:** fonte, versão e licença são não vazias e auditáveis.
- **AC-DATA-002:** cada arquivo descoberto é reconciliado como aceito, excluído
  ou revisão humana.
- **AC-DATA-003:** toda exclusão possui motivo e aparece no relatório.
- **AC-DATA-004:** cada amostra aceita possui classe válida, caminho existente,
  hash e exatamente um split.
- **AC-DATA-005:** o mapeamento é exatamente chihuahua 0 e muffin 1.
- **AC-DATA-006:** nenhuma SHA-256 aceita aparece mais de uma vez.
- **AC-DATA-007:** nenhum `duplicate_group_id` confirmado e aceito atravessa
  splits; candidatos perceptuais não revisados não recebem esse identificador.
- **AC-DATA-008:** recriar o split com as mesmas entradas e seed produz a mesma
  atribuição.
- **AC-DATA-009:** as proporções e a justificativa de desvios estão registradas.
- **AC-DATA-010:** o contrato de entrada é RGB 128 × 128 × 3, padding branco,
  bilinear e `[0, 1]`, com padding normalizado igual a `(1.0, 1.0, 1.0)`.
- **AC-DATA-011:** digests do índice e relatório correspondem ao manifesto.
- **AC-DATA-012:** nenhuma revisão humana exata bloqueante permanece sem decisão
  na versão liberada. A decisão humana documentada pode aceitar suspeitas
  perceptuais não revisadas como limitação de um experimento didático.
- **AC-DATA-013:** o manifesto declara `schema_version = "1.0"` e seu digest é
  reproduzível independentemente da ordem original das chaves.

### Resultado dos critérios no Incremento 8

| Critérios | Situação e evidência |
|---|---|
| `AC-DATA-001` | Atendido: fonte, versão, URL, licença declarada e estado ambíguo permanecem auditáveis no manifesto. |
| `AC-DATA-002`–`003` | Atendidos: 5.918 arquivos reconciliados; 5.917 aceitos e um excluído com motivos no índice e relatório. |
| `AC-DATA-004`–`005` | Atendidos: originais, hashes, classes, mapeamento `chihuahua = 0`, `muffin = 1` e uma atribuição por amostra foram validados. |
| `AC-DATA-006`–`007` | Atendidos: nenhum SHA-256 aceito repetido e nenhum grupo confirmado atravessa splits; suspeitas pHash continuam sem `duplicate_group_id`. |
| `AC-DATA-008`–`009` | Atendidos: atribuições aprovadas são consumidas diretamente, a reexecução é determinística e seed, algoritmo, proporções e contagens estão registrados. |
| `AC-DATA-010` | Atendido: RGB 128 × 128 × 3, EXIF, padding branco, bilinear, modos L/RGBA e normalização `[0, 1]` são verificados. |
| `AC-DATA-011` | Atendido: digests do índice e relatório correspondem aos bytes e ao manifesto. |
| `AC-DATA-012` | Atendido com exceção didática aprovada: não há revisão exata bloqueante; os 719 pares perceptuais permanecem candidatos não revisados e não confirmados. |
| `AC-DATA-013` | Atendido: esquema `1.0`, JSON canônico e digest autorreferente foram reproduzidos pelo validador. |

Os testes automatizados do Incremento 8 cobrem integração sintética completa,
alteração detectável de artefatos e originais, caminhos inseguros, esquema
incompatível, IDs duplicados, divergência de classes e contagens, falha de
publicação, reexecução determinística e comportamento somente leitura do
validador. A validação real comprova separadamente as contagens e os digests
acima; `make check` não depende do dataset nem de internet.

## Casos de borda

- nomes de classes inesperados ou classe ausente;
- arquivos com extensão de imagem e conteúdo inválido;
- orientação EXIF inválida;
- transparência parcial e grayscale com metadados incomuns;
- animações com apenas um frame aparente;
- grupo duplicado que inviabiliza uma estratificação exata;
- duplicatas entre partições oficiais;
- fotografia duplicada com rótulos conflitantes;
- dataset atualizado depois da auditoria;
- licença ausente, ambígua ou incompatível com o uso.

## Critérios ainda pendentes

- eventual revisão jurídica adicional da aplicabilidade da licença declarada;
- eventual revisão futura dos candidatos perceptuais, sem bloquear o
  experimento didático atual;
- eventual aprovação futura de um limiar perceptual para outro experimento;
- decisão futura sobre publicação ou versionamento dos manifestos locais.

## Registro de decisões

- **2026-11-01:** split preferencial 70/15/15 e seed 42.
- **2026-11-01:** partições oficiais válidas têm precedência sobre proporções
  preferenciais.
- **2026-11-01:** entrada RGB 128 × 128 × 3, padding branco RGB 255 e normalizado
  1.0, bilinear e `[0, 1]`.
- **2026-11-01:** EXIF é corrigido; RGBA usa fundo branco; grayscale replica
  canais; animações e arquivos inválidos são excluídos com registro.
- **2026-11-01:** duplicatas exatas da mesma classe são reduzidas a uma; versões
  da mesma fotografia ficam no mesmo split; casos conflitantes exigem revisão.
- **2026-11-01:** imagens originais são preservadas e as transformações são
  aplicadas em leitura via `tf.data`.
- **2026-11-01:** mapeamento imutável `0 = chihuahua`, `1 = muffin`.
- **2026-11-01:** pHash é candidato para triagem conservadora; parâmetros e
  limiar dependem da auditoria e nenhuma proximidade remove imagens sozinha.
- **2026-11-01:** manifesto de dataset usa `schema_version = "1.0"`, serialização
  canônica e digests do índice e relatório; artefatos permanecem locais e
  ignorados pelo Git inicialmente.
- **2026-11-01:** verificação pública pré-download e inspeção real pós-download
  são etapas distintas; licença ausente ou ambígua bloqueia o uso.
- **2026-11-01:** a SPEC 01 define e valida o contrato de pré-processamento; a
  SPEC 02 o aplica com `tf.data` e deve comprovar equivalência por testes.
- **2026-11-01:** JSON e CSV canônicos usam exclusivamente a biblioteca padrão,
  nas convenções documentadas acima; digests são SHA-256 dos bytes exatos.
- **2026-11-01:** `sample_id` deriva da versão imutável e do caminho POSIX
  normalizado com componentes UTF-8 prefixados por comprimento; conteúdo é
  identificado separadamente.
- **2026-11-01:** a API pública da Kaggle confirma dataset ID `2733586` e versão
  2; a plataforma declara CC0, mas a origem descrita como Google Images mantém a
  licença ambígua e bloqueia o download até revisão humana.
- **2026-11-01:** KaggleHub 1.0.3 é o adaptador aprovado; usa handle versionado,
  autorização explícita, destino exclusivo, sem sobrescrita e com limpeza ou
  relato explícito de falhas parciais.
- **2026-11-01:** foi aprovada exceção humana estrita para download e uso
  educacional local do dataset candidato na versão 2, com reconhecimento da
  ambiguidade da licença e sem autorização de redistribuição; o estado da
  licença continua `ambiguous`.
- **2026-11-01:** Pillow 13.0.0 implementa a inspeção inicial e a referência do
  pré-processamento; JPEG/PNG são formatos inicialmente esperados e outros
  formatos ou modos não cobertos exigem revisão.
- **2026-11-01:** no Incremento 4, hashes repetidos são apenas candidatos e os
  relatórios iniciais não realizam deduplicação, split ou manifesto final.
- **2026-11-01:** a versão 2 foi adquirida localmente e revelou 5.917 JPEGs nas
  partições `train` e `test`, além de um marcador KaggleHub não-imagem; a
  auditoria inicial não encontrou corrupção, animações ou SHA-256 repetidos.
- **2026-11-01:** as partições originais `train` e `test` são preservadas; a
  validação contém 15% por classe somente de `train` e foi implementada no
  Incremento 7 com seed 42.
- **2026-11-01:** deduplicação exata usa grupos SHA-256 determinísticos, prefere
  a ocorrência de `test` em grupos de mesmo rótulo entre partições, mantém
  conflitos de rótulo bloqueados e nunca remove originais fisicamente.
- **2026-11-01:** decisões humanas distinguem conclusão sobre duplicação de
  disposição das amostras e usam os estados `confirmed_duplicate`,
  `not_duplicate` e `uncertain` em JSON canônico local.
- **2026-11-01:** a auditoria exata real encontrou zero grupos e não impôs
  obstáculo quantitativo preliminar à validação candidata; a decisão A/B foi
  então adiada até a análise perceptual.
- **2026-11-01:** ImageHash 4.3.2 com pHash de 64 bits (`hash_size = 8`,
  `highfreq_factor = 4`) foi implementado somente para triagem; comparações são
  vetorizadas em blocos e nenhuma distância confirma ou exclui amostras.
- **2026-11-01:** a auditoria perceptual comparou 17.502.486 pares e gerou uma
  fila ampla de 719 pares até distância 12. A faixa principal proposta é 0–10;
  distância 12 é fronteira de calibração e 14 não é recomendada devido ao forte
  crescimento de falsos candidatos.
- **2026-11-01:** o Incremento 6 recomendou o Cenário A condicionado a revisão
  humana e agrupamento; o responsável posteriormente dispensou essa revisão
  para o experimento didático, sem transformar suspeitas em duplicatas.
- **2026-11-01:** o Cenário A foi escolhido definitivamente: 15% por classe do
  `train` original forma a validação, seed 42, e o `test` original permanece
  integral. Os 719 pares perceptuais não revisados não alteram atribuições.
- **2026-11-01:** o algoritmo `stratified_sha256_rank_v1` ordena por chave
  SHA-256 derivada de seed e `sample_id`, garantindo estabilidade sob mudanças
  na ordem de leitura. A possibilidade de sobreposição entre conjuntos é uma
  limitação metodológica aceita, não evidência de ausência de vazamento.
- **2026-11-01:** o Incremento 8 consome inventário e atribuições somente após
  validar seus digests aprovados; ele não recalcula IDs, deduplicação ou split.
  A publicação final usa preparação completa em diretório temporário e
  renomeação atômica, aceita reexecução idêntica e rejeita divergência sem
  sobrescrita.
- **2026-11-01:** `make validate-data` verifica os três artefatos finais e todos
  os originais referenciados sem escrita. Os critérios `AC-DATA-001` a
  `AC-DATA-013` foram atendidos, preservada a exceção didática explícita para
  os 719 candidatos perceptuais não revisados.
