# SPEC 01 — Aquisição e preparação dos dados

**Status:** Draft

**Criada em:** 2026-11-01

**Entrada:** dataset candidato
`samuelcortinhas/muffin-vs-chihuahua-image-classification`

**Consumidor:** SPEC 02 — CNN, treinamento e avaliação

## Objetivo

Produzir um inventário auditável das imagens, um particionamento reproduzível e
um contrato de pré-processamento que possa ser consumido pela SPEC 02 sem
reparticionamento ou decisões implícitas.

Esta SPEC não implementa nem treina um classificador.

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

### Cenário 3 — Produzir splits sem vazamento (P1)

O treinamento precisa de subconjuntos independentes e reproduzíveis.

**Teste independente:** hashes e grupos de versões da mesma fotografia não
atravessam splits; uma nova execução com as mesmas entradas e seed produz as
mesmas atribuições.

**Aceitação:** dado um grupo de versões da mesma fotografia, quando o split for
criado, então todas as ocorrências aceitas do grupo pertencem ao mesmo
subconjunto.

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
- **DATA-015:** um grupo de versões da mesma fotografia não pode atravessar
  treino, validação e teste.
- **DATA-016:** semelhança semântica entre chihuahuas e muffins não caracteriza
  duplicação. O método não pode agrupar imagens apenas por aparência da classe.
- **DATA-017:** distância perceptual próxima não autoriza exclusão. Candidatos
  próximos ao futuro limiar, ambíguos ou com rótulos conflitantes devem ir para
  revisão humana; rótulos não podem ser alterados automaticamente.
- **DATA-018:** metodologia, algoritmo, parâmetros, critérios, grupos,
  exclusões e revisões devem constar no relatório.

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

### Particionamento

- **DATA-019:** antes de dividir, a implementação deve inspecionar e registrar
  partições preexistentes.
- **DATA-020:** partições oficiais de treino, validação e teste devem ser
  preservadas se forem consistentes e livres de vazamento.
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
- **DATA-032:** transformações determinísticas devem ocorrer durante a leitura
  com `tf.data`; não serão produzidas cópias redimensionadas nesta versão.
- **DATA-033:** parâmetros que precisem ser estimados devem usar apenas treino e
  depois ser aplicados sem reajuste a validação e teste.
- **DATA-034:** qualquer aumento de dados futuro deve ser aleatório apenas no
  treino e pertencer à configuração da SPEC 02, não ao split.

## Precedência para criação dos splits

1. Inspecionar estrutura e partições originais.
2. Inventariar arquivos, corrigir metadados para leitura e calcular hashes.
3. Formar candidatos a grupos de duplicatas e resolver revisões necessárias.
4. Verificar vazamento nas partições oficiais.
5. Preservar partições oficiais válidas conforme DATA-020 e DATA-021.
6. Caso não existam partições utilizáveis, aplicar 70%/15%/15%, seed 42, com
   estratificação por classe e indivisibilidade dos grupos.
7. Registrar proporções desejadas, efetivas e sua justificativa.

## Contrato de dados

Artefatos futuros previstos:

- `data/prepared/index.csv`: uma linha por arquivo descoberto;
- `data/prepared/dataset_manifest.json`: identidade, esquema, parâmetros e
  resumo do dataset preparado;
- `data/prepared/quality_report.json`: ocorrências e decisões de qualidade.

Nenhum desses artefatos existe no bootstrap.

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

## Critérios de aceitação verificáveis

- **AC-DATA-001:** fonte, versão e licença são não vazias e auditáveis.
- **AC-DATA-002:** cada arquivo descoberto é reconciliado como aceito, excluído
  ou revisão humana.
- **AC-DATA-003:** toda exclusão possui motivo e aparece no relatório.
- **AC-DATA-004:** cada amostra aceita possui classe válida, caminho existente,
  hash e exatamente um split.
- **AC-DATA-005:** o mapeamento é exatamente chihuahua 0 e muffin 1.
- **AC-DATA-006:** nenhuma SHA-256 aceita aparece mais de uma vez.
- **AC-DATA-007:** nenhum `duplicate_group_id` aceito atravessa splits.
- **AC-DATA-008:** recriar o split com as mesmas entradas e seed produz a mesma
  atribuição.
- **AC-DATA-009:** as proporções e a justificativa de desvios estão registradas.
- **AC-DATA-010:** o contrato de entrada é RGB 128 × 128 × 3, padding branco,
  bilinear e `[0, 1]`, com padding normalizado igual a `(1.0, 1.0, 1.0)`.
- **AC-DATA-011:** digests do índice e relatório correspondem ao manifesto.
- **AC-DATA-012:** nenhuma revisão humana bloqueante permanece sem decisão na
  versão liberada para treinamento completo.
- **AC-DATA-013:** o manifesto declara `schema_version = "1.0"` e seu digest é
  reproduzível independentemente da ordem original das chaves.

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

- versão e licença reais do dataset;
- estrutura e partições reais da fonte;
- algoritmo perceptual, parâmetros, limiar conservador e protocolo de revisão;
- serializador canônico exato a ser implementado e testado;
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
