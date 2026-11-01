# Instruções para agentes

## Missão e estado do projeto

Este repositório desenvolverá, de forma didática e reproduzível, um classificador
binário de imagens de chihuahuas e muffins com Python e TensorFlow/Keras.

O projeto está em implementação incremental da SPEC 01. A presença de uma
interface no `Makefile` não significa que preparação final, validação completa,
treinamento ou avaliação já estejam implementados.

## Leitura obrigatória

Antes de propor ou realizar uma implementação:

1. Leia `README.md`.
2. Leia `docs/specs/01_dataset.md`.
3. Identifique o contrato de dados e seus critérios de aceitação.
4. Leia `docs/specs/02_modelo.md` antes de trabalhar no modelo.
5. Consulte `config/example.toml` apenas como exemplo; ele não é uma
   configuração de execução validada.

Se uma solicitação conflitar com uma SPEC, um contrato ou uma decisão
registrada, interrompa a mudança e peça decisão humana. Não altere escopo ou
critérios silenciosamente.

## Regras de desenvolvimento

- Trabalhe incrementalmente, mantendo cada mudança pequena e verificável.
- Use Python 3.12 e a `.venv` do projeto somente depois de confirmar que o
  ambiente está funcional.
- Use TensorFlow/Keras para o pipeline de dados e o modelo.
- A SPEC 01 é proprietária do particionamento. A SPEC 02 deve consumir o
  manifesto sem redistribuir amostras.
- Preserve `0 = chihuahua` e `1 = muffin`; a saída sigmoide representa a
  probabilidade de muffin.
- Não use o conjunto de teste para selecionar modelos, hiperparâmetros ou
  limiar de classificação.
- Adicione ou atualize testes quando a implementação correspondente existir.
- Não trate execução sem erro como evidência de qualidade preditiva.
- Não introduza arquivos ou diretórios fora do escopo aprovado sem justificar
  a necessidade e obter autorização quando ela ampliar o escopo.

## Ambiente e dependências

- Não instale pacotes fora da `.venv`.
- Não instale ou altere TensorFlow, CUDA, cuDNN, drivers ou configurações de GPU
  sem autorização explícita.
- Não presuma compatibilidade de versões: verifique-a e registre a evidência.
- Não armazene tokens, credenciais Kaggle ou outros segredos no repositório.
- Quando a instalação for explicitamente autorizada, use o Python da `.venv` e
  prefira `python -m pip` para deixar o ambiente de destino inequívoco.

## Harness e verificações

Use apenas alvos que estejam efetivamente implementados. Consulte `make help`
para o estado atual.

Interface estável planejada:

- `make data`
- `make validate-data`
- `make check`
- `make smoke-train`
- `make train`
- `make evaluate`

Enquanto um alvo for um marcador não implementado, sua falha é intencional e
não pode ser descrita como validação bem-sucedida. Quando houver implementação,
execute as verificações relevantes e relate comando, resultado e limitações.

## Evidências e registro

- Registre decisões arquiteturais na seção "Registro de decisões" da SPEC
  correspondente.
- Registre erros, exclusões, limitações e casos para revisão humana; não descarte
  ocorrências silenciosamente.
- Preserve manifestos, configurações efetivas, versões, logs, métricas e
  identificação dos artefatos conforme as SPECs.
- Nunca invente resultados, contagens do dataset, compatibilidade, uso de GPU,
  métricas ou verificações.
- Diferencie claramente: não executado, executado com falha, executado com
  sucesso e não aplicável.

## Segurança e autorização

Obtenha autorização antes de:

- instalar ou atualizar dependências sensíveis;
- modificar a `.venv`, drivers ou bibliotecas de GPU;
- baixar o dataset quando isso ainda não fizer parte da etapa aprovada;
- mudar contratos, rótulos, splits ou critérios de aceitação;
- executar comandos destrutivos ou remover dados e artefatos;
- fazer commits, publicar mudanças ou usar serviços externos com efeito de
  escrita.

Preserve mudanças preexistentes do usuário. Nunca oculte falhas para produzir
uma aparência de conclusão.

## Critério de conclusão de uma tarefa

Uma tarefa só pode ser declarada concluída quando:

1. o escopo solicitado foi implementado;
2. os contratos aplicáveis foram respeitados;
3. as verificações disponíveis e relevantes foram executadas;
4. resultados e falhas foram relatados fielmente;
5. decisões e limitações relevantes foram documentadas.
