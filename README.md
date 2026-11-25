# Projeto Escala Astro IA

**Matéria:** Aprendizado de Máquina  
**Trabalho:** N3  
**Curso:** 8º Semestre de Engenharia de Software  
**Instituição:** Universidade Católica de Santa Catarina — Unidade Jaraguá do Sul  
**Professora:** Fabiola Pinho Magalhaes  
**Alunos:** Jeziel Nogueira, Vinicius Froes, Santhiago Chapiewski, Pedro Santos e João Witor Mueller

---

## Projeto Escala Astro IA — Brasileirão

Aplicação de aprendizado de máquina para simular confrontos do Campeonato Brasileiro. O usuário escolhe os clubes mandante e visitante; o programa estima vitória, empate ou derrota, mostra probabilidades e placar simulado e recomenda uma escalação 4-3-3 para cada clube.

## Problema, público e fontes dos dados

O objetivo é demonstrar como resultados históricos e estatísticas de jogadores podem apoiar uma análise esportiva. A ferramenta pode ser usada por estudantes de aprendizado de máquina, torcedores e analistas como apoio exploratório — não como garantia de resultado nem como ferramenta de apostas.

Os CSVs são baixados automaticamente da API pública do Kaggle na primeira execução:

- [Brasileirão — Player Stats — 2024](https://www.kaggle.com/datasets/eduardopalmieri/brasileiro-player-stats-2024), de Eduardo Palmieri, licença MIT. Contém estatísticas individuais de jogadores do campeonato de 2024.
- [Brazilian Soccer — Brasileirão](https://www.kaggle.com/datasets/gustavomartino/brasileirao), de Gustavo Gaonn, licença CC0. Contém partidas e placares das temporadas de 2007 a 2023.

Os arquivos baixados ficam em `data/raw/`, ignorados pelo Git. Não é necessário cadastrar token do Kaggle. Use `python app.py --refresh-data` para baixá-los novamente.

## Preparação e modelagem

### Tratamento e exploração

- O arquivo de partidas contém 6.460 linhas; uma partida não tem placar e é descartada, deixando 6.459 partidas com resultado conhecido. Temporada, rodada e clubes são validados antes do treino.
- Os nomes dos clubes são normalizados para conciliar diferenças entre as fontes, por exemplo, `Athletico-PR` e `Athletico Paranaense`.
- As 11.682 linhas de estatísticas de jogadores são agrupadas por atleta e clube. Minutos, gols, assistências, xG, xAG, ações de criação e progressão são convertidos em taxas por 90 minutos.
- A escalação recomendada é uma heurística em 4-3-3: seleciona um goleiro, dois zagueiros, um lateral de cada lado, três meio-campistas e três atacantes. A função precisa representar ao menos 25% dos minutos do jogador; atletas com menos de 450 minutos não são elegíveis para a escalação-base. O índice individual pondera estatísticas por posição e é convertido em percentil para comparar a qualidade geral dos elencos. Como o CSV não fornece métricas específicas de goleiros, o goleiro é ranqueado por minutos jogados.
- A variável-alvo do classificador tem três classes: vitória do mandante, empate e vitória do visitante.

### Conceitos de aprendizado de máquina

- **Aprendizado supervisionado:** cada partida histórica é um exemplo rotulado pelo resultado final.
- **Engenharia de atributos:** o modelo usa diferença de Elo (com bônus de mando), aproveitamento e saldo de gols da temporada até aquela rodada e forma dos cinco jogos anteriores. O Elo inicia em 1.500 pontos, é atualizado após cada rodada e regredido parcialmente à média entre temporadas.
- **Prevenção de vazamento de dados:** atributos são construídos em ordem de temporada e rodada. Todos os jogos da rodada recebem os mesmos dados anteriores; os resultados da rodada só atualizam o histórico depois da geração dos atributos.
- **Classificação multiclasse:** uma floresta aleatória (`RandomForestClassifier`) estima as probabilidades das três classes. O ensemble agrega árvores treinadas em amostras aleatórias, captura relações não lineares e usa `class_weight="balanced"` para reduzir o viés para a classe mais frequente. Árvores não precisam de padronização de escala.
- **Simulação do placar:** médias históricas de gols a favor e contra, suavizadas pela média da liga, alimentam distribuições de Poisson. O placar mostrado é compatível com o resultado mais provável.
- **Uso da escalação:** os percentis do elenco de 2024 ajustam as probabilidades e os gols esperados por uma regra explícita. Esse ajuste é heurístico, separado do classificador e não faz parte da avaliação temporal, pois o conjunto de partidas não contém escalações históricas comparáveis.

### Avaliação, resultados e limitações

`python app.py --eval` treina nas temporadas de 2007 a 2021 e testa nas duas temporadas mais recentes, 2022 e 2023. É uma separação temporal mais adequada que embaralhar partidas, pois representa a previsão de temporadas futuras. Dentro do teste, os resultados observados de uma rodada atualizam os atributos das rodadas seguintes, mas o classificador permanece fixo; isso simula previsões sequenciais durante a temporada. O relatório apresenta acurácia, acurácia balanceada, F1 macro, baseline majoritária e matriz de confusão. A escalação heurística não é incluída nessas métricas.

**Resultado medido neste conjunto:** no teste temporal de 760 partidas (2022–2023), o classificador obteve acurácia de **44,6%**, acurácia balanceada de **41,7%** e F1 macro de **0,410**. O baseline que sempre escolhe a classe mais frequente teve **45,5%** de acurácia. A matriz de confusão (real nas linhas; previsto nas colunas, na ordem mandante/empate/visitante) foi `[[199, 54, 93], [77, 45, 84], [68, 45, 95]]`. O modelo identifica as três classes, mas não supera o baseline em acurácia simples; portanto, seu desempenho é limitado e não deve ser apresentado como previsão confiável. `python app.py --eval` refaz a medição nos arquivos baixados.

Limitações importantes: os placares disponíveis terminam em 2023; as estatísticas de jogadores são da temporada de 2024. O ajuste de escalação é uma heurística, não foi validado por uma base histórica de escalações, e não considera lesões, suspensões, mudanças de técnico, tática ou transferências posteriores. As probabilidades do classificador não foram calibradas e o ajuste da escalação é heurístico; interprete-os como estimativas, não como chances confiáveis ou garantias.

## Instalação e execução

Requer Python 3.10 ou superior e conexão à internet na primeira execução.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python web.py
```

Depois de iniciar o servidor, abra **http://127.0.0.1:8000** no navegador, selecione os clubes e clique em **Simular partida**. Na primeira execução, os arquivos do Kaggle são baixados automaticamente; isso requer conexão com a internet. Encerre o servidor com `Ctrl+C`.

Para usar a versão de linha de comando em vez da interface web, execute `python app.py`.

Comandos úteis:

```powershell
python web.py --port 8080
python web.py --refresh-data
python app.py --list-teams
python app.py --eval
python -m unittest discover -s tests -v
```

## Estrutura

- `app.py` — interface de linha de comando e apresentação dos resultados.
- `web.py` — servidor HTTP local e API JSON que conecta o frontend ao modelo.
- `web/` — interface web responsiva para escolha dos clubes e visualização das simulações.
- `service.py` — valida a entrada e conecta dados, modelo e interface.
- `repository.py` — baixa, valida e normaliza os CSVs do Kaggle.
- `model.py` — atributos históricos, floresta aleatória, avaliação temporal, escalações e simulação.
- `tests/` — testes focados de histórico por rodada, escalação e API web.

## Desenvolvimento

Para compartilhar o desenvolvimento, crie uma branch para cada melhoria, rode os testes e a avaliação, e abra um pull request com a alteração e os resultados observados. Não inclua `data/raw/` nos commits.
