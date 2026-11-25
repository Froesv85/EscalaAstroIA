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

### Fontes complementares avaliadas para a análise de 2026

As fontes abaixo complementam a pesquisa, mas **ainda não são baixadas nem utilizadas pela aplicação**:

- [Football Odds History: Brazil (Sample) — Kaggle](https://www.kaggle.com/datasets/archangel0x01/football-odds-brazil-sample), licença MIT. Inclui as 380 partidas do Brasileirão 2025, placares do intervalo e finais, rodada e odds. Pode ampliar o histórico de resultados; odds não devem ser usadas como atributos sem garantir que foram registradas antes do início da partida.
- [Brasileirão 2003–2025 — Kaggle](https://www.kaggle.com/datasets/regazze/brasileirao-dataset-20032025), licença Apache 2.0. Contém estatísticas agregadas por temporada, como pontos, jogos e gols; serve para análises e conferência de classificações, mas não contém uma linha por partida para treinar o classificador.
- [Details for Brasileirão 2025 Games — Kaggle](https://www.kaggle.com/datasets/pigortekids/details-for-brasileiro-2025-games), licença CC0. Contém calendário e metadados de transmissão, sem placares. A rodada foi inferida a partir das datas, portanto deve ser conferida com uma fonte oficial antes de uso.
- [Tabela oficial da Série A 2026 — CBF](https://www.cbf.com.br/futebol-brasileiro/tabelas/campeonato-brasileiro/serie-a/2026). Referência para a classificação atual; é uma página dinâmica, não uma base histórica integrada ao aplicativo.

O frontend exibe um retrato estático das 38 rodadas em [`web/fixtures-2026.json`](web/fixtures-2026.json): resultados das rodadas 1 a 37 compilados do [Brasileirão em Rodadas](https://brasileiraoemrodadas.com.br/) e agenda da rodada 38 conferida no [placar da ESPN de 02/12](https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/scoreboard?dates=20261202&limit=100). O arquivo foi atualizado em 25/11/2026 e não é sincronizado automaticamente; confira as fontes antes de reutilizá-lo.

O painel **Previsão comparada com o realizado** replaya as partidas em ordem de rodada. O classificador permanece treinado no histórico até 2023 e, para prever cada rodada de 2026, só recebe os resultados encerrados das rodadas anteriores; partidas da mesma rodada não atualizam os atributos antes da previsão. Os jogos sem placar ficam pendentes e não entram nas métricas. É possível comparar Random Forest, Regressão Logística e HistGradientBoosting, selecionar os modelos e editar seus parâmetros numéricos na tela. Para cada modelo, o painel apresenta acurácia, acurácia balanceada, F1 macro, matriz de confusão e previsões por partida e por rodada. Também mostra rankings por rodada e por clube, calculados somente com partidas encerradas; cada linha informa acertos/jogos e acurácia, e empates na liderança são identificados. A avaliação usa apenas a classe de resultado do classificador, sem o ajuste heurístico das escalações. Como os dados de 2024 e 2025 não estão integrados, esta é uma comparação retrospectiva limitada, não uma previsão publicada antes do início do campeonato; as probabilidades não são calibradas.

Os seletores de clubes são alimentados pelos nomes presentes no calendário de 2026, garantindo os 20 participantes dessa temporada mesmo quando não há dados de partidas históricas ou jogadores de 2024 para algum clube. Sem escalação de 2024 disponível, a simulação continua usando as probabilidades do classificador sem ajuste de elenco, e a interface informa essa limitação.

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
- **Classificação multiclasse:** cada partida recebe uma das classes vitória do mandante, empate ou vitória do visitante. A simulação individual e a avaliação temporal usam Random Forest; o painel de 2026 permite comparar esse classificador com Regressão Logística e HistGradientBoosting.
- **Simulação do placar:** médias históricas de gols a favor e contra, suavizadas pela média da liga, alimentam distribuições de Poisson. O placar mostrado é compatível com o resultado mais provável.
- **Uso da escalação:** os percentis do elenco de 2024 ajustam as probabilidades e os gols esperados por uma regra explícita. Esse ajuste é heurístico, separado do classificador e não faz parte da avaliação temporal, pois o conjunto de partidas não contém escalações históricas comparáveis.

### Modelos comparados em 2026

Os três classificadores usam os mesmos atributos históricos — diferença de Elo, desempenho na temporada e forma recente — e são treinados com os resultados disponíveis até 2023. Todos usam pesos balanceados por classe (`class_weight="balanced"`) e semente fixa (`random_state=0`). Os parâmetros abaixo são os padrões do painel; podem ser alterados na interface.

- **Random Forest (`RandomForestClassifier`):** treina várias árvores de decisão em amostras aleatórias dos dados e combina suas previsões. Essa combinação captura relações não lineares e tende a reduzir a variância de uma árvore isolada. Padrão: 300 árvores, profundidade máxima 8 e mínimo de 15 exemplos por folha.
- **Regressão Logística (`LogisticRegression`):** aprende uma fronteira linear entre os atributos e as três classes, estimando a probabilidade de cada resultado. É um modelo mais simples e rápido, útil como referência; pode não representar relações complexas entre atributos. A regularização ajuda a limitar o ajuste excessivo. Padrão: `C=1` e `max_iter=1000`.
- **HistGradientBoosting (`HistGradientBoostingClassifier`):** constrói árvores pequenas em sequência, cada uma ajustando os erros que restam no conjunto anterior. Agrupar valores em histogramas torna o treinamento eficiente e as árvores modelam interações não lineares. Padrão: 150 iterações, taxa de aprendizado 0,1 e até 15 folhas por árvore.

Na comparação por rodada, os atributos de cada rodada são construídos antes de incorporar seus resultados; somente partidas encerradas em rodadas anteriores atualizam o histórico. As métricas usam apenas jogos com placar conhecido. O replay mede o desempenho retrospectivo no retrato da temporada, não uma previsão publicada antes do início do campeonato.

### Análise dos resultados da comparação 2026

No retrato atualizado em **25/11/2026**, foram avaliadas 360 partidas encerradas; outras 20 estavam pendentes e foram excluídas. A tabela ordena os modelos pela acurácia. A acurácia balanceada é a média do recall das classes, e o F1 macro dá o mesmo peso ao F1 de cada classe.

| Modelo | Acertos | Acurácia | Acurácia balanceada | F1 macro |
|---|---:|---:|---:|---:|
| Regressão Logística | 160/360 | 44,4% | 42,0% | 0,400 |
| Random Forest | 151/360 | 41,9% | 39,0% | 0,383 |
| HistGradientBoosting | 142/360 | 39,4% | 36,7% | 0,367 |

A Regressão Logística teve o melhor resultado entre os três modelos neste conjunto, tanto em acurácia quanto nas duas métricas que consideram o desempenho por classe. Porém, ela acertou menos que um baseline simples: escolher sempre vitória do mandante, classe mais frequente no histórico de treino, acertaria 175/360 jogos (**48,6%**). Portanto, nenhum classificador superou esse baseline; a diferença entre os modelos não deve ser interpretada como evidência de previsões confiáveis. Os resultados podem mudar com novos placares, outro retrato de dados ou parâmetros diferentes.

### Avaliação, resultados e limitações

`python app.py --eval` treina nas temporadas de 2007 a 2021 e testa nas duas temporadas mais recentes, 2022 e 2023. É uma separação temporal mais adequada que embaralhar partidas, pois representa a previsão de temporadas futuras. Dentro do teste, os resultados observados de uma rodada atualizam os atributos das rodadas seguintes, mas o classificador permanece fixo; isso simula previsões sequenciais durante a temporada. O relatório apresenta acurácia, acurácia balanceada, F1 macro, baseline majoritária e matriz de confusão. A escalação heurística não é incluída nessas métricas.

**Resultado medido neste conjunto:** no teste temporal de 760 partidas (2022–2023), o classificador obteve acurácia de **44,6%**, acurácia balanceada de **41,7%** e F1 macro de **0,410**. O baseline que sempre escolhe a classe mais frequente teve **45,5%** de acurácia. A matriz de confusão (real nas linhas; previsto nas colunas, na ordem mandante/empate/visitante) foi `[[199, 54, 93], [77, 45, 84], [68, 45, 95]]`. O modelo identifica as três classes, mas não supera o baseline em acurácia simples; portanto, seu desempenho é limitado e não deve ser apresentado como previsão confiável. `python app.py --eval` refaz a medição nos arquivos baixados.

Limitações importantes: os placares usados atualmente pela aplicação terminam em 2023; as fontes complementares de 2025 e a tabela da CBF de 2026 listadas acima ainda não foram integradas. As estatísticas de jogadores são da temporada de 2024. O ajuste de escalação é uma heurística, não foi validado por uma base histórica de escalações, e não considera lesões, suspensões, mudanças de técnico, tática ou transferências posteriores. As probabilidades do classificador não foram calibradas e o ajuste da escalação é heurístico; interprete-os como estimativas, não como chances confiáveis ou garantias.

## Instalação e execução

Requer Python 3.10 ou superior e conexão à internet na primeira execução.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python web.py
```

Depois de iniciar o servidor, abra **http://127.0.0.1:8000** no navegador, selecione os clubes e clique em **Simular partida**. Na primeira execução, os dois datasets integrados do Kaggle são baixados automaticamente; isso requer conexão com a internet. Encerre o servidor com `Ctrl+C`.

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
- `web/` — interface web responsiva para simulações, jogos de 2026 e comparação entre previsões e resultados.
- `service.py` — valida a entrada e conecta dados, modelo e interface.
- `repository.py` — baixa, valida e normaliza os CSVs do Kaggle.
- `model.py` — atributos históricos, classificadores, avaliação temporal, replay de temporada, escalações e simulação.
- `tests/` — testes focados de histórico e comparação por rodada, escalação e API web.

## Desenvolvimento

Para compartilhar o desenvolvimento, crie uma branch para cada melhoria, rode os testes e a avaliação, e abra um pull request com a alteração e os resultados observados. Não inclua `data/raw/` nos commits.
