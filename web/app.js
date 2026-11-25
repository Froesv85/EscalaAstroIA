const LABELS = {
  "home win": (home) => `Vitória de ${home}`,
  draw: () => "Empate",
  "away win": (_, away) => `Vitória de ${away}`,
};

const ROLE_ORDER = ["Goleiro", "Defesa", "Meio-campo", "Ataque"];

const form = document.querySelector("#simulation-form");
const homeSelect = document.querySelector("#home-team");
const awaySelect = document.querySelector("#away-team");
const swapButton = document.querySelector("#swap-teams");
const simulateButton = document.querySelector("#simulate-button");
const statusMessage = document.querySelector("#status-message");
const resultSection = document.querySelector("#result-section");
const roundButtons = document.querySelectorAll(".round-button");
const roundMatchList = document.querySelector("#round-match-list");
const roundsStatus = document.querySelector("#rounds-status");
const comparisonRoundSelect = document.querySelector("#comparison-round");
const comparisonMatches = document.querySelector("#comparison-matches");
const comparisonStatus = document.querySelector("#comparison-status");
const comparisonForm = document.querySelector("#comparison-configuration");
const comparisonModelOptions = document.querySelector("#comparison-model-options");
const comparisonModelResults = document.querySelector("#comparison-model-results");
const comparisonRunStatus = document.querySelector("#comparison-run-status");
const runComparisonButton = document.querySelector("#run-comparison");
let roundFixtures = null;
let seasonComparison = null;

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function showStatus(message, state = "") {
  statusMessage.textContent = message;
  statusMessage.dataset.state = state;
}

function setFormEnabled(enabled) {
  homeSelect.disabled = !enabled;
  awaySelect.disabled = !enabled;
  swapButton.disabled = !enabled;
  simulateButton.disabled = !enabled;
}

function setTeams(select, teams, placeholder) {
  select.replaceChildren(new Option(placeholder, ""));
  for (const team of teams) {
    select.add(new Option(team, team));
  }
}

async function loadTeams() {
  try {
    const response = await fetch("/api/teams");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Não foi possível carregar os clubes.");
    }

    if (!Array.isArray(payload.teams) || payload.teams.length < 2) {
      throw new Error("A lista de clubes não tem dados suficientes para simular partidas.");
    }

    setTeams(homeSelect, payload.teams, "Escolha um clube");
    setTeams(awaySelect, payload.teams, "Escolha um clube");
    homeSelect.value = payload.teams.find((team) => team === "Flamengo") || payload.teams[0];
    awaySelect.value = payload.teams.find((team) => team === "Palmeiras")
      || payload.teams.find((team) => team !== homeSelect.value);
    setFormEnabled(true);
    showStatus(
      `Histórico de partidas: ${payload.match_seasons} · estatísticas de jogadores: ${payload.player_season}.`,
    );
  } catch (error) {
    showStatus(
      `${error.message} Confira se o servidor Python está executando em python web.py.`,
      "error",
    );
  }
}

function renderScore(result) {
  const display = document.querySelector("#score-display");
  display.replaceChildren();

  const homeSide = createElement("div", "score-side");
  homeSide.append(
    createElement("div", "score-team", result.home),
    createElement("div", "score-team-caption", "MANDANTE"),
  );
  const numbers = createElement("div", "score-numbers");
  numbers.append(
    createElement("span", "", String(result.score.home_goals)),
    createElement("span", "score-divider", "—"),
    createElement("span", "", String(result.score.away_goals)),
  );
  const awaySide = createElement("div", "score-side");
  awaySide.append(
    createElement("div", "score-team", result.away),
    createElement("div", "score-team-caption", "VISITANTE"),
  );
  display.append(homeSide, numbers, awaySide);

  const predictedLabel = LABELS[result.predicted]?.(result.home, result.away);
  document.querySelector("#score-summary").textContent =
    `Resultado mais provável: ${predictedLabel}.`;

  document.querySelector("#expected-goals").textContent =
    `GOLS ESPERADOS ${result.score.expected.home.toFixed(2)} — ${result.score.expected.away.toFixed(2)}`;
}

function renderProbabilities(result) {
  document.querySelector(".unadjusted-note").textContent =
    result.lineup_adjustment_applied
      ? "Ajustadas pelo índice das escalações · não calibradas"
      : "Sem ajuste de escalações · não calibradas";
  const container = document.querySelector("#probability-list");
  container.replaceChildren();

  const outcomes = [
    { key: "home win", label: `Vitória de ${result.home}` },
    { key: "draw", label: "Empate" },
    { key: "away win", label: `Vitória de ${result.away}` },
  ];

  for (const outcome of outcomes) {
    const value = result.probabilities[outcome.key];
    const card = createElement("article", "probability-card");
    card.dataset.outcome = outcome.key;
    card.dataset.result = outcome.key === result.predicted ? "predicted" : "";

    const header = createElement("div", "probability-header");
    header.append(createElement("span", "probability-label", outcome.label));
    const percentage = createElement("span", "probability-value", `${(value * 100).toFixed(1)}%`);

    const track = createElement("div", "probability-track");
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", `${outcome.label}: ${(value * 100).toFixed(1)} por cento`);
    const fill = createElement("div", "probability-fill");
    fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    track.append(fill);

    const footer = createElement("div", "probability-values");
    footer.append(
      createElement(
        "span",
        "favorite-label",
        outcome.key === result.predicted ? "MAIS PROVÁVEL" : "",
      ),
      createElement("span", "", "Estimativa do modelo"),
    );
    card.append(header, percentage, track, footer);
    container.append(card);
  }
}

function renderLineup(team, lineup, strength) {
  const card = createElement("article", "lineup-card");
  const header = createElement("header", "lineup-card-header");
  header.append(
    createElement("h4", "lineup-team-name", team),
  );
  if (!lineup.length || strength === null) {
    card.append(
      header,
      createElement(
        "p",
        "lineup-unavailable",
        "Sem escalação: não há estatísticas de jogadores 2024 para este clube.",
      ),
    );
    return card;
  }

  const rating = createElement("div", "lineup-rating");
  rating.append(
    createElement("span", "", "ÍNDICE DO ELENCO"),
    createElement("strong", "", strength.toFixed(3)),
  );
  header.append(rating);

  const groups = createElement("div", "lineup-groups");
  for (const role of ROLE_ORDER) {
    const players = lineup.filter((player) => player.role === role);
    if (players.length === 0) continue;

    const group = createElement("section", "lineup-group");
    group.append(createElement("h5", "lineup-role", role));
    const playerList = createElement("div", "lineup-player-list");
    players.forEach((player, index) => {
      const row = createElement("div", "lineup-player");
      row.append(
        createElement("span", "player-number", String(index + 1).padStart(2, "0")),
      );
      const nameBlock = createElement("div");
      nameBlock.append(
        createElement("div", "player-name", player.name),
        createElement("div", "player-position", player.position),
      );
      row.append(nameBlock);
      row.append(createElement("span", "player-minutes", `${player.minutes} min`));
      playerList.append(row);
    });
    group.append(playerList);
    groups.append(group);
  }

  card.append(header, groups);
  return card;
}

function renderLineups(result) {
  const container = document.querySelector("#lineup-grid");
  container.replaceChildren(
    renderLineup(
      result.home,
      result.lineups.home,
      result.lineup_strength.home,
    ),
    renderLineup(
      result.away,
      result.lineups.away,
      result.lineup_strength.away,
    ),
  );
  document.querySelector(".lineup-disclaimer").textContent =
    "Sugestões baseadas nas estatísticas de jogadores de 2024. Quando faltam dados de um dos clubes, o modelo não aplica ajuste de escalação às probabilidades.";
}

function formatKickoff(value) {
  return new Intl.DateTimeFormat("pt-BR", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "America/Sao_Paulo",
  }).format(new Date(value));
}

function renderRound(round) {
  roundMatchList.replaceChildren();

  for (const match of round.matches) {
    const finished = match.status === "finished";
    const card = createElement("article", "fixture-card");
    const meta = createElement("div", "fixture-meta");
    meta.append(
      createElement(
        "span",
        `fixture-status${finished ? " is-finished" : ""}`,
        finished ? "Encerrado" : "Agendado",
      ),
    );
    if (match.kickoff) {
      const kickoff = createElement("time", "fixture-time", formatKickoff(match.kickoff));
      kickoff.dateTime = match.kickoff;
      meta.append(kickoff);
    }

    const home = createElement("span", "fixture-team", match.home);
    const scoreText = match.score
      ? `${match.score.home} — ${match.score.away}`
      : "×";
    const score = createElement("strong", "fixture-score", scoreText);
    const away = createElement("span", "fixture-team", match.away);
    card.append(meta, home, score, away);
    roundMatchList.append(card);
  }

  const completed = round.matches.filter((match) => match.status === "finished").length;
  const scheduled = round.matches.length - completed;
  const completedLabel = `${completed} jogo${completed === 1 ? "" : "s"} encerrado${completed === 1 ? "" : "s"}`;
  const scheduledLabel = `${scheduled} jogo${scheduled === 1 ? "" : "s"} agendado${scheduled === 1 ? "" : "s"}`;
  roundsStatus.textContent = scheduled
    ? `${round.matches.length} jogos · ${completedLabel} · ${scheduledLabel}`
    : `${round.matches.length} jogos encerrados`;

  for (const button of roundButtons) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.round) === round.number));
  }
}

function renderSources(sources, containerId = "rounds-sources") {
  const container = document.querySelector(`#${containerId}`);
  container.replaceChildren(createElement("span", "", "Fontes:"));
  sources.forEach((source, index) => {
    if (index > 0) container.append(document.createTextNode(" · "));
    const link = createElement("a", "", source.label);
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    container.append(link);
  });
}

function formatOutcome(outcome, home, away) {
  if (outcome === "home win") return `Vitória de ${home}`;
  if (outcome === "away win") return `Vitória de ${away}`;
  return "Empate";
}

function formatAccuracy(value) {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function renderComparison() {
  if (!seasonComparison) return;

  const { summary, models } = seasonComparison;
  document.querySelector("#comparison-total").textContent =
    String(summary.total_matches);
  document.querySelector("#comparison-completed").textContent =
    String(summary.completed_matches);
  document.querySelector("#comparison-pending").textContent =
    String(summary.pending_matches);

  const modelEntries = Object.entries(models);
  const referenceRounds = modelEntries[0]?.[1].rounds || [];
  const selectedRound = comparisonRoundSelect.value;
  comparisonMatches.replaceChildren();
  const tableHeader = document.querySelector(".comparison-table thead tr");
  tableHeader.replaceChildren(
    createElement("th", "", "Rodada"),
    createElement("th", "", "Confronto"),
  );
  for (const [, model] of modelEntries) {
    tableHeader.append(createElement("th", "", model.label));
  }
  tableHeader.append(createElement("th", "", "Realizado"));

  const selectedRoundData = referenceRounds.find(
    (round) => String(round.number) === selectedRound,
  );
  comparisonStatus.removeAttribute("data-state");
  if (selectedRoundData) {
    const pending = selectedRoundData.total_matches - selectedRoundData.completed_matches;
    const modelRoundSummaries = modelEntries.map(([modelId, model]) => {
      const round = model.rounds.find((item) => item.number === selectedRoundData.number);
      return `${model.label}: ${round.metrics.correct_predictions}/${round.metrics.samples}`;
    });
    comparisonStatus.textContent =
      `Rodada ${selectedRoundData.number} · ${modelRoundSummaries.join(" · ")} · ` +
      `${pending} pendentes.`;
  } else {
    comparisonStatus.textContent =
      `${summary.completed_matches} jogos avaliados · ${summary.pending_matches} pendentes.`;
  }

  for (const round of referenceRounds) {
    if (selectedRound !== "all" && Number(selectedRound) !== round.number) continue;

    const roundsByModel = modelEntries.map(([modelId, model]) => [
      modelId,
      model.rounds.find((item) => item.number === round.number),
    ]);
    for (let index = 0; index < round.matches.length; index += 1) {
      const match = round.matches[index];
      const row = document.createElement("tr");
      row.append(
        createElement("td", "comparison-round-number", String(round.number)),
        createElement("td", "comparison-teams", `${match.home} × ${match.away}`),
      );
      for (const [modelId, modelRound] of roundsByModel) {
        const modelMatch = modelRound.matches[index];
        const predictionCell = createElement("td", "comparison-prediction");
        predictionCell.append(
          createElement(
            "span",
            "",
            `${formatOutcome(modelMatch.predicted, modelMatch.home, modelMatch.away)} · ` +
              formatAccuracy(modelMatch.confidence),
          ),
        );
        const evaluation = modelMatch.status === "finished"
          ? modelMatch.correct ? "Acertou" : "Errou"
          : "Pendente";
        predictionCell.append(
          createElement(
            "small",
            `comparison-result${modelMatch.correct === true ? " is-correct" : ""}${modelMatch.correct === false ? " is-incorrect" : ""}`,
            evaluation,
          ),
        );
        predictionCell.dataset.model = modelId;
        row.append(predictionCell);
      }

      const actual = match.status === "finished"
        ? `${match.score.home} — ${match.score.away} · ${formatOutcome(
            match.actual,
            match.home,
            match.away,
          )}`
        : "Aguardando resultado";
      row.append(createElement("td", "comparison-actual", actual));
      comparisonMatches.append(row);
    }
  }
}

function renderComparisonControls(options, models) {
  comparisonModelOptions.replaceChildren();
  for (const [modelId, option] of Object.entries(options)) {
    const card = createElement("section", "comparison-model-option");
    const heading = createElement("div", "comparison-model-option-heading");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.dataset.model = modelId;
    const title = createElement("span", "", option.label);
    heading.append(checkbox, title);
    card.append(
      heading,
      createElement("p", "comparison-model-description", option.description),
    );

    const modelParameters = models[modelId]?.parameters || {};
    const parameterGrid = createElement("div", "comparison-parameter-grid");
    for (const [name, parameter] of Object.entries(option.parameters)) {
      const label = createElement("label", "comparison-parameter");
      label.append(createElement("span", "", parameter.label));
      const input = document.createElement("input");
      input.type = "number";
      input.min = String(parameter.min);
      input.max = String(parameter.max);
      input.step = String(parameter.step);
      input.value = String(modelParameters[name] ?? parameter.default);
      input.required = true;
      input.dataset.model = modelId;
      input.dataset.parameter = name;
      label.append(input);
      parameterGrid.append(label);
    }
    card.append(parameterGrid);
    card.append(
      createElement(
        "p",
        "comparison-fixed-parameters",
        `Fixos: ${Object.entries(option.fixed_parameters)
          .map(([name, value]) => `${name}=${value}`)
          .join(" · ")}`,
      ),
    );

    checkbox.addEventListener("change", () => {
      for (const input of parameterGrid.querySelectorAll("input")) {
        input.disabled = !checkbox.checked;
      }
    });
    comparisonModelOptions.append(card);
  }
}

function renderComparisonResults() {
  comparisonModelResults.replaceChildren();
  for (const [modelId, model] of Object.entries(seasonComparison.models)) {
    const card = createElement("article", "comparison-model-card");
    card.dataset.model = modelId;
    card.append(createElement("h3", "", model.label));
    const metrics = createElement("div", "comparison-model-metrics");
    for (const [label, value] of [
      ["Acurácia", formatAccuracy(model.summary.accuracy)],
      ["Acurácia balanceada", formatAccuracy(model.summary.balanced_accuracy)],
      ["F1 macro", model.summary.macro_f1 === null ? "—" : model.summary.macro_f1.toFixed(3)],
    ]) {
      const metric = createElement("div", "comparison-model-metric");
      metric.append(
        createElement("span", "", label),
        createElement("strong", "", value),
      );
      metrics.append(metric);
    }
    card.append(metrics);
    card.append(
      createElement(
        "p",
        "comparison-parameters-used",
        `Parâmetros: ${Object.entries(model.parameters)
          .map(([name, value]) => `${name}=${value}`)
          .join(" · ")}`,
      ),
    );
    comparisonModelResults.append(card);
  }
}

function collectComparisonConfigurations() {
  const configurations = {};
  for (const checkbox of comparisonModelOptions.querySelectorAll(
    'input[type="checkbox"][data-model]',
  )) {
    if (!checkbox.checked) continue;
    const modelId = checkbox.dataset.model;
    configurations[modelId] = {};
    const inputs = comparisonModelOptions.querySelectorAll(
      `input[type="number"][data-model="${modelId}"]`,
    );
    for (const input of inputs) {
      configurations[modelId][input.dataset.parameter] =
        input.step.includes(".") ? Number(input.value) : Number.parseInt(input.value, 10);
    }
  }
  return configurations;
}

async function loadRoundFixtures() {
  try {
    const response = await fetch("/api/fixtures/2026");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Não foi possível carregar os jogos.");
    }
    if (!Array.isArray(payload.rounds) || !Array.isArray(payload.sources)) {
      throw new Error("A lista de jogos está em um formato inválido.");
    }

    roundFixtures = payload;
    const selectedRound = payload.rounds.find((round) => round.number === 37)
      || payload.rounds[0];
    if (!selectedRound || !Array.isArray(selectedRound.matches)) {
      throw new Error("Não há jogos cadastrados para as rodadas 35 a 38.");
    }

    document.querySelector("#rounds-snapshot").textContent =
      `DADOS DE ${formatKickoff(payload.updatedAt)}`;
    document.querySelector("#rounds-description").textContent = payload.note;
    renderSources(payload.sources);
    renderRound(selectedRound);
  } catch (error) {
    roundsStatus.textContent = error.message;
    roundsStatus.dataset.state = "error";
  }
}

async function loadSeasonComparison(configurations) {
  runComparisonButton.disabled = true;
  comparisonRunStatus.removeAttribute("data-state");
  comparisonRunStatus.textContent = "Treinando e comparando os modelos…";
  try {
    const request = configurations
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ models: configurations }),
        }
      : {};
    const response = await fetch("/api/comparison/2026", request);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "Não foi possível gerar o comparativo.");
    }
    if (!payload.models || !payload.model_options || !payload.summary) {
      throw new Error("O comparativo recebido está em um formato inválido.");
    }

    seasonComparison = payload;
    if (!configurations) {
      renderComparisonControls(payload.model_options, payload.models);
      const referenceRounds = Object.values(payload.models)[0]?.rounds || [];
      for (const round of referenceRounds) {
        comparisonRoundSelect.add(new Option(`Rodada ${round.number}`, String(round.number)));
      }
    }
    const snapshotDate = new Intl.DateTimeFormat("pt-BR", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      timeZone: "America/Sao_Paulo",
    }).format(new Date(payload.updatedAt));
    document.querySelector("#comparison-snapshot").textContent =
      `DADOS DE ${snapshotDate}`;
    document.querySelector("#comparison-description").textContent = payload.note;
    renderSources(payload.sources, "comparison-sources");
    renderComparisonResults();
    renderComparison();
    comparisonRunStatus.textContent =
      `Comparação concluída com ${Object.keys(payload.models).length} modelo(s).`;
  } catch (error) {
    comparisonRunStatus.textContent = error.message;
    comparisonRunStatus.dataset.state = "error";
  } finally {
    runComparisonButton.disabled = false;
  }
}

comparisonRoundSelect.addEventListener("change", renderComparison);

comparisonForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!comparisonForm.reportValidity()) return;

  const configurations = collectComparisonConfigurations();
  if (Object.keys(configurations).length === 0) {
    comparisonRunStatus.textContent = "Selecione ao menos um modelo para comparar.";
    comparisonRunStatus.dataset.state = "error";
    return;
  }
  loadSeasonComparison(configurations);
});

roundButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!roundFixtures) return;

    const roundNumber = Number(button.dataset.round);
    const round = roundFixtures.rounds.find((item) => item.number === roundNumber);
    if (!round) {
      roundsStatus.textContent = `A rodada ${roundNumber} não está disponível.`;
      roundsStatus.dataset.state = "error";
      return;
    }
    roundsStatus.removeAttribute("data-state");
    renderRound(round);
  });
});

function renderResult(result) {
  renderScore(result);
  renderProbabilities(result);
  renderLineups(result);
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

swapButton.addEventListener("click", () => {
  const previousHome = homeSelect.value;
  homeSelect.value = awaySelect.value;
  awaySelect.value = previousHome;
  resultSection.hidden = true;
});

homeSelect.addEventListener("change", () => {
  resultSection.hidden = true;
});

awaySelect.addEventListener("change", () => {
  resultSection.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  if (homeSelect.value === awaySelect.value) {
    showStatus("Escolha dois clubes diferentes para simular o confronto.", "error");
    return;
  }

  setFormEnabled(false);
  showStatus("Analisando histórico, escalações e probabilidades…", "loading");

  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ home: homeSelect.value, away: awaySelect.value }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      const suggestions = Object.values(payload.suggestions || {}).flat();
      const suggestionMessage = suggestions.length
        ? ` Talvez você quis dizer: ${suggestions.join(", ")}.`
        : "";
      throw new Error(`${payload.message || "Não foi possível simular o confronto."}${suggestionMessage}`);
    }

    renderResult(payload);
    showStatus("Simulação pronta. Altere os clubes para experimentar outro confronto.");
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    setFormEnabled(true);
  }
});

loadTeams();
loadRoundFixtures();
loadSeasonComparison();
