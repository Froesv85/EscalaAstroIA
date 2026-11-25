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
}

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
