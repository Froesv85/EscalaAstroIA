import argparse
import sys

from service import Service


LABEL_PT = {
    "home win": "Vitória do mandante",
    "draw": "Empate",
    "away win": "Vitória do visitante",
}
ROLE_ORDER = ("Goleiro", "Defesa", "Meio-campo", "Ataque")


def _print_lineup(team, lineup, strength):
    print(f"\nEscalação recomendada de {team} (4-3-3; índice do elenco: {strength:.3f})")
    for role in ROLE_ORDER:
        players = [player for player in lineup if player["role"] == role]
        names = ", ".join(
            f"{player['name']} ({player['position']}, {player['minutes']:.0f} min)"
            for player in players
        )
        print(f"  {role}: {names}")


def _print_prediction(result):
    print(f"\n{result['home']} x {result['away']}")
    print(f"Resultado mais provável: {LABEL_PT[result['predicted']]}")
    print(
        "Probabilidades estimadas (ajustadas pelo índice da escalação; "
        "não calibradas):"
    )
    print(f"  Vitória do mandante: {result['probabilities']['home win']:.1%}")
    print(f"  Empate:              {result['probabilities']['draw']:.1%}")
    print(f"  Vitória do visitante:{result['probabilities']['away win']:.1%}")
    score = result["score"]
    print(
        f"Placar simulado: {result['home']} {score['home_goals']} x "
        f"{score['away_goals']} {result['away']}"
    )
    print(
        "Gols esperados (estimativa): "
        f"{score['expected']['home']:.2f} x {score['expected']['away']:.2f}"
    )
    _print_lineup(
        result["home"],
        result["lineups"]["home"],
        result["lineup_strength"]["home"],
    )
    _print_lineup(
        result["away"],
        result["lineups"]["away"],
        result["lineup_strength"]["away"],
    )
    print()


def _print_evaluation(metrics, quality):
    print("Avaliação temporal do classificador (sem ajuste heurístico das escalações)")
    print(f"Treino: temporadas {metrics['train_seasons'][0]}–{metrics['train_seasons'][-1]}")
    print(f"Teste: temporadas {metrics['test_seasons'][0]}–{metrics['test_seasons'][-1]}")
    print(f"Partidas avaliadas: {metrics['samples']}")
    print(f"Acurácia:           {metrics['accuracy']:.1%}")
    print(f"Acurácia balanceada:{metrics['balanced_accuracy']:.1%}")
    print(f"F1 macro:           {metrics['macro_f1']:.3f}")
    print(f"Baseline majoritária: {metrics['baseline_accuracy']:.1%}")
    print("\nMatriz de confusão (linhas = real; colunas = previsto):")
    print("                   Mandante  Empate  Visitante")
    labels = metrics["labels"]
    row_names = {
        "home win": "Mandante",
        "draw": "Empate",
        "away win": "Visitante",
    }
    for label, row in zip(labels, metrics["confusion_matrix"]):
        print(f"{row_names[label]:>12}       {row[0]:>4}    {row[1]:>4}    {row[2]:>4}")
    print(
        "\nPartidas válidas: {matches_used}/{matches_total} "
        "(removidas por placar ausente: {matches_missing_score}); "
        "linhas de jogadores: {players_used}".format(**quality)
    )


def _resolve_name(service, name):
    if service.is_known(name):
        return name
    suggestions = service.suggest(name)
    if suggestions:
        print(f"Clube não encontrado. Sugestões: {', '.join(suggestions)}")
    else:
        print("Clube não encontrado. Consulte a lista de clubes disponíveis.")
    return None


def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def run_cli(service):
    print("EscalaAstroIA — previsão de partidas do Campeonato Brasileiro")
    print("Resultados históricos até 2023; estatísticas de jogadores da temporada 2024.")
    print("Clubes disponíveis: " + ", ".join(service.list_team_names()))
    print("Digite um clube mandante e um visitante. Deixe vazio para sair.\n")

    while True:
        home = _ask("Mandante: ")
        if not home:
            break
        home = _resolve_name(service, home)
        if home is None:
            continue

        away = _ask("Visitante: ")
        if not away:
            break
        away = _resolve_name(service, away)
        if away is None:
            continue

        result = service.predict(home, away)
        if not result["ok"]:
            print("Escolha dois clubes diferentes.\n")
            continue
        _print_prediction(result)

    print("Até logo.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Simula confrontos do Brasileirão e recomenda escalações."
    )
    parser.add_argument(
        "--eval",
        "--evaluate",
        action="store_true",
        help="avalia o classificador com as duas temporadas mais recentes",
    )
    parser.add_argument(
        "--list-teams",
        action="store_true",
        help="lista os clubes disponíveis e encerra",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="baixa novamente os arquivos públicos do Kaggle",
    )
    args = parser.parse_args(argv)

    try:
        service = Service(refresh_data=args.refresh_data)
        if args.list_teams:
            print("\n".join(service.list_team_names()))
            return 0
        if args.eval:
            print("Preparando a avaliação temporal...")
            _print_evaluation(service.evaluate(), service.data_quality())
            return 0

        service.setup()
        run_cli(service)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Erro ao preparar a simulação: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
