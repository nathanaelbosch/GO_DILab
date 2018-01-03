competition_type = 'playoff'

record_games = True
stderr_to_log = True


players = {
    'gnugo' : Player("gnugo --mode=gtp"),
    'random' : Player("python -m src.play.controller.GTPengine -p RandomBot"),
    'win_pred' : Player("python -m src.play.controller.GTPengine -p WinPredictionBot"),
    'Bot_11' : Player("python -m src.play.controller.GTPengine -p Bot_11"),
    'Bot_12' : Player("python -m src.play.controller.GTPengine -p Bot_12"),
    'Bot_21' : Player("python -m src.play.controller.GTPengine -p Bot_21"),
    'Bot_22' : Player("python -m src.play.controller.GTPengine -p Bot_22"),
    'Bot_31' : Player("python -m s1rc.play.controller.GTPengine -p Bot_31"),
    'Bot_32' : Player("python -m src.play.controller.GTPengine -p Bot_32"),
    }

board_size = 9
komi = 7

matchups = [
    Matchup('random', 'Bot_11', scorer='internal', alternating=True,number_of_games=200),
    Matchup('random', 'Bot_12', scorer='internal', alternating=True,number_of_games=200),
    Matchup('random', 'Bot_21', scorer='internal', alternating=True,number_of_games=200),
    Matchup('random', 'Bot_22', scorer='internal', alternating=True,number_of_games=200),
    Matchup('random', 'Bot_31', scorer='internal', alternating=True,number_of_games=200),
    Matchup('random', 'Bot_32', scorer='internal', alternating=True,number_of_games=200),
    Matchup('random', 'win_pred', scorer='internal', alternating=True, number_of_games=200),
    ]
 