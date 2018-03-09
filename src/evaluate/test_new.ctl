competition_type = 'playoff'

record_games = True
stderr_to_log = True


players = {
    'gnugo_l1' : Player("gnugo --mode=gtp --level=1 --chinese-rules --capture-all-dead"),
    'random' : Player("python -m src.play.controller.GTPengine -p RandomBot"),
    'current_best' : Player("python -m src.play.controller.GTPengine -p ConvBot_policy"),
    'new_bot' : Player("python -m src.play.controller.GTPengine -p NewBot"),
    }

board_size = 9
komi = 7

matchups = [
    Matchup('gnugo_l1', 'current_best', scorer='internal', alternating=True,number_of_games=200),
    Matchup('gnugo_l1', 'new_bot', scorer='internal', alternating=True,number_of_games=200),
#    Matchup('random', 'current_best', scorer='internal', alternating=True,number_of_games=200),
#    Matchup('random', 'new_bot', scorer='internal', alternating=True,number_of_games=200),
]
