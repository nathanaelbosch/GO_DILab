competition_type = 'playoff'

record_games = True
stderr_to_log = True

players = {
    # 'gnugo' : Player("gnugo --mode=gtp"),
    'bot1' : Player("python ../play/controller/GTPengine.py -p RandomBot"),
    'bot2' : Player("python ../play/controller/GTPengine.py -p WinPredictionBot"),
    }

board_size = 9
komi = 6

matchups = [
    Matchup('bot1', 'bot2', scorer='players', number_of_games=1),
    ]
