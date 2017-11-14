import subprocess
import time

# or sys.executable instead of 'python'
player1 = subprocess.Popen(['python', 'GTPplayer.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
player2 = subprocess.Popen(['python', 'GTPplayer.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

current_player = player1
other_player = player2


def get_player_id(player):
    return 'black' if player == player1 else 'white'


def await_response(player):
    _response = ''
    while len(_response) == 0:
        _response = player.stdout.readline().decode('utf-8').strip()
        player.stdout.flush()
    print('[response from ' + get_player_id(player) + ']\t' + _response)
    return _response


def send_command(player, command):
    print('[command to ' + get_player_id(player) + ']\t\t' + command)
    b = bytearray()
    b.extend(map(ord, command + '\n'))
    player.stdin.write(b)
    player.stdin.flush()


for i in range(5):  # while until someone sends quit? doesn't happen for random bots though TODO
    print('\nnext turn\n')
    color = 'b' if current_player == player1 else 'w'

    send_command(current_player, 'genmove ' + color)
    response = await_response(current_player)
    move = response[2:]  # strip away the "= "
    send_command(other_player, 'play ' + color + ' ' + move)
    await_response(other_player)

    # keep track of game on own board to validate moves TODO

    time.sleep(0.5)

    # swap players for next turn
    if current_player == player1:
        current_player = player2
        other_player = player1
    else:
        current_player = player1
        other_player = player2
