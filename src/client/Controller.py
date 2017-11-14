import subprocess
import sys


player1 = subprocess.Popen(
    [sys.executable, 'GTPplayer.py'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

player2 = subprocess.Popen(
    [sys.executable, 'GTPplayer.py'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

current_player = player1
other_player = player2


def to_byte(_str):
    b = bytearray()
    b.extend(map(ord, _str))
    return b

for i in range(1):
    color = 'b' if current_player == player1 else 'w'

    current_player.stdin.write(to_byte('genmove ' + color + '\n'))
    response = current_player.stdout.readline().decode('utf-8').strip()
    print(response)
    move = response[2:]  # strip away the "= "

    other_player.stdin.write(to_byte('play b ' + move + '\n'))
    response = other_player.stdout.readline().decode('utf-8').strip()
    print(response)

    # swap players for next turn
    if current_player == player1:
        current_player = player2
        other_player = player1
    else:
        current_player = player1
        other_player = player2
