from paramiko import SSHClient, AutoAddPolicy


def read(std):
    while True:
        line = std.readline()
        if line:
            print(line.rstrip())
        else:
            break


ssh = SSHClient()
ssh.set_missing_host_key_policy(AutoAddPolicy())

ssh.connect('10.155.208.218', username='player', password='')

command = '''
    SELECT COUNT(*)
    FROM meta
    WHERE meta.all_moves_imported IS NOT 0
    AND meta.elo_black IS NOT ""
    AND meta.elo_black > 2000
    AND meta.elo_white IS NOT ""
    AND meta.elo_white > 2000
    AND meta.turns > 30
'''

stdin, stdout, stderr = ssh.exec_command("sqlite3 db.sqlite '" + command + "'")

read(stderr)
read(stdout)
