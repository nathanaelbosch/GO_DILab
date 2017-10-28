import sgf
from src.play.model.Game import Game
import os
import re


rootdir = 'data/unpacked'
filepaths = []
for root, subdirs, files in os.walk(rootdir):
    filepaths += [os.path.join(root, f) for f in files]

for file in filepaths:
    # file = 'data/unpacked/2017/09/08/338554.sgf'
    print(file)
    filename = re.findall('/([^/]*?).sgf', file)[0]
    with open(file, 'r') as f:
        content = f.read()
        collection = sgf.parse(content)

    # Assume the sgf file contains one game
    game_tree = collection.children[0]
    n_0 = game_tree.nodes[0]
    # n_0.properties contains the initial game setup
    game_id = n_0.properties['GN'][0]
    myfilepath = 'data/myformat/'+game_id+'.csv'
    if os.path.isfile(myfilepath):
        os.remove(myfilepath)

    engine = Game(n_0.properties, show_each_turn=True)
    for n in game_tree.nodes[1:]:
        props = n.properties
        # print(props)
        if 'W' in props.keys():
            engine.w(props['W'][0])
        if 'B' in props.keys():
            engine.b(props['B'][0])
        # print(engine)
        engine.board2file(myfilepath, 'a')
