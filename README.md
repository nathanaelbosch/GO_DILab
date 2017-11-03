# A machine learning how to play GO

### :game_die: TUM Data Innovation Lab WS2017/18, Team Hikaru

## Setup
- We use **pygame** for the GUI. Because of thread-issues (known and unresolved according to some forums online) this doesn't work properly when run from a virtual python environment (at least for the an anaconda environment that's the case), so it must be a normal installation.
- Install _Python 3.6_ if not already present
- Run `pip install -r requirements.txt` (or `pip3` on macOS sometimes) or let your favorite IDE help you with installing the required modules. If a module fails to install via the IDE, try to install it via `pip`/`pip3`.

## Usage
- In `run.py` you can configure the type of GUI and the type of players.
- run `src/play/run.py` either via a run configuration in your IDE or via command line (the location from which you run doesn't matter, the path gets set correctly by the script).

![](https://user-images.githubusercontent.com/5141792/32337814-4fe3d76c-bff3-11e7-9d66-ddaa1e2f4faa.png)
