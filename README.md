# A machine learning how to play GO

### :game_die: TUM Data Innovation Lab WS2017/18, Team Hikaru

## Setup
- install _Python 3.6_ if not already present
- create a virtual environment in the repository: `python -m venv .venv` (or `python3`, check `python --version`)
- activate it: `source activate .venv` on macOS, `source .venv/bin/activate` on Linux or `.\.venv\Scripts\activate` on Windows
- run `pip install -r requirements.txt` or let your IDE help you with installing the required modules

## Usage
To start up a controller with two engines run `src/play/run.py` either via a run configuration in your IDE or via command line. In `run.py` you can configure the type of players.

To start just one engine, run `src/play/controller/GTPengine.py` and use GTP commands to communicate with it.

To build an executable engine (to attach it to a Go GUI or run it by double-clicking), install _pyinstaller_ (`pip install pyinstaller`) and run `pyinstaller --onefile src/play/controller/GTPengine.py`. On Windows this will create an `.exe` in a new `dist/` folder.
