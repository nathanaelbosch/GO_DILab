:: activate the python virtual environment
call .\.venv\Scripts\activate
:: build the .exe into a single file
pyinstaller --onefile .\src\play\controller\GTPengine.py
:: move the .exe from dist/ to toplevel
move .\dist\GTPengine.exe GTPengine.exe
:: remove file and folders that pyinstaller created
del GTPengine.spec
rd /S /Q dist
rd /S /Q build
:: remove (large) cache-files that pyinstaller created
rd /S /Q %APPDATA%\pyinstaller
pause
