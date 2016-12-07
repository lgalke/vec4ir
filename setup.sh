VENV_PATH=$( cd $(dirname "$0"); pwd -P)/venv
command -v virtualenv >/dev/null 2>&1 || { echo >&2 "Required package virtualenv is not installed.  Aborting."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo >&2 "Required package python3 is not installed.  Aborting."; exit 1; }
if [ ! -e $VENV_PATH ]; then
    virtualenv -p python3 $VENV_PATH
fi
echo $VENV_PATH
source $VENV_PATH/bin/activate && python setup.py develop
deactivate 2>/dev/null
