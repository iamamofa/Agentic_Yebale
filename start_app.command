#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo 'Python environment missing. Follow the installation steps in docs/app/README.md.'
  read -r -p 'Press Enter to close.'
  exit 1
fi
.venv/bin/python start_app.py "$@"
launch_status=$?
if [ "$launch_status" -ne 0 ]; then
  read -r -p 'Press Enter to close.'
fi
exit "$launch_status"
