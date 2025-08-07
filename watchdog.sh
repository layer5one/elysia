#!/bin/bash
# watchdog.sh — restart on crash, capture logs

while true; do
  echo "Starting Elysia $(date)" >> watchdog.log
  python3 main_app.py
  code=$?
  if [ $code -ne 0 ]; then
    echo "Crash (exit $code) $(date)" >> watchdog.log
    tail -n 40 elysia.log > last_crash_snippet.log
    # main_app writes crash_info.txt; just pause and relaunch
    sleep 2
  else
    echo "Normal exit $(date)" >> watchdog.log
    break
  fi
done
