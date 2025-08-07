#!/bin/bash
# Simple watchdog to restart Elysia on crash

while true; do
    echo "Starting Elysia at $(date)" >> watchdog.log
    python3 main_app.py
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Elysia crashed (exit code $exit_code) at $(date). Restarting..." >> watchdog.log
        # Capture the last lines of the log for debugging
        tail -n 30 elysia.log > last_crash_snippet.log
        # (crash_info.txt will have been written by main_app on exception)
        sleep 2  # small delay before restart
        continue
    else
        echo "Elysia exited normally at $(date). Watchdog stopping." >> watchdog.log
        break
    fi
done
