#!/usr/bin/env bash
# Kill any auto-sklearn worker over LIMIT_KB resident (auto-sklearn's memory_limit, which macOS can't enforce).
LIMIT_KB=${LIMIT_KB:-3145728}
LOG="$1"
while pgrep -f automl_harness_autosklearn.py >/dev/null; do
  ps -axo pid,rss,command | grep 'envs/autosklearn/bin/python' | grep forkserver | grep -v grep | \
  while read pid rss rest; do
    if [ "$rss" -gt "$LIMIT_KB" ]; then
      echo "$(date '+%H:%M:%S') killed $pid rss=$((rss/1024)) MB" >> "$LOG"; kill -9 $pid
    fi
  done
  sleep 2
done
echo "$(date '+%H:%M:%S') harness gone, watchdog exit" >> "$LOG"
