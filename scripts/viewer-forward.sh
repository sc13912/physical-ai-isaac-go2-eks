#!/bin/bash
# usage: viewer-forward.sh LOCAL_PORT LABEL_SELECTOR  — loopback-only, re-resolves the pod on every reconnect
while true; do
  P=$(kubectl -n isaac get pod -l "$2" --field-selector=status.phase=Running -o name 2>/dev/null | head -1)
  [ -n "$P" ] && kubectl -n isaac port-forward "$P" "$1":8080 --address 127.0.0.1
  sleep 3
done
