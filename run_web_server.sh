#!/bin/bash
cd "$(dirname "$0")"
: "${PUMPKIN_PORT:=5050}"
export PUMPKIN_PORT
exec "$(pwd)"/.venv/bin/python web_server.py
