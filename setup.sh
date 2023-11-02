#!/bin/bash

py -m venv .venv
pip install -r requirements.txt
source .venv/bin/activate
AutoROM -v
