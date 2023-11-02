#!/bin/bash

py -m venv .venv
pip install -r requirements.txt
AutoROM -v
