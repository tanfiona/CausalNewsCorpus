#!/bin/bash
cd "$(dirname "$0")"
pip install --upgrade pip
pip install datasets
pip install seqeval
python3 evaluate.py $1 $2