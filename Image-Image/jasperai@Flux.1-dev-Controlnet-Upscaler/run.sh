#!/bin/sh

export HF_HOME=$PWD
export HF_OFFLINE=1

$PWD\..\..\.venv\Scripts\python $PWD\run.py $1 $2
