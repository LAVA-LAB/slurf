#!/bin/bash
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 50 --Nvalidation 1000 --repeat 10;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 100 --Nvalidation 1000 --repeat 10;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 200 --Nvalidation 1000 --repeat 10;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 400 --Nvalidation 1000 --repeat 10;