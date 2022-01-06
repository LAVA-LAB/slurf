#!/bin/bash
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 50 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 100 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 200 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 400 --Nvalidate 1000;
###
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 50 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 100 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 200 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 400 --Nvalidate 1000;
###
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 50 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 100 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 200 --Nvalidate 1000;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 400 --Nvalidate 1000;
###