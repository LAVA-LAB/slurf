#!/bin/bash
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 25;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 50;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 100;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 200;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N 400;
###
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 25;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 50;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 100;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 200;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 400;
###
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 25;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 50;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 100;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 200;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N 400;
###
python3 runfile.py --model ctmc/epidemic/sir140.sm --N 25;
python3 runfile.py --model ctmc/epidemic/sir140.sm --N 50;
python3 runfile.py --model ctmc/epidemic/sir140.sm --N 100;
python3 runfile.py --model ctmc/epidemic/sir140.sm --N 200;
python3 runfile.py --model ctmc/epidemic/sir140.sm --N 400;