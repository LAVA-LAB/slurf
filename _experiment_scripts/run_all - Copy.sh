#!/bin/bash
cd ..;
# SIR approximate
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [100,200] --prop_file properties25.xlsx --precision 0.01;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [100,200] --prop_file properties25.xlsx --precision 0.01;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N [100,200] --prop_file properties25.xlsx --precision 0.01;
python3 runfile.py --model ctmc/epidemic/sir140.sm --N [100,200] --prop_file properties25.xlsx --precision 0.01;