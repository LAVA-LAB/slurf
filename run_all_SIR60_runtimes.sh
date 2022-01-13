#!/bin/bash
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties25.xlsx --repeat 5; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties50.xlsx --repeat 5; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties100.xlsx --repeat 5; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties200.xlsx --repeat 5; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties400.xlsx --repeat 5; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties800.xlsx --repeat 5; 