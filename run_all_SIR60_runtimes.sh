#!/bin/bash
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties25.xlsx --seeds 3 --rho_min 0.01 --rho_max_iter 1;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties50.xlsx --seeds 3 --rho_min 0.01 --rho_max_iter 1;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties100.xlsx --seeds 3 --rho_min 0.01 --rho_max_iter 1; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties200.xlsx --seeds 3 --rho_min 0.01 --rho_max_iter 1; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties400.xlsx --seeds 3 --rho_min 0.01 --rho_max_iter 1; 
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties800.xlsx --seeds 3 --rho_min 0.01 --rho_max_iter 1; 
#
#python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties25.xlsx --seeds 3 --rho_min 1.2 --rho_max_iter 1;
#python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties50.xlsx --seeds 3 --rho_min 1.2 --rho_max_iter 1;
#python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties100.xlsx --seeds 3 --rho_min 1.2 --rho_max_iter 1; 
#python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties200.xlsx --seeds 3 --rho_min 1.2 --rho_max_iter 1; 
#python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties400.xlsx --seeds 3 --rho_min 1.2 --rho_max_iter 1; 
#python3 runfile.py --model ctmc/epidemic/sir60.sm --N [50,100,200,400,800] --prop_file properties800.xlsx --seeds 3 --rho_min 1.2 --rho_max_iter 1;