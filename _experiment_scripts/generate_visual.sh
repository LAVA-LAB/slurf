#!/bin/bash
cd ..;
# Kanban example figure
python3 runfile.py --model ctmc/kanban/kanban2.sm --no-bisim --N 25
#
# Extinction probability over time for the SIR benchmark
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [400] --prop_file properties25.xlsx --rho [1.2,0.6,0.3,0.15,0.075,0.0375];
#
# Pareto-front for buffer benchmark
python3 runfile.py --model ctmc/buffer/buffer.sm --N [200] --pareto_pieces 5 --rho [0.1,1.1];