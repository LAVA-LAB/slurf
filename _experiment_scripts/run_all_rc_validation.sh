#!/bin/bash
cd ..;
python3 runfile.py --model dft/rc/rc.1-1-hc.dft --N [50,100,200,400,800] --rho 1.1 --Nvalidate 1000 --seeds 10;
#python3 runfile.py --model ctmc/kanban/kanban3.sm --N 100 --no-bisim --Nvalidate 100 --seeds 2;
#python3 runfile.py --model ctmc/kanban/kanban3.sm --N 200 --no-bisim --Nvalidate 100 --seeds 2;
#python3 runfile.py --model ctmc/kanban/kanban3.sm --N 400 --no-bisim --Nvalidate 100 --seeds 2;