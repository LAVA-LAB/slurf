#!/bin/bash
python3 runfile.py --model ctmc/kanban/kanban2.sm --N 50 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban2.sm --N 100 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban2.sm --N 200 --no-bisim;
#python3 runfile.py --model ctmc/kanban/kanban2.sm --N 400 --no-bisim;
###
python3 runfile.py --model ctmc/kanban/kanban3.sm --N 50 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban3.sm --N 100 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban3.sm --N 200 --no-bisim;
#python3 runfile.py --model ctmc/kanban/kanban3.sm --N 400 --no-bisim;
###
python3 runfile.py --model ctmc/kanban/kanban4.sm --N 50 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban4.sm --N 100 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban4.sm --N 200 --no-bisim;
#python3 runfile.py --model ctmc/kanban/kanban4.sm --N 400 --no-bisim;
###
python3 runfile.py --model ctmc/kanban/kanban5.sm --N 50 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban5.sm --N 100 --no-bisim;
python3 runfile.py --model ctmc/kanban/kanban5.sm --N 200 --no-bisim;
#python3 runfile.py --model ctmc/kanban/kanban5.sm --N 400 --no-bisim;