#!/bin/bash
python3 runfile.py --model dft/dcas/dcas.dft --N 50 --Nvalidate 1000;
python3 runfile.py --model dft/dcas/dcas.dft --N 100 --Nvalidate 1000;
python3 runfile.py --model dft/dcas/dcas.dft --N 200 --Nvalidate 1000;
python3 runfile.py --model dft/dcas/dcas.dft --N 300 --Nvalidate 1000;
python3 runfile.py --model dft/dcas/dcas.dft --N 400 --Nvalidate 1000;
python3 runfile.py --model dft/dcas/dcas.dft --N 500 --Nvalidate 1000;
###
python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 50 --Nvalidate 1000;
python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 100 --Nvalidate 1000;
python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 200 --Nvalidate 1000;
python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 300 --Nvalidate 1000;
python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 400 --Nvalidate 1000;
python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 500 --Nvalidate 1000;
###
python3 runfile.py --model dft/nppwp/nppwp.dft --N 50 --Nvalidate 1000;
python3 runfile.py --model dft/nppwp/nppwp.dft --N 100 --Nvalidate 1000;
python3 runfile.py --model dft/nppwp/nppwp.dft --N 200 --Nvalidate 1000;
python3 runfile.py --model dft/nppwp/nppwp.dft --N 300 --Nvalidate 1000;
python3 runfile.py --model dft/nppwp/nppwp.dft --N 400 --Nvalidate 1000;
python3 runfile.py --model dft/nppwp/nppwp.dft --N 500 --Nvalidate 1000;
###
python3 runfile.py --model dft/rc/rc.1-1-hc.dft --N 50 --Nvalidate 1000;
python3 runfile.py --model dft/rc/rc.1-1-hc.dft --N 100 --Nvalidate 1000;
python3 runfile.py --model dft/rc/rc.1-1-hc.dft --N 200 --Nvalidate 1000;
python3 runfile.py --model dft/rc/rc.1-1-hc.dft --N 300 --Nvalidate 1000;
python3 runfile.py --model dft/rc/rc.1-1-hc.dft --N 400 --Nvalidate 1000;
python3 runfile.py --model dft/rc/rc.1-1-hc.dft --N 500 --Nvalidate 1000;