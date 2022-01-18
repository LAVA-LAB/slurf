#!/bin/bash
cd ..;
python3 runfile.py --model dft/dcas/dcas.dft --N [50,100,200,400,800] --prop_file properties25.xlsx --rho 0.1 --seeds 3;
python3 runfile.py --model dft/dcas/dcas.dft --N [50,100,200,400,800] --prop_file properties50.xlsx --rho 0.1 --seeds 3;
python3 runfile.py --model dft/dcas/dcas.dft --N [50,100,200,400,800] --prop_file properties100.xlsx --rho 0.1 --seeds 3;
python3 runfile.py --model dft/dcas/dcas.dft --N [50,100,200,400,800] --prop_file properties200.xlsx --rho 0.1 --seeds 3;
python3 runfile.py --model dft/dcas/dcas.dft --N [50,100,200,400,800] --prop_file properties400.xlsx --rho 0.1 --seeds 3;
python3 runfile.py --model dft/dcas/dcas.dft --N [50,100,200,400,800] --prop_file properties800.xlsx --rho 0.1 --seeds 3;
