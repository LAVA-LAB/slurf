#!/bin/bash
cd ..;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [50,100,200,400,800] --rho 0.1 --seeds 1;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties50.xlsx --N [50,100,200,400,800] --rho 0.1 --seeds 1;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties100.xlsx --N [50,100,200,400,800] --rho 0.1 --seeds 1;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties200.xlsx --N [50,100,200,400,800] --rho 0.1 --seeds 1;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties400.xlsx --N [50,100,200,400,800] --rho 0.1 --seeds 1;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties800.xlsx --N [50,100,200,400,800] --rho 0.1 --seeds 1;