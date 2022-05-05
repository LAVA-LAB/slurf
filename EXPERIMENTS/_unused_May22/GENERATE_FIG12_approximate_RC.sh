#!/bin/bash
cd ..;
python runfile.py --model dft/rc_for_imprecise/rc.2-2-hc_parametric.dft --param_file rc.2-2-hc_parameters.xlsx --prop_file properties2.xlsx --precision 0.1 --N 200 --refine --refine_precision 0.001 --plot_timebounds [3,6] --rho 0.4 --Nvalidate 1000;