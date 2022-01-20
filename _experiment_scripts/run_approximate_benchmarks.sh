#!/bin/bash
cd ..;
# RC n=100
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --precision 0.01;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.2-2-hc_parametric.dft --param_file rc.2-2-hc_parameters.xlsx --precision 0.01;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.3-3-hc_parametric.dft --param_file rc.3-3-hc_parameters.xlsx --precision 0.01;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.4-4-hc_parametric.dft --param_file rc.4-4-hc_parameters.xlsx --precision 0.01;
#
# HECS n=100
timeout 3600s python runfile.py --model dft/hecs/hecs_2_1.dft --precision 0.01;
timeout 3600s python runfile.py --model dft/hecs/hecs_2_2.dft --param_file hecs_2_2_parameters.xlsx --precision 0.01;
#
#
# RC n=200
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --precision 0.01 --N 200;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.2-2-hc_parametric.dft --param_file rc.2-2-hc_parameters.xlsx --precision 0.01 --N 200;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.3-3-hc_parametric.dft --param_file rc.3-3-hc_parameters.xlsx --precision 0.01 --N 200;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.4-4-hc_parametric.dft --param_file rc.4-4-hc_parameters.xlsx --precision 0.01 --N 200;
#
# HECS n=200
timeout 3600s python runfile.py --model dft/hecs/hecs_2_1.dft --precision 0.01 --N 200;
#
#
# BONUS BIG MODEL: RC 4-4-hc
timeout 3600s python runfile.py --model dft/hecs/hecs_2_2.dft --param_file hecs_2_2_parameters.xlsx --precision 0.01 --N 200;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.4-4-hc_parametric.dft --param_file rc.4-4-hc_parameters.xlsx --precision 0.01;