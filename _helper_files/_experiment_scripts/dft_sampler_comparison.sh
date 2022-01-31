#!/bin/bash
cd ..;
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [100] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [100] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N [100] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N [100] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [100] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [100] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [100] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [100] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [100] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [100] --dft_checker 'concrete';