#!/bin/bash
cd ..;
# === Fault trees ===
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
#
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [200] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [200] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N [200] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N [200] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [200] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [200] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [200] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [200] --dft_checker 'concrete';
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [200] --dft_checker 'parametric';
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [200] --dft_checker 'concrete';
#
#
# === CTMCS ===
# SIR
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [100,200] --prop_file properties25.xlsx;
python3 runfile.py --model ctmc/epidemic/sir60.sm --N [100,200] --prop_file properties25.xlsx;
python3 runfile.py --model ctmc/epidemic/sir100.sm --N [100,200] --prop_file properties25.xlsx;
python3 runfile.py --model ctmc/epidemic/sir140.sm --N [100,200] --prop_file properties25.xlsx;
#
# buffer
python3 runfile.py --model ctmc/buffer/buffer.sm --N [100,200] --pareto_pieces 5 --rho [1.1,0.75,0.4,0.2,0.1];
#
# tandem
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem15.sm --N [100,200];
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem31.sm --N [100,200];
#
# embedded
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.64.prism --N [100,200];
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.256.prism --N [100,200];
#
# kanban and polling
python3 runfile.py --model ctmc/kanban/kanban3.sm --no-bisim --N [100,200]
python3 runfile.py --model ctmc/polling/polling.3.prism --N [100,200];
python3 runfile.py --model ctmc/polling/polling.9.prism --N [100,200];
python3 runfile.py --model ctmc/kanban/kanban5.sm --no-bisim --N [100,200]
python3 runfile.py --model ctmc/polling/polling.15.prism --N [100];
