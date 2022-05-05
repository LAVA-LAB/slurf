#!/bin/bash
cd ..;
echo "++++++++ CREATE FIGURE 10: SIR (60), N=400 ++++++++";
python3 runfile.py --model ctmc/epidemic/sir60.sm --N 400 --prop_file properties25.xlsx --rho [1.2,0.6,0.3,0.15,0.075,0.0375];
#
echo "++++++++ CREATE FIGURE 11: Pareto front for buffer, N=200 ++++++++";
python3 runfile.py --model ctmc/buffer/buffer.sm --N 200 --pareto_pieces 5 --rho [0.1,1.1];
#
echo "++++++++ CREATE FIGURE 12: Refining imprecise solution vectors for RC (2,2), N=100 ++++++++";
python runfile.py --model dft/rc_for_imprecise/rc.2-2-hc_parametric.dft --param_file rc.2-2-hc_parameters.xlsx --prop_file properties2.xlsx --precision 0.1 --N 100 --refine --refine_precision 0.001 --plot_timebounds [3,6] --rho 0.4;
#
echo "++++++++ CREATE FIGURE 14 (CTMC benchmarks) ++++++++";
python3 runfile.py --model ctmc/kanban/kanban3.sm --no-bisim --N 200;
python3 runfile.py --model ctmc/polling/polling.9.prism --N 200;
python3 runfile.py --model ctmc/tandem/tandem15.sm --N 100;
python3 runfile.py --model ctmc/embedded/embedded.64.prism --N 100;
#
echo "++++++++ CREATE FIGURE 15 (DFT benchmarks) ++++++++";
python3 runfile.py --model sft/pcs/pcs.dft --N 200 --dft_checker 'concrete';
python3 runfile.py --model sft/rbc/rbc.dft --N 200 --dft_checker 'concrete';
python3 runfile.py --model dft/dcas/dcas.dft --N 200 --dft_checker 'concrete';
python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 200 --dft_checker 'concrete';
echo "++++++++ DONE CREATING FIGURES (EXPORTED TO SUBFOLDERS IN 'OUTPUT/') ++++++++";