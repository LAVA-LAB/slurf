#!/bin/bash
cd ..;
echo "++++++++ CREATE EMPTY TABLE TO EXPORT RESULTS TO ++++++++";
python3 create_empty_table.py;
#
# === CTMCS ===
# SIR
echo "++++++++ START SIR BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir20.sm --N [100,200] --prop_file properties.xlsx --export_table;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir60.sm --N [100,200] --prop_file properties.xlsx --export_table;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir100.sm --N [100,200] --prop_file properties.xlsx --export_table;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir100.sm --N [100,200] --prop_file properties.xlsx --precision 0.01 --export_table;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir140.sm --N [100,200] --prop_file properties.xlsx --export_table;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir140.sm --N [100,200] --prop_file properties.xlsx --precision 0.01 --export_table;
#
# kanban
echo "++++++++ START KANBAN BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban3.sm --no-bisim --N [100,200] --export_table;
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban5.sm --no-bisim --N [100,200] --export_table;
#
# polling
echo "++++++++ START POLLING BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/polling/polling.3.prism --N [100,200] --export_table;
timeout 3600s python3 runfile.py --model ctmc/polling/polling.9.prism --N [100,200] --export_table;
timeout 3600s python3 runfile.py --model ctmc/polling/polling.15.prism --N [100] --export_table;
#
# buffer
echo "++++++++ START BUFFER BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/buffer/buffer.sm --N [100,200] --pareto_pieces 5 --rho [1.1,0.75,0.4,0.2,0.1] --export_table;
#
# tandem
echo "++++++++ START TANDEM BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem15.sm --N [100,200] --export_table;
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem31.sm --N [100,200] --export_table;
#
# embedded
echo "++++++++ START EMBEDDED BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.64.prism --N [100,200] --export_table;
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.256.prism --N [100,200] --export_table;
#
# === Fault trees ===
echo "++++++++ START FAULT TREE BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N [100,200] --dft_checker 'parametric' --export_table;
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N [100,200] --dft_checker 'concrete' --export_table;
#
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [100,200] --dft_checker 'parametric' --export_table;
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [100,200] --dft_checker 'concrete' --export_table;
#
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [100,200] --dft_checker 'parametric' --export_table;
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [100,200] --dft_checker 'concrete' --export_table;
#
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [100,200] --dft_checker 'parametric' --export_table;
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [100,200] --dft_checker 'concrete' --export_table;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --precision 0.01 --N [100,200] --export_table;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.2-2-hc_parametric.dft --param_file rc.2-2-hc_parameters.xlsx --precision 0.01 --N [100,200] --export_table;
#
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [100,200] --dft_checker 'parametric' --export_table;
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [100,200] --dft_checker 'concrete' --export_table;
timeout 3600s python runfile.py --model dft/hecs/hecs_2_1.dft --precision 0.01 --N [100,200] --export_table;
timeout 3600s python runfile.py --model dft/hecs/hecs_2_2.dft --param_file hecs_2_2_parameters.xlsx --precision 0.01 --N [100,200] --export_table;
#