#!/bin/bash
cd ..;
echo -e "RUN COMPLETE SET OF BENCHMARKS\n";
#
# === CTMCS ===
# SIR
echo "++++++++ START SIR BENCHMARKS ++++++++";
# SIR 20
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir20.sm --N 100 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir20_100.json;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir20.sm --N 200 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir20_200.json;
# SIR 60
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir60.sm --N 100 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir60_100.json;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir60.sm --N 200 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir60_200.json;
# SIR 100
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir100.sm --N 100 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir100_100.json;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir100.sm --N 200 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir100_200.json;
# SIR 100 approximate
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir100.sm --N 100 --prop_file properties.xlsx --precision 0.01 --seed 1 --export_stats output/stats_benchmarks/sir100_100.json;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir100.sm --N 200 --prop_file properties.xlsx --precision 0.01 --seed 1 --export_stats output/stats_benchmarks/sir100_200.json;
# SIR 140
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir140.sm --N 100 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir140_100.json;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir140.sm --N 200 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir140_200.json;
# SIR 140 approximate
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir140.sm --N 100 --prop_file properties.xlsx --precision 0.01 --seed 1 --export_stats output/stats_benchmarks/sir100_140.json;
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir140.sm --N 200 --prop_file properties.xlsx --precision 0.01 --seed 1 --export_stats output/stats_benchmarks/sir200_140.json;
#
# kanban
echo "++++++++ START KANBAN BENCHMARKS ++++++++";
# kanban 3
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban3.sm --no-bisim --N 100 --seed 1 --export_stats output/stats_benchmarks/kanban3_100.json;
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban3.sm --no-bisim --N 200 --seed 1 --export_stats output/stats_benchmarks/kanban3_200.json;
# kanban 5
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban5.sm --no-bisim --N 100 --seed 1 --export_stats output/stats_benchmarks/kanban5_100.json;
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban5.sm --no-bisim --N 200 --seed 1 --export_stats output/stats_benchmarks/kanban5_200.json;
#
# polling
echo "++++++++ START POLLING BENCHMARKS ++++++++";
# polling 3
timeout 3600s python3 runfile.py --model ctmc/polling/polling.3.prism --N 100 --seed 1 --export_stats output/stats_benchmarks/polling3_100.json;
timeout 3600s python3 runfile.py --model ctmc/polling/polling.3.prism --N 200 --seed 1 --export_stats output/stats_benchmarks/polling3_200.json;
# polling 9
timeout 3600s python3 runfile.py --model ctmc/polling/polling.9.prism --N 100 --seed 1 --export_stats output/stats_benchmarks/polling9_100.json;
timeout 3600s python3 runfile.py --model ctmc/polling/polling.9.prism --N 200 --seed 1 --export_stats output/stats_benchmarks/polling9_200.json;
# polling 15
timeout 3600s python3 runfile.py --model ctmc/polling/polling.15.prism --N 100 --seed 1 --export_stats output/stats_benchmarks/polling15_100.json;
timeout 3600s python3 runfile.py --model ctmc/polling/polling.15.prism --N 200 --seed 1 --export_stats output/stats_benchmarks/polling15_200.json;
#
# buffer
echo "++++++++ START BUFFER BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/buffer/buffer.sm --N 100 --pareto_pieces 5 --rho [1.1,0.75,0.4,0.2,0.1] --seed 1 --export_stats output/stats_benchmarks/buffer_100.json;
timeout 3600s python3 runfile.py --model ctmc/buffer/buffer.sm --N 200 --pareto_pieces 5 --rho [1.1,0.75,0.4,0.2,0.1] --seed 1 --export_stats output/stats_benchmarks/buffer_200.json;
#
# tandem
echo "++++++++ START TANDEM BENCHMARKS ++++++++";
# tandem 15
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem15.sm --N 100 --seed 1 --export_stats output/stats_benchmarks/tandem15_100.json;
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem15.sm --N 200 --seed 1 --export_stats output/stats_benchmarks/tandem15_200.json;
# tandem 31
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem31.sm --N 100 --seed 1 --export_stats output/stats_benchmarks/tandem31_100.json;
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem31.sm --N 200 --seed 1 --export_stats output/stats_benchmarks/tandem31_200.json;
#
# embedded
echo "++++++++ START EMBEDDED BENCHMARKS ++++++++";
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.64.prism --N 100 --seed 1 --export_stats output/stats_benchmarks/embedded64_100.json;
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.64.prism --N 200 --seed 1 --export_stats output/stats_benchmarks/embedded64_200.json;
# embedded 256
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.256.prism --N 100 --seed 1 --export_stats output/stats_benchmarks/embedded256_100.json;
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.256.prism --N 200 --seed 1 --export_stats output/stats_benchmarks/embedded256_200.json;
#
# === Fault trees ===
echo "++++++++ START FAULT TREE BENCHMARKS ++++++++";
# pcs
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/pcs_100_parametric.json;
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N 200 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/pcs_200_parametric.json;
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N 100 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/pcs_100_concrete.json;
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N 200 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/pcs_200_concrete.json;
#
# rbc
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/rbc_100_parametric.json;
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N 200 --dft_checker 'parametric' --export_table stats_benchmarks/rbc_200_parametric.json;
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N 100 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/rbc_100_concrete.json;
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N 200 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/rbc_200_concrete.json;
#
# dcas
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/dcas_100_parametric.json;
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N 200 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/dcas_200_parametric.json;
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N 100 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/dcas_100_concrete.json;
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N 200 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/dcas_200_concrete.json;
#
# rc (1-1)
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/hc1-1_100_parametric.json;
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N 200 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/hc1-1_200_parametric.json;
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N 100 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/rc1-1_100_concrete.json;
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N 200 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/rc1-1_200_concrete.json;
# rc (1-1, approximate)
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --precision 0.01 --N 100 --seed 1 --export_stats output/stats_benchmarks/rc1-1_100_approximate.json;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --precision 0.01 --N 200 --seed 1 --export_stats output/stats_benchmarks/rc1-1_200_approximate.json;
# rc (2-2)
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.2-2-hc_parametric.dft --param_file rc.2-2-hc_parameters.xlsx --precision 0.01 --N 100 --seed 1 --export_stats output/stats_benchmarks/rc2-2_100_approximate.json;
timeout 3600s python runfile.py --model dft/rc_for_imprecise/rc.2-2-hc_parametric.dft --param_file rc.2-2-hc_parameters.xlsx --precision 0.01 --N 200 --seed 1 --export_stats output/stats_benchmarks/rc2-2_200_approximate.json;
#
# hecs (2-1)
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/hecs2-1_100_parametric.json;
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 200 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/hecs2-1_200_parametric.json;
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 100 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/hecs2-1_100_concrete.json;
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 200 --dft_checker 'concrete' --seed 1 --export_stats output/stats_benchmarks/hecs2-1_200_concrete.json;
# hecs (2-1 approximate)
timeout 3600s python runfile.py --model dft/hecs/hecs_2_1.dft --precision 0.01 --N 100 --seed 1 --export_stats output/stats_benchmarks/hecs2-1_100_approximate.json;
timeout 3600s python runfile.py --model dft/hecs/hecs_2_1.dft --precision 0.01 --N 200 --seed 1 --export_stats output/stats_benchmarks/hecs2-1_200_approximate.json;
# hecs (2-2, approximate)
timeout 3600s python runfile.py --model dft/hecs/hecs_2_2.dft --param_file hecs_2_2_parameters.xlsx --precision 0.01 --N 100 --seed 1 --export_stats output/stats_benchmarks/hecs2-2_100_approximate.json;
timeout 3600s python runfile.py --model dft/hecs/hecs_2_2.dft --param_file hecs_2_2_parameters.xlsx --precision 0.01 --N 200 --seed 1 --export_stats output/stats_benchmarks/hecs2-2_200_approximate.json;
#
#
echo -e "\n++++++++ GRAB RESULTS AND CREATE TABLE... ++++++++\n";
python generate_table.py --folder 'output/stats_benchmarks/' --outfile 'output/benchmark_statistics.csv' --mode statistics;
#
echo -e "\n++++++++ SCRIPT DONE ++++++++\n";