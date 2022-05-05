#!/bin/bash
cd ..;
echo -e "RUN PARTIAL SET OF BENCHMARKS\n";
#
# === CTMCS ===
# SIR
echo -e "\n++++++++ START SIR BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir20.sm --N 100 --prop_file properties.xlsx --seed 1 --export_stats output/stats_benchmarks/sir20_100.json;
#
# kanban
echo -e "\n++++++++ START KANBAN BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban3.sm --no-bisim --N 100 --seed 1 --export_stats output/stats_benchmarks/kanban3_100.json;
#
# polling
echo -e "\n++++++++ START POLLING BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/polling/polling.3.prism --N 100 --seed 1 --export_stats output/stats_benchmarks/polling3_100.json;
#
# buffer
echo -e "\n++++++++ START BUFFER BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/buffer/buffer.sm --N 100 --pareto_pieces 5 --rho [1.1,0.75,0.4,0.2,0.1] --seed 1 --export_stats output/stats_benchmarks/buffer_100.json;
#
# tandem
echo -e "\n++++++++ START TANDEM BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem15.sm --N 100 --seed 1 --export_stats output/stats_benchmarks/tandem15_100.json;
#
# embedded
echo -e "\n++++++++ START EMBEDDED BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.64.prism --N 100 --seed 1 --export_stats output/stats_benchmarks/embedded64_100.json;
#
# === Fault trees ===
echo -e "\n++++++++ START FAULT TREE BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/pcs_100_parametric.json;
#
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/rbc_100_parametric.json;
#
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/dcas_100_parametric.json;
#
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/hc1-1_100_parametric.json;
#
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N 100 --dft_checker 'parametric' --seed 1 --export_stats output/stats_benchmarks/hecs2-1_100_parametric.json;
#
echo -e "\n++++++++ GRAB RESULTS AND CREATE TABLE... ++++++++\n";
python generate_table.py --folder 'output/stats_benchmarks/' --outfile 'experiments/results/benchmark_statistics_partial.csv' --mode statistics;
#
echo -e "\n++++++++ SCRIPT DONE ++++++++\n";