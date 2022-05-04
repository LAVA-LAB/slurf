#!/bin/bash
cd ..;
OutputFile='partial_benchmark_stats.csv';
if [ -f "$OutputFile" ] ; then
    rm "$OutputFile";
fi
echo -e "RUN PARTIAL SET OF BENCHMARKS\n";
#
# === CTMCS ===
# SIR
echo -e "\n++++++++ START SIR BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/epidemic/sir20.sm --N [100,200] --prop_file properties.xlsx --export_stats partial_benchmark_stats.csv;
#
# kanban
echo -e "\n++++++++ START KANBAN BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/kanban/kanban3.sm --no-bisim --N [100,200] --export_stats partial_benchmark_stats.csv;
#
# polling
echo -e "\n++++++++ START POLLING BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/polling/polling.3.prism --N [100,200] --export_stats partial_benchmark_stats.csv;
#
# buffer
echo -e "\n++++++++ START BUFFER BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/buffer/buffer.sm --N [100,200] --pareto_pieces 5 --rho [1.1,0.75,0.4,0.2,0.1] --export_stats partial_benchmark_stats.csv;
#
# tandem
echo -e "\n++++++++ START TANDEM BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/tandem/tandem15.sm --N [100,200] --export_stats partial_benchmark_stats.csv;
#
# embedded
echo -e "\n++++++++ START EMBEDDED BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model ctmc/embedded/embedded.64.prism --N [100,200] --export_stats partial_benchmark_stats.csv;
#
# === Fault trees ===
echo -e "\n++++++++ START FAULT TREE BENCHMARKS ++++++++\n";
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N 100 --dft_checker 'parametric' --export_stats $OutputFile;
timeout 3600s python3 runfile.py --model sft/pcs/pcs.dft --N 200 --dft_checker 'parametric' --export_stats $OutputFile;
#
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [100] --dft_checker 'parametric' --export_stats $OutputFile;
timeout 3600s python3 runfile.py --model sft/rbc/rbc.dft --N [200] --dft_checker 'parametric' --export_stats $OutputFile;
#
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [100] --dft_checker 'parametric' --export_stats $OutputFile;
timeout 3600s python3 runfile.py --model dft/dcas/dcas.dft --N [200] --dft_checker 'parametric' --export_stats $OutputFile;
#
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [100] --dft_checker 'parametric' --export_stats $OutputFile;
timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --N [200] --dft_checker 'parametric' --export_stats $OutputFile;
#
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [100] --dft_checker 'parametric' --export_stats $OutputFile;
timeout 3600s python3 runfile.py --model dft/hecs/hecs_2_1.dft --N [200] --dft_checker 'parametric' --export_stats $OutputFile;
#