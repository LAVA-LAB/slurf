#!/bin/bash
cd ..;
echo -e "++++++++ RUN SIR(20) BENCHMARK ++++++++\n";
reps=3
for (( i=0; i < $reps; ++i )); do
	for N in 100 200; do
		for props in 50 100 200; do
			timeout 3600s python3 runfile.py --model ctmc/epidemic/sir20.sm --N $N --prop_file properties${props}.xlsx --seed $i --rho 0.1 --export_stats output/stats_runtime/sir/sir20_N${N}_props${props}_s${i}.json;
		done
	done
done
python generate_table.py --folder 'output/stats_runtime/sir/' --outfile 'output/table3_sir20_partial.csv' --mode scen_opt_time;
#
echo -e "++++++++ RUN RC(1-1) BENCHMARK ++++++++\n";
for (( i=0; i < $reps; ++i )); do
	for N in 100 200; do
		for props in 50 100 200; do
			timeout 3600s python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --N $N --param_file rc.1-1-hc_parameters.xlsx --prop_file properties${props}.xlsx --seed $i --rho 0.1 --export_stats output/stats_runtime/rc/rc1-1_N${N}_props${props}_s${i}.json;
		done
	done
done
python generate_table.py --folder 'output/stats_runtime/rc/' --outfile 'output/table3_rc1-1_partial.csv' --mode scen_opt_time;