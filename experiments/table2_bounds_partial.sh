#!/bin/bash
cd ..;
echo -e "++++++++ RUN KANBAN(3) BENCHMARK ++++++++\n";
reps=3
for (( i=0; i < $reps; ++i )); do
	for N in 100 200; do
		python3 runfile.py --model ctmc/kanban/kanban3.sm --N $N --rho 1.1 --no-bisim --Nvalidate 500 --seed $i --export_stats output/stats_bounds/kanban/kanban3_${N}_s${i}.json;
	done
done
python generate_table.py --folder 'output/stats_bounds/kanban/' --outfile 'output/table2_kanban_partial.csv' --mode lower_bounds;
#
echo -e "\n++++++++ RUN KANBAN(3) BENCHMARK ++++++++\n";
reps=3
for (( i=0; i < $reps; ++i )); do
	for N in 100 200; do
		python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --N $N --rho 1.1 --Nvalidate 500 --seed $i --export_stats output/stats_bounds/rc/rc1-1_${N}_s$i.json;
	done
done
python generate_table.py --folder 'output/stats_bounds/rc/' --outfile 'output/table2_rc_partial.csv' --mode lower_bounds;