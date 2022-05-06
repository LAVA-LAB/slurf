#!/bin/bash
cd ..;
echo -e "++++++++ RUN COMPARISON TO NAIVE MONTE CARLO BASELINE ++++++++\n";
reps=10
for (( i=0; i < $reps; ++i )); do
	for N in 100 200 400 800; do
		python3 runfile.py --model sft/pcs/pcs.dft --N $N --naive_baseline --prop_file properties${props}.xlsx --seed $i --rho 2 --export_stats output/stats_naive/pcs/pcs_N${N}_s${i}.json; 
	done
done
#
echo -e "\n++++++++ GRAB RESULTS AND CREATE TABLE... ++++++++\n";
python generate_table.py --folder 'output/stats_naive/pcs/' --outfile 'output/naive_baseline.csv' --mode naive_comparison;
#
echo -e "\n++++++++ SCRIPT DONE ++++++++\n";
