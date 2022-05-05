#!/bin/bash
echo -e "Start the full set of benchmarks...\n\n";

echo -e "\nStart recreating all figures...\n\n";
bash create_all_figures.sh;

echo -e "\nStart reproducing benchmark statistics (Tables 1 and 4)...\n\n";
bash table4_statistics_partial.sh;

echo -e "\nStart reproducing table of lower bounds (Table 2)...\n\n";
bash table2_bounds_partial.sh; 

echo -e "\nStart reproducing table of scenario optimization run times (Table 3)...\n\n";
bash table3_runtime_partial.sh;

echo -e "\nStart reproducing comparison with naive monte carlo baseline (Table 5)...\n\n";
bash table5_baseline_partial.sh

echo -e "\nBenchmarks completed!";