#!/bin/bash
cd ..;
OutputFile='sir20_Table3.csv';
if [ -f "$OutputFile" ] ; then
    rm "$OutputFile";
fi
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [50,100,200,400,800] --prop_file properties25.xlsx --seeds 3 --rho 0.1;
python3 gen_run_times_table.py --file 'output/sir20_run_times.csv' --outfile $OutputFile --no_props 25;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [50,100,200,400,800] --prop_file properties50.xlsx --seeds 3 --rho 0.1;
python3 gen_run_times_table.py --file 'output/sir20_run_times.csv' --outfile $OutputFile--no_props 50;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [50,100,200,400,800] --prop_file properties100.xlsx --seeds 3 --rho 0.1;
python3 gen_run_times_table.py --file 'output/sir20_run_times.csv' --outfile $OutputFile --no_props 100;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [50,100,200,400,800] --prop_file properties200.xlsx --seeds 3 --rho 0.1;
python3 gen_run_times_table.py --file 'output/sir20_run_times.csv' --outfile $OutputFile --no_props 200;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [50,100,200,400,800] --prop_file properties400.xlsx --seeds 3 --rho 0.1;
python3 gen_run_times_table.py --file 'output/sir20_run_times.csv' --outfile $OutputFile --no_props 400;
python3 runfile.py --model ctmc/epidemic/sir20.sm --N [50,100,200,400] --prop_file properties800.xlsx --seeds 3 --rho 0.1;
python3 gen_run_times_table.py --file 'output/sir20_run_times.csv' --outfile $OutputFile --no_props 800;
#
OutputFile='hc1-1_Table3.csv';
if [ -f "$OutputFile" ] ; then
    rm "$OutputFile";
fi
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --N [50,100,200,400,800] --param_file rc.1-1-hc_parameters.xlsx --prop_file properties.xlsx --rho 0.1 --seeds 1;
python3 gen_run_times_table.py --file 'output/rc.1-1-hc_parametric_run_times.csv' --outfile $OutputFile --no_props 25;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --N [50,100,200,400,800] --param_file rc.1-1-hc_parameters.xlsx --prop_file properties50.xlsx --rho 0.1 --seeds 1;
python3 gen_run_times_table.py --file 'output/rc.1-1-hc_parametric_run_times.csv' --outfile $OutputFile --no_props 50;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --N [50,100,200,400,800] --param_file rc.1-1-hc_parameters.xlsx --prop_file properties100.xlsx --rho 0.1 --seeds 1;
python3 gen_run_times_table.py --file 'output/rc.1-1-hc_parametric_run_times.csv' --outfile $OutputFile --no_props 100;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --N [50,100,200,400,800] --param_file rc.1-1-hc_parameters.xlsx --prop_file properties200.xlsx --rho 0.1 --seeds 1;
python3 gen_run_times_table.py --file 'output/rc.1-1-hc_parametric_run_times.csv' --outfile $OutputFile --no_props 200;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --N [50,100,200,400,800] --param_file rc.1-1-hc_parameters.xlsx --prop_file properties400.xlsx --rho 0.1 --seeds 1;
python3 gen_run_times_table.py --file 'output/rc.1-1-hc_parametric_run_times.csv' --outfile $OutputFile --no_props 400;
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --N [50,100,200,400,800] --param_file rc.1-1-hc_parameters.xlsx --prop_file properties800.xlsx --rho 0.1 --seeds 1;
python3 gen_run_times_table.py --file 'output/rc.1-1-hc_parametric_run_times.csv' --outfile $OutputFile --no_props 800;