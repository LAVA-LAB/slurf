#!/bin/bash
cd ..;
echo -e "++++++++ RUN KANBAN(3) BENCHMARK ++++++++\n";
OutputFile='kanban3_Table2.csv';
if [ -f "$OutputFile" ] ; then
    rm "$OutputFile";
fi
BoundsFile='kanban3_bounds_Table2.csv';
if [ -f "$BoundsFile" ] ; then
    rm "$BoundsFile";
fi
#
python3 runfile.py --model ctmc/kanban/kanban3.sm --N [100,200] --rho 1.1 --no-bisim --Nvalidate 1000 --seeds 2 --export_bounds $BoundsFile;
python3 gen_lower_bounds_table.py --file $BoundsFile --outfile $OutputFile;
#
#
echo -e "\n++++++++ RUN KANBAN(3) BENCHMARK ++++++++\n";
OutputFile='rc.1-1-hc_parametric_Table2.csv';
if [ -f "$OutputFile" ] ; then
    rm "$OutputFile";
fi
BoundsFile='rc1-1_bounds_Table2.csv';
if [ -f "$BoundsFile" ] ; then
    rm "$BoundsFile";
fi
#
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --N [100,200] --rho 1.1 --Nvalidate 1000 --seeds 10 --export_bounds $BoundsFile;
python3 gen_lower_bounds_table.py --file $BoundsFile --outfile $OutputFile;