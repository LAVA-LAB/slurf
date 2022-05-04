#!/bin/bash
cd ..;
echo -e "++++++++ RUN KANBAN(3) BENCHMARK ++++++++\n";
BoundsFile='output/kanban3_bounds.csv';
if [ -f "$BoundsFile" ] ; then
    rm "$BoundsFile";
fi
OutputFile='kanban3_Table2.csv';
if [ -f "$OutputFile" ] ; then
    rm "$OutputFile";
fi
#
python3 runfile.py --model ctmc/kanban/kanban3.sm --N [100,200,400,800] --rho 1.1 --no-bisim --Nvalidate 1000 --seeds 10 --export_bounds $BoundsFile;
python3 gen_lower_bounds_table.py --file $BoundsFile --outfile $OutputFile;
#
#
echo -e "\n++++++++ RUN KANBAN(3) BENCHMARK ++++++++\n";
BoundsFile='output/rc1-1_bounds.csv';
if [ -f "$BoundsFile" ] ; then
    rm "$BoundsFile";
fi
OutputFile='rc.1-1-hc_parametric_Table2.csv';
if [ -f "$OutputFile" ] ; then
    rm "$OutputFile";
fi
#
python3 runfile.py --model dft/rc/rc.1-1-hc_parametric.dft --param_file rc.1-1-hc_parameters.xlsx --N [100,200,400,800] --rho 1.1 --Nvalidate 1000 --seeds 10 --export_bounds $BoundsFile;
python3 gen_lower_bounds_table.py --file $BoundsFile --outfile $OutputFile;