#!/bin/bash
cd ..;
python3 runfile.py --model sft/cabinets/cabinets.2-3_no_inspection.dft --param_file parameters.csv --prop_file properties.csv --N 4 --beta 0.99
python3 runfile.py --model sft/cabinets/cabinets.2-3_low_inspection.dft --param_file parameters.csv --prop_file properties.csv --N 4 --beta 0.99