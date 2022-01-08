import os
import pathlib
import pandas as pd

from slurf.commons import path

def convert_dft_to_parametric(model, param_list = None, 
                              dist_type = 'gaussian', stdev_factor = 0.1,
                              interval_halfwidth = 0.1):

    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to model
    model = 'models/ffort/cabinets/cabinets.2-1.dft'
    model_path = path(root_dir, '', model)
    
    prefix = []
    
    param_dic = {}
    
    with open(model_path) as f:
        lines = f.readlines()
        
    for i,line in enumerate(lines):
        if not ('lambda' in line or 'prob' in line):
            continue
        
        if 'lambda' in line:
            split = 'lambda'
        else:
            split = 'prob'
        
        BE_name = line.split('"')[1]
        param_name = BE_name.replace('-', '_')
        BR_rate_mean = float( line.split(split+'=')[1].split(' ')[0] )
        
        print('Basic event identified:',BE_name)
        
        prefix += ['param '+str(param_name)+';']
        
        lines[i] = line.replace(split+'='+str(BR_rate_mean), split+'='+param_name)
        
        if dist_type == 'gaussian':
            std = stdev_factor * BR_rate_mean
            
            param_dic[param_name] = {'type': 'gaussian',
                                     'mean': BR_rate_mean,
                                     'std': std,
                                     'inverse': 'False'}
        elif dist_type == 'normal':
            lb = BR_rate_mean - interval_halfwidth
            ub = BR_rate_mean + interval_halfwidth
            param_dic[param_name] = {'type': 'interval',
                                     'lb': lb,
                                     'ub': ub,
                                     'inverse': 'False'}
        else:
            print('ERROR: Wrong parameter distribution provided:',dist_type)
            assert False
     
    # Combine prefix with actual model definition
    output_model = prefix + lines
    
    # Export parametric file
    _path = pathlib.Path(model)
    parent_folder = _path.parents[0]

    output_filename = _path.stem + '_parametric' + _path.suffix
    output_path = path(root_dir, parent_folder, output_filename)

    if pathlib.Path(output_path).is_file():
        print ("ERROR: File `"+str(output_filename)+"` already exists")
    else:
        with open(output_path, 'w') as f:
            f.writelines(output_model)
            
            print('- Parametric model exported to',output_path)

    # Create the corresponding Excel file as well
    param_df = pd.DataFrame(param_dic)
    param_df = param_df.T
    param_df.index.name = 'name'
    
    excel_filename = _path.stem + '_parameters.xlsx'
    excel_path = path(root_dir, parent_folder, excel_filename)
    
    # Write to Excel
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    param_df.to_excel(writer, sheet_name='Parameters')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    print('- Parameters Excel file exported to',excel_path)
    
    return param_df
   
model = 'models/ffort/cabinets/cabinets.2-1.dft' 
param_df = convert_dft_to_parametric(model)
