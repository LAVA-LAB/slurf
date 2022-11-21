import argparse
import pathlib
from ast import literal_eval
import statistics

def parse_arguments(manualModel=None, nobisim=False):
    """
    Function to parse arguments provided

    Parameters
    ----------
    :manualModel: Override model as provided as argument in the command
    :nobisim: Override bisimulatoin option as provided as argument in the command

    Returns
    -------
    :args: Dictionary with all arguments

    """
    
    parser = argparse.ArgumentParser(description="Sampling-based verifier for upCTMCs")
    # Scenario problem main arguments
    parser.add_argument('--N', type=int, action="store", dest='Nsamples', 
                        default=100, help="Number of samples to compute")
    parser.add_argument('--beta', type=str, action="store", dest='beta', 
                        default='[0.9,0.99,0.999]', help="Number of samples to compute")
    # Scenario problem optional arguments
    parser.add_argument('--rho', type=str, action="store", dest='rho_list', 
                        default=None, help="List of cost of violation")
    
    # Argument for model to load
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=manualModel, help="Model file to load")
    # Set a manual parameter distribution file
    parser.add_argument('--param_file', type=str, action="store", dest='param_file', 
                        default=None, help="Parameter distribution file (CSV or Excel)")
    # Set a manual parameter distribution file
    parser.add_argument('--prop_file', type=str, action="store", dest='prop_file', 
                        default=None, help="Properties file (CSV or Excel)")
    
    # Enable/disable bisimulation
    parser.add_argument('--no-bisim', dest='bisim', action='store_false',
                        help="Disable bisimulation")
    parser.set_defaults(bisim=True)
    
    # Enable/disable cache usage
    parser.add_argument('--no-cache', dest='cache', action='store_false',
                        help="Disable usage of solution cache")
    parser.set_defaults(cache=True)
    
    # Number of repetitions
    parser.add_argument('--seed', type=int, action="store", dest='seed', 
                        default=1, help="Random seed to set in the experiment")
    
    # Number of validation samples (0 by default, i.e. no validation)
    parser.add_argument('--Nvalidate', type=int, action="store", dest='Nvalidate', 
                        default=0, help="Number of samples to validate confidence regions with")
    
    # Argument for type of model checking
    parser.add_argument('--dft_checker', type=str, action="store", dest='dft_checker', 
                        default='concrete', help="Type of DFT model checker to use")
    parser.add_argument('--precision', type=float, action="store", dest='precision', 
                        default=0, help="Initial precision to be used for computing solutions (0 is exact)")
    
    parser.add_argument('--naive_baseline', action="store_true", dest='naive_baseline', 
                        help="Enable naive baseline that analyzes each measure independently")
    parser.set_defaults(naive_baseline=False)
    
    # Allow for exporting tabular results
    parser.add_argument('--export_stats', type=str, action="store", dest='export_stats',
                        help="Export benchmark statistics to specified file")
    parser.set_defaults(export_stats=None)
    
    # Allow for exporting tabular results
    parser.add_argument('--export_filetype', type=str, action="store", dest='export_filetype',
                        help="Extension ('csv' or 'xlsx') of files to export results to")
    parser.set_defaults(export_filetype='csv')
    
    # Enable/disable refinement and set precision for refinement
    parser.add_argument('--refine', dest='refine', action='store_true',
                        help="Iterative refinement scheme")
    parser.set_defaults(refine=False)
    parser.add_argument('--refine_precision', type=float, action="store", dest='refine_precision', 
                        default=0, help="Refinement precision to be used for refining solutions (0 is exact)")
    
    parser.add_argument('--plot_timebounds', type=str, action="store", dest='plot_timebounds', 
                        default=None, help="List of two timebounds to create 2D plot for")
    
    # Plot optional arguments
    parser.add_argument('--curve_plot_mode', type=str, action="store", dest='curve_plot_mode', 
                        default='conservative', help="Plotting mode for reliability curve")
    
    # Number of pareto pieces is the number of liens making up the right-top
    # of the pareto curve
    parser.add_argument('--pareto_pieces', type=int, action="store", dest='pareto_pieces', 
                        default=0, help="Number of Pareto-front pieces")

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()    

    if args.dft_checker not in ['concrete','parametric','simulation']:
        print('ERROR: the DFT model checker argument should be "concrete", "parametric" or "simulation"; current value:', args.dft_checker)
        assert False

    assert args.export_filetype in ['csv', 'xlsx']

    # Interprete some arguments as lists
    if args.rho_list:
        try:
            args.rho_list = [float(args.rho_list)]
        except:
            args.rho_list = list(literal_eval(args.rho_list))
    
    if args.plot_timebounds:
        try:
            args.plot_timebounds = [float(args.plot_timebounds)]
        except:
            args.plot_timebounds = list(literal_eval(args.plot_timebounds))
    
    try:
        args.beta = [float(args.beta)]
    except:
        args.beta = list(literal_eval(args.beta))
        
    args.Nsamples = int(args.Nsamples)
        
    args.beta2plot = str(statistics.median(args.beta))

    if args.model == None:
        print('ERROR: No model specified')
        assert False
        
    _path = pathlib.Path(args.model)
    args.suffix = _path.suffix
    args.modelfile_nosuffix = _path.stem
    if args.suffix == '.dft':
        args.model_type = 'DFT'
    else:
        args.model_type = 'CTMC'
        
    args.modelfolder, args.modelfile = args.model.rsplit('/', 1)
    
    if not args.bisim or nobisim:
        args.bisim = False
        print('- Bisimulation is disabled')
        
    if args.refine:
        print('- Iterative refinement scheme is enabled')
        print('--- Refinement precision is:',args.refine_precision)
    
    if args.precision == 0:
        args.exact = True
        print('- Compute exact solution vectors')
    else:
        args.exact = False
        print('- Computing imprecise solution vectors (precision: '+str(args.precision)+')')
    
    return args