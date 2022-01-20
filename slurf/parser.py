import argparse
import pathlib
from ast import literal_eval
import statistics

def parse_arguments(manualModel=None, nobisim=False):
    
    parser = argparse.ArgumentParser(description="Sampling-based verifier for upCTMCs")
    # Scenario problem main arguments
    parser.add_argument('--N', type=str, action="store", dest='Nsamples', 
                        default=100, help="Number of samples to compute")
    parser.add_argument('--beta', type=str, action="store", dest='beta', 
                        default='[0.9,0.99,0.999]', help="Number of samples to compute")
    
    # Number of validation samples (0 by default, i.e. no validation)
    parser.add_argument('--Nvalidate', type=int, action="store", dest='Nvalidate', 
                        default=0, help="Number of samples to validate confidence regions with")
    
    # Number of repetitions
    parser.add_argument('--seeds', type=int, action="store", dest='seeds', 
                        default=1, help="Number of repetitions (to compute average results over)")
    
    # Argument for model to load
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=manualModel, help="Model file to load")
    
    # Argument for type of model checking
    parser.add_argument('--dft_checker', type=str, action="store", dest='dft_checker', 
                        default='concrete', help="Type of DFT model checker to use")
    parser.add_argument('--precision', type=float, action="store", dest='precision', 
                        default=0, help="Switch between exact and approximate results")
    
    parser.add_argument('--naive_baseline', action="store_true", dest='naive_baseline', 
                        help="Enable naive baseline that analyzes each measure independently")
    parser.set_defaults(naive_baseline=False)
    
    # Set a manual parameter distribution file
    parser.add_argument('--param_file', type=str, action="store", dest='param_file', 
                        default=None, help="Parameter distribution Excel file")
    # Set a manual parameter distribution file
    parser.add_argument('--prop_file', type=str, action="store", dest='prop_file', 
                        default=None, help="Properties Excel file")
    
    # Enable/disable bisimulation
    parser.add_argument('--no-bisim', dest='bisim', action='store_false',
                        help="Disable bisimulation")
    parser.set_defaults(bisim=True)
        
    # # Scenario problem optional arguments
    parser.add_argument('--rho', type=str, action="store", dest='rho_list', 
                        default=None, help="List of cost of violation")
    parser.add_argument('--rho_steps', type=int, action="store", dest='rho_steps', 
                        default=10, help="Number of values for rho to run for")
    # parser.add_argument('--rho_min', type=float, action="store", dest='rho_min', 
    #                     default=0.0001, help="Minimum cost of violation")
    # parser.add_argument('--rho_incr', type=float, action="store", dest='rho_incr', 
    #                     default=1.5, help="Increment factor for the cost of violation")
    # parser.add_argument('--rho_max_iter', type=int, action="store", dest='rho_max_iter', 
    #                     default=20, help="Maximum number of iterations to perform")
    
    parser.add_argument('--plot-timebounds', type=str, action="store", dest='plot_timebounds', 
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

    if args.dft_checker not in ['concrete','parametric']:
        print('ERROR: the DFT model checker argument should be "concrete" or "parametric"; current value:', args.dft_checker)
        assert False

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
        
    try:
        args.Nsamples = [int(args.Nsamples)]
    except:
        args.Nsamples = list(literal_eval(args.Nsamples))   
        args.Nsamples = [int(n) for n in args.Nsamples]
        
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
    
    if args.precision == 0:
        args.exact = True
    else:
        args.exact = False
    
    return args