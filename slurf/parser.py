import argparse
import pathlib

def parse_arguments(manualModel=None, nobisim=True):
    
    parser = argparse.ArgumentParser(description="Sampling-based verifier for upCTMCs")
    # Scenario problem main arguments
    parser.add_argument('--N', type=int, action="store", dest='Nsamples', 
                        default=100, help="Number of samples to compute")
    parser.add_argument('--beta', type=float, action="store", dest='beta', 
                        default=0.99, help="Number of samples to compute")
    
    # Argument for model to load
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=manualModel, help="Model file to load")
    
    parser.add_argument('--no-bisim', dest='bisim', action='store_false',
                        help="Disable bisimulation")
    parser.set_defaults(bisim=True)
        
    # Scenario problem optional arguments
    parser.add_argument('--rho_min', type=float, action="store", dest='rho_min', 
                        default=0.0001, help="Minimum cost of violation")
    parser.add_argument('--rho_incr', type=float, action="store", dest='rho_incr', 
                        default=1.5, help="Increment factor for the cost of violation")
    parser.add_argument('--rho_max_iter', type=int, action="store", dest='rho_max_iter', 
                        default=20, help="Maximum number of iterations to perform")
    
    # Plot optional arguments
    parser.add_argument('--curve_plot_mode', type=str, action="store", dest='curve_plot_mode', 
                        default='conservative', help="Plotting mode for reliability curve")

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()    

    if args.model == None:
        print('ERROR: No model specified')
        assert False
        
    args.suffix = pathlib.Path(args.model).suffix
    if args.suffix == '.dft':
        args.model_type = 'DFT'
    else:
        args.model_type = 'CTMC'
    
    if not args.bisim or nobisim:
        args.bisim = False
        print('- Bisimulation is disabled')
    
    return args