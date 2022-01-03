import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Sampling-based verifier for upCTMCs")
    parser.add_argument('--N', type=int, action="store", dest='Nsamples', 
                        default=100, help="Number of samples to compute")
    parser.add_argument('--beta', type=float, action="store", dest='beta', 
                        default=0.99, help="Number of samples to compute")
    parser.add_argument('--model', type=str, action="store", dest='model', 
                        default=None, help="Model file to load")
    
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
    
    return args