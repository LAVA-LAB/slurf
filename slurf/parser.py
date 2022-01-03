import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Sampling-based verifier for upCTMCs")
    parser.add_argument('--N', type=int, action="store", dest='Nsamples', 
                        default=100, help="Number of samples to compute")
    parser.add_argument('--beta', type=float, action="store", dest='beta', 
                        default=0.99, help="Number of samples to compute")
    parser.add_argument('--folder', type=str, action="store", dest='folder', 
                        default=None, help="Subfolder where model is located")
    parser.add_argument('--model', type=str, action="store", dest='file', 
                        default=None, help="Model file to load")

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()
    
    return args