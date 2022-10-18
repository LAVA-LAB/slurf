import numpy as np
import ast

def load_fixed_valuations(param_dic, N):
    """
    Load a fixed set of parameter valuations
    :param_dic: Parameter valuation dictionary
    """
    
    # Interpet string of valuations as a nested list
    enum = [ast.literal_eval(v['values']) for p,v in param_dic.items()]    
    
    # Convert nested list of valuations into a single matrix
    # Each column represents a parameter; each row a valuation
    enum_concat = np.vstack(enum).T
    
    assert len(enum_concat) == N
    
    return enum_concat