import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
import itertools

from slurf.commons import getDateTime, path

def plot_results(root_dir, args, regions, solutions, reliability,
                 prop_labels=None, Tlist=None):
    
    # Plot the solution set
    if reliability:
        # As reliability over time (if properties object is a tuple)
        plot_reliability(Tlist, regions, solutions, args.beta, 
                         mode=args.curve_plot_mode, plotSamples=True)
        
        # Save figure
        exp_file = args.model.rsplit('/', 1)[1] + '_' + \
                       str(getDateTime()+'.pdf')
        filename = path(root_dir, "output", exp_file)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(' - Reliability plot exported to:',exp_file)
        
    else:
        # As a solution set (if properties object is a list of properties)    
        for idx_pair in itertools.combinations(np.arange(len(prop_labels)), 2):
            # Plot the solution set for every combination of 2 properties
            
            plot_solution_set_2d(idx_pair, prop_labels, regions, solutions, 
                                 args.beta, plotSamples=True)
    
            # Save figure
            exp_file = args.model.rsplit('/', 1)[1] + '_' + \
                           str(getDateTime() + '_' + str(idx_pair)+'.pdf')
            filename = path(root_dir, "output", exp_file)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            print(' - 2D plot exported to:',exp_file)


def make_conservative(low, upp):
    '''
    Make a region conservative (such that the smooth curve is guaranteed to
    contain the actual curve).

    Parameters
    ----------
    low : Lower bound (array)
    upp : Upper bound (array)

    Returns
    -------
    x_low : Conservative lower bound
    x_upp : Conservative upper bound

    '''
    
    x_low = np.array([np.min(low[i-1:i+2]) if i > 0 and i < len(low) else 
                          low[i] for i in range(len(low))])
    x_upp = np.array([np.max(upp[i-1:i+2]) if i > 0 and i < len(upp) else 
                          upp[i] for i in range(len(upp))])
    
    return x_low, x_upp


def plot_reliability(Tlist, regions, samples, beta, plotSamples=False, 
                     mode='conservative', annotate=False):
    
    assert mode in ['optimistic', 'conservative']
    
    # Create plot
    fig, ax = plt.subplots()

    # Set colors and markers
    color_map = sns.color_palette("Blues_r", as_cmap=True)
    
    for i, item in sorted(regions.items(), reverse=True):
        
        color = color_map(1 - item['Pviolation'])
        
        if mode == 'conservative':
            x_low, x_upp = make_conservative(item['x_low'], item['x_upp'])
        else:
            x_low, x_upp = item['x_low'], item['x_upp']
        
        plt.fill_between(Tlist, x_low, x_upp, color=color)
        
        j = int( len(Tlist)/2 + 3 - i  )
        t = Tlist[j]
        y = x_low[j]
        
        xy = (t-1, y)
        xytext = (50, -15)
        
        if annotate:
            plt.annotate(r'$\eta=$'+str(np.round(1-item['Pviolation'], 2)), 
                         xy=xy, xytext=xytext,
                         ha='left', va='center', textcoords='offset points',
                         arrowprops=dict(arrowstyle="-|>",mutation_scale=12, facecolor='black'),
                         )
        
    if plotSamples:
        plt.plot(Tlist, samples.T, color='k', lw=0.3, ls='dotted', alpha=0.5)
        
    plt.xlabel('Time')
    plt.ylabel('Probability of zero infected')

    ax.set_title("Solution sets over time (confidence beta={}; N={} samples)".
                 format(beta, len(samples)))
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)
    

def plot_solution_set_2d(idxs, prop_names, regions, samples, beta,
                         plotSamples=True):
    
    X, Y = idxs
    
    # Create plot
    fig, ax = plt.subplots()

    # Set colors and markers
    color_map = sns.color_palette("Blues_r", as_cmap=True)
    
    for i, item in sorted(regions.items(), reverse=True):
        
        color = color_map(1 - item['Pviolation'])
        
        diff = item['x_upp'] - item['x_low']
        
        rect = patches.Rectangle(item['x_low'][[X,Y]], diff[X], diff[Y], 
                                 linewidth=0, edgecolor='none', facecolor=color)

        # Add the patch to the Axes
        ax.add_patch(rect)
        
    if plotSamples:
        plt.scatter(samples[:,X], samples[:,Y], color='k', s=10, alpha=0.5)
        
    plt.xlabel(prop_names[X])
    plt.ylabel(prop_names[Y])

    ax.set_title("Solution sets (confidence beta={}; N={} samples)".
                 format(beta, len(samples)))
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)