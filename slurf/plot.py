import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns

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
                     mode='conservative'):
    
    assert mode in ['smooth', 'step', 'conservative']
    
    # Create plot
    fig, ax = plt.subplots()
    if plotSamples:
        plt.plot(Tlist, samples.T, color='k', lw=0.3, ls='dotted', alpha=0.3)

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
        
        plt.annotate(r'$\eta=$'+str(np.round(1-item['Pviolation'], 2)), 
                     xy=xy, xytext=xytext,
                     ha='left', va='center', textcoords='offset points',
                     arrowprops=dict(arrowstyle="-|>",mutation_scale=12, facecolor='black'),
                     )
        
    plt.xlabel('Time')
    plt.ylabel('Probability of zero infected')

    ax.set_title("Confidence regions on a randomly sampled curve (confidence beta={}; N={} samples)".
                 format(len(samples), beta))
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)
    
    plt.show()
    
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
        
        rect = patches.Rectangle(item['x_low'], diff[0], diff[1], 
                                 linewidth=0, edgecolor='none', facecolor=color)

        # Add the patch to the Axes
        ax.add_patch(rect)
        
    if plotSamples:
        plt.scatter(samples[:,X], samples[:,Y], color='k', s=10, alpha=0.5)
        
    plt.xlabel(prop_names[0])
    plt.ylabel(prop_names[1])

    ax.set_title("Solution sets (confidence beta={}; N={} samples)".
                 format(len(samples), beta))
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)
    
    plt.show()