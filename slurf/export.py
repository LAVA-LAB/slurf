import numpy as np
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
import itertools

from slurf.commons import path

def plot_results(output_dir, args, regions, solutions, reliability,
                 prop_labels=None, timebounds=None):
    
    # Plot the solution set
    if reliability:
        # As reliability over time (if properties object is a tuple)
        plot_reliability(timebounds, regions, solutions, args.beta, 
                         mode=args.curve_plot_mode, plotSamples=False)
        
        # Save figure
        exp_file = args.modelfile + '.pdf'
        filename = path(output_dir, "", exp_file)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(' - Reliability plot exported to:',exp_file)
        
    else:
        # As a solution set (if properties object is a list of properties)    
        for idx_pair in itertools.combinations(np.arange(len(prop_labels)), 2):
            # Plot the solution set for every combination of 2 properties
            
            plot_solution_set_2d(idx_pair, prop_labels, regions, solutions, 
                                 args.beta, plotSamples=True)
    
            # Save figure
            exp_file = args.modelfile + "_" + str(idx_pair) + '.pdf'
            filename = path(output_dir, "", exp_file)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            print(' - 2D plot exported to:',exp_file)
            
            


def save_results(output_path, dfs, modelfile_nosuffix, N, beta):
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    xlsx_file = modelfile_nosuffix + "_N=" + str(N) + "_beta=" \
                    + str(beta) + "_results.xlsx"
    xlsx_path = path(output_path, "", xlsx_file)
    writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')
    
    # Write each dataframe to a different worksheet.
    for name, df in dfs.items():
        df.to_excel(writer, sheet_name=str(name))
        
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    print('- Results exported to:',xlsx_path)
    
    return


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


def plot_reliability(timebounds, regions, samples, beta, plotSamples=False, 
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
        
        plt.fill_between(timebounds[:-1], x_low[:-1], x_upp[:-1], color=color)
        
        if annotate:
            j = int( len(timebounds)/2 + 3 - i  )
            t = timebounds[j]
            y = x_low[j]
            
            xy = (t-1, y)
            xytext = (50, -15)
            
            plt.annotate(r'$\eta=$'+str(np.round(1-item['Pviolation'], 2)), 
                         xy=xy, xytext=xytext,
                         ha='left', va='center', textcoords='offset points',
                         arrowprops=dict(arrowstyle="-|>",mutation_scale=12, facecolor='black'),
                         )
        
    if plotSamples:
        plt.plot(timebounds[:-1], samples.T[:-1], color='k', lw=0.3, ls='dotted', alpha=0.5)
        
    plt.xlabel('Time')
    plt.ylabel('Value')

    ax.set_title("Solution sets over time (confidence beta={}; N={} samples)".
                 format(beta, len(samples)))
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)
    

def plot_pareto(idxs, prop_names, regions, samples, beta, plotSamples=True, 
                plotSampleID = True):
    
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
        
        # Check if imprecise samples are used
        if samples.ndim == 3:
            # If imprecise, plot as boxes
            for i,sample in enumerate(samples):
                s_low = sample[:,0]
                s_upp = sample[:,1]
                diff  = s_upp - s_low
                
                rect = patches.Rectangle(s_low[[X,Y]], diff[X], diff[Y], 
                                         linewidth=0.5, edgecolor='red', 
                                         linestyle='dashed', facecolor='none')
                
                # Add the patch to the Axes
                ax.add_patch(rect)
                
                if plotSampleID:
                    plt.text(s_upp[X], s_upp[Y], i, fontsize=6, color='r')
            
        else:
            # Else, plot as points
            plt.scatter(samples[:,X], samples[:,Y], color='k', s=10, alpha=0.5)
            
            if plotSampleID:
                for i,sample in enumerate(samples):
                    
                    plt.text(samples[i,X], samples[i,Y], i, fontsize=6, color='r')
        
    plt.xlabel(prop_names[X])
    plt.ylabel(prop_names[Y])

    ax.set_title("Solution sets (confidence beta={}; N={} samples)".
                 format(beta, len(samples)))
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)


def plot_solution_set_2d(idxs, prop_names, regions, samples, beta,
                         plotSamples=True, plotSampleID = True):
    
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
        
        # Check if imprecise samples are used
        if samples.ndim == 3:
            # If imprecise, plot as boxes
            for i,sample in enumerate(samples):
                s_low = sample[:,0]
                s_upp = sample[:,1]
                diff  = s_upp - s_low
                
                rect = patches.Rectangle(s_low[[X,Y]], diff[X], diff[Y], 
                                         linewidth=0.5, edgecolor='red', 
                                         linestyle='dashed', facecolor='none')
                
                # Add the patch to the Axes
                ax.add_patch(rect)
                
                if plotSampleID:
                    plt.text(s_upp[X], s_upp[Y], i, fontsize=6, color='r')
            
        else:
            # Else, plot as points
            plt.scatter(samples[:,X], samples[:,Y], color='k', s=10, alpha=0.5)
            
            if plotSampleID:
                for i,sample in enumerate(samples):
                    
                    plt.text(samples[i,X], samples[i,Y], i, fontsize=6, color='r')
        
    plt.xlabel(prop_names[X])
    plt.ylabel(prop_names[Y])

    ax.set_title("Solution sets (confidence beta={}; N={} samples)".
                 format(beta, len(samples)))
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)