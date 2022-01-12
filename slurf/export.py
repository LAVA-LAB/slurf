import numpy as np
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
import itertools
from scipy.spatial import HalfspaceIntersection, ConvexHull

from slurf.commons import path

def plot_results(output_dir, args, regions, solutions, reliability,
                 prop_labels=None, timebounds=None):
    
    # Plot the solution set
    if reliability:
        # As reliability over time (if properties object is a tuple)
        plot_reliability(timebounds, regions, solutions, args.beta2plot, 
                         mode=args.curve_plot_mode, plotSamples=False)
        
        # Save figure
        exp_file = args.modelfile + '.pdf'
        filename = path(output_dir, "", exp_file)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(' - Reliability plot exported to:',exp_file)
        
        plt.show()
        
    else:
        # As a solution set (if properties object is a list of properties)    
        for idx_pair in itertools.combinations(np.arange(len(prop_labels)), 2):
            # Plot the solution set for every combination of 2 properties
            
            if args.pareto_pieces > 0:
                region_list = list(regions.keys())
            else:
                region_list = [None]
            for R in region_list:
            
                # Plot 2D confidence regions
                plot_2D(args, idx_pair, prop_labels, regions, solutions, 
                        R, plotSamples=True, plotSampleID=True)
        
                exp_file = args.modelfile + "_" + str(idx_pair)
                if R != None:
                    exp_file += "_rho=" + '{:0.4f}'.format(regions[R]['rho'])
                exp_file += '.pdf'    
                
                filename = path(output_dir, "", exp_file)
                plt.savefig(filename, format='pdf', bbox_inches='tight')
                print(' - 2D plot exported to:',exp_file)
                
                plt.show()


def save_results(output_path, dfs, modelfile_nosuffix, N):
    '''
    Export the results of the current execution to an Excel file
    '''
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    xlsx_file = modelfile_nosuffix + "_N=" + str(N) + "_results.xlsx"
    xlsx_path = path(output_path, "", xlsx_file)
    writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')
    
    # Write each dataframe to a different worksheet.
    for name, df in dfs.items():
        df.to_excel(writer, sheet_name=str(name))
        
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    print('- Results exported to:',xlsx_path)
    
    return


def save_validation_results(args, output_path, val_dfs):
    '''
    Export the average empirical validation results into a .csv file
    '''
    
    for rho, df in val_dfs.items():    
        
        if len(df) < args.repeat:
            continue

        prob_list = list(np.mean(df, axis=0))

        csv_file = 'validation_' + '{:.4f}'.format(rho) + '.csv'
        csv_path = path(output_path,'',csv_file)
        
        with open(csv_path,'w') as file:
            
            prob = ' & '.join('{:.3f}'.format(float(i)) for i in prob_list)
            string = 'N='+str(args.Nsamples)+' & '+prob+' \\\\'
            file.write(string)
            
    print('- Validatoin results exported to:',csv_path)


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
        
        color = color_map(item['satprob_beta='+str(beta)])
        
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
            
            plt.annotate(r'$\eta=$'+str(np.round(item['satprob_beta='+str(beta)], 2)), 
                         xy=xy, xytext=xytext,
                         ha='left', va='center', textcoords='offset points',
                         arrowprops=dict(arrowstyle="-|>",mutation_scale=12, facecolor='black'),
                         )
        
    if plotSamples:
        plt.plot(timebounds[:-1], samples.T[:-1], color='k', lw=0.3, ls='dotted', alpha=0.5)
        
    plt.xlabel('Time')
    plt.ylabel('Value')

    ax.set_title("Confidence regions over time (beta={}; N={})".
                 format(beta, len(samples)), fontsize=8)
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)
    

def plot_2D(args, idxs, prop_names, regions, samples, R=None,
            plotSamples=True, plotSampleID=True, title=False):
    
    beta = args.beta2plot
    if args.pareto_pieces > 0:
        pareto = True
    else:
        pareto = False
    
    X, Y = idxs
    
    # Create plot
    fig, ax = plt.subplots()

    # Set colors and markers
    color_map = sns.color_palette("Blues_r", as_cmap=True)
    
    for i, item in sorted(regions.items(), reverse=True):
        
        # Check if we should plot this region index
        if type(R)==list and i not in R:
            continue
        if type(R)==int and i != R:
            continue
        
        color = color_map(item['satprob_beta='+str(beta)])
        
        if pareto:
            # Plot Pareto-front confidence region
            
            # Convert halfspaces to verticers of polygon
            feasible_point = item['x_low'][[X,Y]] + 1e-6
            poly_vertices = HalfspaceIntersection(item['halfspaces'], 
                                                  feasible_point).intersections
            hull = ConvexHull(poly_vertices) 
            vertices = hull.points[hull.vertices]
            
            print(vertices)
            
            polygon = patches.Polygon(vertices, True, 
                                  linewidth=0, edgecolor='none', facecolor=color)
            ax.add_patch(polygon)
            
        else:
            # Plot default rectangular confidence region
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
            
            if type(R) == int:
                plt.scatter(samples[regions[R]['critical_set'],X], 
                            samples[regions[R]['critical_set'],Y], 
                            color='r', s=10, alpha=0.8)
                
            if plotSampleID:
                for i,sample in enumerate(samples):
                    
                    plt.text(samples[i,X], samples[i,Y], i, fontsize=6, color='r')
        
    plt.xlabel(prop_names[X], fontsize=24)
    plt.ylabel(prop_names[Y], fontsize=24)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    if pareto:
        plt.gca().set_xlim(left=np.min(samples[:,0]))
        plt.gca().set_ylim(bottom=np.min(samples[:,1]))
        
        plt.xticks([260, 300, 340, 380, 420])

    if title:
        if pareto:
            ax.set_title("Pareto-front (beta={}; N={})".
                     format(beta, len(samples)), fontsize=22)
        else:       
            ax.set_title("Confidence regions (beta={}; N={})".
                     format(beta, len(samples)), fontsize=22)
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    cbar = ax.figure.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=22)