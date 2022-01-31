import numpy as np
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
import itertools
from scipy.spatial import HalfspaceIntersection, ConvexHull

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

from slurf.commons import path, append_new_line

def plot_results(output_dir, args, regions, solutions, file_suffix=None):
    
    reliability = args.reliability
    prop_labels = args.prop_labels
    timebounds  = args.timebounds
    
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
        
        if args.plot_timebounds:
            
            idx_pair   = (timebounds.index(args.plot_timebounds[0]),
                          timebounds.index(args.plot_timebounds[1]))
            
            idx_prop = [prop_labels[idx_pair[0]]] + [prop_labels[idx_pair[1]]]
        
            # Plot 2D confidence regions
            plot_2D(args, idx_pair, idx_prop, regions, solutions, 
                    regions, plotSamples=True, plotSampleID=False)
        
            exp_file = args.modelfile
            if file_suffix != None:
                exp_file += "_" + str(file_suffix)
            exp_file += '.pdf'   
            filename = path(output_dir, "", exp_file)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            
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
                
                idx_prop = [prop_labels[idx_pair[0]]] + [prop_labels[idx_pair[1]]]
                
                # Plot 2D confidence regions
                plot_2D(args, idx_pair, idx_prop, regions, 
                        solutions, R, plotSamples=True, plotSampleID=False)
        
                exp_file = args.modelfile + "_" + str(idx_pair)
                if R != None:
                    exp_file += "_rho=" + '{:0.4f}'.format(regions[R]['rho'])
                exp_file += '.pdf'    
                
                filename = path(output_dir, "", exp_file)
                plt.savefig(filename, format='pdf', bbox_inches='tight')
                print(' - 2D plot exported to:',exp_file)
                
                plt.show()
                
            # Plot 2D confidence regions
            plot_2D(args, idx_pair, idx_prop, regions, 
                    solutions, plotSamples=True, plotSampleID=False)
    
            exp_file = args.modelfile + "_" + str(idx_pair) + '.pdf'    
            
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


class Cases():
    
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
    
    def init_eta(self, eta_csv, eta_cols):
        
        self.eta_csv = path(self.root_dir, 'output', eta_csv)
        self.eta_df = pd.DataFrame(columns = eta_cols)
        
        append_new_line(self.eta_csv, ';'.join(self.eta_df.columns))
        
    def add_eta_row(self, q, args, regions, emp_satprob):
        
        for i, region in regions.items():
            
            self.eta_df.loc[q, 'N'] = args.Nsamples
            self.eta_df.loc[q, '#'] = q
            self.eta_df.loc[q, 'seed'] = args.seeds
            self.eta_df.loc[q, 'rho'] = np.round(region['rho'], 6)
            
            betas = list(map(str, args.beta))
            self.eta_df.loc[q, betas] = np.round(list(region['eta_series']), 6)
            
            self.eta_df.loc[q, 'empirical'] = emp_satprob[i]
                
            row = self.eta_df.loc[q]
            
            # Write row to file
            append_new_line(self.eta_csv, ';'.join(map(str, row)))
        
    def init_time(self, time_csv, time_cols):
        
        self.time_csv = path(self.root_dir, 'output', time_csv)
        self.time_df = pd.DataFrame(columns = time_cols)
        self.time_df.index.name = '#'
        
    def add_time_row(self, args, time):
        
        self.time_df.loc[args.seeds, str(args.Nsamples)] = np.round(time, 5)
        
    def write_time(self):
        
        # Write DataFrame to file
        self.time_df.to_csv(self.time_csv, sep=';')


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
                     mode='conservative', annotate=False, title=False):
    
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
        
    if plotSamples and len(samples.shape) == 2:
        plt.plot(timebounds[:-1], samples.T[:-1], color='k', lw=0.3, ls='dotted', alpha=0.5)
        
    plt.xlabel('Time', fontsize=24)
    plt.ylabel('Value', fontsize=24)
    
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    if title:
        ax.set_title("Confidence regions over time (beta={}; N={})".
                 format(beta, len(samples)), fontsize=22)
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0,1))
    sm.set_array([])
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.06, ax.get_position().height])
    ax.figure.colorbar(sm, cax=cax)
    

def plot_2D(args, idxs, prop_labels, regions, samples, R=None,
            plotSamples=True, plotSampleID=False, title=False):
    
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
            feasible_point = item['x_low'][[X,Y]] + 1e-3
            poly_vertices = HalfspaceIntersection(item['halfspaces'], 
                                                  feasible_point).intersections
            hull = ConvexHull(poly_vertices) 
            vertices = hull.points[hull.vertices]
            
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
                
                # Still plot as point if the samples is fully refined
                if all(diff < 1e-3):
                    plt.scatter(s_upp[X], s_upp[Y], color='k', s=10, alpha=0.5)
                    
                else:
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
        
    plt.xlabel(prop_labels[0], fontsize=24)
    plt.ylabel(prop_labels[1], fontsize=24)

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