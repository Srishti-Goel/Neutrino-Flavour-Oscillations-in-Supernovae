import numpy as np
import matplotlib.pyplot as plt

legends = [r'$\alpha_e$', r'$\alpha_x$', r'$\alpha_{xb}$', r'$F_{\nu_e}$', r'$F_{\bar\nu_e}$', r'$F_{\nu_x}$', r'$F_{\bar\nu_x}$']

def plot_misclassifications(x, y_preds, y_true, x_axis_idx, y_axis_idx, other_component_vals = None, legends = legends):
    """
    Plots the mis-classified points in parameter space.

    Parameters:
    x (ndarray)         : The list of inputs
    y_preds (ndarray)   : The list of predicted classes
    y_true (ndarray)    : The list of true classes
    x_axis_idx (int)    : The component to be plotted on X-axis
    y_axis_idx (int)    : The component to be plotted on Y-axis
    other_component_vals (list | None): The value of the other components (in order)
                        : Value None projects the misclassifications onto the 2D space (x_axis_idx,y_axis_idx)
    legends (list)      : The names of the components
    """
    
    fig = plt.figure(figsize=(4,4))


    if other_component_vals is None:
        points_misclassified = abs(y_preds - y_true) > 1e-3
        plt.scatter(x[points_misclassified, x_axis_idx], 
                    x[points_misclassified, y_axis_idx], 
                    marker = 'o', cmap=plt.cm.Set1, 
                    edgecolor='k', s = 20, alpha = 1 
                )
    else:
        for i in range(len(y_preds)):
            if abs(y_preds[i] - y_true[i]) > 1e-3:
                k = 0
                plot_point = True
                for j in range(len(legends)):
                    if j != x_axis_idx and j != y_axis_idx:
                        if abs(x[i, j] - other_component_vals[k]) > 1e-3:
                            plot_point = False
                            break
                        k += 1
                if plot_point:
                    plt.scatter(x[i, x_axis_idx],
                    x[points_misclassified, y_axis_idx], 
                    marker = 'o', cmap=plt.cm.Set1, 
                    edgecolor='k', s = 20, alpha = 1 
                )
    plt.title('Misclassifications', size=20)
    plt.xlabel(legends[x_axis_idx],size=18)
    plt.ylabel(legends[y_axis_idx],size=18)
    plt.minorticks_on()
    plt.tick_params(axis='y',which='both',left=True,right=True,labelleft=True, direction='in')
    plt.tick_params(axis='x',which='both',top=True,bottom=True,labelbottom=True, direction='in')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.0001,1.)
    plt.xlim(-0.0001,1.)
    plt.show()


def plot_reduced_data(x_reduced, y, x_axis_idx, y_axis_idx, reduction_name):

    """
    Plots the data after Dimensionality Reduction

    Parameters:
    x_reduced (ndarray) : The data-points to be plotted in the reduced parameter space
    y (ndarray)         : The classes of the various data-points
    x_axis_idx (int)    : The component to be plotted on X-axis
    y_axis_idx (int)    : The component to be plotted on Y-axis
    reduction_name (str): The name of the dimensionality reduction technique
                        : Used in labelling only


    """
    fig = plt.figure(figsize=(6,6))
    plt.scatter(x_reduced[:, x_axis_idx], x_reduced[:, y_axis_idx], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(reduction_name + ' Component ' + str(x_axis_idx + 1))
    plt.ylabel(reduction_name + ' Component ' + str(y_axis_idx + 1))
    plt.title(reduction_name + ' Dimensionality Reduction')
    plt.show()

def make_subplots(data_gaussian, data_max_ent, legend=False, x_idx = 3, y_idx = 4, other_points = [.5,.5,.5,.5,.5,.5], only_first_class = True):
    """ 
    Makes a scatter plot of where crossings exist
    """

    N_POINTS = max(len(data_gaussian), len(data_max_ent))
    for i in range(0, N_POINTS):

        # Making sure the other elements are in the right order
        j = -1

        data_max_plot = True
        data_gauss_plot = True
        for k in range(len(legends)):
            if k != x_idx and k != y_idx:
                j += 1
                if abs(data_max_ent[i,k] - other_points[j]) > 0.5:
                    data_max_plot = False
                
                if abs(data_gaussian[i,k] - other_points[j]) > 0.5:
                    data_gauss_plot = False

        if only_first_class:
            if data_max_plot and data_max_ent[i,-1] == 1:
                plt.scatter(data_max_ent[ i,x_idx], data_max_ent[ i,y_idx], marker='o', rasterized=True, label='',color='tomato',s=10, alpha=1)
            
            if data_gauss_plot and data_gaussian[i,-1] == 1:
                plt.scatter(data_gaussian[ i,x_idx], data_gaussian[ i,y_idx], marker='x', rasterized=True, label='',color='darkblue',s=10, alpha=1)
        else:
            if data_max_plot and data_max_ent[i,-1] > 0:
                plt.scatter(data_max_ent[ i,x_idx], data_max_ent[ i,y_idx], marker='o', rasterized=True, label='',color='tomato',s=10, alpha=1)
            
            if data_gauss_plot and data_gaussian[i,-1] > 0:
                plt.scatter(data_gaussian[ i,x_idx], data_gaussian[ i,y_idx], marker='x', rasterized=True, label='',color='darkblue',s=10, alpha=1)
    
    
    plt.xlabel(legends[x_idx],size=18)
    plt.ylabel(legends[y_idx],size=18)
    plt.minorticks_on()
    plt.tick_params(axis='y',which='both',left=True,right=True,labelleft=True, direction='in')
    plt.tick_params(axis='x',which='both',top=True,bottom=True,labelbottom=True, direction='in')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.0001,1.)
    plt.xlim(-0.0001,1.)
    
    if legend: plt.legend(("max-ent","Gaussian"), loc = 'lower right', fontsize=14, fancybox=True, framealpha=1.)