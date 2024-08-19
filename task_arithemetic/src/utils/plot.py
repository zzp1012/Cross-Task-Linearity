import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def scatter_plot_with_regression(y: np.ndarray,
                                 x: np.ndarray, 
                                 save_path: str,
                                 fname: str,
                                 xlabel: str=None,
                                 ylabel: str=None,
                                 title: str=None,
                                 xlim: tuple=None,
                                 ylim: tuple=None,
                                 figsize: tuple=(7, 7)):
    """plot the scatter plot with linear regression

    Args:
        y (np.ndarray): the y values.
        x (np.ndarray): the x values.
        save_path (str): the save path.
        fname (str): the filename.
        xlabel (str, optional): the xlabel. Defaults to None.
        ylabel (str, optional): the ylabel. Defaults to None.
        title (str, optional): the title. Defaults to None.
        xlim (tuple, optional): the xlim. Defaults to None.
        ylim (tuple, optional): the ylim. Defaults to None.
        figsize (tuple, optional): the figsize. Defaults to (7, 7).
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=figsize)
    # Perform linear regression to get slope and intercept
    slope, intercept, _, _, _ = linregress(x, y)

    # Create a scatter plot
    ax.scatter(x, y, label="Data points")

    # Plot the regression line
    regression_line = slope * np.array(x) + intercept
    ax.plot(x, regression_line, color='red', 
            label=f'Linear Regression (slope={slope:.2f}, intercept={intercept:.2f})')

    # Add labels and legend
    ax.grid()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    
    # set the xlim and ylim
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=True, prop={'size': 10})

    # save the fig
    path = os.path.join(save_path, fname)
    fig.savefig(path)
    plt.close()


def plot_multiple_curves(Y: dict,
                         x: np.ndarray,
                         save_path: str, 
                         fname: str,
                         xlabel: str,
                         ylabel: str,
                         title: str = None,
                         ylim: list = None,
                         figsize: tuple = (7, 5)) -> None:
    """plot curves in one figure for each key in dictionary.
    Args:
        Y (dict): the dictionary of the curves.
        x (np.ndarray): the x axis.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        xlabel (str): the label of x axis.
        ylabel (str): the label of y axis.
        title (str): the title of the figure.
        ylim (list): the range of y axis.
        figsize (tuple): the size of the figure.
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw all the lines in the same plot, assigning a label for each one to be
    #   shown in the legend.
    for label, y in Y.items():
        ax.plot(x, y, label=label)
    
    # Add labels and legend
    ax.grid()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=True, prop={'size': 10})
    
    # set the ylim
    plt.ylim(ylim)
    
    # save the fig
    path = os.path.join(save_path, fname)
    fig.savefig(path)
    plt.close()