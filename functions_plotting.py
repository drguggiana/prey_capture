import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv
import sklearn.cluster as clu
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from matplotlib import animation, ticker
from bokeh.themes.theme import Theme
from bokeh.plotting import show as bokeh_show

import processing_parameters


# define in to cm
constant_in2cm = 2.54

# Define standard colors for holoviews
hv_blue_rgb = tuple(np.array([42, 158, 210]) / 255.)
hv_blue_hex = '#2a9ed2'
hv_orange_rgb = tuple(np.array([255, 191, 134]) / 255.)
hv_orange_hex = '#ffbf86'
hv_mpi_green_rgb = tuple(np.array([0, 136, 122]) / 255.)
hv_mpi_green_hex = '#00887a'
hv_mpi_yellow_rgb = tuple(np.array([203, 219, 42]) / 255.)
hv_mpi_yellow_hex = '#cbdb2a'
hv_gray_hex = '#bcbcbc'
hv_yellow_hex = '#fae365'

hv_white_hex = '#FFFFFF'
hv_red_hex = '#db3c30'

violin_palette = {'Freely Moving': hv_red_hex, 'Head Fixed': hv_blue_hex}
swarm_pallete = {'Freely Moving': hv_white_hex, 'Head Fixed': hv_white_hex}  

# define the standard font sizes
paper = '12pt'
poster = '15pt'
screen = '18pt'
small = '7pt'
# define the conversion constant from pt to cm
constant_pt2cm = 0.0352778
# define the font size default dictionary for figures
font_sizes_raw = {
    'paper': {
        'xlabel': paper,
        'ylabel': paper,
        'zlabel': paper,
        'labels': paper,
        'xticks': paper,
        'yticks': paper,
        # 'zticks': small-2,
        'ticks': paper,
        'minor_xticks': paper,
        'minor_yticks': paper,
        'minor_ticks': paper,
        'title': paper,
        # 'legend': small-1,
        'legend_title': paper,
    },
    'poster': {
        'xlabel': poster,
        'ylabel': poster,
        'zlabel': poster,
        'labels': poster,
        'xticks': '13pt',
        'yticks': poster,
        'zticks': poster,
        'ticks': poster,
        'minor_xticks': poster,
        'minor_yticks': poster,
        'minor_ticks': poster,
        'title': poster,
        'legend': '13pt',
        'legend_title': poster,
    },
    'screen': {
        'xlabel': screen,
        'ylabel': screen,
        'zlabel': screen,
        'labels': screen,
        'xticks': screen,
        'yticks': screen,
        'zticks': screen,
        'ticks': screen,
        'minor_xticks': screen,
        'minor_yticks': screen,
        'minor_ticks': screen,
        'title': screen,
        # 'legend': screen - 1,
        'legend_title': screen,
    },
    'small': {
        'xlabel': small,
        'ylabel': small,
        'zlabel': small,
        'labels': small,
        'xticks': small,
        'yticks': small,
        'zticks': small,
        'ticks': small,
        'minor_xticks': small,
        'minor_yticks': small,
        'minor_ticks': small,
        'title': small,
        # 'legend': screen - 1,
        'legend_title': small,
    }
}

attr_dict = {
    'attrs': {
        'Figure': {
            'background_fill_color': '#FFFFFF',
            'border_fill_color': '#FFFFFF',
            'outline_line_color': None,
        },
        # 'Grid': {
        #     'grid_line_dash': [6, 4],
        #     'grid_line_alpha': .3,
        # },
        'Text':
            {
                'text_font': 'arial',
                'text_color': 'black',
                'text_font_style': 'normal',
                'text_alpha': 1.0,
                # 'text_font_size': 20,
            },

        'Axis': {
            'major_label_text_color': 'black',
            'axis_label_text_color': 'black',
            'major_tick_line_color': 'black',
            'axis_line_color': "black",
            'axis_line_width': 2,
            'axis_line_cap': 'round',
            'axis_label_text_font_size': '18pt',
            'major_label_text_font_size': '18pt',
            'minor_tick_line_color': None,
            'major_tick_in': 0,
            'major_tick_line_cap': 'round',
            'major_label_text_font_style': 'normal',
            'axis_label_text_font_style': 'normal',
        },
        'Legend': {
            'label_text_font_size': '18pt',
        },
        'Title': {
            'text_color': 'black',
        },
    }
}


def set_theme():
    """set the default theme for the figures"""
    theme = Theme(
        json=attr_dict
    )

    hv.renderer('bokeh').theme = theme
    return theme


def plot_2d(data_in, rows=1, columns=1, labels=None, markers=None, linestyle='-', color=None,
            xerr=None, yerr=None, fig=None, fontsize=None, dpi=None, **kwargs):
    """Wrapper for 2D plotting data into subplots"""
    # create a new figure window
    if fig is None:
        if dpi is None:
            dpi = 300
        else:
            dpi = dpi
        fig = plt.figure(dpi=dpi)
    # initialize a plot counter
    plot_counter = 1
    # for all the rows
    for row in range(rows):
        # for all the columns
        for col in range(columns):
            # if there's no plot in this position, skip it
            if len(data_in) < plot_counter:
                continue
            # add the subplot
            ax = fig.add_subplot(rows, columns, plot_counter)
            # for all the lines in the list
            for count, lines in enumerate(data_in[plot_counter - 1]):
                if color is not None:
                    c = color[plot_counter - 1]
                else:
                    c = 'b'
                # plot x,y ot just y depending on the size of the data
                if len(lines.shape) == 2:
                    # line2d = ax.plot(lines[:, 0], lines[:, 1])
                    if xerr is not None:
                        if yerr is not None:
                            line2d = ax.errorbar(lines[:, 0], lines[:, 1], xerr=xerr[plot_counter - 1][count]
                                                 , yerr=yerr[plot_counter - 1][count])
                        else:
                            line2d = ax.errorbar(lines[:, 0], lines[:, 1], xerr=xerr[plot_counter - 1][count]
                                                 , yerr=None)
                    else:
                        if yerr is not None:
                            # line2d = ax.errorbar(range(lines.shape[0]), lines, xerr=None
                            #                      , yerr=yerr[plot_counter - 1][count])

                            line2d = ax.plot(lines[:, 0], lines[:, 1], color=c)
                            y_error = yerr[plot_counter - 1][count]
                            ax.fill_between(lines[:, 0], lines[:, 1]-y_error, lines[:, 1]+y_error, alpha=0.5, color=c)
                        else:
                            line2d = ax.errorbar(lines[:, 0], lines[:, 1], xerr=None
                                                 , yerr=None)
                else:
                    if xerr is not None:
                        if yerr is not None:
                            line2d = ax.errorbar(range(lines.shape[0]), lines, xerr=xerr[plot_counter - 1][count]
                                                 , yerr=yerr[plot_counter - 1][count])
                        else:
                            line2d = ax.errorbar(range(lines.shape[0]), lines, xerr=xerr[plot_counter - 1][count]
                                                 , yerr=None)
                    else:
                        if yerr is not None:
                            # line2d = ax.errorbar(range(lines.shape[0]), lines, xerr=None
                            #                      , yerr=yerr[plot_counter - 1][count])

                            line2d = ax.plot(range(lines.shape[0]), lines, color=c)
                            y_error = yerr[plot_counter - 1][count]
                            ax.fill_between(range(lines.shape[0]), lines-y_error, lines+y_error, alpha=0.5, color=c)
                        else:
                            line2d = ax.errorbar(range(lines.shape[0]), lines, xerr=None
                                                 , yerr=None)

                # change the marker if provided, otherwise use dots
                if markers is not None:
                    line2d[0].set_marker(markers[plot_counter - 1][count])
                else:
                    line2d[0].set_marker('.')
                if linestyle != '-':
                    line2d[0].set_linestyle(linestyle[plot_counter - 1][count])
                # change the font size if provided
                if fontsize is not None:
                    ax[0].set_fontsize(fontsize[plot_counter - 1][count])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            # add labels if provided
            if labels is not None:
                plt.legend(labels[plot_counter - 1])

            # apply kwargs
            # TODO: implement kwargs
            # update the plot counter
            plot_counter += 1
    return fig


def plot_3d(data_in, fig=None):
    """Wrapper for 3D plotting data"""
    # create a new figure window
    if fig is None:
        fig = plt.figure()
    # add the subplot
    ax = fig.add_subplot(111, projection='3d')
    # for all the lines in the list
    for lines in data_in:
        ax.plot(lines[:, 0], lines[:, 1], lines[:, 2], marker='.')
    return fig


def animation_plotter(motivedata, bonsaidata, cricket_data, xlim, ylim, interval=10):
    """Plot animations from the motive, bonsai and cricket"""
    # TODO: generalize function to any number of lines to plot
    # First set up the figure, the axis, and the plot element we want to animate
    fig0 = plt.figure()
    ax0 = plt.axes(xlim=xlim, ylim=ylim)
    line0, = ax0.plot([], [], lw=2)
    line1, = ax0.plot([], [], lw=2)
    line2, = ax0.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line0.set_data([], [])
        line1.set_data([], [])
        line2.set_data([], [])
        return line0, line1, line2

    # animation function.  This is called sequentially
    def animate(i):
        # x = np.linspace(0, 2, 1000)
        # y = np.sin(2 * np.pi * (x - 0.01 * i))

        line0.set_data(motivedata[:i, 0], motivedata[:i, 1])
        line1.set_data(bonsaidata[:i, 0], bonsaidata[:i, 1])
        line2.set_data(cricket_data[:i, 0], cricket_data[:i, 1])

        return line0, line1, line2

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig0, animate, init_func=init,
                                   frames=motivedata.shape[0], interval=interval, blit=True)

    return anim


def simple_animation(data_in, interval=10):
    """Animate the trajectories given"""
    # First set up the figure, the axis, and the plot element we want to animate
    fig0 = plt.figure()
    # ax0 = plt.axes(xlim=xlim, ylim=ylim)
    xlim = [np.min(data_in[[0, 2, 4, 6, 8], :]), np.max(data_in[[0, 2, 4, 6, 8], :])]
    ylim = [np.min(data_in[[1, 3, 5, 7, 9], :]), np.max(data_in[[1, 3, 5, 7, 9], :])]
    ax0 = plt.axes(xlim=xlim, ylim=ylim)
    # line_list = [ax0.plot([], [], lw=2)[0] for el in np.arange(data_in.shape[0])]
    line0, = ax0.plot([], [], 'bo', lw=2)
    line1, = ax0.plot([], [], 'ko', lw=2)
    line2, = ax0.plot([], [], 'mo', lw=2)
    line3, = ax0.plot([], [], 'go', lw=2)
    line4, = ax0.plot([], [], 'ro', lw=2)

    # initialization function: plot the background of each frame
    def init():
        line0.set_data([], [])
        line1.set_data([], [])
        line2.set_data([], [])
        # line_list = [el.set_data([], []) for el in line_list]
        return line0, line1, line2, line3, line4
        # return line_list

    # animation function.  This is called sequentially
    def animate(i):
        # x = np.linspace(0, 2, 1000)
        # y = np.sin(2 * np.pi * (x - 0.01 * i))

        line0.set_data(data_in[0, i], data_in[1, i])
        line1.set_data(data_in[2, i], data_in[3, i])
        line2.set_data(data_in[4, i], data_in[5, i])
        line3.set_data(data_in[6, i], data_in[7, i])
        line4.set_data(data_in[8, i], data_in[9, i])

        return line0, line1, line2, line3, line4

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig0, animate, init_func=init,
                                   frames=data_in.shape[1], interval=interval, blit=True)
    return anim


def spike_raster(data_in, cells=None):
    if cells is None:
        cells = [el for el in data_in.columns if 'cell' in el]

    spikes = data_in.loc[:, cells]

    im = hv.Image((data_in.time_vector, np.arange(len(cells)), spikes.values.T),
                  kdims=['Time (s)', 'Cells'], vdims=['Activity (a.u.)'])
    im.opts(width=600)  # , cmap='Purples')
    return im


def trace_raster(data, cells=None, ds_factor=1):
    if cells is None:
        cells = [el for el in data.columns if 'cell' in el]

    trace = data.loc[:, cells]
    max_std = trace.std().max()

    lines = {i: hv.Curve((data.time_vector, trace.iloc[:, i].values.T + i * max_std)) for i in np.arange(len(cells))}
    lineoverlay = hv.NdOverlay(lines, kdims=['Time (s)']).opts(height=500, width=800)
    return lineoverlay


def rand_jitter(arr):
    stdev = .001 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot_tuning_curve(tuning_curve, error, plot_fit=True, plot_trials=False, plot_pref_angle=False,
                      fit=None, trials=None, pref_angle=None, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure(dpi=300, figsize=(10, 6))
        ax = fig.add_subplot(111)

    tuning = ax.errorbar(tuning_curve[:, 0], tuning_curve[:, 1],
                         c='k', alpha=1, yerr=error, elinewidth=1,
                         **kwargs)

    if (fit is not None) and plot_fit:
        ax.plot(fit[:, 0], fit[:, 1], c='#1f77b4')

    if (pref_angle is not None) and plot_pref_angle:
        ax.axvline(pref_angle, color='r', linewidth=1)

    if (trials is not None) and plot_trials:
        ax.scatter(rand_jitter(trials[:, 0]), trials[:, 1], marker='.', c='k', alpha=0.5, edgecolor='none')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    return tuning


def plot_polar_tuning_curve(tuning_curve, error, plot_fit=True, plot_trials=False, plot_pref_angle=True,
                            fit=None, trials=None, pref_angle=None, ax=None, **kwargs):
    theta_max = kwargs.pop('theta_max', 360)
    font_size = kwargs.pop('font_size', 'poster')
    font_size = int(font_sizes_raw[font_size]['xlabel'][:-2])

    if ax is None:
        fig = plt.figure(dpi=300, figsize=(5, 5))
        ax = fig.add_subplot(111, projection='polar')
        ax.grid(linewidth=0.25, zorder=0)

    ax.errorbar(np.deg2rad(tuning_curve[:, 0]), tuning_curve[:, 1],
                c='k', alpha=1, linewidth=2, yerr=error, elinewidth=1,
                zorder=2, **kwargs)

    if (fit is not None) and plot_fit:
        ax.plot(np.deg2rad(fit[:, 0]), fit[:, 1], c='#1f77b4', linewidth=1, zorder=2.5)

    if (trials is not None) and plot_trials:
        ax.scatter(rand_jitter(np.deg2rad(trials[:, 0])), trials[:, 1], 
                   marker='.', color='k', alpha=0.5, edgecolor='none',
                   zorder=2)

    if (pref_angle is not None) and plot_pref_angle:
        ax.axvline(np.deg2rad(pref_angle), color='r', linewidth=1, zorder=2)

    ax.set_thetamax(theta_max)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    # ax.set_rorigin(0)
    radial_ticks = [0.0, 0.5,1.0]
    ax.set_rticks(radial_ticks, color='black')
    ax.set_yticklabels([], color='black')
    ax.set_rlabel_position(0)
    # ax.set_xticklabels(['0', '', '90', '', '180', '', '270', ''], color='black', fontsize=font_size)
    ax.set_xticklabels([], color='black', fontsize=font_size)

    # ax.yaxis.set_label_position('right')

    return ax


def plot_tuning_curve_hv(tuning_curve, error, fit=None, trials=None, pref_angle=None, **kwargs):
    overlay = []

    tuning = hv.Curve(tuning_curve).opts(width=600, height=300, **kwargs)
    overlay.append(tuning)

    error_plot = hv.Spread((*tuning_curve.T, error)).opts(fill_alpha=0.25)
    overlay.append(error_plot)

    if fit is not None:
        fit_plot = hv.Curve(fit)
        overlay.append(fit_plot)

    if trials is not None:
        trials_plot = hv.Scatter(trials).opts(color='k', size=3)
        overlay.append(trials_plot)

    if pref_angle is not None:
        pref_plot = hv.VLine(pref_angle).opts(color='k', line_width=1)
        overlay.append(pref_plot)

    return hv.Overlay(overlay)


def plot_tuning_with_stats(dataset, cell, tuning_kind='direction', error='std', 
                           norm=True, polar=True, subfig=None,
                           plot_selectivity=True, plot_gof=False,
                           **kwargs):

    num_subplots = 1 + plot_selectivity + plot_gof
    
    data_cols = ['mean', error, 'resp']
    fit_cols = ['fit_curve', 'pref']

    if 'direction' in tuning_kind:
        multiplier = 1.
    else:
        multiplier = 2.

    if norm:
        data_cols = [el + '_norm' for el in data_cols]

    columns = data_cols + fit_cols
    columns = [el + '_' + tuning_kind[:3] for el in columns]

    if subfig is None:
        figsize = kwargs.pop('figsize', (5 * constant_in2cm, 5 * constant_in2cm))
        fig = plt.figure(layout='constrained', figsize=figsize)
        fig.suptitle(f"Cell {cell}", fontsize='x-large')
        subfig = fig.subfigures(nrows=1, ncols=1)

    if polar:
        ax1 = subfig.add_subplot(int(f'1{num_subplots}1'), projection="polar")  # tuning
    else:
        ax1 = subfig.add_subplot(int(f'1{num_subplots}1'))  # tuning

    axes = [ax1]

    # Plot the TC
    if polar:
        tuning_plot_func = plot_polar_tuning_curve
    else:
        tuning_plot_func = plot_tuning_curve

    ds = dataset.loc[cell, :]
    plot_kwargs = {'theta_max': 360 / multiplier}
    plot_kwargs.update(kwargs)
    # Plot directions
    _ = tuning_plot_func(ds[columns[0]], ds[columns[1]],
                         trials=ds[columns[2]],
                         fit=ds[columns[3]],
                         pref_angle=ds[columns[4]],
                         ax=ax1,
                         **plot_kwargs
                         )

    # Plot dsi or osi
    if plot_selectivity:
        ax2 = subfig.add_subplot(int(f'1{num_subplots}2'))  # resp
        axes.append(ax2)
        if 'direction' in tuning_kind:
            si = 'dsi_abs'
            title = 'DSI'
            xlims = (-0.05, 1.001)
            step = 0.025
        else:
            si = 'osi'
            title = 'OSI'
            xlims = (-0.05, 1.001)
            step = 0.025

        hist_resp = np.abs(ds[f'bootstrap_{si}'])
        hist_resp[np.isnan(hist_resp)] = -0.05
        real_resp = ds[si]
        p_resp = ds[f'bootstrap_p_{si}']
        edges = np.arange(*xlims, step)
        ax2.hist(hist_resp, bins=edges, edgecolor="black", color=hv_blue_rgb)
        ax2.axvline(x=real_resp, color='r', linestyle='dashed', linewidth=2)
        # axes[1].text(1.0, 1.0, f"%ile={p_resp: .2f}", size=10, ha='right', va='bottom', transform=ax2.transAxes)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.set_title(title)

    # # Plot goodness of fit
    # if plot_gof:
        # ax3 = subfig.add_subplot(int(f'1{num_subplots}3'))  # resp
        # axes.append(ax3)
        # hist_gof = ds['bootstrap_gof']
        # hist_gof[np.isnan(hist_gof)] = -0.1
        # real_gof = ds['gof']
        # p_gof = ds['p_gof']
        # edges = np.arange(-0.1, 1.001, 0.025)
        # ax3.hist(hist_gof, bins=edges, edgecolor="black", color=holoviews_blue_rgb)
        # ax3.axvline(x=real_gof, color='r', linestyle='dashed', linewidth=2)
        # # ax3.text(1.0, 1.0, f"%ile={p_gof: .2f}", size=10, ha='right', va='bottom', transform=ax3.transAxes)
        # ax3.spines['right'].set_visible(False)
        # ax3.spines['top'].set_visible(False)
        # ax3.xaxis.set_ticks_position('bottom')
        # ax3.yaxis.set_ticks_position('left')
        # ax3.set_title(gof_type.upper())

    return axes


def histogram(data_in, rows=1, columns=1, bins=50, fig=None, color=None, fontsize=None, dpi=None):
    """Wrapper for the histogram function in subplots"""
    # create a new figure window
    if fig is None:
        if dpi is None:
            dpi = 300
        else:
            dpi = dpi
        fig = plt.figure(dpi=dpi)
    # initialize a plot counter
    plot_counter = 1
    # for all the rows
    for row in range(rows):
        # for all the columns
        for col in range(columns):
            # add the subplot
            ax = fig.add_subplot(rows, columns, plot_counter)
            # for all the lines in the list
            for count, lines in enumerate(data_in[plot_counter - 1]):
                if color is not None:
                    fc = color[plot_counter - 1]
                else:
                    fc = 'b'
                # plot the histogram of the data
                ax.hist(lines, bins=bins, density=True, alpha=0.5, fc=fc)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                # change the font size if provided
                if fontsize is not None:
                    ax.set_fontsize(fontsize[plot_counter - 1][count])
            # update the plot counter
            plot_counter += 1
    return fig


def violin_swarm(ds, save_path, backend='hvplot', save=False,
                 cmap='blue', xlabel='', ylabel='',
                 width=1500, height=1000, 
                 font_size='screen', dpi=800, **kwargs):
    """
    Generate a violin plot with swarm plot overlay using either hvplot or seaborn backend.

    Parameters:
    - ds (pandas.DataFrame): The input dataset.
    - save_path (str): The path to save the generated plot.
    - backend (str, optional): The plotting backend to use. Default is 'hvplot'.
    - save (bool, optional): Whether to save the plot. Default is False.
    - cmap (str, optional): The color map to use for the violin plot. Default is 'blue'.
    - xlabel (str, optional): The label for the x-axis. Default is an empty string.
    - ylabel (str, optional): The label for the y-axis. Default is an empty string.
    - width (int, optional): The width of the plot in pixels. Default is 1500.
    - height (int, optional): The height of the plot in pixels. Default is 1000.
    - font_size (str, optional): The font size for the axis labels. Default is 'screen'.
    - dpi (int, optional): The DPI (dots per inch) for saving the plot. Default is 800.

    Returns:
    - violinplot (object): The generated violin plot.

    Raises:
    - Exception: If an invalid backend is specified.
    """

    ylim = kwargs.pop('ylim', (-0.05, 1.05))

    rename_dict = dict(zip(list(ds.columns), [processing_parameters.wf_label_dictionary_wo_units[col] for col in list(ds.columns)]))
    ds = ds.rename(columns=rename_dict)

    if backend == 'hvplot':
        violinplot = ds[list(rename_dict.values())].hvplot.violin(legend=False, inner='quartiles', color=cmap)
        violinplot.opts(xlabel=xlabel, ylabel=ylabel, ylim=ylim, xrotation=45, width=width, height=height)
        if save:
            violinplot = save_figure(violinplot, save_path=save_path, fig_width=width, dpi=dpi, fontsize='screen',
                                     target='both', display_factor=0.1)
        else:
            violinplot = save_figure(violinplot, save_path=save_path, fig_width=width, dpi=dpi, fontsize='screen',
                                     target='screen', display_factor=0.1)
        return violinplot

    # elif backend=='holoviews':
    
    elif backend == 'seaborn':
        swarm_palette = {k: 'k' for k in rename_dict.values()}
        fig, ax = plt.subplots(figsize=(width, height))
        violinplot = sns.violinplot(data=ds[list(rename_dict.values())], color=cmap, native_scale=True, width=1)
        violinplot = sns.stripplot(data=ds[list(rename_dict.values())], size=2, palette=swarm_palette, marker="x", linewidth=1)
        ax.set_ylim(ylim)

        violinplot.spines[['right', 'top']].set_visible(False)
        font_size = int(font_sizes_raw[font_size]['xlabel'][:-2])

        if len(list(rename_dict.values())) == 1:
            ax.spines[['left']].set_visible(False)
            ax.set(xticklabels=[])
            ax.set(xlabel=None)  # remove the y-axis label
            ax.set(yticklabels=[])
            ax.set(ylabel=None)  # remove the y-axis label
            ax.tick_params(left=False)  # remove the ticks
        else:
            violinplot.set_ylabel(ylabel, fontsize=font_size)
            violinplot.set_xlabel(xlabel, fontsize=font_size)
            plt.xticks(rotation=45)

        plt.tight_layout()

        if save:
            plt.savefig(save_path, dpi=dpi, format='png')

        return violinplot
    else:
        return Exception('Invalid backend')
    

def hv_hist(ds, key, label, drop_na=True, xlabel=''):
    data = ds[key].copy()

    if drop_na:
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        data = data[data >= 0]
        
    frequencies, edges = np.histogram(data, 20)
    hist = hv.Histogram((edges, frequencies), label=label).opts(xlabel=xlabel, ylabel='Freq.')
    return hist


def plot_arrow(trajectory, centers, heading, head, cricket, angles, angles2, fig=None):
    """Draw the animal trajectory and the heading"""
    # create a new figure window
    if fig is None:
        fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(trajectory[:, 0], trajectory[:, 1])

    ax.plot(cricket[:, 0], cricket[:, 1], linestyle='None', marker='o')
    plt.quiver(centers[:, 0], centers[:, 1], heading[:, 0], heading[:, 1], angles=angles, pivot='mid')
    plt.quiver(centers[:, 0], centers[:, 1], heading[:, 0], heading[:, 1], angles=angles2, pivot='mid', color='r')

    plt.axis('equal')

    return fig


def animate_hunt(trajectory, heading, head, cricket, xlim, ylim, interval=10):
    """Animate the full hunting sequence, including movement and head direction"""
    # First set up the figure, the axis, and the plot element we want to animate
    fig0 = plt.figure()
    ax0 = plt.axes(xlim=xlim, ylim=ylim)
    line0, = ax0.plot([], [], lw=2)
    line1, = ax0.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line0.set_data([], [])
        line1.set_data([], [])
        line2 = ax0.quiver(trajectory[0, 0], trajectory[0, 1], np.ones_like(trajectory[0, 0]),
                           np.ones_like(trajectory[0, 0]),
                           angles=heading[0], pivot='mid')
        line3 = ax0.quiver(trajectory[0, 0], trajectory[0, 1], np.ones_like(trajectory[0, 0]),
                           np.ones_like(trajectory[0, 0]),
                           angles=head[0], pivot='tail', color='r')
        return line0, line1, line2, line3

    # animation function.  This is called sequentially
    def animate(i):
        line0.set_data(trajectory[:i, 0], trajectory[:i, 1])
        line1.set_data(cricket[:i, 0], cricket[:i, 1])
        # line2.set_data(trajectory[i, 0], trajectory[i, 1])
        # line2.set_marker((3, 0, heading[i]))
        # line2.set_markersize(20)
        line2 = ax0.quiver(trajectory[i, 0], trajectory[i, 1], np.ones_like(trajectory[i, 0]),
                           np.ones_like(trajectory[i, 0]),
                           angles=heading[i], pivot='mid')
        line3 = ax0.quiver(trajectory[i, 0], trajectory[i, 1], np.ones_like(trajectory[i, 0]),
                           np.ones_like(trajectory[i, 0]),
                           angles=head[i], pivot='tail', color='r')

        return line0, line1, line2, line3

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig0, animate, init_func=init,
                                   frames=trajectory.shape[0], interval=interval, blit=True)
    return anim


def plot_polar(data_in, fig=None, color=None, fontsize=None):
    """Make a polar plot"""
    if fig is None:
        fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='polar')
    array_len = list(range(data_in.shape[0])) + [0]
    array_len = np.array(array_len)
    if color is None:
        color = 'b'
    else:
        color = color
    plt.polar(np.deg2rad(data_in[array_len, 0]), data_in[array_len, 1], color=color)
    ax.tick_params(labelsize=fontsize, pad=15)
    return fig


def plot_image(data_in, rows=1, columns=1, fig=None, colormap=None, colorbar=None, dpi=100):
    """Wrapper for the imshow function in subplots"""
    # create a new figure window
    if fig is None:
        # fig = plt.figure(dpi=300)
        fig, ax = plt.subplots(nrows=rows, ncols=columns, squeeze=False, dpi=dpi)
    else:
        ax = fig.subplots(nrows=rows, ncols=columns, squeeze=False)
    # initialize a plot counter
    plot_counter = 1
    # for all the rows
    for row in range(rows):
        # for all the columns
        for col in range(columns):
            # add the subplot
            # ax = fig.add_subplot(rows, columns, plot_counter)
            # for all the lines in the list
            lines = data_in[plot_counter - 1]
            # plot the data
            im = ax[row, col].imshow(lines, interpolation='nearest', cmap=colormap)
            ax[row, col].set_xlabel('Time')
            ax[row, col].set_ylabel('Trials')
            # if there are colorbars, use them
            if colorbar is not None:
                cbar = fig.colorbar(im, ax=ax[row, col], shrink=0.5)
                cbar.set_label(colorbar[plot_counter - 1], rotation=-90, labelpad=13)

            # update the plot counter
            plot_counter += 1
    return fig


def plot_scatter(data_in, rows=1, columns=1, labels=None, markers=None, color=None,
                 fig=None, fontsize=None, dpi=None):
    """Draw a scatter plot"""
    # create a new figure window
    if fig is None:
        if dpi is None:
            dpi = 300
        else:
            dpi = dpi
        fig = plt.figure(dpi=dpi)
    # initialize a plot counter
    plot_counter = 1
    # for all the rows
    for row in range(rows):
        # for all the columns
        for col in range(columns):
            # if there's no plot in this position, skip it
            if len(data_in) < plot_counter:
                continue
            # add the subplot
            ax = fig.add_subplot(rows, columns, plot_counter)
            # for all the lines in the list
            for count, lines in enumerate(data_in[plot_counter - 1]):
                if color is not None:
                    c = color[plot_counter - 1]
                else:
                    c = 'b'
                # plot x,y ot just y depending on the size of the data
                if len(lines.shape) == 2:
                    # line2d = ax.plot(lines[:, 0], lines[:, 1])
                    scatter2d = ax.scatter(lines[:, 0], lines[:, 1], c=c)
                else:
                    scatter2d = ax.scatter(range(lines.shape[0]), lines, c=c)

                # # change the marker if provided, otherwise use dots
                # if markers is not None:
                #     scatter2d.set_marker(markers[plot_counter - 1][count])
                # else:
                #     scatter2d.set_marker('.')
                # # change the font size if provided
                if fontsize is not None:
                    ax[0].set_fontsize(fontsize[plot_counter - 1][count])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            # add labels if provided
            if labels is not None:
                plt.legend(labels[plot_counter - 1])
            # update the plot counter
            plot_counter += 1
    return fig


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def preprocessing_figure(filtered_traces, real_crickets, vr_crickets, corners):
    """Generate the figure with the preprocessing summary"""
    # save the filtered trace
    fig_final = plt.figure()
    ax = fig_final.add_subplot(111)
    # TODO: define the plotting for the VWHeel trials
    if 'mouse_x' in filtered_traces.columns:

        # plot the filtered trace
        a = ax.scatter(filtered_traces.mouse_x, filtered_traces.mouse_y,
                       c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Blues')
        cbar = fig_final.colorbar(a, ax=ax)
        cbar.set_label('Time (s)')
        ax.axis('equal')

        # for all the real crickets
        for real_cricket in range(real_crickets):
            ax.scatter(filtered_traces['cricket_' + str(real_cricket) + '_x'],
                       filtered_traces['cricket_' + str(real_cricket) + '_y'],
                       c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Oranges')

        # for all the virtual crickets or virtual targets
        for vr_cricket in range(vr_crickets):
            try:
                ax.scatter(filtered_traces['vrcricket_' + str(vr_cricket) + '_x'],
                           filtered_traces['vrcricket_' + str(vr_cricket) + '_y'],
                           c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Greens')
            except:
                ax.scatter(filtered_traces['target_x_m'], filtered_traces['target_y_m'],
                           c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Greens')

    # plot the found corners if existent
    if len(corners) > 0:
        for corner in corners:
            ax.scatter(corner[0], corner[1], c='black')
    return fig_final


def plot_heatmap(values, xlabels, ylabels, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    cmap_min = kwargs.get('cmap_min', 0)
    cmap_max = kwargs.get('cmap_max', np.max(values))

    im = ax.imshow(values, cmap='viridis', vmin=cmap_min, vmax=cmap_max)
    _ = annotate_heatmap(im, valfmt="{x:.2f}")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()

    return im, ax


def show():
    """Wrapper for plt.show"""
    return plt.show()


def sort_traces(data_in, nclusters=10):
    """Sort the traces in a data matrix based on hierarchical clustering"""

    # cluster the data and return the labels
    labels = clu.AgglomerativeClustering(n_clusters=nclusters).fit_predict(data_in)
    # get the sorted idx of labels
    sorted_idx = np.argsort(labels)
    # return the sorted matrix
    return data_in[sorted_idx, :], sorted_idx, labels


def margin(plot, element):
    """Function to prevent me from clipping the xlabel when saving the fig"""
    plot.handles['plot'].min_border_bottom = 50
    plot.handles['plot'].min_border_top = 50
    plot.handles['xaxis'].axis_line_width = 5
    plot.handles['yaxis'].axis_line_width = 5


def cm2pt(cm_value, dpi=600):
    """Function to convert figure sizes in cm to pixels based on a dpi requirement"""
    return int(np.round((cm_value/2.54)*dpi))


def px2pt(px_value, dpi=600, scale_factor=1):
    """Function to convert sizes from pixels to dots at a defined dpi, or just scale"""
    if scale_factor == 1:
        conversion = int(np.round((px_value*constant_pt2cm/2.54)*dpi))
    else:
        conversion = int(np.ceil(px_value * scale_factor))
    return conversion


def search2path(search_string):
    """Turn the input search string into a path entry for figures"""
    search_string = search_string.replace(':', '_')
    search_string = search_string.replace('=', '')
    search_string = search_string.replace(', ', '_')
    return search_string


def format_label(label):
    """Format labels for plots"""
    new_label = label.replace('cricket_0', 'cricket_')
    new_label = new_label.split('_')
    new_label = ' '.join([el.capitalize() for el in new_label])
    return new_label


def get_figure_dimensions(render_fig):
    """Get a rendered figures pixel dimensions"""

    # get the original width and height of the figure
    px_width = render_fig.properties_with_values()['width']
    # set flag for which dim to change later
    flag_width = 'plot'
    # get the frame width if the plot one wasn't defined
    if px_width is None:
        px_width = render_fig.properties_with_values()['frame_width']
        flag_width = 'frame'
    # repeat for height
    px_height = render_fig.properties_with_values()['height']
    flag_height = 'plot'
    if px_height is None:
        px_height = render_fig.properties_with_values()['frame_height']
        flag_height = 'frame'
    return px_width, px_height, flag_width, flag_height


def format_figure(fig, **kwargs):
    """Apply basic figure formattings"""
    # # rotate x axis labels
    # fig.opts(xrotation=45)

    # Format the title and axis labels
    xlabel = str(fig.kdims[0])
    if xlabel != 'x':
        xlabel = format_label(xlabel)
    else:
        xlabel = ''
    fig.opts(xlabel=xlabel)

    # define which label to grab depending on the number of kdims
    if len(fig.kdims) == 1:
        ylabel = str(fig.vdims[0])
    else:
        ylabel = str(fig.kdims[1])
    if ylabel != 'y':
        ylabel = format_label(ylabel)
    else:
        ylabel = ''
    fig.opts(ylabel=ylabel)

    # activate the hover tool
    fig.opts(tools=['hover'])

    # pass the rest of the kwargs
    fig.opts(**kwargs)

    return fig


def format_axis_hook(plot, element, dpi=600, scale_factor=1):
    """Hook to rescale axis components"""

    # get the plot dict
    b = plot.state

    # if it's not an image
    if ('Image' not in element._group_param_value) and ('Raster' not in element._group_param_value):
        # scale the axis line width
        current_axis_width = b.below[0].axis_line_width
        b.below[0].axis_line_width = px2pt(current_axis_width, dpi, scale_factor)
        b.left[0].axis_line_width = px2pt(current_axis_width, dpi, scale_factor)
        # scale the outer tick length
        current_tick_length = b.below[0].major_tick_out
        b.below[0].major_tick_out = px2pt(current_tick_length, dpi, scale_factor)
        b.left[0].major_tick_out = px2pt(current_tick_length, dpi, scale_factor)
        # scale the tick width
        current_tick_width = b.below[0].major_tick_line_width
        b.below[0].major_tick_line_width = px2pt(current_tick_width, dpi, scale_factor)
        b.left[0].major_tick_line_width = px2pt(current_tick_width, dpi, scale_factor)
    else:
        # scale the axis line width
        b.below[0].axis_line_width = 0
        b.left[0].axis_line_width = 0
        b.below[0].major_tick_out = 0
        b.left[0].major_tick_out = 0
        b.below[0].major_tick_line_width = 0
        b.left[0].major_tick_line_width = 0
    # scale the tick standoff
    current_tick_standoff = b.below[0].major_label_standoff
    b.below[0].major_label_standoff = px2pt(current_tick_standoff, dpi, scale_factor)
    b.left[0].major_label_standoff = px2pt(current_tick_standoff, dpi, scale_factor)
    # scale the axis label standoff
    current_label_standoff = b.below[0].axis_label_standoff
    b.below[0].axis_label_standoff = px2pt(current_label_standoff, dpi, scale_factor)
    b.left[0].axis_label_standoff = px2pt(current_label_standoff, dpi, scale_factor)
    plot.outline_line_color = None

    # detect if there's a colorbar
    if len(b.right) > 0:
        if 'ColorBar' in str(type(b.right[0])):
            # scale the distance between labels and bar
            label_standoff = b.right[0].label_standoff
            b.right[0].label_standoff = px2pt(label_standoff, dpi, scale_factor)
            # scale the bar width
            width = b.right[0].width
            if width == 'auto':
                width = current_tick_standoff
            b.right[0].width = px2pt(width, dpi, scale_factor)
            # scale the figure padding so the bar labels don't end up outside the figure
            padding = b.right[0].padding
            b.right[0].padding = px2pt(padding, dpi, scale_factor)
            # remove the tick marks
            b.right[0].major_tick_in = 0
            # remove the border
            b.right[0].bar_line_width = 0
            # scale the standoff
            title_standoff = b.right[0].title_standoff
            b.right[0].title_standoff = px2pt(title_standoff, dpi, scale_factor)
            b.right[0].title_text_baseline = 'middle'

    # # check for boxplot
    # if 'BoxWhisker' in str(type(element)):
    #     print()


def holoviews_mods(figure_in, dpi, scale_factor):
    """Apply plot modifications via holoviews"""
    # select whether image or not
    if ('Image' not in str(type(figure_in))) & ('Raster' not in str(type(figure_in))):
        # get the properties
        props = figure_in.opts.get()[0]
        if 'line_width' in props.keys():
            current_line_width = props['line_width']
            figure_in.opts(line_width=px2pt(current_line_width, dpi, scale_factor))
        # check for scatter
        if 'Scatter' in str(type(figure_in)):
            # also scale the dot size
            dot_size = props['size']
            figure_in.opts(size=px2pt(dot_size, dpi, scale_factor))
        # check for BoxWhisker plot
        if 'BoxWhisker' in str(type(figure_in)):
            box_line_width = props.get('box_line_width', 1)
            figure_in.opts(box_line_width=px2pt(box_line_width, dpi, scale_factor))
            whisker_line_width = props.get('whisker_line_width', 1)
            figure_in.opts(whisker_line_width=px2pt(whisker_line_width, dpi, scale_factor))
            outlier_line_width = props.get('outlier_line_width', 1)
            figure_in.opts(outlier_line_width=px2pt(outlier_line_width, dpi, scale_factor))
        if 'Violin' in str(type(figure_in)):
            violin_line_width = props.get('violin_line_width', 1)
            figure_in.opts(box_line_width=px2pt(violin_line_width, dpi, scale_factor))
            stats_line_width = props.get('stats_line_width', 0.5)
            figure_in.opts(stats_line_width=px2pt(stats_line_width, dpi, scale_factor))
            outline_line_width = props.get('outline_line_width', 1)
            figure_in.opts(box_line_width=px2pt(outline_line_width, dpi, scale_factor))
            box_line_width = props.get('box_line_width', 1)
            figure_in.opts(box_line_width=px2pt(box_line_width, dpi, scale_factor))

    return figure_in


def scale_figure(figure_in, target, dpi, fontsize, mode='convert'):
    """Scale a figure according to its target size"""
    # render the plot as a bokeh element to get the inner features
    render_fig = hv.render(figure_in)
    # get the pixel dimensions
    px_width, px_height, flag_width, flag_height = get_figure_dimensions(render_fig)

    # get their ratio
    h_w_ratio = px_height / px_width

    # scale the dimensions of the figure
    if mode == 'convert':
        # set the conversion mode
        new_width = cm2pt(target, dpi)
        new_height = cm2pt(target * h_w_ratio, dpi)
        scale_factor = 1
    elif mode == 'scale':
        # set the scaling mode
        new_width = int(px_width * target)
        new_height = int(px_height * target)
        scale_factor = target
    else:
        raise KeyError("Unrecognized scaling mode")

    if flag_width == 'plot':
        figure_in.opts(width=new_width)
    else:
        figure_in.opts(frame_width=new_width)

    if flag_height == 'plot':
        figure_in.opts(height=new_height)
    else:
        figure_in.opts(frame_height=new_height)

    # deal with the fonts

    # scale the font sizes
    scaled_fontsizes = {}
    if mode == 'convert':
        current_fontsize_dict = font_sizes_raw[fontsize]
    elif mode == 'scale':
        # get the current font sizes
        current_fontsize_dict = figure_in.opts.get()[0]['fontsize']
    else:
        raise KeyError('Unrecognized scale value')

    for key, value in current_fontsize_dict.items():
        number = int(value[:-2])
        number = px2pt(number, dpi, scale_factor)
        scaled_fontsizes[key] = str(number) + 'pt'

    # apply the scaling for the final figure
    figure_in.opts(fontsize=scaled_fontsizes)
    # hardcoded scaling factor to correct too large fonts (maybe holoviews bug)
    figure_in.opts(fontscale=0.745)
    # check if legend is present
    try:
        is_legend = True if figure_in.opts.get()[0]['show_legend'] else False
    except KeyError:
        is_legend = False
    if is_legend:
        # scale legend
        current_fontsize = float(render_fig.legend[0].label_text_font_size[:-2])
        figure_in.opts(legend_opts={'label_text_font_size': str(px2pt(current_fontsize-5, dpi, scale_factor)) + 'pt'})

    # select between singular and nested structures
    if ('Overlay' in str(type(figure_in))) | ('Layout' in str(type(figure_in))):
        # for all the plot components
        for idx, el in enumerate(figure_in.values()):

            # if line_width is there, scale it
            el = holoviews_mods(el, dpi, scale_factor)

            if idx == 0:
                # apply the hooks only to the first one
                el.opts(hooks=[partial(format_axis_hook, dpi=dpi, scale_factor=scale_factor)])
    else:
        # apply common scaling
        figure_in = holoviews_mods(figure_in, dpi, scale_factor)
        # format with hooks also
        figure_in.opts(hooks=[partial(format_axis_hook, dpi=dpi, scale_factor=scale_factor)])
    return figure_in


def save_figure(fig, save_path=None, fig_width=5, dpi=600, fontsize='paper', target='screen', display_factor=0.1):
    """Save figure for publication"""

    # scale the figure for saving
    fig = scale_figure(fig, fig_width, dpi, fontsize, mode='convert')

    # if the save flag is on
    if target in ['save', 'both']:
        # if not save path was provided, throw an exception
        assert save_path is not None, 'Please provide a save path'
        # save the figure
        hv.save(fig, save_path, backend='bokeh', dpi=dpi)
        print(f'Figure saved!: {save_path}')

    # scale the figure for display
    fig = scale_figure(fig, display_factor, dpi, fontsize, mode='scale')

    if target in ['screen', 'both']:
        # display the figure
        bokeh_show(hv.render(fig))

    return fig
