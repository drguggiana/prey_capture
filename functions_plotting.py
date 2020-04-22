from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def plot_2d(data_in, rows=1, columns=1, labels=None, markers=None, linestyle='-', color=None,
            xerr=None, yerr=None, fig=None, fontsize=None, dpi=None):
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
                if linestyle is not '-':
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


def plot_image(data_in, rows=1, columns=1, fig=None, colormap=None, colorbar=None):
    """Wrapper for the imshow function in subplots"""
    # create a new figure window
    if fig is None:
        # fig = plt.figure(dpi=300)
        fig, ax = plt.subplots(nrows=rows, ncols=columns, squeeze=False, dpi=300)
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


def show():
    """Wrapper for plt.show"""
    return plt.show()

