from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sklearn.cluster as clu
import holoviews as hv
from bokeh.themes.theme import Theme

# define the standard font sizes
small = '7pt'
medium = 10
large = 12
# define the conversion constant from pt to cm
pt2cm = 0.0352778
# define the font size default dictionary for figures
font_sizes_raw = {
    'small': {
        'xlabel': small,
        'ylabel': small,
        'zlabel': small,
        'labels': small,
        'xticks': small,
        'yticks': small,
        # 'zticks': small-2,
        'ticks': small,
        'minor_xticks': small,
        'minor_yticks': small,
        'minor_ticks': small,
        'title': small,
        # 'legend': small-1,
        'legend_title': small,
    },
    'medium': {
        'xlabel': medium,
        'ylabel': medium,
        'zlabel': medium,
        'labels': medium,
        'xticks': medium,
        'yticks': medium,
        'zticks': medium,
        'ticks': medium,
        'minor_xticks': medium,
        'minor_yticks': medium,
        'minor_ticks': medium,
        'title': medium,
        'legend': medium-1,
        'legend_title': medium,
    },
    'large': {
        'xlabel': large,
        'ylabel': large,
        'zlabel': large,
        'labels': large,
        'xticks': large,
        'yticks': large,
        'zticks': large,
        'ticks': large,
        'minor_xticks': large,
        'minor_yticks': large,
        'minor_ticks': large,
        'title': large,
        'legend': large-1,
        'legend_title': large,
    },
}

attr_dict = {
    'attrs': {
        'Figure': {
            'background_fill_color': '#FFFFFF',
            'border_fill_color': '#FFFFFF',
            'outline_line_color': '#FFFFFF',
        },
        # 'Grid': {
        #     'grid_line_dash': [6, 4],
        #     'grid_line_alpha': .3,
        # },
        'Text':
            {
                'text_font': 'Arial',
                # 'text_font_size': 20,
            },

        'Axis': {
            'major_label_text_color': 'black',
            'axis_label_text_color': 'black',
            'major_tick_line_color': 'black',
            'axis_line_color': "black",
            'axis_line_width': 2,
            'axis_line_cap': 'round',
            'axis_label_text_font_size': '14pt',
            'major_label_text_font_size': '14pt',
            'minor_tick_line_color': None,
            'major_tick_in': 0,
            'major_tick_line_cap': 'round',
        },
        'Legend': {
            'label_text_font_size': '7pt',
        }
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


def pix(cm_value, dpi=600):
    """Function to convert figure sizes in cm to pixels based on a dpi requirement"""
    return int(np.round((cm_value/2.54)*dpi))


def search2path(search_string):
    """Turn the input search string into a path entry for figures"""
    search_string = search_string.replace(':', '_')
    search_string = search_string.replace('=', '')
    search_string = search_string.replace(', ', '_')
    return search_string


def format_label(label):
    """Format labels for plots"""
    new_label = label.replace('cricket_0', 'cricket_-')
    new_label = new_label.split('_')
    new_label = ' '.join([el.capitalize() for el in new_label])
    return new_label


def format_figure(fig, width=None, frame_width=None, height=None, frame_height=None, fontsize='small'):
    """Apply basic figure formattings"""
    # rotate x axislabels
    fig.opts(xrotation=45)
    # apply sizing
    if width is not None:
        fig.opts(width=width)
    elif frame_width is not None:
        fig.opts(frame_width=frame_width)
    if height is not None:
        fig.opts(height=height)
    elif frame_height is not None:
        fig.opts(frame_height=frame_height)
    # Format the title and axis labels
    xlabel = str(fig.kdims[0])
    xlabel = format_label(xlabel)
    fig.opts(xlabel=xlabel)

    ylabel = str(fig.vdims[0])
    ylabel = format_label(ylabel)
    fig.opts(ylabel=ylabel)

    # label = fig.label
    # label = format_label(label)
    # fig.opts(title=label)

    # apply font sizing
    fig.opts(fontsize=font_sizes_raw[fontsize])
    # hardcoded scaling factor to correct too large fonts (maybe holoviews bug)
    fig.opts(fontscale=0.745)

    return fig


def format_axis_hook(plot, element):
    """Hook to rescale axis components"""
    # define the dpi (alternatively, global from outside?)
    # TODO: make dpi definition prettier/more universal
    dpi = 600
    # get the plot dict
    b = plot.state
    # scale the axis line width
    current_axis_width = b.below[0].axis_line_width
    b.below[0].axis_line_width = pix(current_axis_width * pt2cm, dpi)
    b.left[0].axis_line_width = pix(current_axis_width * pt2cm, dpi)
    # scale the outer tick length
    current_tick_length = b.below[0].major_tick_out
    b.below[0].major_tick_out = pix(current_tick_length * pt2cm, dpi)
    b.left[0].major_tick_out = pix(current_tick_length * pt2cm, dpi)
    # scale the tick width
    current_tick_width = b.below[0].major_tick_line_width
    b.below[0].major_tick_line_width = pix(current_tick_width * pt2cm, dpi)
    b.left[0].major_tick_line_width = pix(current_tick_width * pt2cm, dpi)
    # scale the tick standoff
    current_tick_standoff = b.below[0].major_label_standoff
    b.below[0].major_label_standoff = pix(current_tick_standoff * pt2cm, dpi)
    b.left[0].major_label_standoff = pix(current_tick_standoff * pt2cm, dpi)
    # scale the axis label standoff
    current_label_standoff = b.below[0].axis_label_standoff
    b.below[0].axis_label_standoff = pix(current_label_standoff * pt2cm, dpi)
    b.left[0].axis_label_standoff = pix(current_label_standoff * pt2cm, dpi)


def save_figure(fig, save_path, fig_width=5, dpi=600, fontsize='small'):
    """Save figure for publication"""

    # fig = original_fig.opts(clone=True)
    # print(fig.opts.info())
    # fig
    # render the plot as a bokeh element to get the inner features
    render_fig = hv.render(fig)
    # get the original width and height of the figure
    px_width = render_fig.properties_with_values()['plot_width']
    # set flag for which dim to change later
    flag_width = 'plot'
    # get the frame width if the plot one wasn't defined
    if px_width is None:
        px_width = render_fig.properties_with_values()['frame_width']
        flag_width = 'frame'
    # repeat for height
    px_height = render_fig.properties_with_values()['plot_height']
    flag_height = 'plot'
    if px_height is None:
        px_height = render_fig.properties_with_values()['frame_height']
        flag_height = 'frame'

    # get their ratio
    h_w_ratio = px_height / px_width

    # scale the dimensions of the figure
    if flag_width == 'plot':
        fig.opts(width=pix(fig_width, dpi))
    else:
        fig.opts(frame_width=pix(fig_width, dpi))

    if flag_height == 'plot':
        fig.opts(height=pix(fig_width * h_w_ratio, dpi))
    else:
        fig.opts(frame_height=pix(fig_width * h_w_ratio, dpi))

    # scale the font sizes
    scaled_fontsizes = {}
    for key, value in font_sizes_raw[fontsize].items():
        number = int(value[:-2])
        number = pix(number * pt2cm, dpi)
        scaled_fontsizes[key] = str(number) + 'pt'
    # apply the scaling for the final figure
    fig.opts(fontsize=scaled_fontsizes)
    # hardcoded scaling factor to correct too large fonts (maybe holoviews bug)
    fig.opts(fontscale=0.745)
    # check if legend is present
    try:
        is_legend = True if fig.opts.get()['show_legend']['show_legend'] else False
    except KeyError:
        is_legend = False
    if is_legend:
        # scale legend
        current_fontsize = float(render_fig.above[0].label_text_font_size[:-2])
        fig.opts(legend_opts={'label_text_font_size': str(pix(current_fontsize*pt2cm, dpi))+'pt'})

    # scale line width
    try:

        current_line_width = render_fig.renderers[0].glyph.line_width
        fig.opts(line_width=pix(current_line_width*pt2cm, dpi))
        fig.opts(hooks=[format_axis_hook])

    except ValueError:
        for idx, el in enumerate(fig.items()):
            render_el = hv.render(el[1])
            current_line_width = render_el.renderers[0].glyph.line_width
            el[1].opts(line_width=pix(current_line_width*pt2cm, dpi))
            if idx == 0:
                el[1].opts(hooks=[format_axis_hook])

    # save the figure
    hv.save(fig, save_path, backend='bokeh', dpi=dpi)
    return fig
