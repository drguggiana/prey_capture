from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_2d(data_in, rows=1, columns=1, labels=None, markers=None):
    # create a new figure window
    fig = plt.figure()
    # initialize a plot counter
    plot_counter = 1
    # for all the rows
    for row in range(rows):
        # for all the columns
        for col in range(columns):
            # add the subplot
            ax = fig.add_subplot(str(rows)+str(columns)+str(plot_counter))
            # for all the lines in the list
            for count, lines in enumerate(data_in[plot_counter-1]):
                # plot x,y ot just y depending on the size of the data
                if len(lines.shape) == 2:
                    line2d = ax.plot(lines[:, 0], lines[:, 1])
                else:
                    line2d = ax.plot(lines)
                # change the marker if provided, otherwise use dots
                if markers is not None:
                    line2d[0].set_marker(markers[plot_counter-1][count])
                else:
                    line2d[0].set_marker('.')
            # add labels if provided
            if labels is not None:
                plt.legend(labels[plot_counter-1])
            # update the plot counter
            plot_counter += 1
    return fig


def plot_3d(data_in):
    # create a new figure window
    fig = plt.figure()
    # add the subplot
    ax = fig.add_subplot(111, projection='3d')
    # for all the lines in the list
    for lines in data_in:
        ax.plot(lines[:, 0], lines[:, 1], lines[:, 2], marker='.')
    return fig


def animation_plotter(motivedata, bonsaidata, cricket_data, xlim, ylim, interval=10):
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


def histogram(data_in, rows=1, columns=1, bins=50):
    # create a new figure window
    fig = plt.figure()
    # initialize a plot counter
    plot_counter = 1
    # for all the rows
    for row in range(rows):
        # for all the columns
        for col in range(columns):
            # add the subplot
            ax = fig.add_subplot(str(rows) + str(columns) + str(plot_counter))
            # for all the lines in the list
            for lines in data_in[plot_counter - 1]:
                # plot the histogram of the data
                ax.hist(lines, bins=bins)
            # update the plot counter
            plot_counter += 1
    return fig
