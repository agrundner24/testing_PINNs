import sys
sys.path.insert(0, '../')

import numpy as np
import common_methods as com
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D

def generate(options):
    """
    Generating data / function-values on a regular grid of space-time, adding noise and taking a batch of
    down-sampled regular sub-grids of this grid. This batch will contain the samples to train our network with.

    :param options: The dictionary of user-specified options (cf. main.py). Contains e.g. the grid-dimensions and noise
    :return: A batch (as a list) of samples (as dictionaries), that in turn consist of (noisy) function values on
             down-sampled sub-grids for all dt-layers.
    """
    # u_t = a*u_x + b*u_y + c*u_{xx} + d*u_{yy}

    # Variable declarations
    nx = options['mesh_size'][0]
    ny = options['mesh_size'][1]
    nt = options['layers']
    dt = options['dt']

    # Really good with a = b = 2
    a = 2
    b = 2
    c = 0.5
    d = 0.5

    dx = 2 * np.pi / (nx - 1)
    dy = 2 * np.pi / (ny - 1)

    ## Needed for plotting:
    # x = np.linspace(0, 2*np.pi, num = nx)
    # y = np.linspace(0, 2*np.pi, num = ny)
    # X, Y = np.meshgrid(x, y)

    # Assign initial function:
    u = com.initgen(options['mesh_size'], freq=4, boundary='Periodic')

    ## Plotting the initial function:
    # fig = plt.figure(figsize=(11,7), dpi=100)
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
    #
    # plt.show()

    xy_range = np.arange(0, 2 * np.pi + dx, dx)

    sample = {}
    sample['u_star'] = np.zeros((nx*ny, nt))
    sample['u_star'][:,0] = u.flatten(order = 1)
    sample['X_star'] = np.column_stack((np.tile(xy_range, nx), np.repeat(xy_range, nx)))
    sample['t'] = np.linspace(0, dt*(nt - 1), nt)[np.newaxis, :].T

    for n in range(nt - 1):
        un = com.pad_input_2(u, 2)[1:, 1:]

        u = (un[1:-1, 1:-1] + c * dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
             + d * dt / dy ** 2 * (un[2:, 1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
             + a * dt / dx * (un[1:-1, 2:] - un[1:-1, 1:-1])
             + b * dt / dy * (un[2:, 1:-1] - un[1:-1, 1:-1]))[:-1, :-1]

        sample['u_star'][:, n + 1] = u.flatten(order = 1)



    ## Plotting the function values from the last layer:
    # fig2 = plt.figure()
    # ax2 = fig2.gca(projection='3d')
    # surf2 = ax2.plot_surface(X, Y, u, cmap=cm.viridis)
    #
    # plt.show()



    return sample
