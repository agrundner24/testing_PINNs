"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import generate_data as gd

# np.random.seed(1234)
# tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t, u, layers):
        
        X = np.concatenate([x, y, t], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        
        self.u = u
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize parameters
        self.c0 = tf.Variable([0], dtype=tf.float32)
        self.c00 = tf.Variable([0], dtype=tf.float32)
        self.c10 = tf.Variable([0], dtype=tf.float32)
        self.c01 = tf.Variable([0], dtype=tf.float32)
        self.c20 = tf.Variable([0], dtype=tf.float32)
        self.c11 = tf.Variable([0], dtype=tf.float32)
        self.c02 = tf.Variable([0], dtype=tf.float32)
        self.c30 = tf.Variable([0], dtype=tf.float32)
        self.c21 = tf.Variable([0], dtype=tf.float32)
        self.c12 = tf.Variable([0], dtype=tf.float32)
        self.c03 = tf.Variable([0], dtype=tf.float32)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.u_pred, self.f_u_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred))
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, x, y, t):
        c0 = self.c0
        c00 = self.c00
        c10 = self.c10
        c01 = self.c01
        c20 = self.c20
        c11 = self.c11
        c02 = self.c02
        c30 = self.c30
        c21 = self.c21
        c12 = self.c12
        c03 = self.c03
        
        u = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_xy = tf.gradients(u_x, y)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxy = tf.gradients(u_xx, y)[0]
        u_xyy = tf.gradients(u_yy, x)[0]
        u_yyy = tf.gradients(u_yy, y)[0]

        f_u = u_t - c00*u - c10*u_x - c01*u_y - c20*u_xx - c02*u_yy - c11*u_xy - c30*u_xxx - c21*u_xxy - c12*u_xyy \
              - c03*u_yyy - c0*tf.sin(u)
        
        return u, f_u
    
    def callback(self, loss, c0, c00, c10, c01, c20, c11, c02, c30, c21, c12, c03):
        print('Loss: %.3e, c0:%.5f, c00:%.5f, c10:%.5f, c01:%.5f, c20:%.5f, c11:%.5f, c02:%.5f, c30:%.5f, c21:%.5f, '
              'c12:%.5f, c03:%.5f' % (loss, c0, c00, c10, c01, c20, c11, c02, c30, c21, c12, c03))
      
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                c0_value = self.sess.run(self.c0)
                c00_value = self.sess.run(self.c00)
                c10_value = self.sess.run(self.c10)
                c01_value = self.sess.run(self.c01)
                c20_value = self.sess.run(self.c20)
                c11_value = self.sess.run(self.c11)
                c02_value = self.sess.run(self.c02)
                c30_value = self.sess.run(self.c30)
                c21_value = self.sess.run(self.c21)
                c12_value = self.sess.run(self.c12)
                c03_value = self.sess.run(self.c03)
                print('It: %d, Loss: %.3e, c0:%.5f, c00:%.5f, c10:%.5f, c01:%.5f, c20:%.5f, c11:%.5f, c02:%.5f, '
                      'c30:%.5f, c21:%.5f, c12:%.5f, c03:%.5f, Time: %.2f' %
                      (it, loss_value, c0_value, c00_value, c10_value, c01_value, c20_value, c11_value, c02_value,
                       c30_value, c21_value, c12_value, c03_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.c0, self.c00, self.c10, self.c01, self.c20, self.c11,
                                           self.c02, self.c30, self.c21, self.c12, self.c03],
                                loss_callback = self.callback)
            
    
    def predict(self, x_star, y_star, t_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        
        return u_star


def plot_solution(X_star, u_star, index, isPred):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure()
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')

    if isPred is True:
        plt.savefig('./DiffusionNonlinear_prediction_%d' % index)
    else:
        plt.savefig('./DiffusionNonlinear_%d' % index)
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
if __name__ == "__main__":

    t0 = time.time()
      
    N_train = 22500  # Have 540.000 data points over in PDE-Net

    options = {'mesh_size': [250, 250],  # How large is the (regular) 2D-grid of function values for a fixed t.
               # Keep mesh_size[0] = mesh_size[1]
               'layers': 8,  # Layers of the NN. Also counting the initial layer!
               'dt': 0.003,  # Time discretization. We step dt*(layers - 1) forward in time.
               'batch_size': 1,  # We take a batch of sub-grids in space
               'downsample_by': 1,  # Size of sub-grids (in space) * downsample_by = mesh_size
               'boundary_cond': 'PERIODIC'
               # Set to 'PERIODIC' if data has periodic bdry condition to use periodic padding
               }

    data = gd.generate(options)
    
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    UU = data['u_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    
    ######################################################################
    ######################## Noiseless Data ##############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, layers)
    model.train(200000) # Default: 200000

    t_complete = time.time() - t0

    for i in range(options['layers']):
        # Test Data
        snap = np.array([i])
        x_star = X_star[:,0:1]
        y_star = X_star[:,1:2]
        t_star = TT[:,snap]

        u_star = UU[:,snap]

        # Prediction
        u_pred = model.predict(x_star, y_star, t_star)

        # Error
        error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

        with open('results.txt', 'a') as file:
            file.write('Relative L2-error of u for t = %.3f: %e\n' % (i * options['dt'], error_u))

        print('Error u: %e' % (error_u))

        # Plot Results
        plot_solution(X_star, u_pred, i, True)
        plot_solution(X_star, u_star, i, False)

    c0_value = model.sess.run(model.c0)
    c00_value = model.sess.run(model.c00)
    c10_value = model.sess.run(model.c10)
    c01_value = model.sess.run(model.c01)
    c20_value = model.sess.run(model.c20)
    c11_value = model.sess.run(model.c11)
    c02_value = model.sess.run(model.c02)
    c30_value = model.sess.run(model.c30)
    c21_value = model.sess.run(model.c21)
    c12_value = model.sess.run(model.c12)
    c03_value = model.sess.run(model.c03)

    error_c0 = (c0_value - 15)**2
    error_c00 = (c00_value - 0) ** 2
    error_c10 = (c10_value - 0) ** 2
    error_c01 = (c01_value - 0) ** 2
    error_c20 = (c20_value - 0.3) ** 2
    error_c11 = (c11_value - 0) ** 2
    error_c02 = (c02_value - 0.3) ** 2
    error_c30 = (c30_value - 0) ** 2
    error_c21 = (c21_value - 0) ** 2
    error_c12 = (c12_value - 0) ** 2
    error_c03 = (c03_value - 0) ** 2

    print('Error c0: %.5f' % (error_c0))
    print('Error c00: %.5f' % (error_c00))
    print('Error c10: %.5f' % (error_c10))
    print('Error c01: %.5f' % (error_c01))
    print('Error c20: %.5f' % (error_c20))
    print('Error c11: %.5f' % (error_c11))
    print('Error c02: %.5f' % (error_c02))
    print('Error c30: %.5f' % (error_c30))
    print('Error c21: %.5f' % (error_c21))
    print('Error c12: %.5f' % (error_c12))
    print('Error c03: %.5f' % (error_c03))

    print('Time needed to run the algorithm: ', t_complete)

    with open('results.txt', 'a') as file:
        file.write('Inferred c0: %.5f\n' % (c0_value))
        file.write('Inferred c00: %.5f\n' % (c00_value))
        file.write('Inferred c10: %.5f\n' % (c10_value))
        file.write('Inferred c01: %.5f\n' % (c01_value))
        file.write('Inferred c20: %.5f\n' % (c20_value))
        file.write('Inferred c11: %.5f\n' % (c11_value))
        file.write('Inferred c02: %.5f\n' % (c02_value))
        file.write('Inferred c30: %.5f\n' % (c30_value))
        file.write('Inferred c21: %.5f\n' % (c21_value))
        file.write('Inferred c12: %.5f\n' % (c12_value))
        file.write('Inferred c03: %.5f\n' % (c03_value))

        file.write('Squared error c0: %.5f\n' % (error_c0))
        file.write('Squared error c00: %.5f\n' % (error_c00))
        file.write('Squared error c10: %.5f\n' % (error_c10))
        file.write('Squared error c01: %.5f\n' % (error_c01))
        file.write('Squared error c20: %.5f\n' % (error_c20))
        file.write('Squared error c11: %.5f\n' % (error_c11))
        file.write('Squared error c02: %.5f\n' % (error_c02))
        file.write('Squared error c30: %.5f\n' % (error_c30))
        file.write('Squared error c21: %.5f\n' % (error_c21))
        file.write('Squared error c12: %.5f\n' % (error_c12))
        file.write('Squared error c03: %.5f\n' % (error_c03))

        file.write('Time needed to run the algorithm: %d seconds' % t_complete)

    # plt.show()

    # # ######################################################################
    # # ############################# Plotting ###############################
    # # ######################################################################
    #
    # fig, ax = plotting.newfig(1.0, 1.2)
    # ax.axis('off')
    #
    # ######## Row 2: Pressure #######################
    # ########      Predicted p(t,x,y)     ###########
    # gs2 = gridspec.GridSpec(1, 2)
    # gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
    # ax = plt.subplot(gs2[:, 0])
    # h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow',
    #               extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    #
    # fig.colorbar(h, cax=cax)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_aspect('equal', 'box')
    # ax.set_title('Predicted u', fontsize = 10)
    #
    # ########     Exact p(t,x,y)     ###########
    # ax = plt.subplot(gs2[:, 1])
    # h = ax.imshow(u_star, interpolation='nearest', cmap='rainbow',
    #               extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    #
    # fig.colorbar(h, cax=cax)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_aspect('equal', 'box')
    # ax.set_title('Exact u', fontsize = 10)
    #
    # plt.savefig('./AdvectionDiffusion_prediction')
