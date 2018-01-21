import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

alpha = 0.5

def generate_wave_set(n_support=100, n_train=25, std=0.3):
    rgen = np.random.RandomState(42)
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['values'] = np.sin(data['support']) + 10
    data['x_train'] = np.sort(rgen.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']) + 10 + rgen.normal(0, std, size=data['x_train'].shape[0])
    return data


def plot_ground_truth():
    margin = 3
    plt.plot(data['support'], data['values'], 'b--', alpha=0.5, label='manifold')
    plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')
    # plt.xlim(data['x_train'].min() - margin, data['x_train'].max() + margin)
    # plt.ylim(data['y_train'].min() - margin, data['y_train'].max() + margin)
    plt.legend(loc='upper right', prop={'size': 20})
    plt.title('True manifold and noised data')
    plt.xlabel('x')
    plt.ylabel('y')



def plot_line(w):
    plot_ground_truth()
    plt.plot(data['x_train'], lin_reg(w))
    plt.show()


def normal_eq():
    inverted = np.linalg.inv(X.T.dot(X))
    return inverted.dot(X.T).dot(y)


def lin_reg(w):
    return X.dot(w).flatten()


def mse(w):
    y_hat = lin_reg(w)
    return ((y - y_hat)**2).mean()


def mse_gradient(w):
    y_hat = lin_reg(w)
    return 1.0 / n * X.T.dot(y_hat - y).reshape(w.shape)


def gd(w):
    gd_trace = np.zeros((1000, 2))
    for i in range(0, 1000):
        grad = mse_gradient(w)
        w = w - alpha * grad
        gd_trace[i] = w.T
    return w, gd_trace


def mse_data_for_plot():
    ww0 = np.linspace(-500, 500, 100)
    ww1 = np.linspace(-500, 500, 100)
    W0, W1 = np.meshgrid(ww0, ww1)
    wx_space = list(it.product(ww0, ww1))
    wx_space = np.array(wx_space)
    ZZ = ((y[:, np.newaxis] - X.dot(wx_space.T))**2).mean(axis=0)
    ZZ = ZZ.reshape(W0.shape)
    return W0, W1, ZZ


def plot_paraboloid(surface=False, contour_x=False, contour_y=False, contour_z=False):
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    # Data for a three-dimensional line
    xx = np.linspace(-50, 50, 100)
    yy = np.linspace(-50, 50, 100)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = XX ** 2 + YY ** 2
    if surface:
        ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    else:
        ax.contour3D(XX, YY, ZZ, 50, cmap='viridis')
    if contour_x:
        cset = ax.contour(XX, YY, ZZ, 25, zdir='x', offset=-50, cmap=cm.coolwarm)
    if contour_y:
        cset = ax.contour(XX, YY, ZZ, 25, zdir='y', offset=50, cmap=cm.coolwarm)
    if contour_z:
        cset = ax.contour(XX, YY, ZZ, 25, zdir='z', offset=0, cmap=cm.coolwarm)
    plt.show()


def plot_mse_3d(w, surface=False, gd_trace=None, show_mse_sample=False, contour_x=False, contour_y=False, contour_z=False):
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    # Data for a three-dimensional line
    W0, W1, ZZ = mse_data_for_plot()
    ax.set_xlabel('$\omega_1$')
    ax.set_ylabel('$\omega_0$')
    ax.set_zlabel('MSE')
    if surface:
        ax.plot_surface(W0, W1, ZZ, rstride=8, cstride=8, alpha=0.3)
    else:
        ax.contour3D(W0, W1, ZZ, 50, cmap='viridis')
    if contour_x:
        cset = ax.contour(W0, W1, ZZ, 25, zdir='x', offset=-500, cmap=cm.coolwarm)
    if contour_y:
        cset = ax.contour(W0, W1, ZZ, 25, zdir='y', offset=500, cmap=cm.coolwarm)
    if contour_z:
        cset = ax.contour(W0, W1, ZZ, 25, zdir='z', offset=0, cmap=cm.coolwarm)
    if gd_trace is not None:
        gd_mse = ((y[:, np.newaxis] - X.dot(gd_trace.T))**2).mean(axis=0)
        if show_mse_sample:
            ax.plot3D([gd_trace[1, 1], gd_trace[1, 1]], 
                [gd_trace[1, 0], gd_trace[1, 0]], 
                [gd_mse[1], 0], c='red')
        ax.plot3D(gd_trace[:, 1], gd_trace[:, 0], gd_mse)
    ax.scatter3D(w[1], w[0], mse(w), s=10, c='black')
    plt.show()


def plot_mse_contour(w, gd_trace=None):
    W0, W1, ZZ = mse_data_for_plot()
    plt.axes().set_xlabel('$\omega_1$')
    plt.axes().set_ylabel('$\omega_0$')
    plt.axvline(0, color='black', linestyle='-', label='origin')
    plt.axhline(0, color='black', linestyle='-')
    plt.scatter(w[1], w[0], s=10, c='black')
    if gd_trace is not None:
        plt.plot(gd_trace[:, 1], gd_trace[:, 0])
    plt.contour(W0, W1, ZZ, 50, cmap='viridis');
    plt.colorbar();
    plt.show()


def draw_mishra_bird():
    fig = plt.figure(figsize=(14, 10))
    x = np.arange(-10, 1, 0.1)
    y = np.arange(-6, 0.5, 0.1)
    X, Y = np.meshgrid(x, y)
    ax = plt.gca(projection='3d')
    Z = np.sin(Y) * np.exp((1 - np.cos(X)) ** 2) + np.cos(X) * np.cos(X) * np.exp((1 - np.sin(Y)) ** 2) + (X - Y) ** 2
    print(len(X), len(Y), len(Z))
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, 50, zdir='z', offset=-30, cmap=cm.coolwarm)
    ax.view_init(20, -60)
    plt.show()


n = 100
data = generate_wave_set(n, 25)

X = data['x_train']
y = data['y_train']

X = X[:, np.newaxis]
o = np.ones((X.shape[0], 1))
X = np.hstack([o, X])

w = np.ones((X.shape[1], 1)) * 500
