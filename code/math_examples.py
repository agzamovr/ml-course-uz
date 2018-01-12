import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def draw_parabola(steps=50):
    x = np.linspace(-4, 4, steps)
    plt.plot(x, x ** 2)
    plt.axvline(x=0, color='b', linestyle='dashed')


def draw_paraboloid(steps=50):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    x = np.linspace(-1, 1, steps)
    y = np.linspace(-1, 1, steps)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)


def draw_mishra_bird():
    fig = plt.figure(figsize=(14, 10))
    x = np.arange(-10, 1, 0.1)
    y = np.arange(-6, 0.5, 0.1)
    X, Y = np.meshgrid(x, y)
    ax = plt.gca(projection='3d')
    Z = np.sin(Y) * np.exp((1 - np.cos(X)) ** 2) + np.cos(X) * np.cos(X) * np.exp((1 - np.sin(Y)) ** 2) + (X - Y) ** 2
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.view_init(20, -60)


def draw_hyperbolic_paraboloid():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 - Y ** 2
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)