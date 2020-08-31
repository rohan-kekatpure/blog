import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import simps
from isoperimetric_nnet import nnet_optimizer

def generate_points(n_points):
    eps = 0.01
    theta = np.linspace(-np.pi, np.pi + eps, n_points)
    radius = np.random.uniform(0.5, 1.5, size=(n_points,))
    return radius, theta

def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x = np.hstack((x, x[0]))
    y = np.hstack((y, y[0]))
    return x, y

def perimeter(r, theta):
    x, y = pol2cart(r, theta)
    dx = np.diff(x)
    dy = np.diff(y)
    perim = np.sqrt((dx*dx + dy * dy)).sum()
    return perim


def area(r, theta):
    return simps(r * r, theta) / 2.0

def save_image(r, theta, x0, y0, image_index, out_dir):
    x1, y1 = pol2cart(r, theta)
    pl.close('all')
    _, ax = pl.subplots()
    ax.plot(x0, y0, 'r-', lw=1)
    ax.plot(x1, y1, 'g-', lw=1)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect('equal', 'box')
    pl.savefig('{}/image_{}.png'.format(out_dir, image_index))

def loss(r, theta, constraint_val, lambda1, lambda2):
    C = constraint_val
    A = area(r, theta)
    S = perimeter(r, theta)
    x, y = pol2cart(r, theta)
    return -A + lambda1 * (S - C) ** 2 + lambda2 * (x.mean() ** 2 + y.mean() ** 2)

def greedy(r, theta, lambda1, lambda2, n_iterations, save_images, save_every):
    n_points = r.shape[0]
    C = 2 * np.pi
    best_r = r.copy()
    best_loss = loss(best_r, theta, C, lambda1, lambda2)
    x0, y0 = pol2cart(r, theta)

    for i in range(n_iterations):
        old_r = r.copy()
        j = np.random.randint(0, n_points)
        r[j] += np.random.uniform(-0.01, 0.01)
        new_loss = loss(r, theta, C, lambda1, lambda2)

        if new_loss < best_loss:
            best_loss = new_loss
            best_r = r
        else:
            r = old_r

        if save_images and (i % save_every == 0):
            save_image(r, theta, x0, y0, i, 'images/greedy')

        if i % 1000 == 0:
            print('iteration: {}, loss: {}'.format(i, best_loss))

    return best_r, theta

def gradient_descent(r, theta, learning_rate, lambda1, lambda2, n_iterations, save_images, save_every):
    C = 2 * np.pi
    x0, y0 = pol2cart(r, theta)
    d_theta = theta[1] - theta[0]
    for i in range(n_iterations):
        S = perimeter(r, theta)
        grad_J = -d_theta * r + 2 * d_theta * lambda1 * (S - C)
        r += learning_rate * grad_J
        new_loss = loss(r, theta, C, lambda1, lambda2)

        if save_images and ((i < 100) or (i % save_every == 0)):
            save_image(r, theta, x0, y0, i, 'images/gradient_descent')

        if i % 10 == 0:
            print('iteration: {}, loss: {}'.format(i, new_loss))

    return r, theta

def main():
    # Make folders for saving images
    shutil.rmtree('images')
    (Path('images') / 'greedy').mkdir(parents=True)
    (Path('images') / 'gradient_descent').mkdir(parents=True)
    (Path('images') / 'nnet').mkdir(parents=True)

    r_, theta_ = generate_points(64)
    P, A = perimeter(r_, theta_), area(r_, theta_)
    r_ *= (2 * np.pi) / P
    print(perimeter(r_, theta_), area(r_, theta_))

    best_r, best_theta = greedy(
        r_.copy(),
        theta_.copy(),
        lambda1=10.,
        lambda2=10.,
        n_iterations=100000,
        save_images=True,
        save_every=1000
    )

    print(perimeter(best_r, best_theta), area(best_r, best_theta))

    # best_r, best_theta = gradient_descent(
    #     r_.copy(),
    #     theta_.copy(),
    #     learning_rate=0.01,
    #     lambda1=10.,
    #     lambda2=0.,
    #     n_iterations=5000,
    #     save_images=True,
    #     save_every=50
    # )
    #
    # print(perimeter(best_r, best_theta), area(best_r, best_theta))

    # best_r, best_theta = nnet_optimizer(
    #     r_.copy(),
    #     theta_.copy(),
    #     learning_rate=1e-4,
    #     lambda1=1.,
    #     lambda2=5.,
    #     n_iterations=5000,
    #     save_images=True,
    #     save_every=20
    # )

    pl.close('all')

if __name__ == '__main__':
    main()
