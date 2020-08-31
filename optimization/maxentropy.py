import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import simps
import torch
import torch.nn as nn


def generate_points(xmin, xmax, n_points):
    eps = 0.01
    x = np.linspace(xmin, xmax + eps, n_points)
    y = np.random.uniform(0, 0.8, size=(n_points,))
    # y = np.exp(-1. * (x - 1) ** 2)
    return x, y

def save_image(x, y, x0, y0, image_index, output_dir):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().numpy()
        y = y.clone().detach().numpy()

    pl.close('all')
    _, ax = pl.subplots()
    ax.fill_between(x0, y0, y2=0, color='r', alpha=0.3)
    ax.fill_between(x, y, y2=0, color='g', alpha=0.7)
    ax.plot(x, y, 'g-', lw=1)
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, 1])
    pl.savefig('{}/image_{}.png'.format(output_dir.as_posix(), image_index))

def entropy(x, y):
    return simps(-y * np.log(y), x)

def mean(x, y):
    return simps(x * y, x)

def stddev(x, y):
    mu = mean(x, y)
    var = simps(x * x * y, x) - mu * mu
    return np.sqrt(var)

def norm(x, y):
    return simps(y, x)

def loss(x, y, mu0, sigma0, lambda1, lambda2, lambda3):
    H = entropy(x, y)
    M = mean(x, y)
    S = stddev(x, y)
    N = norm(x, y)
    L = -H + lambda1 * (M - mu0) ** 2 \
           + lambda2 * (S - sigma0) ** 2 \
           + lambda3 * (N - 1.0) ** 2
    return L

class TorchFuncs:
    @staticmethod
    def entropy(x, y):
        # return torch.trapz(-y * torch.log(y), x)
        return (-y * torch.log(y)).sum() * (x[1] - x[0])
    @staticmethod
    def mean(x, y):
        # return torch.trapz(x * y, x)
        return (x * y).sum() * (x[1] - x[0])

    @staticmethod
    def stddev(x, y):
        mu = TorchFuncs.mean(x, y)
        # var = torch.trapz(x * x * y, x) - mu * mu
        var = (x * x * y).sum() * (x[1] - x[0]) - mu ** 2
        return torch.sqrt(var)

    @staticmethod
    def norm(x, y):
        # return torch.trapz(y, x)
        return y.sum() * (x[1] - x[0])

    @staticmethod
    def loss(x, y, mu0, sigma0, lambda1, lambda2, lambda3):
        H = TorchFuncs.entropy(x, y)
        M = TorchFuncs.mean(x, y)
        S = TorchFuncs.stddev(x, y)
        N = TorchFuncs.norm(x, y)
        L = -H + lambda1 * (M - mu0) ** 2 \
            + lambda2 * (S - sigma0) ** 2 \
            + lambda3 * (N - 1.0) ** 2
        if torch.isnan(L):
            print('nan loss')
            from IPython import embed; embed(); exit(0)
        return L

class Inet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Inet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out, bias=False)
        )

    def forward(self, input_):
        return self.main(input_)

def greedy(x, y, mu0, sigma0, lambda1, lambda2, lambda3,
           n_iterations, output_dir, save_images, save_every):
    n_points = x.shape[0]
    best_y = y.copy()
    best_loss = loss(x, best_y, mu0, sigma0, lambda1, lambda2, lambda3)
    x0, y0 = x.copy(), y.copy()
    epsy = 1e-8
    for i in range(n_iterations):
        old_y = y.copy()
        j = np.random.randint(0, n_points)
        y[j] += np.random.uniform(-0.005, 0.005)
        y = np.clip(y, epsy, None)
        new_loss = loss(x, y, mu0, sigma0, lambda1, lambda2, lambda3)

        if new_loss < best_loss:
            best_loss = new_loss
            best_y = y
        else:
            y = old_y

        if save_images and (i % save_every == 0):
            save_image(x, y, x0, y0, i, output_dir)

        if i % 1000 == 0:
            print('iteration: {}, loss: {:0.3f}, mu: {:0.3f}, sigma: {:0.3f}, norm: {:0.3f}'\
                  .format(i, best_loss, mean(x, y), stddev(x, y), norm(x, y))
            )

    return x, best_y

def gradient_descent(x, y, mu0, sigma0, lambda1, lambda2, lambda3,
                     learning_rate,  n_iterations, output_dir, save_images, save_every):
    x0, y0 = x.copy(), y.copy()
    dx = x[1] - x[0]
    epsy = 1e-12
    for i in range(n_iterations):
        N = norm(x, y)
        M = mean(x, y)
        S = stddev(x, y)

        grad_L = (1 + np.log(y)) \
                 + 2. * lambda1 * x * (M - mu0) \
                 + lambda2 * x * x * (1 - sigma0 / S) \
                 + 2. * lambda3 * (N - 1.)

        y -= learning_rate * dx * grad_L
        y = np.clip(y, epsy, None)

        save_every_ = 10 if i < 1000 else save_every
        if save_images and (i % save_every_ == 0):
            save_image(x, y, x0, y0, i, output_dir)

        if i % 10 == 0:
            new_loss = loss(x, y, mu0, sigma0, lambda1, lambda2, lambda3)
            print('iteration: {}, loss: {:0.3f}, mu: {:0.3f}, sigma: {:0.3f}, norm: {:0.3f}'\
                  .format(i, new_loss, M, S, N)
            )

    return x, y

def nnet(x, y, mu0, sigma0, lambda1, lambda2, lambda3,
         learning_rate, n_iterations, output_dir, save_images,
         save_every):
    num_points = x.shape[0]
    x, y = torch.tensor(x), torch.tensor(y)
    x0 = x.clone().detach().numpy()
    y0 = y.clone().detach().numpy()
    epsy = 1e-12
    model = Inet(1, 256, num_points)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    torch.autograd.set_detect_anomaly(True)

    batch_size = 16
    n_samples = x.size()[0]
    for i in range(n_iterations):
        permutation = torch.randperm(n_samples)
        for k in range(0, n_samples, batch_size):
            indices = permutation[k: k + batch_size]
            newy = model(torch.tensor([1.])).double()
            # y = newy.clone().clamp(epsy)
            y = y.clone() + 0
            y[indices] = newy.clone()[indices]
            y = y.clamp(epsy)
            loss_value = TorchFuncs.loss(x, y, mu0, sigma0, lambda1, lambda2, lambda3)
            loss_value.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        if save_images and ((i < 100) or (i % save_every == 0)):
            save_image(x, y, x0, y0, i, output_dir)
        # if save_images and (i % save_every == 0):
        #     save_image(x, y, x0, y0, i, output_dir)

        if i % 10 == 0:
            H = TorchFuncs.entropy(x, y)
            M = TorchFuncs.mean(x, y)
            S = TorchFuncs.stddev(x, y)
            N = TorchFuncs.norm(x, y)
            print('iteration: {}, loss: {:0.3f}, mu: {:0.3f}, sigma: {:0.3f}, norm: {:0.3f}, entropy: {:0.3f}'\
                  .format(i, loss_value, M, S, N, H)
            )

    return x, y

def main():
    # Make folders for saving images
    root = Path('images') / 'maxentropy'
    shutil.rmtree(root)
    output_dir_greedy = root / 'greedy'
    output_dir_gradient_descent = root / 'gradient_descent'
    output_dir_nnet = root / 'nnet'
    output_dir_greedy.mkdir(parents=True)
    output_dir_gradient_descent.mkdir(parents=True)
    output_dir_nnet.mkdir(parents=True)
    x, y = generate_points(-5, 5, 128)
    n_iterations_gradient_descent = 100000
    n_iterations_nnet = 10000

    # best_x, best_y = greedy(
    #     x.copy(),
    #     y.copy(),
    #     mu0=1.,
    #     sigma0=0.5,
    #     lambda1=100.,
    #     lambda2=100.,
    #     lambda3=100.,
    #     n_iterations=n_iterations,
    #     save_images=True,
    #     save_every=(n_iterations // 100)
    # )

    # best_x, best_y = gradient_descent(
    #     x.copy(),
    #     y.copy(),
    #     mu0=1.,
    #     sigma0=0.5,
    #     lambda1=20.,
    #     lambda2=20.,
    #     lambda3=5.,
    #     learning_rate=5e-4,
    #     n_iterations=n_iterations_gradient_descent,
    #     output_dir=output_dir_gradient_descent,
    #     save_images=True,
    #     save_every=n_iterations_gradient_descent//100
    # )

    best_x, best_y = nnet(
        x.copy(),
        y.copy(),
        mu0=1.,
        sigma0=0.5,
        lambda1=1,
        lambda2=2,
        lambda3=1,
        learning_rate=1e-4,
        n_iterations=n_iterations_nnet,
        output_dir=output_dir_nnet,
        save_images=True,
        save_every=n_iterations_nnet//200
    )

    pl.close('all')

if __name__ == '__main__':
    main()
