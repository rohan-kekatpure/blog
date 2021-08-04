import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pl

def save_image(r, theta, x0, y0, image_index, out_dir):
    x0 = x0.detach().numpy()
    y0 = y0.detach().numpy()
    x1, y1 = pol2cart(r, theta)
    x1 = x1.detach().numpy()
    y1 = y1.detach().numpy()
    pl.close('all')
    _, ax = pl.subplots()
    ax.plot(x0, y0, 'r-', lw=1)
    ax.plot(x1, y1, 'g-', lw=1)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect('equal', 'box')
    pl.savefig('{}/image_{}.png'.format(out_dir, image_index))

def pol2cart(r, theta):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    x = torch.cat((x, torch.tensor([x[0]])))
    y = torch.cat((y, torch.tensor([y[0]])))
    return x, y

def perimeter(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    perim = torch.sqrt((dx*dx + dy * dy)).sum()
    return perim

class Inet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Inet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out, bias=False)
        )

    def forward(self, input_):
        return self.main(input_)

def iso_loss(r, theta, lambda1, lambda2):
    P0 = torch.tensor(2 * np.pi)
    area = 0.5 * torch.trapz(r * r, theta)
    x, y = pol2cart(r, theta)
    perim = perimeter(x, y)
    loss = -area + lambda1 * (perim - P0) ** 2 + lambda2 * (x.mean() ** 2 + y.mean() ** 2)
    return loss

def nnet_optimizer(r, theta, learning_rate, lambda1, lambda2, n_iterations, save_images, save_every):
    num_points = r.shape[0]
    theta = torch.tensor(theta)
    r = torch.tensor(r)
    x0, y0 = pol2cart(r, theta)
    model = Inet(1, 100, num_points)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    const = torch.tensor([1.])
    for i in range(n_iterations):
        r = model(const)
        loss = iso_loss(r, theta, lambda1, lambda2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if save_images and ((i < 100) or (i % save_every == 0)):
            save_image(r, theta, x0, y0, i, 'img/nnet')
        print('iteration: {}, loss: {}'.format(i, loss.item()))

    return r, theta
