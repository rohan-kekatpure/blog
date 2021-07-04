import matplotlib.pyplot as pl
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

import numpy as np

def main():
    zlist = [5, 10, 20, 100]
    fig, ax = pl.subplots(nrows=2, ncols=2)
    ax = ax.ravel()
    for i, z in enumerate(zlist):
        d = 3.0 * np.sqrt(2 * z)
        x = np.linspace(max(0, z - d), z + d, 200)
        y = np.power(x, z) * np.exp(-x)
        g = 1./np.sqrt(2 * np.pi * z) * np.exp(-(x - z) ** 2 / (2 * z))
        ax[i].plot(x, y / y.max(), 'k')
        ax[i].plot(x, g / g.max(), color=(0.5, 0.5, 0.5))
        ax[i].text(0.05, 0.8, f'$z = {z}$', transform=ax[i].transAxes, fontsize=12)
        ax[i].set_xlabel('$x$', fontsize=12)
        ax[i].set_ylabel(f'$g(x)$', fontsize=12)
        if i == 0:
            ax[i].legend([f'$g(x)$', r'$\mathcal{N}(z, z)$'], frameon=False)
    fig.tight_layout()
    fig.savefig('images/gamma_evolution.png')

if __name__ == '__main__':
    main()
