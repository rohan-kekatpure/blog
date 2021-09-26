import numpy as np
import matplotlib.pyplot as pl
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

def main():
    x = np.linspace(-5, 5, 100)
    y0 = np.exp(-x * x)
    y1 = 1. / (2 + x * x)
    y2 = np.power(np.cosh(x), -2)
    y3 = np.sin(3 * x) / (3 * x)

    mosaic = \
    '''
    ABC
    '''
    ax = pl.figure(constrained_layout=True, figsize=(9, 3)).subplot_mosaic(mosaic=mosaic)

    # Rational
    ax['A'].plot(x, y0, 'k-', lw=2, alpha=0.3)
    ax['A'].plot(x, y1, 'r-', lw=2)
    ax['A'].text(0.7, 0.9, r'$$\frac{\alpha_m}{\beta_m + x^2}$$', fontsize=14, transform=ax['A'].transAxes, color='k')

    # Sech2
    ax['B'].plot(x, y0, 'k-', lw=2, alpha=0.3)
    ax['B'].plot(x, y2, 'g-', lw=2)
    ax['B'].text(0.7, 0.9, r'$\textrm{sech}^2(mx)$', fontsize=14, transform=ax['B'].transAxes, color='k')

    # Sinc
    ax['C'].plot(x, y0, 'k-', lw=2, alpha=0.3)
    ax['C'].plot(x, y3, 'b-', lw=2)
    ax['C'].text(0.7, 0.9, r'$$\frac{\sin (mx)}{mx}$$', fontsize=14, transform=ax['C'].transAxes, color='k')

    for _, a in ax.items():
        a.xaxis.set_ticklabels([])
        a.yaxis.set_ticklabels([])
        a.set_ylim([-0.3, 1.2])
    pl.savefig('gaussian_approx_candidates.png')

if __name__ == '__main__':
    main()