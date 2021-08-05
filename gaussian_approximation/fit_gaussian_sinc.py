import matplotlib.pyplot as pl
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
import numpy as np
from pathlib import Path
import shutil
from scipy.linalg import solve_triangular

def _generate_figures(coef_list, images_dir):
    N = len(coef_list)
    T, _ = coef_list[0].shape
    x = np.linspace(-20, 20, 200)

    for t in range(T):
        pl.close('all')
        if N == 1:
            figsize = (10, 5)
        else:
            figsize = (8, 10)
        fig, ax = pl.subplots(nrows=N, ncols=2, figsize=figsize)

        for j in range(N):
            coefs = coef_list[j]
            M = coefs.shape[1]
            q = np.arange(1, M + 1, dtype=float)

            # Choose axes
            if N == 1:
                ax0 = ax[0]
                ax1 = ax[1]
            else:
                ax0 = ax[j, 0]
                ax1 = ax[j, 1]

            # Plot orbit
            alphas = coefs
            markerline, stemlines, baseline = ax0.stem(
                q, alphas[t], linefmt='grey', markerfmt='o', bottom=0,
                use_line_collection=True
            )

            markerline.set_markerfacecolor('grey')
            ax0.set_ylim([-0.1, 0.5])
            val = np.sqrt(np.pi) * (alphas[t, :] / q).sum()
            txt = r'$$\sum_{m = 1}^{M}\frac{\alpha_m\sqrt{\pi}}{m}= %0.4f$$' % val
            ax0.text(0.4, 0.8, txt, transform=ax0.transAxes, fontsize=14)
            ax0.set_xlabel(r'$m$', fontsize=16)
            ax0.set_ylabel(r'$\alpha_m$', fontsize=16)

            # Plot convergence to Gaussian
            # Plot convergence to Gaussian
            a = alphas[t, :].reshape(-1, 1)
            qx = q.reshape(-1, 1) * x
            y = (a * np.sin(qx) / qx).sum(axis=0)
            z = np.exp(-x * x)
            ax1.plot(x, z, 'k-')
            ax1.plot(x, y, 'r-')
            ax1.set_xlim([-20, 20])
            ax1.set_ylim([-0.1, 1.25])
            ax1.set_xlabel(r'$x$', fontsize=16)
            ax1.set_ylabel(r'$f(x), g(x)$', fontsize=16)

            fig.tight_layout()
            # pl.subplots_adjust(wspace=0.1, hspace=None)
    
        # Save
        figname = f'i_{t:04d}.png'
        figpath = images_dir / figname
        fig.savefig(figpath.as_posix())

def _fit_sinc_gradient_descent(x, num_base_functions, niters,
                               learning_rate, regularization_param):
    x = np.clip(x, 1e-6, None)
    M = num_base_functions
    reg = regularization_param
    alphas = np.random.random((M, 1))
    b = np.arange(1, M + 1, dtype=float).reshape(-1, 1)

    coefs = []
    v_low, v_high = -3., 3.
    num_images = 256
    tsteps = np.unique(np.logspace(0, np.log(niters), num_images).astype(int))

    for t in range(niters):
        bx = b * x
        sincbx = np.sin(bx) / bx
        v1 = alphas * sincbx
        r = np.exp(-x * x) - v1.sum(axis=0)
        v2 = -2.0 * r * sincbx

        # Learn alphas
        dla = v2.sum(axis=1).reshape(-1, 1)
        alphas -= learning_rate * dla + reg * alphas
        alphas = np.clip(alphas, v_low, v_high)

        if t in tsteps:
            coefs.append(alphas.tolist())

        if t % 10000 == 0:
            print(f'iteration: {t}')
            # print(f'coefs = \n{alphas}')

    coefs = np.array(coefs).reshape(-1, M)
    np.save('_coefs.npy', np.array(coefs))
    return alphas

def _fit_sinc_analytical(num_base_functions):
    """
    Compute coefficients of a sinc fit to Gaussian analytically
    based on matrix equations
    """
    M = num_base_functions
    q = np.arange(1, M + 1, dtype=float)

    # Construct b
    b = np.exp(-q * q / 4) / np.sqrt(np.pi)

    # Construct A
    v = 1. / q
    A = np.tile(v, (M, 1))
    A = np.triu(A, 0)
    E = np.ones((M, M))
    np.fill_diagonal(E, 0.5)
    A *= E

    # Now solve the upper triangular system
    alphas = solve_triangular(A, b, lower=False)
    print(f'cs = {np.sqrt(np.pi) * (alphas/q).sum()}')
    return alphas

def _fit_sync_linear_regression(x, num_base_functions):

    M = num_base_functions
    x = np.clip(x, 1e-6, None)
    mvals = np.arange(1, M + 1, dtype=float).reshape(-1, 1)
    mx = (mvals * x).T
    X = np.sin(mx) / mx
    y = np.exp(-x * x)
    xtx = X.T @ X
    xty = X.T @ y
    alphas = np.linalg.inv(xtx) @ xty
    return alphas

def plot_analytical_fit(M, alphas):
    alphas = alphas.reshape(-1, 1)
    q = np.arange(1, M + 1, dtype=float).reshape(-1, 1)
    x_min, x_max = -20.0, 20.0
    x = np.linspace(x_min, x_max, 200)
    qx = q * x
    S = alphas * np.sin(qx) / qx
    y = np.exp(-x * x)
    yfit = S.sum(axis=0)
    # pl.plot(x, y, 'k')
    pl.plot(x, yfit - y, 'r')
    pl.show()

def main():
    num_base_functions = 8
    MILLION = 1000000
    niters = int(0.05 * MILLION)
    learning_rate = 1e-4
    regularization_param = 0
    x = np.linspace(0, 6, 256)

    # Solve for alphas using gradient descent
    alphas = _fit_sinc_gradient_descent(x, num_base_functions, niters,
                                   learning_rate, regularization_param)

    # Solve for alpha using linear regression normal equation
    alphas = _fit_sync_linear_regression(x, num_base_functions)

    # Solve for alpha using
    alphas = _fit_sinc_analytical(num_base_functions)

    images_dir = Path('_images')
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir()

    coefs = np.load('_coefs.npy')
    # _generate_figures([coefs], images_dir)

    # Generate fig for M = 8, 16, 32; uncomment to generate figure
    c1 = np.load('_coefs_sinc_m8.npy')
    c2 = np.load('_coefs_sinc_m16.npy')
    c3 = np.load('_coefs_sinc_m32.npy')
    coefs = [c1, c2, c3]
    _generate_figures(coefs, images_dir)

if __name__ == '__main__':
    main()
