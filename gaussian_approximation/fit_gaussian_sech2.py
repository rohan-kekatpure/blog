import matplotlib.pyplot as pl
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
import numpy as np
from pathlib import Path
import shutil

def _generate_figures(coef_list, images_dir):
    N = len(coef_list)
    T, _ = coef_list[0].shape
    x = np.linspace(-5, 5, 200)

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
            ax0.set_ylim([-3, 3])
            val = 2./(np.sqrt(np.pi)) * (alphas[t, :] / q).sum()
            txt = r'$$\sum_{m = 1}^{%d}\frac{2\alpha_m}{m\sqrt{\pi}}= %0.4f$$' % (M, val)
            ax0.text(0.4, 0.8, txt, transform=ax0.transAxes, fontsize=14)
            ax0.set_xlabel(r'$m$', fontsize=16)
            ax0.set_ylabel(r'$\alpha_m$', fontsize=16)

            # Plot convergence to Gaussian
            # Plot convergence to Gaussian
            a = alphas[t, :].reshape(-1, 1)
            qx = q.reshape(-1, 1) * x
            sech2qx = np.power(np.cosh(qx), -2)
            y = (a * sech2qx).sum(axis=0)
            z = np.exp(-x * x)
            ax1.plot(x, z, 'k-')
            ax1.plot(x, y, 'r-')
            ax1.set_xlim([-5, 5])
            ax1.set_ylim([-0.1, 1.25])
            ax1.set_xlabel(r'$x$', fontsize=16)
            ax1.set_ylabel(r'$f(x), g(x)$', fontsize=16)

            fig.tight_layout()
            # pl.subplots_adjust(wspace=0.1, hspace=None)
    
        # Save
        figname = f'i_{t:04d}.png'
        figpath = images_dir / figname
        fig.savefig(figpath.as_posix())

def _fit_sech2_gradient_descent(x, num_base_functions, niters,
                                learning_rate, regularization_param):
    x = np.clip(x, 1e-6, None)
    M = num_base_functions
    reg = regularization_param
    alphas = np.random.random((M, 1))
    b = np.arange(1, M + 1, dtype=float).reshape(-1, 1)

    coefs = []
    v_low, v_high = -10., 10.
    num_images = 256
    tsteps = np.unique(np.logspace(0, np.log(niters), num_images).astype(int))

    for t in range(niters):
        bx = b * x
        sech2bx = np.power(np.cosh(bx), -2)
        v1 = alphas * sech2bx
        r = np.exp(-x * x) - v1.sum(axis=0)
        v2 = -2.0 * r * sech2bx

        # Learn alphas
        dla = v2.sum(axis=1).reshape(-1, 1)
        alphas -= learning_rate * dla + reg * alphas
        alphas = np.clip(alphas, v_low, v_high)

        if t in tsteps:
            coefs.append(alphas.tolist())

        if t % 10000 == 0:
            print(f'iteration: {t}')
            print(f'coefs = \n{alphas}')

    coefs = np.array(coefs).reshape(-1, M)
    np.save('_coefs.npy', np.array(coefs))
    return alphas

def main():
    num_base_functions = 64
    MILLION = 1000000
    niters = int(5 * MILLION)
    learning_rate = 1e-2
    regularization_param = 0
    x = np.linspace(0, 6, 128)

    # Solve for alphas using gradient descent
    # alphas = _fit_sech2_gradient_descent(x, num_base_functions, niters,
    #                                      learning_rate, regularization_param)

    images_dir = Path('_images')
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir()

    coefs_m8 = np.load('_coefs_m8.npy')
    coefs_m16 = np.load('_coefs_m16.npy')
    coefs_m32 = np.load('_coefs_m32.npy')
    coefs_m64 = np.load('_coefs_m64.npy')
    _generate_figures([coefs_m8, coefs_m16, coefs_m32, coefs_m64], images_dir)

if __name__ == '__main__':
    main()
