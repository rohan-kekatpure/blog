import matplotlib.pyplot as pl
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
import numpy as np
from pathlib import Path
import shutil

def _generate_composite_figure(coef_list, images_dir):
    N = len(coef_list)
    T, _ = coef_list[0].shape
    x = np.linspace(-5, 5, 100)

    for t in range(T):
        pl.close('all')
        fig, ax = pl.subplots(nrows=N, ncols=2, figsize=(8, 10))
            
        for j in range(N):
            coefs = coef_list[j]
            M = coefs.shape[1] // 2
            alphas = coefs[:, :M]
            betas = coefs[:, M:]
            ax0 = ax[j, 0]
            ax1 = ax[j, 1]

            # Plot orbit
            for i in range(M):
                ax0.plot(alphas[:t, i], betas[:t, i], 'k-', alpha=0.3)

            # Plot coordinates
            for i in range(M):
                ax0.plot(alphas[t, i], betas[t, i], 'ko', ms=7, alpha=0.6)

            ax0.set_xlim([-1, 1])
            ax0.set_ylim([-0, 2])
            ax0.set_xticks([-1, -0.5, 0, 0.5, 1])
            val = np.sqrt(np.pi) * (alphas[t, :] / np.sqrt(betas[t, :])).sum()
            txt = r'$$\sum_{m = 1}^{%d}\frac{\alpha_m\sqrt{\pi}}{\sqrt{\beta_m}}= %0.4f$$' % (M, val)
            ax0.text(0.5, 0.85, txt, transform=ax0.transAxes, fontsize=12)
            ax0.set_xlabel(r'$\alpha_m$', fontsize=16)
            ax0.set_ylabel(r'$\beta_m$', fontsize=16)
            ax0.tick_params(axis='x', labelsize=12)
            ax0.tick_params(axis='y', labelsize=12)

            # Plot convergence to Gaussian
            a = alphas[t, :].reshape(-1, 1)
            b = betas[t, :].reshape(-1, 1)
            y = (a / (b + x ** 2)).sum(axis=0)
            z = np.exp(-x * x)
            ax1.plot(x, z, 'k-')
            ax1.plot(x, y, 'r-')
            ax1.set_xlim([-5, 5])
            ax1.set_ylim([-0.1, 1.25])
            ax1.set_xlabel(r'$x$', fontsize=16)
            ax1.set_ylabel(r'$f(x), g(x)$', fontsize=16)
            ax1.tick_params(axis='x', labelsize=12)
            ax1.tick_params(axis='y', labelsize=12)

            fig.tight_layout()
            # pl.subplots_adjust(wspace=0.1, hspace=None)
    
        # Save
        figname = f'i_{t:04d}.png'
        figpath = images_dir / figname
        fig.savefig(figpath.as_posix())
        pl.close('all')

def _generate_media(coefs, images_dir):
    T, num_base_functions = coefs.shape
    M = num_base_functions // 2
    alphas = coefs[:, :M]
    betas = coefs[:, M:]
    x_min, x_max = -5, 5
    x = np.linspace(x_min, x_max, 200)
    for t in range(T):
        pl.close('all')
        fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(10, 5))    

        # Plot orbit
        for i in range(M):
            ax[0].plot(alphas[:t, i], betas[:t, i], 'k-', alpha=0.3)

        # Plot coordinates
        for i in range(M):
            ax[0].plot(alphas[t, i], betas[t, i], 'ko', ms=7, alpha=0.6)

        ax[0].set_xlim([-1, 1])
        ax[0].set_ylim([0.0, 2.0])
        ax[0].set_xticks([-1, -0.5, 0, 0.5, 1])
        val = np.sqrt(np.pi) * (alphas[t, :]/np.sqrt(betas[t, :])).sum()
        txt = r'$$\sum_{m = 1}^{M}\frac{\alpha_m\sqrt{\pi}}{\sqrt{\beta_m}}= %0.4f$$' % val
        ax[0].text(0.5, 0.9, txt, transform=ax[0].transAxes, fontsize=14)
        ax[0].set_xlabel(r'$\alpha_m$', fontsize=16)
        ax[0].set_ylabel(r'$\beta_m$', fontsize=16)

        # Plot convergence to Gaussian
        a = alphas[t, :].reshape(-1, 1)
        b = betas[t, :].reshape(-1, 1)
        y = (a / (b + x ** 2)).sum(axis=0)
        z = np.exp(-x * x)
        ax[1].plot(x, z, 'k-')
        ax[1].plot(x, y, 'r-')
        ax[1].set_xlim([x_min, x_max])
        ax[1].set_ylim([-0.1, 1.25])
        ax[1].set_xlabel(r'$x$', fontsize=16)
        ax[1].set_ylabel(r'$f(x), g(x)$', fontsize=16)
        fig.tight_layout()

        # Save
        figname = f'i_{t:04d}.png'
        figpath = images_dir / figname
        fig.savefig(figpath.as_posix())

def _fit_rational(x, num_base_functions, niters,
                  learning_rate, regularization_param):
    M = num_base_functions
    reg = regularization_param
    alphas = np.random.random((M, 1))
    betas = np.random.random((M, 1))
    coefs = []
    v_low, v_high = -3., 3.
    num_images = 200
    tsteps = np.unique(np.logspace(0, np.log(niters), num_images).astype(int))

    for t in range(niters):
        v1 = alphas / (betas + x * x)
        r = np.exp(-x * x) - v1.sum(axis=0)
        q = betas + x * x
        v3 = -2.0 / q * r
        v4 = (2.0 * alphas) / (q * q) * r

        # Learn alphas
        dla = v3.sum(axis=1).reshape(-1, 1)
        alphas -= learning_rate * dla + reg * alphas
        alphas = np.clip(alphas, v_low, v_high)

        # Learn betas
        dlb = v4.sum(axis=1).reshape(-1, 1)
        betas -= learning_rate * dlb + reg * betas
        betas = np.clip(betas, v_low, v_high)

        if t in tsteps:
            c = alphas.tolist() + betas.tolist()
            coefs.append(c)

        if t % 100000 == 0:
            print(f'\niteration: {t}')
            ab = np.column_stack((alphas, betas))
            print(f'coefs = \n{ab}')

    coefs = np.array(coefs).reshape(-1, 2 * M)
    np.save('_coefs.npy', np.array(coefs))
    return alphas, betas

def main():
    num_base_functions = 8
    MILLION = 1000000
    niters = int(1 * MILLION)
    learning_rate = 1e-4
    regularization_param = 0
    x = np.linspace(0, 6, 256)

    _, _ = _fit_rational(x, num_base_functions, niters,
                         learning_rate, regularization_param)

    images_dir = Path('_images')
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir()

    coefs = np.load('_coefs.npy')
    _generate_media(coefs, images_dir)

    # Generate fig for M = 20; uncomment to generate figure
    # c1 = np.load('_coefs_m20_1.npy')
    # c2 = np.load('_coefs_m20_2.npy')
    # c3 = np.load('_coefs_m20_3.npy')
    # coefs20 = [c1, c2, c3]
    # _generate_composite_figure(coefs20, images_dir)

    # Generate fig for M = 8, 20, 64; uncomment to generate figure
    # c1 = np.load('_coefs_m8.npy')
    # c2 = np.load('_coefs_m20.npy')
    # c3 = np.load('_coefs_m64.npy')
    # coefs = [c1, c2, c3]
    # _generate_composite_figure(coefs, images_dir)

if __name__ == '__main__':
    main()
