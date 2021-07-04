import matplotlib.pyplot as pl
import numpy as np
from pathlib import Path
import shutil

def _compute_fit():
    M = 8
    alphas = np.ones(M)
    betas = np.ones(M)
    niters = 10000
    learning_rate = 5e-4
    x = np.linspace(-5, 5, 1000)

    for t in range(niters):
        for m in range(M):
            dla = dlb = 0.0
            am = alphas[m]
            bm = betas[m]
            for i in range(len(x)):                
                U = np.sum(alphas/(betas + x[i] ** 2))
                V = np.exp(-x[i] * x[i]) - U
                dla += -2.0 / (bm + x[i] ** 2) * V
                dlb += (2.0 * am) / ((bm + x[i] ** 2) ** 2) * V 

            alphas[m] -= learning_rate * dla
            betas[m] -= learning_rate * dlb

        if t % 100 == 0:
            print(f'\niteration: {t}')
            print(f'alphas = {alphas}')
            print(f'betas = {betas}')

    return alphas, betas

def generate_media(coefs, images_folder):
    T, num_base_functions = coefs.shape
    M = num_base_functions // 2
    alphas = coefs[:, :M]
    betas = coefs[:, M:]
    x = np.linspace(-5, 5, 100)

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
        ax[0].set_ylim([-0, 2])
        ax[0].set_xticks([-1, -0.5, 0, 0.5, 1])

        # Plot convergence to Gaussian
        a = alphas[t, :].reshape(-1, 1)
        b = betas[t, :].reshape(-1, 1)
        y = (a / (b + x ** 2)).sum(axis=0)
        z = np.exp(-x * x)
        ax[1].plot(x, z, 'k-')
        ax[1].plot(x, y, 'r-')
        ax[1].set_xlim([-5, 5])
        ax[1].set_ylim([-0.1, 1.25])

        fig.tight_layout()

        # Save
        figname = f'i_{t:04d}.png'
        figpath = images_folder/figname
        fig.savefig(figpath.as_posix())

def compute_fit(x, num_base_functions, niters, 
                learning_rate, regularization_param):
    M = num_base_functions
    reg = regularization_param
    alphas = np.random.random((M, 1))
    betas = np.random.random((M, 1))
    coefs = []
    v_low, v_high = -3., 3.
    tsteps = np.unique(np.logspace(0, np.log(niters), 200).astype(int))
    for t in range(niters):
        v1 = alphas / (betas + x * x)
        r = np.exp(-x * x) - v1.sum(axis=0)
        q = betas + x * x
        v3 = -2.0 / q * r
        v4 = (2.0 * alphas) / (q * q) * r
        dla = v3.sum(axis=1).reshape(-1, 1)
        dlb = v4.sum(axis=1).reshape(-1, 1)
        alphas -= learning_rate * dla + reg * alphas
        betas -= learning_rate * dlb + reg * betas
        alphas = np.clip(alphas, v_low, v_high)
        betas = np.clip(betas, v_low, v_high)

        if t in tsteps:
            c = alphas.tolist() + betas.tolist()
            coefs.append(c)

        if t % 10000 == 0:
            print(f'\niteration: {t}')
            ab = np.column_stack((alphas, betas))
            print(f'coefs = \n{ab}')

    coefs = np.array(coefs).reshape(-1, 2 * M)
    np.save('_coefs.npy', np.array(coefs))
    return alphas, betas

def main():
    num_base_functions = 32
    MILLION = 2000000
    niters = int(1 * MILLION)
    learning_rate = 1e-4
    regularization_param = 0
    x = np.linspace(0, 5, 200)

    images_dir = Path('_images')
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir()

    alphas, betas = compute_fit(x, num_base_functions, niters, 
                                learning_rate, regularization_param)

    coefs = np.load('_coefs.npy')
    generate_media(coefs, images_dir)


if __name__ == '__main__':
    main()
