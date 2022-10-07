import numpy as np
import matplotlib.pyplot as plt

def plotCovariance(covariance_estimates):
    fig, axs = plt.subplots(5,5)
    DT = 0.1
    x = DT * np.arange(covariance_estimates.shape[2])
    for i in range(5):
        for j in range(5):
            y = covariance_estimates[i, j, :]
            axs[i,j].plot(x, y)
            axs[i,j].set_title('$\Sigma_{t'+ str((i,j)) + '}$')
    
    for ax in axs.flat:
        ax.set(xlabel='time (s)', ylabel='variance ($m^2$)')

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    
