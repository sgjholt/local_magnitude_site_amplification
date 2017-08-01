from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import parsers as par
import plot_routines as plr

# comment/uncomment as necessary
# ------------------Distance/Mw/DeltaMl Plots------------------------------------------------------#
# path = '/home/james/Dropbox/ML_sims/f0_10hz/'


#for path in sorted(glob(path+'*.dat'), key=lambda x: (int(x.split('/')[-1].split('_')[0]),
#                                                           float(x.split('/')[-1].split('_')[1]),
#                                                           int(x.split('/')[-1].split('_')[2]))):

#    print('processing: {}'.format(path.split('/')[-1]))
#    plr.distance_v_deltaML(par.parse_deltaml_distance(path))
#    plr.deltaML_v_ML(par.parse_deltaml_distance(path))

# ------------------Synthetic Amplification Function Plots------------------------------------------------------#
path = '/home/james/Dropbox/ML_sims/ampl/'

fig, ax = plt.subplots(5, 2, sharex=True, sharey=True)
ax = ax.ravel()
for i, path in enumerate(sorted(glob(path+'*.ampl'), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1].split('h')[0]))):
    dd = par.parse_ampl(path)
    print(dd)
    ax.ravel()[i].loglog(dd['data'][:, 0], dd['data'][:, 1], 'r', linewidth=3)
    ax.ravel()[i].hlines(1, 1, 10, colors='k', linestyles='dashed', linewidth=2)
    plt.ylim([0.8, 35])
    ax.ravel()[i].set_xticks([1, 5, 10])
    ax.ravel()[i].set_yticks([1, 10, 30])
    ax.ravel()[i].set_title(r'$f_0: {}$'.format(dd['f0'].upper()))

    if i in [4]:
        ax.ravel()[i].set_ylabel(r'$Amplification$')
    if i in [8, 9]:
        ax.ravel()[i].set_xlabel(r'$Frequency$ $[Hz]$')
    ax.ravel()[i].grid(which='both')
    plt.ylim([0, 40])
    ax.ravel()[i].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    ax.ravel()[i].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
plt.show()
#plt.savefig('plots/ampls.eps')