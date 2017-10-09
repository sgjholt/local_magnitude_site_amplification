# -*- coding: utf-8 -*-
# plot_wa_sens.py
# Plots the earthquake scenario sensitivity curves (deltaML vs PG/D/V/A)
# from the time domain simulations. (stochastic method)


import os, shutil, errno
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt



def ignore_absent_file(func, path, exc_inf):
    except_instance = exc_inf[1]
    if isinstance(except_instance, OSError) and \
        except_instance.errno == errno.ENOENT:
        return
    raise except_instance

in_file = 'wa_disp.csv'

df = pd.read_csv(in_file, names=['dML', 'surf_motion', 'bh_motion', 'Mw', 'SD',
                                 'R', 'Q', 'ko', 'fo'])
tmp_dir = 'tmp/disp/'

try:
    os.makedirs(tmp_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

N = df.Mw.unique()
cmap = plt.get_cmap('jet',len(N))
norm = mpl.colors.Normalize(vmin=N.min(), vmax=N.max())
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array(N)

for fo in [1, 3, 10]:
    print('working on fo = {0}'.format(fo))
    for ko in df.ko.unique():
        for SD in df.SD.unique():
            for Q in df.Q.unique():
                for R in df.R.unique():
                    subset = df.query(
                            'SD=={0} & Q =={1} & fo=={2} & ko=={3} & R=={4}'.format(
                                    SD, Q, fo, ko, R))
                    
                    fig, ax = plt.subplots()
                    ax.scatter(subset.surf_motion, subset.dML, 20, subset.Mw, cmap=cmap)
                    ax.scatter(subset.bh_motion, subset.dML, 20, subset.Mw, cmap=cmap, marker=',')
                    ax.set_xscale("log")
                    mn, mx = (np.min([subset.surf_motion.values.min(),subset.bh_motion.values.min()]), 
                                        np.max([subset.surf_motion.values.max(), subset.bh_motion.values.max()]))    
                    ax.set_xlim([mn/1.5,mx*1.5])
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: str(x)))
                    ax.grid(which='both', alpha=0.4)
                    ax.set_ylabel(r'$\Delta$ $M_L$')
                    ax.set_xlabel(r'$PGD$ $[cm]$')
                    ax.legend(['srf', 'bh'])
                    cb = fig.colorbar(sm, ticks=N, boundaries=np.arange(0.75,7.25,0.5), ax=ax)
                    cb.set_label(r'$M_w$')
                    plt.suptitle(
                            'Simulated ML amplification: ' \
                            + r'$SD={0}$, $Q={1}$, $f_0={2}$, $\kappa^0={3}$, $R={4}$ '.format(
                                    SD, Q, fo, ko, R))
                    
                    
                    
                    plt.savefig('tmp/disp/sens_{0}_{1}_{2}_{3}_{4}.pdf'.format(SD, Q, fo, ko, R))
                    plt.close()
                    
#shutil.rmtree(tmp_dir, onerror=ignore_absent_file)
