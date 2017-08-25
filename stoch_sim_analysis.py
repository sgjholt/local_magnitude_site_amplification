#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:43:08 2017

@author: james
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

cmap = cm.get_cmap('jet')

titles = ['dMl', 'Mw', 'SD', 'R', 'Q', 'ko', 'fo']

df = pd.read_csv('test.out', names=titles)
df.R = df.R.apply(np.log10)

save=True

for fo in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for ko in [0.005, 0.01, 0.02, 0.04]:
        for SD in [10, 50, 100, 200]:
            #for dist in [2, 20, 200]:
            for Q in [600, 1200, 2400]:
                plot = df.query('SD=={0} & Q =={1} & fo=={2} & ko=={3}'.format(SD, Q, fo, ko)).plot(
                    x='Mw', y='dMl', c='R', kind='scatter', title=
                    'Simulated ML amplification: ' + r'$SD={0}$, $Q={1}$, $f_0={2}$, $\kappa^0={3}$ '.format(SD, Q, fo, ko), cmap=cmap)
                plt.ylabel(r'$\Delta$ $M_L$')
                plt.xlabel(r'$M_w$')
                if save:
                    fig = plot.get_figure()
                    fig.savefig('plots/fo={0}_ko={1}_SD={2}_Q={3}_DMl_Mw.eps'.format(fo, ko, SD, Q))