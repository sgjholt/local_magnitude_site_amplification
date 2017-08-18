#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:43:08 2017

@author: james
"""

import pandas as pd
import numpy as np
from matplotlib import cm
cmap = cm.get_cmap('jet')

titles = ['dMl', 'Mw', 'SD', 'R', 'Q', 'ko', 'fo']

df = pd.read_csv('test.out', names=titles)


for fo in [1, 10]:
    for ko in [0.005, 0.04]:
        for SD in [10, 100]:
            #for dist in [2, 20, 200]:
            for Q in [1200]:
                plot = df.query('SD=={0} & Q =={1} & fo=={2} & ko=={3}'.format(SD, Q, fo, ko)).plot(
                    x='Mw', y='dMl', c='R', kind='scatter', title=
                    'Simulated ML amplification: ' + r'$SD={0}$, $Q={1}$, $f_0={2}$, $\kappa^0={3}$ '.format(SD, Q, fo, ko), cmap=cmap)
                plt.ylabel(r'$\Delta$ $M_L$')
                