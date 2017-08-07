import pandas as pd
from glob import glob
import subprocess
from parsers import parse_for_pandas
import matplotlib.pyplot as plt

for f in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:

    path = '/home/james/Dropbox/ML_sims/f0_{0}hz/'.format(f)

    first = True
    for path in sorted(glob(path+'*.dat'), key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                                               float(x.split('/')[-1].split('_')[1]),
                                                               int(x.split('/')[-1].split('_')[2]))):
        if first:
            df = pd.DataFrame(parse_for_pandas(path))
            first = False

        else:
            tmp = pd.DataFrame(parse_for_pandas(path))
            df = pd.concat([df, tmp], axis=0)


    # once data frame is constructed - plot and save the figures (deltaML v stress drop)
    for mw in [2, 4, 6]:
        for dist in [10, 50, 100]:
            for q in [600, 1200, 2400]:
                plot = df[((df.distance-dist)/dist).abs() <= 0.26].query('mw=={0} & q =={1}'.format(mw, q)).plot(
                    x='sd', y='deltaML', c='k0', kind='scatter', title=
                    'Simulated ML amplification: ' + r'$M_w={0}$, $R_{{hyp}}={1}$, $Q={2}$ '.format(mw, dist, q))
                plt.ylabel(r'$\Delta$ $M_L$')
                plt.xlabel(r'$\Delta\sigma$')
                fig = plot.get_figure()
                fig.savefig('plots/mw={0}_dist={1}_q={2}_sdVk0.eps'.format(mw, dist, q))
                plt.close()

    subprocess.check_call(
        'gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile={0}hz_sdvk0.pdf plots/*.eps'.format(f),
        shell=True)
    subprocess.check_call('mv {0}hz_sdvk0.pdf plots/ampl_plots/.'.format(f), shell=True)

subprocess.check_call('rm plots/*.eps', shell=True)