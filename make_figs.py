from glob import glob
import pandas as pd
import parsers as par
import plot_routines as plr


path = '/home/james/Dropbox/ML_sims/'


for path in sorted(glob(path+'*.dat'), key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                                           float(x.split('/')[-1].split('_')[1]),
                                                           int(x.split('/')[-1].split('_')[2]))):



    #print('processing: {}'.format(path.split('/')[-1]))
    #plr.distance_v_deltaML(par.parse_deltaml_distance(path))
    #plr.deltaML_v_ML(par.parse_deltaml_distance(path))

    pass
