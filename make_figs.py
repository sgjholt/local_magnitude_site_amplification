import parsers as par
import plot_routines as plr
import matplotlib.pyplot as plt

path = '/home/james/Desktop/ML_sims/200_0.005_deltaMl.dat'


plr.distance_v_deltaML(par.parse_deltaml_distance(path))

plr.deltaML_v_ML(par.parse_deltaml_distance(path))
