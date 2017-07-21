import numpy as np
import matplotlib.pyplot as plt


def distance_v_deltaML(parsed_data_file, logD=True):
    data = parsed_data_file['data']
    meta = parsed_data_file['meta']
    plt.scatter(data[:, 1], data[:, 2], 20, data[:, 0], alpha=0.5, )
    plt.title('Simulated ML amplification: '+r'$\Delta\sigma={0}$, $\kappa0={1}$ '.format(meta['sd'], meta['k0']))
    plt.ylabel(r'$\Delta$ $M_L$')
    if logD:
        plt.xlabel(r'$ln(R_{hyp})$')
        plt.gca().set_xscale('log')
    else:
        plt.xlabel(r'$R_{hyp}$')
    plt.grid(which='both')
    plt.colorbar()
    plt.savefig('plots/dML_v_rhyp_SD={0}_kappa0={1}.eps'.format(meta['sd'], meta['k0']))
    plt.close()


def deltaML_v_ML(parsed_data_file):
    data = parsed_data_file['data']
    meta = parsed_data_file['meta']
    plt.scatter(data[:, 0], data[:, 2], 20, data[:, 1], alpha=0.5)
    plt.title('Simulated ML amplification (bedrock to surface): '+r'$\Delta\sigma={0}$, $\kappa0={1}$ '.format(meta['sd'], meta['k0']))
    plt.ylabel(r'$\Delta$ $M_L$')
    plt.xlabel(r'$M_L$')
    plt.grid(which='both')
    plt.colorbar()
    plt.savefig('plots/dML_v_ML_SD={0}_kappa0={1}.eps'.format(meta['sd'], meta['k0']))
    plt.close()