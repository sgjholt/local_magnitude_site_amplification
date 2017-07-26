import matplotlib.pyplot as plt


def distance_v_deltaML(parsed_data_file, logD=True):
    data = parsed_data_file['data']
    meta = parsed_data_file['meta']
    plt.scatter(data[:, 1], data[:, 2], 20, data[:, 0], alpha=0.5, )
    plt.title(
        'Simulated ML amplification: ' + r'$\Delta\sigma={0}$, $\kappa^0={1}$, $Q={2}$ '.format(meta['sd'], meta['k0'],
                                                                                               meta['q']))
    plt.ylabel(r'$\Delta$ $M_L$')
    plt.xlabel(r'$R_{hyp}$')
    if logD:
        plt.gca().set_xscale('log')
    plt.grid(which='both')
    plt.colorbar()
    plt.savefig('plots/dML_v_rhyp_SD={0}_kappa0={1}_Q={2}.eps'.format(meta['sd'], meta['k0'], meta['q']))
    plt.close()


def deltaML_v_ML(parsed_data_file):
    data = parsed_data_file['data']
    meta = parsed_data_file['meta']
    plt.scatter(data[:, 0], data[:, 2], 20, data[:, 1], alpha=0.5)
    plt.title(
        'Simulated ML amplification: ' + r'$\Delta\sigma={0}$, $\kappa^0={1}$, $Q={2}$ '.format(meta['sd'], meta['k0'],
                                                                                               meta['q']))
    plt.ylabel(r'$\Delta$ $M_L$')
    plt.xlabel(r'$M_w$')
    plt.grid(which='both')
    plt.colorbar()
    plt.savefig('plots/dML_v_ML_SD={0}_kappa0={1}_Q={2}.eps'.format(meta['sd'], meta['k0'], meta['q']))
    plt.close()