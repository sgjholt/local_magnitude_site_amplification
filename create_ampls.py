from SiteMethods import ShTransferFunction
from stoch_sim import QWL_amp
import numpy as np

num_samps = 50
mod = {'Vs': [100, 1500], 'Dn': [1600, 2600], 'Hl': [25, 0], 'Qs': [10, 100],
       'Freqs': np.logspace(np.log10(0.01), np.log10(100), num_samps).tolist()}

#for Vs in [x*100 for x in range(1, 11)]:
#    mod['Vs'][0] = Vs
#    # print(mod['Vs'][0])
#    out = np.abs(ShTransferFunction(mod['Hl'], mod['Vs'], mod['Dn'], mod['Qs'], mod['Freqs']))
#    np.savetxt('f0_{:02d}hz.ampl'.format(int(Vs/100)), np.hstack((np.array(mod['Freqs'])[::, None], out)), fmt='%02.3f', header='50', comments='')


amps = [QWL_amp(x, mod['Freqs']) for x in range(1,11)]
for i, amp in enumerate(amps):
        np.savetxt('/home/james/Dropbox/ML_sims/ampl/f0_{:02d}hz.ampl'.format(int(i+1)), np.hstack(
                (np.array(
                        mod['Freqs'])[::, None], amp[::, None])), fmt='%02.3f', header='50', comments='')
    