from SiteMethods import ShTransferFunction

mod = {'Vs': [100, 1500], 'Dn': [1600, 2600], 'Hl': [25, 0], 'Qs': [10, 100],
       'Freqs': np.logspace(np.log10(1), np.log10(10), 50).tolist()}

for Vs in [x*100 for x in range(1, 11)]:
    mod['Vs'][0] = Vs
    # print(mod['Vs'][0])
    out = np.abs(ShTransferFunction(mod['Hl'], mod['Vs'], mod['Dn'], mod['Qs'], mod['Freqs']))
    np.savetxt('f0_{:02d}hz.ampl'.format(int(Vs/100)), np.hstack((np.array(mod['Freqs'])[::, None], out)), fmt='%02.3f')
