#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:46:32 2017

@author: james
"""
import time
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from SiteMethods import ShTransferFunction

def expon_filter(t, T, ftm=2, eps=0.2, eta=0.05, plot=False):
    """

    :param t:
    :param T:
    :param ftm:
    :param eps:
    :param eta:
    :param plot:
    :return:
    """
    
    b = -(eps*np.log(eta))/(1+eps*(np.log(eps)-1))
    c = b/eps
    a = (np.exp(1)/eps)**b
    tn = ftm*T
    
    if plot:
        plt.plot(t/tn, (a*(t/tn)**b)*(np.exp(-c*(t/tn))), 'r-')
        plt.grid(which='both')
        plt.xlabel(r'$t/t_{n}$')
        plt.ylabel('Amplitude')
        
    else:
        return (a*(t/tn)**b)*(np.exp(-c*(t/tn)))
    

def stoch_signal_spectrum(fs=200, secs=20, plot=False):
    """
    
    :param fs:
    :param plot:
    :return: Displacement 
    """
    # rfft used as real signal - only real freqs used
    t = np.linspace(0, secs, fs*secs) # define arbitrary time vector
    t_sig = np.random.randn(fs*secs)*expon_filter(t, secs/2)
    t_sig_pad = np.pad(t_sig, int(fs*(secs/2)), 'constant') # pad the signal with zeros
    sig = np.fft.rfft(t_sig_pad)[1:]# abs(fft) of windowed rand signal (normal, sig=1)
    freq = np.fft.rfftfreq(len(t_sig_pad), d=1/fs)[1:] # the first value is 0 so ignore it (otherwise problems)
    sig /= (np.complex(1j)*2*np.pi*freq)**2 # integration in time domain (acc->disp = integrate twice)
    sig /= np.sqrt(np.mean(sig)**2) # normalise the signal such that the RMS=1
    
    if plot:
        fig, ax = plt.subplots(2, 1)
        ax.ravel()[0].plot(np.arange(0, secs, 1/fs), t_sig, 'r')
        ax.ravel()[0].set_title('time signal')
        ax.ravel()[0].set_xlabel('time [s]')
        ax.ravel()[0].set_ylabel('amplitude')
        ax.ravel()[1].loglog(freq, np.abs(sig), 'r')
        ax.ravel()[1].set_title('frequency signal')
        ax.ravel()[1].set_xlabel('frequency [Hz]')
        ax.ravel()[1].set_ylabel('normalised amplitude')
        fig.tight_layout()
    else:
        return freq, sig
    
    
def many_signals(num=100, fs=200, secs=20):
    """
    
    :param fs:
    :param plot:
    :return: Displacement 
    """
    
    #TODO: change this function to return a N*f set of arrays to be convolved with the rest of the things 
        
    # rfft used as real signal - only real freqs used
    t = np.linspace(0, secs, fs*secs) # define arbitrary time vector
    t_sig = np.random.randn((num,fs*secs)*expon_filter(t, secs/2)
    t_sig_pad = np.pad(t_sig, int(fs*(secs/2)), 'constant') # pad the signal with zeros
    sig = np.fft.rfft(t_sig_pad)[1:]# abs(fft) of windowed rand signal (normal, sig=1)
    freq = np.fft.rfftfreq(len(t_sig_pad), d=1/fs)[1:] # the first value is 0 so ignore it (otherwise problems)
    sig /= (np.complex(1j)*2*np.pi*freq)**2 # integration in time domain (acc->disp = integrate twice)
    sig /= np.sqrt(np.mean(sig)**2) # normalise the signal such that the RMS=1
    
    return freq, sig


def brune_source(Mw, SD, f, plot=False):
    """

    :param Mw:
    :param SD:
    :param f:
    :param plot:
    :return: displacement (cm)
    """
    # as defined in Boore (2003) Pure and Applied Geophysics
    # https://link.springer.com/article/10.1007/PL00012553
    Mo = 10**((Mw+10.7)*1.5)  # dyne.cm
    # print('Seismic Moment:'+str(Mo))
    C = (0.55*0.71*2)/(4*np.pi*2.8*(3.5**3))*10E-20  # the multiplication factor (10E-20) is a tiny note on page 642 ...
    fo = 4.9E6*3.5*(SD/Mo)**(1/3)
    # print('Corner Freq:'+str(fo))

    if plot:
        plt.loglog(f, C*Mo*(2*np.pi*f)**2 / (1 + (f/fo)**2), 'r')
        plt.plot(fo, C*Mo*(2*np.pi*fo)**2 / (1 + (fo/fo)**2), 'ok')
        plt.title(r'Brune Source Acceleration Spectra: $M_w={0}$, $\Delta\sigma={1}$, $f_0={2}$'.format(Mw, SD, np.round(fo, 2)))
        plt.ylabel(r'$Amplitude$ $[cm/s]$')
        plt.xlabel(r'$Frequency$ $[Hz]$')
        plt.grid(which='both')
    else:
        return (C*Mo*(2*np.pi*f)**2 / (1 + (f/fo)**2)) / (np.complex(1j)*2*np.pi*f)
    

def whole_atten(R, Q, f, ko, plot=False):
    """

    :param R:
    :param Q:
    :param f:
    :param ko:
    :return:
    """
    if plot:
        plt.title(r'$Attentuation$ $filter$ $for$ $R={} km$, $Q={}$, $\kappa^0={}$'.format(R, Q, ko))
        plt.loglog(f, R * np.exp(np.pi*f*((R/(Q*3.5))+ko)), 'r')
        plt.grid(which='both')
        plt.xlabel(r'$Frequency$ $[Hz]$')
        plt.ylabel(r'$Amplitude$')
    else:    
        return R * np.exp(np.pi*f*((R/(Q*3.5))+ko))


def path_atten(R,Q,f, plot=False):
    """
    
    :param R:
    :param Q:
    :param f:
    :return:
    """
    if plot:
        plt.title(r'$Attentuation$ $filter$ $for$ $R={} km$, $Q={}$'.format(R, Q))
        plt.loglog(f, R * np.exp(np.pi*f*(R/(Q*3.5))), 'r')
        plt.grid(which='both')
        plt.xlabel(r'$Frequency$ $[Hz]$')
        plt.ylabel(r'$Amplitude$')
    else:
        return R * np.exp(np.pi*f*(R/(Q*3.5)))


def site_amp(f0, f, plot=False):
    """

    :param f0:
    :param f:
    :return:
    """
    up_vs = {1: 100, 2: 200, 3: 300, 4: 400, 5: 500,
             6: 600, 7: 700, 8: 800, 9: 900, 10: 1000}
    th = [25, 0]
    dn = [1500, 2600]
    vs = [up_vs[f0], 1500]
    qs = [10, 100]
    
    if plot:
        plt.title(r'Single layer (25 m) over halfspace 1D-SHTF: $f_0={}$'.format(f0))
        plt.loglog(f, np.abs(ShTransferFunction(th, vs, dn, qs, f)).ravel(), 'r')
        plt.grid(which='both')
        plt.xlabel(r'$Frequency$ $[Hz]$')
        plt.ylabel(r'$Amplitude$')
    else:
        return np.abs(ShTransferFunction(th, vs, dn, qs, f)).ravel()


def wood_and_filt(freqs, plot=False):
    """
    Calculate the Wood-Anderson seismometer response (displacement - mm) for a given range of frequencies.
    :param f:
    :return:
    """
    paz = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
           'zeros': [0+0j, 0+0j],
           'gain': 1.0,
           'sensitivity': 2080}

    resp = np.zeros(len(freqs), dtype='complex')
    for i, freq in enumerate(freqs):
        jw = np.complex(0, 2 * np.pi * freq)  # angular frequency
        fac = np.complex(1, 0)
        for zero in paz['zeros']:  # numerator
            fac *= jw - zero
        for pole in paz['poles']:  # denominator
            fac /= jw - pole
        resp[i] = fac * paz['gain']
    if plot:
        plt.title('Displacement Response Spectrum for Wood-Anderson Seismometer')
        plt.loglog(freqs, resp, 'r')
        plt.grid(which='both')
        plt.ylabel(r'$Displacement$ $[cm.s]$')
        plt.xlabel(r'$Frequency$ $[Hz]$')
    else:
        return resp/10  # divide by 10 to output response in cm (orig in mm - I think)


def check_distance(r, mw):
    """
    Basic check to ensure that given the size of the event the observation point is greater than the fault radius.
    :param r:
    :param mw:
    :return:
    """
    a, b = 4.07, 0.98  # Wells and Coppersmith (1994) empirical fault area.

    r_tip = np.sqrt(10**((mw-a)/b)/np.pi)
    changed = False
    if r <= r_tip:
        r = r_tip + 1  # add one km to the rupture tip - reference distance is at 1 km
        changed=True

    return np.round(r, 2), changed 


if __name__ == '__main__':
    
    bucket_size = 100
    results = []
    now = time.time()
    # ----------script begin------------#
    f = np.linspace(0.1, 25, 1000)
    # f = np.linspace(0, 25, 100)
    wood_and = wood_and_filt(f)
    plot = False
    save = True
    #plt.ion()
    for fo in [1, 2, 3, 5, 10]:
        site_a = site_amp(fo, f)
        print(fo)
        for Mw in [1,2,3,4,5,6,7]:
            for SD in [10, 50, 100, 200]:
                #dist_check = [] # ***uncomment this if you want to use fault parrallel observation points/comment for fault normal
                for R in [2, 5, 10, 20, 40, 80, 160, 320]:
                    #R, changed = check_distance(R, Mw) #***
                    #if changed: #***
                    #    if R in dist_check: #***
                    #        R = dist_check[-1] + 10 #***
                    #    else:    #***
                    #        dist_check.append(R) #***
                    #    print('observation point < radius of rupture - new observation point @ {} km'.format(R)) #***
                    for Q in [2400, 1200, 600]:
                        for ko in [0.005, 0.01, 0.04]:
                            
                            sim_bucket_surf = np.zeros((bucket_size, len(f)), dtype='complex')
                            sim_bucket_bh = sim_bucket_surf.copy()
                            for i in range(bucket_size):
                                freq, stoch_sig = stoch_signal_spectrum()
                                stoch_sig = np.interp(f, freq, stoch_sig)
        
                                sim_bucket_surf[i] = ((stoch_sig*brune_source(Mw, SD, f)/whole_atten(
                                    R+0.0025, Q, f, ko))*site_a) #/(np.complex(1j)*2*np.pi)**2  # displacement
                                #sim_bucket_surf[i] *= wood_and  # * (2*np.pi*f)**2
                                
                                sim_bucket_bh[i] = (stoch_sig*brune_source(Mw, SD, f)/path_atten(R, Q, f))#/(np.complex(1j)*2*np.pi)**2
                                #sim_bucket_bh[i] *= wood_and  # * (2*np.pi*f)**2
                                
                            sim_bucket_surf *= wood_and
                            sim_bucket_bh *= wood_and
        
                            t_surf = np.fft.irfft(sim_bucket_surf) * len(sim_bucket_surf[0]) # normalised by 1/N remove norm by multiplying by the length
                            #t_surf = t_surf[0:int(len(eq_spec_surf) / 2)]
                            t_bh = np.fft.irfft(sim_bucket_bh) * len(sim_bucket_bh[0]) # normalised by 1/N remove norm by multiplying by the length
                            #t_bh = t_bh[0:int(len(eq_spec_bh) / 2)]
                            
                            results.append(list((np.round(np.mean(np.log10(t_surf.max(axis=1) / t_bh.max(axis=1))), 2), Mw, SD, R, Q, ko, fo)))
                            
                            #print('Delta ML:{}, Mw:{}, SD:{}, R:{}, Q:{}, ko:{}, fo:{}'.format((np.round(np.mean(np.log10(t_surf.max(axis=1) / t_bh.max(axis=1))), 2)), Mw, SD, R, Q, ko, fo))
                            
                            if plot:
                                fig, ax = plt.subplots(2, 1)
                                fig.suptitle('Simulated Wood Anderson Spectra: Surface and Borehole')
                                
                                
                                #ax.ravel()[0].set_title(
                                #    r'$M_w:{}$, $\Delta\sigma:{}$ $Bars$, $R_{{hyp}}:{}$ $km$, $Q:{}$, $\kappa^0:{}$, $f_0:{}$ $Hz$'.format(
                                #        Mw, SD, R, Q, ko, fo))
                                ax.ravel()[0].loglog(1/f, np.abs(sim_bucket_surf[0]), 'r')
                                ax.ravel()[0].loglog(1/f, np.abs(sim_bucket_bh[0]), 'k')
                                # plt.xlabel(r'$Frequency$ $[Hz]$')
                                ax.ravel()[0].set_xlabel(r'$Period$ $[s]$')
                                ax.ravel()[0].set_ylabel(r'$Displacement$ $Spectra$ $[cm s]$')
                                # plt.legend(loc=2)
                                ax.ravel()[0].grid(which='both')
                            
                        
                                #ax.ravel()[1].set_title('Delta ML:{}, Mw:{}, SD:{}, R:{}, Q:{}, ko:{}, fo:{}'.format(np.log10(t_surf.max() / t_bh.max()), Mw, SD, R, Q, ko, fo))
                                ax.ravel()[1].plot(t_surf[0], 'r')
                                ax.ravel()[1].plot(t_bh[0], 'k-')
                                ax.ravel()[1].set_ylabel(r'$Displacement$ $cm$')
                           
                                plt.show()
                                #_ = input('Press ENTER to continue.')
                                #time.sleep(5)
                                plt.close()
    print('Finished:{} seconds elapsed'.format(np.round(time.time()-now, 2)))
    if save:
        np.savetxt('test.out', np.array(results), delimiter=',', fmt='%02.3f')