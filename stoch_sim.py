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

def expon_filter(t, T, ftm=2, eps=0.2, eta=0.05):
    
    b = -(eps*np.log(eta))/(1+eps*(np.log(eps)-1))
    c = b/eps
    a = (np.exp(1)/eps)**b
    tn = ftm*T
    
    return (a*(t/tn)**b)*(np.exp(-c*(t/tn)))
    

def stoch_signal_spectrum(fs=200, secs=20, plot=False):
    """

    :param fs:
    :param plot:
    :return:
    """
    # rfft used as real signal - only real freqs used
    t = np.linspace(0, secs, fs*secs)
    t_sig = np.random.randn(fs*secs)*expon_filter(t, 10)
    t_sig_pad = np.pad(t_sig, 1000, 'constant')
    sig = np.abs(np.fft.rfft(t_sig_pad))  # abs(fft) of windowed rand signal (normal, sig=1)
    sig /= np.sqrt(np.mean(sig**2))  # normalised such that RMS = 1
    freq = np.fft.rfftfreq(len(t_sig_pad), d=1/fs)  # frequencies of fft

    # print(len(sig), len(freq))

    if plot:
        fig, ax = plt.subplots(2, 1)
        ax.ravel()[0].plot(np.arange(0, secs, 1/fs), t_sig, 'r')
        ax.ravel()[0].set_title('time signal')
        ax.ravel()[0].set_xlabel('time [s]')
        ax.ravel()[0].set_ylabel('amplitude')
        ax.ravel()[1].loglog(freq, sig, 'r')
        ax.ravel()[1].set_title('frequency signal')
        ax.ravel()[1].set_xlabel('frequency [Hz]')
        ax.ravel()[1].set_ylabel('normalised amplitude')
    else:
        return freq, sig


def brune_source(Mw, SD, f, plot=False):
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
        return C*Mo*(2*np.pi*f)**2 / (1 + (f/fo)**2)
    

def whole_atten(R, Q, f, ko):
    """

    :param R:
    :param Q:
    :param f:
    :param ko:
    :return:
    """
    return R * np.exp(np.pi*f*((R/(Q*3.5))+ko))


def path_atten(R,Q,f):
    """
    """
    return R * np.exp(np.pi*f*(R/(Q*3.5)))


def site_amp(f0, f):
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

    return np.abs(ShTransferFunction(th, vs, dn, qs, f)).ravel()


def wood_and_filt(freqs, plot=False):
    """
    Calculate the Wood-Anderson seismometer response (displacement - mm) for a given range of frequencies.
    :param f:
    :return:
    """
    paz = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
           'zeros': [0 + 0j],
           'gain': 1.0,
           'sensitivity': 2080}

    resp = np.zeros(len(freqs))
    for i, freq in enumerate(freqs):
        jw = np.complex(0, 2 * np.pi * freq)  # angular frequency
        fac = np.complex(1, 0)
        for zero in paz['zeros']:  # numerator
            fac *= jw - zero
        for pole in paz['poles']:  # denominator
            fac /= jw - pole
        resp[i] = abs(fac) * paz['gain']
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

    if r <= r_tip:
        r = r_tip + 1  # add one km to the rupture tip - reference distance is at 1 km
        print('observation point < radius of rupture - new observation point @ {} km'.format(r))

    return np.round(r, 2)

# ----------script begin------------#
f = np.linspace(0.1, 25, 1000)
# f = np.linspace(0, 25, 100)
plot = False
#plt.ion()
for Mw in [2, 4, 6]:
    for SD in [10, 100]:
        for R in [2, 20, 200]:
            R = check_distance(R, Mw)
            for Q in [1200]:
                for ko in [0.005, 0.04]:
                    for fo in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        #for _ in range(10):
                        freq, stoch_sig = stoch_signal_spectrum()
                        stoch_sig = np.interp(f, freq, stoch_sig)

                        eq_spec_surf = ((stoch_sig*brune_source(Mw, SD, f)/whole_atten(
                            R+0.025, Q, f, ko))*site_amp(fo, f))/(2*np.pi*f)**2  # displacement
                        eq_spec_surf *= wood_and_filt(f)  # * (2*np.pi*f)**2

                        eq_spec_bh = (stoch_sig*brune_source(Mw, SD, f)/path_atten(R, Q, f))/(2*np.pi*f)**2
                        eq_spec_bh *= wood_and_filt(f)  # * (2*np.pi*f)**2

                        if plot:
                            plt.suptitle('Simulated Wood Anderson Spectra: Surface and Borehole')
                            plt.title(
                                r'$M_w:{}$, $\Delta\sigma:{}$ $Bars$, $R_{{hyp}}:{}$ $km$, $Q:{}$, $\kappa^0:{}$, $f_0:{}$ $Hz$'.format(
                                    Mw, SD, R, Q, ko, fo))
                            plt.loglog(1/f, eq_spec_surf, 'r')
                            plt.loglog(1/f, eq_spec_bh, 'k')
                            # plt.xlabel(r'$Frequency$ $[Hz]$')
                            plt.xlabel(r'$Period$ $[s]$')
                            plt.ylabel(r'$Displacement$ $Spectra$ $[cm s]$')
                            # plt.legend(loc=2)
                            plt.grid(which='both')
                        t_surf = np.fft.irfft(eq_spec_surf)[0:int(len(eq_spec_surf) / 2)]
                        t_bh = np.fft.irfft(eq_spec_bh)[0:int(len(eq_spec_bh) / 2)]
                        print('Delta ML:{}, Mw:{}, SD:{}, R:{}, Q:{}, ko:{}, fo:{}'.format(np.round(np.log10(t_surf.max() / t_bh.max()), 2), Mw, SD, R, Q, ko, fo))
                        if plot:
                            plt.title('Delta ML:{}, Mw:{}, SD:{}, R:{}, Q:{}, ko:{}, fo:{}'.format(
                            np.log10(t_surf.max() / t_bh.max()), Mw, SD, R, Q, ko, fo))
                            plt.plot(t_surf, 'r')
                            plt.plot(t_bh, 'k-')
                            plt.ylabel(r'$Displacement$ $cm$')
                       
                            plt.show()
                            #_ = input('Press ENTER to continue.')
                            #time.sleep(5)
                            plt.close()
