#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:46:32 2017

@author: james
"""
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from SiteMethods import ShTransferFunction


def stoch_signal_spectrum(fs=200, secs=10, plot=False):
    """

    :param fs:
    :param plot:
    :return:
    """
    # rfft used as real signal - only real freqs used
    t_sig = np.random.randn(fs*secs)
    t_sig_pad = np.pad(t_sig, 1000, 'constant')
    sig = np.abs(np.fft.rfft(t_sig_pad*sg.hann(len(t_sig_pad))))  # abs(fft) of windowed rand signal (normal, sig=1)
    sig /= np.sqrt(np.mean(sig**2))  # normalised such that RMS = 1
    freq = np.fft.rfftfreq(len(t_sig_pad), d=1/fs)  # frequencies of fft

    print(len(sig), len(freq))

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
    print('Seismic Moment:'+str(Mo))
    C = (0.55*0.71*2)/(4*np.pi*2.8*(3.5**3))*10E-20  # the multiplication factor (10E-20) is a tiny note on page 642 ...
    fo = 4.9E6*3.5*(SD/Mo)**(1/3)
    print('Corner Freq:'+str(fo))

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
    return R**-1 * np.exp(np.pi*f*((R/(Q*3.5))+ko))


def path_atten(R,Q,f):
    """
    """
    return R**-1 * np.exp(np.pi*f*(R/(Q*3.5)))


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


def wood_and_filt(f):
    WOODANDERSON = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
                    'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}
    resp = np.zeros(len(f))

    for i, freq in enumerate(f):
        resp[i] = paz_2_amplitude_value_of_freq_resp(WOODANDERSON, freq)

    return resp


def paz_2_amplitude_value_of_freq_resp(paz, freq):
    """
    Returns Amplitude at one frequency for the given poles and zeros
    :param paz: Given poles and zeros
    :param freq: Given frequency
    The amplitude of the freq is estimated according to "Of Poles and
    Zeros", Frank Scherbaum, p 43.
    .. rubric:: Example
    #>>> paz = {'poles': [-4.44 + 4.44j, -4.44 - 4.44j],
    #...        'zeros': [0 + 0j, 0 + 0j],
    #...        'gain': 0.4}
    #>>> amp = paz_2_amplitude_value_of_freq_resp(paz, 1)
    #>>> print(round(amp, 7))
    0.2830262
    """
    jw = complex(0, 2 * np.pi * freq)  # angular frequency
    fac = complex(1, 0)
    for zero in paz['zeros']:  # numerator
        fac *= jw - zero
    for pole in paz['poles']:  # denominator
        fac /= jw - pole
    return abs(fac) * paz['gain']



Mw = 3.5
SD = 10
f = np.logspace(np.log10(0.1), np.log10(25), 100)
R = 5  # hypocentral distance
Q = 600
ko = 0.02
fo = 1

for _ in range(10):
    freq, stoch_sig = stoch_signal_spectrum()
    stoch_sig = np.interp(f, freq, stoch_sig)

    eq_spec_surf = ((stoch_sig*brune_source(Mw, SD, f)/whole_atten(R, Q, f, ko))*site_amp(fo, f))/(2*np.pi*f)**2  # displacement
    eq_spec_surf *= wood_and_filt(f)

    eq_spec_bh = (stoch_sig*brune_source(Mw, SD, f)/path_atten(R, Q, f))/(2*np.pi*f)**2
    eq_spec_bh *= wood_and_filt(f)

    plt.title(r'$M_w:{}$, $\Delta\sigma:{}$ $Bars$, $R_{{hyp}}:{}$ $km$'.format(Mw, SD, R))
    plt.loglog(f, eq_spec_surf, label='surface')
    plt.loglog(f, eq_spec_bh, label='borehole')
    plt.xlabel(r'$Frequency$ $[Hz]$')
    plt.ylabel(r'$Displacement$ $Spectra$ $[mm]$')
    plt.legend(loc=2)
    plt.grid(which='both')
plt.show()