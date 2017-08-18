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
from SiteMethods import ShTransferFunction, QwlApproxSolver, QwlImpedance

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
    

def stoch_signal_spectrum(fs=500, secs=40, plot=False):
    """
    
    :param fs:
    :param plot:
    :return: Displacement 
    """
    # rfft used as real signal - only real freqs used
    t = np.linspace(0, secs, int(fs*secs)) # define arbitrary time vector
    t_sig = np.random.randn(int(fs*secs))*expon_filter(t, secs)
    t_sig_pad = np.pad(t_sig, int(fs*(secs/2)), 'constant') # pad the signal with zeros
    sig = np.fft.rfft(t_sig_pad)[1:]# abs(fft) of windowed rand signal (normal, sig=1)
    freq = np.fft.rfftfreq(len(t_sig_pad), d=1/fs)[1:] # the first value is 0 so ignore it (otherwise problems)
    sig /= (np.complex(1j)*2*np.pi*freq)**2 # integration in time domain (acc->disp = integrate twice)
    sig /= np.sqrt(np.mean(sig**2)) # normalise the signal such that the RMS=1
    
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
    

def next_pow_2(x):
    if x == x**np.log2(x):
        return x
    else:
        return int(2**np.ceil(np.log2(x)))
    
def many_stoch_signals(num=100, fs=1000, secs=100, f_cap=1000):
    # TODO: This is broken, you need to sort out the pad widths - for some reason it is different when only selecting one trace...
    """

    :param fs:
    :param plot:
    :return: Displacement 
    """
     
    # rfft used as real signal - only real freqs used
    t = np.linspace(0, secs, int(fs*secs)) # define arbitrary time vector
    t_sig = np.random.randn(num,int(fs*secs))*expon_filter(t, secs/2)
    np2 = next_pow_2(len(t_sig)+100)
    
    print(len(t_sig), np2)
    #if np2 != len(t_sig):
    t_sig_pad = np.pad(t_sig, [(0, 0), (int(np2/2), int(np2/2))],'constant') # pad the signal with zeros
    #else:
    #    t_sig_pad = np.pad(t_sig, [(0, 0), (int((fs*len(t_sig))/8), int((fs*len(t_sig))/8))],'constant')
        
    print(len(t_sig_pad))
    sig = np.fft.rfft(t_sig_pad)[:,1:]# abs(fft) of windowed rand signal (normal, sig=1)
    freq = np.fft.rfftfreq(len(t_sig_pad[0]), d=1/fs)[1:] # the first value is 0 so ignore it (otherwise problems)
    #sig /= np.sqrt(np.mean(np.abs(sig)**2))
    sig /= np.sqrt(np.mean(np.abs(sig)**2, axis=1)[::,None]) 
    #sig /= (np.complex(1j)*2*np.pi*freq)**2 # integration in time domain (acc->disp = integrate twice)
   
    # normalise the signal such that the RMS=1
    f_cap = np.argwhere(freq<=f_cap).ravel()
    return freq[f_cap], sig[:,f_cap]


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
        plt.loglog(f, (C*Mo / (1 + (f/fo)**2)), 'r')
        #plt.plot(fo, (C*Mo / (1 + (f/fo)**2)), 'ok')
        plt.title(r'Brune Source Acceleration Spectra: $M_w={0}$, $\Delta\sigma={1}$, $f_0={2}$'.format(Mw, SD, np.round(fo, 2)))
        plt.ylabel(r'$Amplitude$ $[cm/s]$')
        plt.xlabel(r'$Frequency$ $[Hz]$')
        plt.grid(which='both')
    else:
        return (C*Mo / (1 + (f/fo)**2)) * (2*np.pi*f)**2
    

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
        plt.loglog(f, np.abs(ShTransferFunction(th, vs, dn, qs, f, Elastic=False)).ravel(), 'r') # change back to elastic
        plt.grid(which='both')
        plt.xlabel(r'$Frequency$ $[Hz]$')
        plt.ylabel(r'$Amplitude$')
    else:
        return np.abs(ShTransferFunction(th, vs, dn, qs, f, Elastic=False)).ravel()
    
def duration_calc(Mw,  R, SD=100):
    
    Mo = 10**((Mw+10.7)*1.5)  # dyne.cm
    
    fo = 4.9E6*3.5*(SD/Mo)**(1/3)

    swiss_dur, r = np.array([0, 5.3, 15.3, 17.3]), np.array([0, 38, 100, 200])
    if R <= r[-1]:
        dur = np.interp(R, r, swiss_dur)
    else: 
        m_after = 0.025
        c = swiss_dur[-1]
        dur = m_after*R+c
    #print('source duration :{}'.format(1/fo))
    
    return dur + 1/fo
    
    
    
def QWL_amp(f0, f, fs=1000, plot=False):
    
    fre = np.logspace(np.log10(0.01), np.log10(fs/2), 200)
    
    up_vs = {1: 100, 2: 200, 3: 300, 4: 400, 5: 500,
             6: 600, 7: 700, 8: 800, 9: 900, 10: 1000}
    th = [25, 0]
    dn = [1500, 2600]
    vs = [up_vs[f0], 1500]
    
    _, qwvs, qwdn = QwlApproxSolver(th, vs, dn, fre)
    amp = QwlImpedance(qwvs, qwdn, 3500, 2800)
    
    amp = np.interp(f, fre, amp)
    
    if plot:
        plt.title(r'Single layer (25 m) over halfspace 1D-SHTF: $f_0={}$'.format(f0))
        plt.loglog(f, amp, 'r')
        plt.grid(which='both')
        plt.xlabel(r'$Frequency$ $[Hz]$')
        plt.ylabel(r'$Amplitude$')
    else:
        return amp
    


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
        return resp  # divide by 10 to output response in cm (orig in mm - I think)
    
    
def resp(f, V=1, damp=1, T=0.8, n=0, kind='instrument', plot=False):
    """
    Simulated instrument response.
    # as defined in Boore (2003) Pure and Applied Geophysics
    # https://link.springer.com/article/10.1007/PL00012553
    :param f: Frequencies to be calculated over.
    :param V: Gain of instrument (1 for Wood-And and for Response Spectra).
    :param damp: Damping of instrument.
    :param T: Natural period (seconds) of instrument.
    :param n: 0 for displacment, 1 for velocity, 2 for acceleration.
    :param plot: Plot the response.
    :return: Response of instrument or ground in disp/vel/acc as desired.
    """
    
    if kind != 'instrument': # this is for ground motion assumes source given in displacement
        return (2*np.pi*f*np.complex(1j))**n
    
    resp = -V*(f**2)/(((f**2)-(1/T)**2) - 2*f*(1/T)*np.complex(1j)*damp)
    
    if plot:
        plt.loglog(f, np.abs(resp/(2*np.pi*f*np.complex(1j))**n))
        if n==0:
            plt.ylabel('Displacement')
        if n==1:
            plt.ylabel('Velocity')
        if n==2:
            plt.ylabel('Acceleration')
        plt.xlabel('Frequency [Hz]')
        plt.grid(which='both')
    else:
        return resp / (2*np.pi*f*np.complex(1j))**n


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
    
    # preamble - parameter selection
    plot = False 
    save = False
    
    fs = 1000
    #Dur = 100

    kind = 'instrument'
    unit=0 #0 = displacement cm, 1 = velocity cm/s, 2 = acceleration cm/s/s
    
    now = time.time()
    # ----------script begin------------#
    if unit == 0:
        u = 'cm'
    elif unit == 1:
        u = 'cm/s'
    else:
        u = 'cm/s/s'
    if save:
        results = []
        #issue_bin = []
    #first=True
    #plt.ion()
    for fo in [1, 10]:
        for Mw in [2, 3, 4, 5, 6]:
            #dist_check = [] # ***uncomment this if you want to use fault parrallel observation points/comment for fault normal
            for R in [10, 160, 320]:
                #R, changed = check_distance(R, Mw) #***
                    #if changed: #***
                    #    if R in dist_check: #***
                    #        R = dist_check[-1] + 10 #***
                    #    else:    #***
                    #        dist_check.append(R) #***
                for SD in [10, 100]:
                    Dur = duration_calc(Mw, R, SD=SD)
                    f, _ = many_stoch_signals(num=1, fs=fs, secs=Dur)
                    wood_and = resp(f, kind=kind, n=unit)
                    site_a = QWL_amp(fo, f)
                    
                    #    print('observation point < radius of rupture - new observation point @ {} km'.format(R)) #***
                    for Q in [1200]:
                        for ko in [0.005, 0.04]:
                            
                            #sim_bucket_surf = np.zeros((bucket_size, len(f)), dtype='complex')
                            #sim_bucket
                            #for i in range(bucket_size):
                            _, stoch_sig = many_stoch_signals(num=100, fs=fs, secs=Dur)
                            #stoch_sig = np.interp(f, freq, stoch_sig)
        
                            sim_bucket_surf = (stoch_sig*brune_source(Mw, SD, f)/whole_atten(R, Q, f, ko)*site_a) / (2*np.pi*f*np.complex(1j))**2  # displacement
                                #sim_bucket_surf[i] *= wood_and  # * (2*np.pi*f)**2
                                
                            sim_bucket_bh = (stoch_sig*brune_source(Mw, SD, f)/whole_atten(R, Q, f, ko)) / (2*np.pi*f*np.complex(1j))**2
                                #sim_bucket_bh[i] *= wood_and  # * (2*np.pi*f)**2
                                
                            sim_bucket_surf *= wood_and
                            sim_bucket_bh *= wood_and
        
                            t_surf = np.fft.irfft(sim_bucket_surf) * len(sim_bucket_surf[0]) # normalised by 1/N remove norm by multiplying by the length
                            #t_surf = t_surf[0:int(len(eq_spec_surf) / 2)]
                            t_bh = np.fft.irfft(sim_bucket_bh) * len(sim_bucket_bh[0]) # normalised by 1/N remove norm by multiplying by the length
                            #t_bh = t_bh[0:int(len(eq_spec_bh) / 2)]
                            
                            if save:
                                #if first:
                                #    first=False
                                #    if np.round(np.mean(np.log10(t_surf.max(axis=1) / t_bh.max(axis=1))), 2) == np.nan:
                                #        issue_bin.append(t_surf)
                                #        issue_bin.append(t_bh)
                                results.append(list((np.round(np.mean(np.log10(t_surf.max(axis=1) / t_bh.max(axis=1))), 2), Mw, SD, R, Q, ko, fo)))
                            
                            #if np.random.rand() < 0.25:
                            if kind == 'instrument':
                                print('Delta ML:{}, Mw:{}, SD:{}, R:{}, Q:{}, ko:{}, fo:{}'.format((np.round(np.mean(np.log10(np.abs(t_surf.max(axis=1)) / np.abs(t_bh.max(axis=1)))), 2)), Mw, SD, R, Q, ko, fo))
                            if kind == 'ground': 
                                print('Surf:{0} {1}, Bh:{2} {1}, Mw:{3}, SD:{4}, R:{5}, Q:{6}, ko:{7}, fo:{8}'.format(np.round(np.mean(np.abs(t_surf.max(axis=1))), 2), u, np.round(np.mean(np.abs(t_bh.max(axis=1))), 2), Mw, SD, R, Q, ko, fo))
                                
                            
                            if plot:
                                fig, ax = plt.subplots(2, 1)
                                fig.suptitle('Simulated Wood Anderson Spectra: Surface and Borehole')
                                
                                
                                #ax.ravel()[0].set_title(
                                #    r'$M_w:{}$, $\Delta\sigma:{}$ $Bars$, $R_{{hyp}}:{}$ $km$, $Q:{}$, $\kappa^0:{}$, $f_0:{}$ $Hz$'.format(
                                #        Mw, SD, R, Q, ko, fo))
                                ax.ravel()[0].loglog(f, np.abs(sim_bucket_surf[0]), 'r')
                                ax.ravel()[0].loglog(f, np.abs(sim_bucket_bh[0]), 'k')
                                # plt.xlabel(r'$Frequency$ $[Hz]$')
                                ax.ravel()[0].set_xlabel(r'$Period$ $[s]$')
                                ax.ravel()[0].set_ylabel(r'$Displacement$ $Spectra$ $[cm s]$')
                                # plt.legend(loc=2)
                                ax.ravel()[0].grid(which='both')
                            
                        
                                #ax.ravel()[1].set_title('Delta ML:{}, Mw:{}, SD:{}, R:{}, Q:{}, ko:{}, fo:{}'.format(np.log10(t_surf.max() / t_bh.max()), Mw, SD, R, Q, ko, fo))
                                ax.ravel()[1].plot(np.arange(0, len(t_surf[0]))/1000, t_surf[0], 'r')
                                ax.ravel()[1].plot(np.arange(0, len(t_surf[0]))/1000, t_bh[0], 'k-')
                                ax.ravel()[1].set_ylabel(r'$Displacement$ $cm$')
                           
                                plt.show()
                                #_ = input('Press ENTER to continue.')
                                #time.sleep(5)
                                plt.close()
    print('Finished:{} seconds elapsed'.format(np.round(time.time()-now, 2)))
    if save:
        np.savetxt('test.out', np.array(results), delimiter=',', fmt='%02.3f')
        
        
        
        
        
        
        
    