#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:18:52 2026

@author: susannakeith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, expon

plt.rcParams.update({'font.size': 14})  # increases everything (titles, labels, ticks, legend)

# ==========================================================================
# Define costs 
#    - these are just initial values to produce a plot and can be altered.
# ==========================================================================

C_HL = 3000000    # human loss              £3 million
C_RB = 2000000    # rebuild bridge          £2 million
C_CB = 15000        # 15000     # close bridge (~3 days)  £5000 per day?
C_S  = 5000       # 10000     # monitoring              £10,000   
C_RR = 80000      # riprap                  £30,000

# Normalising all costs to cost of rebuilding the bridge
gamma_HL = C_HL / C_RB
gamma_CB = C_CB / C_RB
gamma_RR = C_RR / C_RB
gamma_S = C_S / C_RB


def DN(Pf):
    v0 = 5
    lambda_F = v0 * Pf
    return (1 + gamma_HL)*lambda_F / (lambda_F + 0.05)

def RR(Pf):
    return Pf*gamma_RR / Pf

def f_y(y):
    return (1/0.25) * np.exp(-(y-1.086)/0.25)

def pf_y(y):
    y = np.array(y)
    return 0.1236556 + (0.002565349 - 0.1236556)/(1 + (y/8.51939)**1.038562)


def PVCM(PF_post, lambda_F, lambda_C, lambda_NF):
    lambda_F =np.array(lambda_F)
    return gamma_S + (1*lambda_F + gamma_CB*lambda_C + gamma_HL * lambda_NF)/ (lambda_F + 0.05)

def run_DT_failure_model(
    waterlevel_file,
    y0,
    y200,
    Wp,
    f_PS,
    f_PA,
    f_d,
    bias,
    sigma_DT_error,
    mu,
    sigma,
    DF_known,
    DFmedian,
    sigma_ln_DF,
    T=30,
    n_samples=10000,
    dt_range=(0,7),
    seed=1
):

    np.random.seed(seed)

    # -----------------------------------
    # 1. Load water levels & fit exponential exceedance
    # -----------------------------------
    df = pd.read_csv(waterlevel_file)
    data = df["Water level from Q (m)"].dropna()

    above = data > y0
    events_start = above & (~above.shift(1, fill_value=False))
    num_events = events_start.sum()

    nu = num_events / T

    exceedances = data[data > y0] - y0
    beta_exp = exceedances.mean()

    x = np.linspace(0,7,10000)
    G = 1 - expon.cdf(x, scale=beta_exp)
    ccdf_30years = 1 - np.exp(-nu * G * T)
    CDF = 1 - ccdf_30years   # Single event 


    # -----------------------------------
    # 2. D_T equation
    # -----------------------------------
    def D_T(y_SP):
        if y_SP <= 2.6 * Wp:
            f_y = 0.78 * (y_SP / Wp) ** 0.255
        else:
            f_y = 1
        return (1.5 * Wp * f_PS * f_PA * f_y * f_d) * bias

    # -----------------------------------
    # 3. Sample water levels & DT error
    # -----------------------------------    
    def T_of_y(y):
        f_y = np.where(y <= 2.6*Wp, 0.78 * (y / 2) ** 0.255, 1.0)
        return 3 * f_y

    def y_of_T(t):
        return 2 * (t / 2.34) ** (1 / 0.255)
          
    G_single_event =  1 - expon.cdf(x, scale=0.25)
    
    CDF_single_event = 1-G_single_event

    u = np.random.rand(10000)
    y_samples = np.interp(u, CDF, x) + y0  #this is 30 year Cdf
    
    dt_with_error = []
    for y in y_samples:
        error = np.random.normal(0, sigma_DT_error)
        dt = D_T(y) * (1+error)
        dt = max(dt,0.0)
        dt_with_error.append(dt)

    dt_with_error = np.array(dt_with_error)
    
    
    CDF_single_event = 1-G_single_event

    y_samples_single = np.interp(u, CDF_single_event, x) + y0# think

    dt_single = []
    for y in y_samples_single:
        error = np.random.normal(0, sigma_DT_error)
        dt = D_T(y) * (1+error)
        dt = max(dt,0.0)

        dt_single.append(dt)

    dt_single = np.array(dt_single)
    
    # -----------------------------------
    # 4. CCDF of DT
    # -----------------------------------
    dt_vals = np.linspace(dt_range[0], dt_range[1], 100)

    sorted_dt = np.sort(dt_with_error)

    n = len(sorted_dt)

    cdf_dt = np.arange(1,n+1)/n
    ccdf_dt = 1-cdf_dt

    ccdf_interp = np.interp(dt_vals, sorted_dt, ccdf_dt, left=1, right=0)
    ccdf_interp = np.maximum.accumulate(ccdf_interp[::-1])[::-1]
    
    # -----------------------------------
    # 4 (b). CCDF of DT Single event
    # -----------------------------------
    dt_vals = np.linspace(dt_range[0], dt_range[1], 100)

    sorted_dt_s = np.sort(dt_single)

    n = len(sorted_dt_s)

    cdf_dt_s = np.arange(1,n+1)/n
    ccdf_dt_s = 1-cdf_dt_s

    ccdf_interp_s = np.interp(dt_vals, sorted_dt_s, ccdf_dt_s, left=1, right=0)
    ccdf_interp_s = np.maximum.accumulate(ccdf_interp_s[::-1])[::-1]
    #plt.plot(dt_vals, ccdf_interp_s)

    #ccdf_interp_s = 1 - np.exp(-5*30*(ccdf_interp_s))
    
    #plt.plot(dt_vals, ccdf_interp_s)
    
    # -----------------------------------
    # 5. Fragility calibration
    # -----------------------------------
    median_DR = mu
    cv_DR = sigma/median_DR 

    sigma_ln_DR = np.sqrt(np.log(1+cv_DR**2))
    
    # PRIOR
    sigma_ln_DT = np.sqrt(sigma_ln_DR**2 + sigma_ln_DF**2)
    mu_ln_DT = np.log(median_DR) + np.log(DFmedian)
    scale_DT = np.exp(mu_ln_DT)

    pdf_prior = lognorm.pdf(dt_vals, s=sigma_ln_DT, scale=scale_DT)
    Pf_prior = np.trapz(pdf_prior*ccdf_interp, dt_vals)
    Pf_prior_s = np.trapz(pdf_prior*ccdf_interp_s, dt_vals)
    
    
    Pf_prior = Pf_prior_s   # PF prior = single

    # POST
    P_fail_post = []
    m, dn, rr = [],[],[]
    MM, DD, R =[], [], []
    MinUt = []
    
    N_MC = 20000
    
    
    
    DF_list = np.random.lognormal(mean=np.log(DFmedian), sigma=sigma_ln_DF, size=N_MC)
    #DF_list = np.minimum(DF_list, 7)

    #DF_list = np.linspace(0.5, 2, 16)#[0.974, 0.975, 0.976]
    
    #DF_list = np.linspace(0.00001, 3.9, N_MC)
    v0 = 5
    y0 = 1.086

    lambda_list = []
    PF_list = []
    
    
    for DF in DF_list:
        
        scale_DT_post = median_DR * DF#np.exp(mu_ln_DT_post)

        pdf_post = lognorm.pdf(dt_vals, s=sigma_ln_DR, scale=scale_DT_post)   # sigma DR as no uncert in DF
        Pf_post = np.trapz(pdf_post*ccdf_interp_s, dt_vals)
        
        P_fail_post.append(Pf_post)
        
        
        lambda_F = v0 * Pf_post

        y_bar = y200
        
        # P_F closed
        
        DT_bar = D_T(y_bar)
        
        mask = dt_vals <= DT_bar

        Pf_below = np.trapz(
            pdf_post[mask] * ccdf_interp_s[mask],
            dt_vals[mask]
        )


        lambda_NF = v0 * Pf_below

        P_y_lt_ybar = 0.19090105916394617 #P(Y<ybar = 1.5m) #  quad(lambda y: f_y(y), y_bar, np.inf)[0]
        
        lambda_C = v0 *P_y_lt_ybar
        
           
        PF_list.append(Pf_post)
        lambda_list.append(lambda_F)
 
        cost_M = PVCM(Pf_post, lambda_F, lambda_C, lambda_NF)

        cost_DN = DN(Pf_post)
        cost_RR = RR(Pf_post)
        
        
        MM.append(cost_M)
        
        DD.append(cost_DN)

        R.append(cost_RR)
    
        minimum_ut = min(cost_M, cost_DN, cost_RR)
    
        MinUt.append(minimum_ut)
        
    
        if minimum_ut == cost_M:
            m.append(1)
            dn.append(0)
            rr.append(0)
        elif minimum_ut == cost_DN:
            #print('DN')
            m.append(0)
            dn.append(1)
            rr.append(0)
        elif minimum_ut == cost_RR:
            #print('RR')
            m.append(0)
            dn.append(0)
            rr.append(1)
        #print(lambda_F)
        #print()


    E_Pf_post = sum(P_fail_post)/N_MC
    
    no_dn = sum(dn)
    no_m = sum(m)
    no_rr = sum(rr)
    
    EC_pp = sum(MinUt)/N_MC
            
    mu_ln_DT_post_known = np.log(median_DR*DF_known) - 0.5*sigma_ln_DR**2
    scale_DT_post_known = np.exp(mu_ln_DT_post_known)

    pdf_post_known_DF = lognorm.pdf(dt_vals, s=sigma_ln_DR, scale=scale_DT_post_known)   # sigma DR as no uncert in DF
    Pf_post_known_DF = np.trapz(pdf_post_known_DF*ccdf_interp, dt_vals)
    
    # -----------------------------------
    # 6. Plot
    # -----------------------------------
    fig, axes = plt.subplots(1,2,figsize=(10,5),sharey=True)

    axes[0].plot(dt_vals, pdf_prior, label="Capacity PDF")
    #axes[0].plot(dt_vals, ccdf_interp, label="CCDF - 30 years")
    axes[0].plot(dt_vals, ccdf_interp_s, label="Demand CCDF")
    
    
    axes[0].set_title(f"PRIOR | Pf={Pf_prior:.2e}")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_ylabel('Probability')
    axes[0].set_xlabel(f'$D_T$ (m)')
    axes[1].set_xlabel(f'$D_T$')

    axes[1].plot(dt_vals, pdf_post_known_DF, label="PDF post")
    axes[1].plot(dt_vals, ccdf_interp, label="CCDF")
    axes[1].set_title(f"POST DF={DF_known:.1f}m | Pf={Pf_post_known_DF:.2e}")
    axes[1].grid(True)
    axes[1].legend()
    

    plt.tight_layout()
    plt.show()
    
    # --------------------------------------------------------------
    # EXPECTED COST PRIOR
    # --------------------------------------------------------------
    lambdaF_prior = Pf_prior * v0
    EC_M_prior = PVCM(Pf_prior, lambdaF_prior, lambda_C, 0.98*lambdaF_prior)
    
    #EC_M_30years(100, y0, beta_exp, nu, Pf_prior_8197, C_S)
        
    EC_DN_prior = DN(Pf_prior)
    EC_RR_prior = RR(Pf_prior)
    
    EC_prior = min(EC_M_prior, EC_DN_prior, EC_RR_prior)
    
    results = {
        "Pf_prior": Pf_prior,
        "EC_M":EC_M_prior,
        "EC_DN_prior":EC_DN_prior,
        "EC_RR_prior":EC_RR_prior,
        "EC_prior": EC_prior,
        "Expected_Pf_Post": E_Pf_post,
        "EC_pp": EC_pp,
        "Number of optimal DN": no_dn,
        "Number of optimal M": no_m,
        "Number of optimal RR": no_rr,
        "VoI": EC_prior - EC_pp,
        "VoI £":(EC_prior - EC_pp)*C_RB
   
    } 
    return results, Pf_prior, E_Pf_post, P_fail_post, DF_list
