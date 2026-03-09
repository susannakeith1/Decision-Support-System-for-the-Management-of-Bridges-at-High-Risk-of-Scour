#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:25:22 2026

@author: susannakeith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, expon

C_HL = 3000000   # human loss              £3 million
C_RB = 2000000   # rebuild bridge          £2 million
C_CB = 15000     # close bridge (~3 days)  £5000 per day?
C_M  = 10000     # monitoring              £10,000   
C_RR = 40000     # riprap                  £30,000

# Normalising all costs to cost of rebuilding the bridge
gamma_HL = C_HL / C_RB
gamma_CB = C_CB / C_RB
gamma_M  = C_M / C_RB
gamma_RR = C_RR / C_RB


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
    seed=0
):

    np.random.seed(seed)

    # -----------------------------------
    # 1. Load water levels & fit exponential exceedance
    # -----------------------------------
    df = pd.read_csv(waterlevel_file)
    data = df["Water level (meters)"].dropna()

    above = data > y0
    events_start = above & (~above.shift(1, fill_value=False))
    num_events = events_start.sum()

    nu = num_events / T

    exceedances = data[data > y0] - y0
    beta_exp = exceedances.mean()

    x = np.linspace(0,7,10000)
    G = 1 - expon.cdf(x, scale=beta_exp)
    ccdf_30years = 1 - np.exp(-nu * G * T)
    CDF = 1 - ccdf_30years
    
    P_y_gt_y200 = expon.sf(y200 - y0, scale=beta_exp)
    P_y_lt_y200 = 1 - P_y_gt_y200

    nu_low = nu * P_y_lt_y200
    nu_high = nu * P_y_gt_y200
    
    cdf_30_low = 1 - np.exp(-nu_low * G * T)
    cdf_30_high = 1 - np.exp(-nu_high * G * T)
    
    CDF_low = 1 - cdf_30_low
    CDF_high = 1 - cdf_30_high
    
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
    u = np.random.rand(n_samples)
    y_samples = np.interp(u, CDF, x)
    
    y_samp_low = np.interp(u, CDF_low, x)
    y_samp_high = np.interp(u, CDF_high, x)

    dt_with_error = []
    for y in y_samples:
        error = np.random.normal(0, sigma_DT_error)
        dt = D_T(y) * (1+error)
        dt = max(dt,0.0)
        dt_with_error.append(dt)

    dt_with_error = np.array(dt_with_error)
    
    dt_with_error_low = []
    for y in y_samp_low:
        error = np.random.normal(0, sigma_DT_error)
        dt = D_T(y) * (1+error)
        dt = max(dt,0.0)
        dt_with_error_low.append(dt)

    dt_with_error_low = np.array(dt_with_error_low)
    
    dt_with_error_high = []
    for y in y_samp_high:
        error = np.random.normal(0, sigma_DT_error)
        dt = D_T(y) * (1+error)
        dt = max(dt,0.0)
        dt_with_error_high.append(dt)

    dt_with_error_high = np.array(dt_with_error_high)

    # -----------------------------------
    # 4. CCDF of DT
    # -----------------------------------
    dt_vals = np.linspace(dt_range[0], dt_range[1], 10000)

    sorted_dt = np.sort(dt_with_error)
    sorted_dt_low = np.sort(dt_with_error_low)
    sorted_dt_high = np.sort(dt_with_error_high)

    n = len(sorted_dt)

    cdf_dt = np.arange(1,n+1)/n
    ccdf_dt = 1-cdf_dt

    ccdf_interp = np.interp(dt_vals, sorted_dt, ccdf_dt, left=1, right=0)
    ccdf_interp = np.maximum.accumulate(ccdf_interp[::-1])[::-1]
    
    ccdf_interp_low = np.interp(dt_vals, sorted_dt_low, ccdf_dt, left=1, right=0)
    ccdf_interp_low = np.maximum.accumulate(ccdf_interp_low[::-1])[::-1]
    
    ccdf_interp_high = np.interp(dt_vals, sorted_dt_high, ccdf_dt, left=1, right=0)
    ccdf_interp_high = np.maximum.accumulate(ccdf_interp_high[::-1])[::-1]

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
    
    P_fail_low = np.trapz(pdf_prior*ccdf_interp_low, dt_vals)
    P_fail_high = np.trapz(pdf_prior*ccdf_interp_high, dt_vals)

    # ==========================================================================
    # 6. pi great and pi less
    # ==========================================================================

    P_F = Pf_prior
    P_NF = 1 - P_F

    P_F_y_low = P_fail_low

    P_NF_y_high = 1 - P_fail_high #0.999941

    P_NF_y_high_AND_y_high = P_NF_y_high * P_y_gt_y200  #  P(NF, y > y200) = P(NF | y > y200) * P(y > y200)

    P_F_y_low_AND_y_low = P_F_y_low * P_y_lt_y200

    pi_great = P_NF_y_high_AND_y_high/P_NF
    pi_low = P_F_y_low_AND_y_low/P_F

    # POST
    P_fail_post = []
    m, dn, rr = [],[],[]
    MM, DD, R =[], [], []
    MinUt = []
    
    N_MC = 20000
    
    DF_list = np.random.lognormal(mean=np.log(DFmedian), sigma=sigma_ln_DF, size=N_MC)
    #DF_list = np.minimum(DF_list, 7)

    
    #DF_list = #np.linspace(0.00001, 3.9, N_MC)
    
    def DN(Pf):
        return Pf * (1 + gamma_HL)

    def RR(Pf):
        return Pf*gamma_RR / Pf


    def M(Pf):

        pi_great = 0.020864 / (1-Pf_prior)     
        pi_low = 0.170729 / Pf_prior
        
        return Pf * (1 + pi_low * gamma_HL - pi_great * gamma_CB) + gamma_M + gamma_CB * pi_great


    def M_p(Pf, pi_great, pi_low):

        return Pf * (1 + pi_low * gamma_HL - pi_great * gamma_CB) + gamma_M + gamma_CB * pi_great
    

    
    for DF in DF_list:
        
        mu_ln_DT_post = np.log(median_DR*DF) - 0.5*sigma_ln_DR**2
        scale_DT_post = median_DR * DF#np.exp(mu_ln_DT_post)

        pdf_post = lognorm.pdf(dt_vals, s=sigma_ln_DR, scale=scale_DT_post)   # sigma DR as no uncert in DF
        Pf_post = np.trapz(pdf_post*ccdf_interp, dt_vals)
        
        P_fail_post.append(Pf_post)
        
        Pf_post_low = np.trapz(pdf_post*ccdf_interp_low, dt_vals)
        Pf_post_high = np.trapz(pdf_post*ccdf_interp_high, dt_vals)
        
        pi_great_post = (1-Pf_post_high) * P_y_gt_y200/(1-Pf_post)
        pi_low_post = Pf_post_low * P_y_lt_y200/Pf_post
          
        cost_M = M_p(Pf_post, pi_great_post, pi_low_post)
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
            m.append(0)
            dn.append(1)
            rr.append(0)
        elif minimum_ut == cost_RR:
            m.append(0)
            dn.append(0)
            rr.append(1)
    
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

    axes[0].plot(dt_vals, pdf_prior, label="PDF prior")
    axes[0].plot(dt_vals, ccdf_interp, label="CCDF")
    axes[0].set_title(f"PRIOR | Pf={Pf_prior:.2e}")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_ylabel('Probability')
    axes[0].set_xlabel(f'$D_T$')
    axes[1].set_xlabel(f'$D_T$')

    axes[1].plot(dt_vals, pdf_post_known_DF, label="PDF post")
    axes[1].plot(dt_vals, ccdf_interp, label="CCDF")
    axes[1].set_title(f"POST DF={DF_known:.1f}m | Pf={Pf_post_known_DF:.2e}")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    EC_M_prior = M(Pf_prior)
    EC_DN_prior = DN(Pf_prior)
    EC_RR_prior = RR(Pf_prior)
    
    EC_prior = min(EC_M_prior, EC_DN_prior, EC_RR_prior)
    
    results = {
        "Pf_prior": Pf_prior,
        "pi_great": pi_great,
        "pi_low": pi_low,
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
        "VoI £":round(EC_prior - EC_pp,3)*C_RB
   
    }

    return results, Pf_prior, E_Pf_post, P_fail_post, DF_list
