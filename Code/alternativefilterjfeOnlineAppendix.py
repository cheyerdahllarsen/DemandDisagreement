# -*- coding: utf-8 -*-
"""AlternativeFilterJFE.py

Local version of the filtering script for JFE Online Appendix.
Originally based on Google Colab notebook.
"""

# Note: Make sure to install required packages locally:
# pip install filterpy arch statsmodels openpyxl torch scipy matplotlib pandas numpy

# --- Define Constant Beta ---
BETA_CONSTANT = 0.0094 #Based on a preregression
UseMacroDis = True #This is the one you change to false to use the PCA based version
print(f"Using fixed BETA_CONSTANT = {BETA_CONSTANT}")

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import gc
import pandas as pd
import os
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from arch import arch_model # Though not used in this UKF path
from scipy.optimize import minimize # Though not used in this UKF path
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

# Set a seed for reproducibility
seed = 42 # You can choose any integer value
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Create output directory if it doesn't exist
output_dir = os.path.join('Data', 'Model Disagreement')
os.makedirs(output_dir, exist_ok=True)

# --- Core Model Functions ---
def getExpectedShortRate(f, l, tau, pars, M, dt, device):
    torch.manual_seed(42)
    rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars
    NT = round(tau / dt)
    ft_val = f.item() if isinstance(f, torch.Tensor) else float(f)
    lt_val = l.item() if isinstance(l, torch.Tensor) else float(l)

    ft_t = torch.tensor(ft_val, dtype=torch.float32, device=device).repeat(M)
    fat = torch.tensor(ft_val, dtype=torch.float32, device=device).repeat(M)
    fbt = torch.tensor(ft_val, dtype=torch.float32, device=device).repeat(M)
    lt_t = torch.tensor(lt_val, dtype=torch.float32, device=device).repeat(M)
    lat = torch.tensor(lt_val, dtype=torch.float32, device=device).repeat(M)
    lbt = torch.tensor(lt_val, dtype=torch.float32, device=device).repeat(M)

    dZ = torch.sqrt(torch.tensor(dt, dtype=torch.float32, device=device)) * torch.randn(NT, M, device=device)
    dZa = dZ - 0.5 * DEL * dt # Simplified tensor math
    dZb = dZ + 0.5 * DEL * dt # Simplified tensor math
    rbar = muY - sigY**2

    for i in range(NT):
        alpt = 1 / (1 + torch.exp(-lt_t))
        phit = ft_t / (rhoA + nu) + (1 - ft_t) / (rhoB + nu)
        betAt = (rhoA + nu) * phit
        betBt = (rhoB + nu) * phit
        muft = nu * (alpt * betAt * (1 - ft_t) - (1 - alpt) * betBt * ft_t) + (rhoB - rhoA) * ft_t * (1 - ft_t) + DEL**2 * (0.5 - ft_t) * ft_t * (1 - ft_t)
        sigft = ft_t * (1 - ft_t) * DEL
        dft = muft * dt + sigft * dZ[i, :]
        dlt = kap * (lbar - lt_t) * dt + sig_l * dZ[i, :]
        lt_t += dlt; ft_t += dft; ft_t = torch.clamp(ft_t, 0, 1)
        r = rbar + rhoA * ft_t + rhoB * (1 - ft_t) + nu * (1 - alpt * betAt - (1 - alpt) * betBt)

        alpt_a = 1 / (1 + torch.exp(-lat)); phit_a = fat / (rhoA + nu) + (1 - fat) / (rhoB + nu)
        betAt_a = (rhoA + nu) * phit_a; betBt_a = (rhoB + nu) * phit_a
        mufat = nu * (alpt_a * betAt_a * (1 - fat) - (1 - alpt_a) * betBt_a * fat) + (rhoB - rhoA) * fat * (1 - fat) + DEL**2 * (0.5 - fat) * fat * (1 - fat)
        sigfat = fat * (1 - fat) * DEL
        dfat = mufat * dt + sigfat * dZa[i, :]; dlat = kap * (lbar - lat) * dt + sig_l * dZa[i, :]
        lat += dlat; fat += dfat; fat = torch.clamp(fat, 0, 1)
        ra = rbar + rhoA * fat + rhoB * (1 - fat) + nu * (1 - alpt_a * betAt_a - (1 - alpt_a) * betBt_a)

        alpt_b = 1 / (1 + torch.exp(-lbt)); phit_b = fbt / (rhoA + nu) + (1 - fbt) / (rhoB + nu)
        betAt_b_belief = (rhoA + nu) * phit_b; betBt_b_belief = (rhoB + nu) * phit_b
        mufbt = nu * (alpt_b * betAt_b_belief * (1 - fbt) - (1 - alpt_b) * betBt_b_belief * fbt) + (rhoB - rhoA) * fbt * (1 - fbt) + DEL**2 * (0.5 - fbt) * fbt * (1 - fbt)
        sigfbt = fbt * (1 - fbt) * DEL
        dfbt = mufbt * dt + sigfbt * dZb[i, :]; dlbt = kap * (lbar - lbt) * dt + sig_l * dZb[i, :]
        lbt += dlbt; fbt += dfbt; fbt = torch.clamp(fbt, 0, 1)
        rb = rbar + rhoA * fbt + rhoB * (1 - fbt) + nu * (1 - alpt_b * betAt_b_belief - (1 - alpt_b) * betBt_b_belief)

    DIS = 0.5 * torch.abs(torch.mean(rb) - torch.mean(ra))
    return torch.mean(r).item(), torch.mean(ra).item(), torch.mean(rb).item(), DIS.item()

def getYield(f, l, tau, pars, M, dt, device):
    rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars
    NT = round(tau / dt)
    ft_val = f.item() if isinstance(f, torch.Tensor) else float(f)
    lt_val = l.item() if isinstance(l, torch.Tensor) else float(l)

    ft_t = torch.tensor(ft_val, dtype=torch.float32, device=device).repeat(M)
    lt_t = torch.tensor(lt_val, dtype=torch.float32, device=device).repeat(M)
    DFt = torch.ones(M, device=device)
    dZ = torch.sqrt(torch.tensor(dt, dtype=torch.float32, device=device)) * torch.randn(NT, M, device=device)

    for i in range(NT):
        theta_alp = DEL * (0.5 - ft_t)
        alpt = 1 / (1 + torch.exp(-lt_t))
        phit = ft_t / (rhoA + nu) + (1 - ft_t) / (rhoB + nu)
        betAt = (rhoA + nu) * phit; betBt = (rhoB + nu) * phit
        muft = nu * (alpt * betAt * (1 - ft_t) - (1 - alpt) * betBt * ft_t) + (rhoB - rhoA) * ft_t * (1 - ft_t) + DEL**2 * (0.5 - ft_t) * ft_t * (1 - ft_t)
        sigft = ft_t * (1 - ft_t) * DEL
        dft = muft * dt + sigft * (dZ[i, :] - theta_alp * dt)
        dlt = kap * (lbar - lt_t) * dt + sig_l * (dZ[i, :] - theta_alp * dt)
        lt_t += dlt; ft_t += dft; ft_t = torch.clamp(ft_t, 0, 1)
        rbar = muY - sigY**2
        r = rbar + rhoA * ft_t + rhoB * (1 - ft_t)  + nu * (1 - alpt * betAt - (1 - alpt) * betBt)
        DFt *= torch.exp(-r * dt)
    B = torch.mean(DFt)
    return -torch.log(B).item() / tau if B.item() > 0 else np.nan # Added .item() and check for B > 0

def compute_mu_f(l, f, pars):
    rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars
    l_scalar = l.item() if isinstance(l, torch.Tensor) else float(l)
    f_scalar = f.item() if isinstance(f, torch.Tensor) else float(f)
    alpha = 1 / (1 + np.exp(-l_scalar))
    phit = f_scalar / (rhoA + nu) + (1 - f_scalar) / (rhoB + nu)
    beta_a = (rhoA + nu) * phit; beta_b = (rhoB + nu) * phit
    mu_f = (nu * (alpha * beta_a * (1 - f_scalar) - (1 - alpha) * beta_b * f_scalar) +
            (rhoB - rhoA) * f_scalar * (1 - f_scalar) +
            DEL**2 * (0.5 - f_scalar) * f_scalar * (1 - f_scalar))
    return mu_f

def fx_new(X, dt, pars_model, ar_params_m): # State: [l, f, m]
    l, f, m = X
    _ , _, _, DEL_param, kap_param, lbar_param, _, _, _ = pars_model # Corrected unpacking for DEL
    a_m, rho_m, _ = ar_params_m
    mu_f_val = compute_mu_f(l, f, pars_model)
    l_next = l + kap_param * (lbar_param - l) * dt
    f_next = f + mu_f_val * dt
    f_next = np.clip(f_next, 0, 1)
    m_next = a_m + rho_m * m
    return np.array([l_next, f_next, m_next])

def hx_new(X, pars_model, M_sim, dt_sim_obs, device_sim, ar_params_m_dummy, fixed_beta): # State: [l, f, m]
    l, f, m = X
    dis_original = getExpectedShortRate(f, l, 1, pars_model, M_sim, dt_sim_obs, device_sim)[3]
    obs_r_DIS_Q4 = dis_original + fixed_beta * m
    obs_TIPSY02 = getYield(f, l, 2, pars_model, M_sim, dt_sim_obs, device_sim)
    obs_DIS_PCA_a = m
    return np.array([obs_r_DIS_Q4, obs_TIPSY02, obs_DIS_PCA_a])

# --- Global Model Parameters ---
DEL_global = 0.8 # Renamed to avoid conflict with DEL in pars_model unpacking in fx_new
rhoA = -0.015; rhoB = 0.025; nu = 0.02; kap = 0.01; lbar = 0.0
sig_l = 0.1; muY = 0.02; sigY = 0.033
pars_global = [rhoA, rhoB, nu, DEL_global, kap, lbar, sig_l, muY, sigY]

dt_ukf = 1/4; M_simulation = 50000; dt_simulation_obs = 1/12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preparation ---
print("Attempting to load data...")
# Load data from local files
yields_file_path = os.path.join('Data', 'DisagreementandRealYields.xlsx')
disagreement_file_path = os.path.join('Data', 'SPF', '2024', 'MergedDisagreementALL_TS.xlsx')

try:
    # Load yields data
    yields_df = pd.read_excel(yields_file_path, engine='openpyxl')
    print(f"Successfully loaded yields data from: {yields_file_path}")
    print(f"Yields data shape: {yields_df.shape}")
    
    # Load disagreement data
    disagreement_df = pd.read_excel(disagreement_file_path, engine='openpyxl')
    print(f"Successfully loaded disagreement data from: {disagreement_file_path}")
    print(f"Disagreement data shape: {disagreement_df.shape}")
    
    # Merge the dataframes on YEAR and QUARTER
    merged_df = pd.merge(yields_df, disagreement_df, on=['YEAR', 'QUARTER'], how='inner', suffixes=('', '_y'))
    print(f"Merged data shape: {merged_df.shape}")
    print("First few rows:")
    print(merged_df.head())
    
    # Handle duplicate column names - keep the first occurrence (from yields_df)
    # Remove duplicate columns that have '_y' suffix
    cols_to_drop = [col for col in merged_df.columns if col.endswith('_y')]
    merged_df = merged_df.drop(columns=cols_to_drop)
    print(f"After removing duplicate columns, shape: {merged_df.shape}")
    print("Available columns:")
    print(merged_df.columns.tolist())
    
except FileNotFoundError as e:
    print(f"Error: Data file not found: {e}")
    print("Please ensure the files exist in the Data folder.")
    raise
except Exception as e:
    print(f"Error reading the excel data from local files: {e}")
    raise

# Create datetime_index from YEAR and QUARTER
if 'YEAR' in merged_df.columns and 'QUARTER' in merged_df.columns:
    merged_df['Period'] = merged_df['YEAR'].astype(str) + 'Q' + merged_df['QUARTER'].astype(str)
    try:
        merged_df['datetime_index'] = pd.to_datetime(pd.PeriodIndex(merged_df['Period'], freq='Q').start_time)
        print("Successfully created 'datetime_index'.")
    except Exception as e:
        print(f"Error creating 'datetime_index': {e}. Using numeric index for time.")
        merged_df['datetime_index'] = pd.Series(pd.NaT, index=merged_df.index) # Assign NaT then handle
else:
    print("WARNING: 'YEAR' and/or 'QUARTER' columns not found. Time axis will be numeric.")
    merged_df['datetime_index'] = pd.Series(pd.NaT, index=merged_df.index)

expected_dis_pca_col = 'DIS_PCA_a' # IMPORTANT: Change if your column name is different
print(expected_dis_pca_col not in merged_df.columns)
print(merged_df.columns)

# Define the dependent variable (y) and independent variables (X)
y = merged_df['r_DIS_Q4']
macro_cols = ['DIS_gRGDPa', 'DIS_gRCONSUMa', 'DIS_gCPROFa',
              'DIS_gRFEDGOVa', 'DIS_gRRESINVa', 'DIS_gRNRESINa',
              'DIS_UNEMP6', 'DIS_HOUSING6', 'DIS_CPI6']
X = merged_df[macro_cols]

# Handle potential NaN values by removing rows with NaNs in y or any column of X
# We create a combined DataFrame for easy NaN handling
combined_df = pd.concat([y, X], axis=1).dropna()

y_filtered = combined_df['r_DIS_Q4']
X_filtered = combined_df[macro_cols]

# Add a constant to the independent variables (for the intercept)
X_filtered = sm.add_constant(X_filtered)

# Fit the OLS model
model = sm.OLS(y_filtered, X_filtered)
results = model.fit()

# Print the regression summary
print("Regression results for r_DIS_Q4 on macroeconomic disagreement variables:")
print(results.summary())

# Get the predicted values for the *entire* merged_df (including rows with NaNs)
# Use the original X with a constant
X_full = sm.add_constant(merged_df[macro_cols])
merged_df['macroDis'] = results.predict(X_full)

print("\nCreated new column 'macroDis' with predicted values.")
# Display the head of the DataFrame to show the new column
print(merged_df.head())

# Ensure 'DIS_PCA_a' column exists (or your chosen name for the third observation)
if UseMacroDis:
    print("Using macroDis as expected_dis_pca_col.")
    expected_dis_pca_col = 'macroDis'
    BETA_CONSTANT = 1
else:
  expected_dis_pca_col = 'DIS_PCA_a' # IMPORTANT: Change if your column name is different
if expected_dis_pca_col not in merged_df.columns:
    print(f"WARNING: Column '{expected_dis_pca_col}' not found.")
    if 'DISQ4' in merged_df.columns: # Fallback, verify if this is intended
        merged_df[expected_dis_pca_col] = merged_df['DISQ4']
        print(f"Using data from 'DISQ4' as '{expected_dis_pca_col}'. Please verify this is correct.")
    else:
        raise KeyError(f"Required column '{expected_dis_pca_col}' and fallback 'DISQ4' not found in DataFrame.")

startObs = 70
dis_pca_series_raw = merged_df[expected_dis_pca_col].values[startObs:] / 100.0
dis_pca_series_for_ar = dis_pca_series_raw[~np.isnan(dis_pca_series_raw)]

a_m_for_fx_global = -0.03; rho_m_global = 0.5; sigma_m_residuals_global = 0.01 # Defaults
if len(dis_pca_series_for_ar) > 100: #Just don't do it right now
    ar_model_m = AutoReg(dis_pca_series_for_ar, lags=1, trend='c')
    ar_results_m = ar_model_m.fit()
    if len(ar_results_m.params) == 2:
        a_m_for_fx_global = ar_results_m.params[0]
        rho_m_global = ar_results_m.params[1]
        sigma_m_residuals_global = np.std(ar_results_m.resid)
    print(f"AR(1) params for m_t: a={a_m_for_fx_global:.4f}, rho={rho_m_global:.4f}, sigma_resid={sigma_m_residuals_global:.4f}")
else: print("Using default AR(1) parameters for m_t due to insufficient data.")
ar_params_m_global = (a_m_for_fx_global, rho_m_global, sigma_m_residuals_global)

observed_data_full = merged_df[['r_DIS_Q4', 'TIPSY02', expected_dis_pca_col]].iloc[startObs:].values / 100.0
merged_df[['r_DIS_Q4', 'TIPSY02', expected_dis_pca_col]].iloc[startObs:].head()

# Extract the columns for regression
# r_DIS_Q4 is the first column (index 0) in observed_data_full
y = observed_data_full[:, 0]
# expected_dis_pca_col is the third column (index 2)
X = observed_data_full[:, 2]

# Handle potential NaN values by removing rows with NaNs in either y or X
nan_mask = ~np.isnan(y) & ~np.isnan(X)
y_filtered = y[nan_mask]
X_filtered = X[nan_mask]

# Add a constant to the independent variable (for the intercept)
X_filtered = sm.add_constant(X_filtered)

# Fit the OLS model
model = sm.OLS(y_filtered, X_filtered)
results = model.fit()

# Print the regression summary
print("Regression results for r_DIS_Q4 on expected_dis_pca_col:")
print(results.summary())

# --- UKF Setup ---
points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2., kappa=0.)
ukf = UKF(dim_x=3, dim_z=3, dt=dt_ukf, fx=fx_new, hx=hx_new, points=points)

num_timesteps = len(observed_data_full)
if UseMacroDis:
  R_diag = [
      1e-11,
      5e-6, #
      5e-7 #
  ]
else:
  R_diag = [
      1e-12,
      1.5e-4,
      1.5e-4
  ]
ukf.R = np.diag(R_diag)
print(f"Measurement Noise Variances (R diag): {R_diag}")

ukf.x = np.array([0.0, 0.29, np.nanmean(dis_pca_series_for_ar) if len(dis_pca_series_for_ar) > 0 else 0.0])
ukf.P = np.diag([0.0001, 0.0001, 0.0001])
# --- Filtering Loop ---
estimated_l_list, estimated_f_list, estimated_m_list = [], [], []
fitted_r_DIS_Q4_list, fitted_TIPSY02_list, fitted_DIS_PCA_a_list = [], [], []
scale_factor_Q = 1.0

fx_args_dict = {'pars_model': pars_global, 'ar_params_m': ar_params_m_global}
hx_args_dict = {'pars_model': pars_global, 'M_sim': M_simulation,
                'dt_sim_obs': dt_simulation_obs, 'device_sim': device,
                'ar_params_m_dummy': None, 'fixed_beta': BETA_CONSTANT}

print("\nStarting Kalman Filtering...")
for t in range(num_timesteps):
    current_obs_row = observed_data_full[t, :]
    if np.any(np.isnan(current_obs_row)):
        print(f"  t={t}: Skipping update due to NaN in observation: {current_obs_row}")
        if t > 0 and estimated_l_list: # Append previous if available
            estimated_l_list.append(estimated_l_list[-1]); estimated_f_list.append(estimated_f_list[-1]); estimated_m_list.append(estimated_m_list[-1])
            fitted_r_DIS_Q4_list.append(fitted_r_DIS_Q4_list[-1]); fitted_TIPSY02_list.append(fitted_TIPSY02_list[-1]); fitted_DIS_PCA_a_list.append(fitted_DIS_PCA_a_list[-1])
        else: # Append NaNs or initial state based placeholders
            estimated_l_list.append(ukf.x[0]); estimated_f_list.append(ukf.x[1]); estimated_m_list.append(ukf.x[2])
            fitted_r_DIS_Q4_list.append(np.nan); fitted_TIPSY02_list.append(np.nan); fitted_DIS_PCA_a_list.append(np.nan)
        continue

    current_f_state = ukf.x[1]
    sigma_f_val = current_f_state * (1 - current_f_state) * DEL_global # Use DEL_global

    q_l = pars_global[6]**2 * dt_ukf
    q_f = sigma_f_val**2 * dt_ukf
    q_m = ar_params_m_global[2]**2 * dt_ukf

    ukf.Q = scale_factor_Q * np.diag([max(q_l, 1e-12), max(q_f, 1e-12), max(q_m, 1e-12)])
    cov_lf = scale_factor_Q * pars_global[6] * sigma_f_val * dt_ukf
    ukf.Q[0,1] = ukf.Q[1,0] = cov_lf

    ukf.predict(**fx_args_dict)
    ukf.update(current_obs_row, **hx_args_dict)
    ukf.x[1] = np.clip(ukf.x[1], 0, 1) # Assuming f is at index 1


    estimated_l_list.append(ukf.x[0]); estimated_f_list.append(ukf.x[1]); estimated_m_list.append(ukf.x[2])

    # Get fitted values from the posterior state
    fitted_values_current = hx_new(ukf.x, **hx_args_dict)
    fitted_r_DIS_Q4_list.append(fitted_values_current[0])
    fitted_TIPSY02_list.append(fitted_values_current[1])
    fitted_DIS_PCA_a_list.append(fitted_values_current[2])

    if t % (num_timesteps // 5) == 0 and t > 0 : print(f"  Processed t={t}/{num_timesteps}")
print("Kalman Filtering finished.")

# --- Post-processing and Plotting ---
estimated_l_arr = np.array(estimated_l_list)
estimated_f_arr = np.array(estimated_f_list)
estimated_m_arr = np.array(estimated_m_list)

fitted_r_DIS_Q4_arr = np.array(fitted_r_DIS_Q4_list)
fitted_TIPSY02_arr = np.array(fitted_TIPSY02_list)
fitted_DIS_PCA_a_arr = np.array(fitted_DIS_PCA_a_list)

# Create Time Axis for Plotting
time_axis_plot = np.arange(len(estimated_l_arr)) # Default to numeric
if 'datetime_index' in merged_df.columns and not merged_df['datetime_index'].isnull().all():
    if len(merged_df['datetime_index']) >= startObs + len(estimated_l_arr):
        time_axis_plot_dt = merged_df['datetime_index'].iloc[startObs : startObs + len(estimated_l_arr)]
        if not time_axis_plot_dt.isnull().all():
            time_axis_plot = time_axis_plot_dt
            print("Using datetime_index for plotting time axis.")
        else:
             print("datetime_index contained all NaNs after slicing, using numeric index.")
    else:
        print("Not enough valid dates in 'datetime_index' slice, using numeric index.")

# Plot 1: Estimated Latent States
plt.figure(figsize=(12, 7))
plt.subplot(3, 1, 1); plt.plot(time_axis_plot, estimated_l_arr, label='Estimated $l$'); plt.ylabel('$l$'); plt.legend(); plt.grid(True)
plt.title(f'Estimated Latent States (Fixed Beta = {BETA_CONSTANT})')
plt.subplot(3, 1, 2); plt.plot(time_axis_plot, estimated_f_arr, label='Estimated $f$'); plt.ylabel('$f$'); plt.legend(); plt.grid(True)
plt.subplot(3, 1, 3); plt.plot(time_axis_plot, estimated_m_arr, label='Estimated $m_t$'); plt.ylabel('$m_t$'); plt.xlabel('Time'); plt.legend(); plt.grid(True)
plt.tight_layout()
# Save plot to output directory
plot1_path = os.path.join(output_dir, 'estimated_latent_states.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot1_path}")
plt.show()


# Plot 2: Extended DIS Calculation and Comparison (as requested previously)
print("\n--- Starting Extended DIS Calculation and Comparison ---")
def calculate_dis_extended(f_val, l_val, pars, M, dt_sim, device):
    _, _, _, dis_value = getExpectedShortRate(f_val, l_val, tau=1, pars=pars, M=M, dt=dt_sim, device=device)
    return dis_value

if len(estimated_l_arr) > 0 and len(estimated_f_arr) > 0:
    extended_dis_series = np.zeros(len(estimated_l_arr))
    print(f"Calculating extended DIS for {len(estimated_l_arr)} (l,f) pairs...")
    for t_idx in range(len(estimated_l_arr)):
        if np.isnan(estimated_l_arr[t_idx]) or np.isnan(estimated_f_arr[t_idx]):
            extended_dis_series[t_idx] = np.nan; continue
        extended_dis_series[t_idx] = calculate_dis_extended(estimated_f_arr[t_idx], estimated_l_arr[t_idx], pars_global, M_simulation, dt_simulation_obs, device)
    print("Finished calculating extended DIS series.")

    #original_disq4_comp = observed_data_full[:len(extended_dis_series), 0] # Align length
    original_disq4_comp = merged_df['DISQ4'].iloc[startObs : startObs + len(extended_dis_series)].values / 100.0


    plt.figure(figsize=(12, 6))
    valid_comp_idx = ~np.isnan(original_disq4_comp) & ~np.isnan(extended_dis_series)

    if isinstance(time_axis_plot, pd.Series) or isinstance(time_axis_plot, pd.DatetimeIndex) :
        plot_time_comp = time_axis_plot[valid_comp_idx]
    else: # numpy array
        plot_time_comp = time_axis_plot[:np.sum(valid_comp_idx)] if len(time_axis_plot) >= np.sum(valid_comp_idx) else np.arange(np.sum(valid_comp_idx))


    if len(plot_time_comp) != np.sum(valid_comp_idx): # Fallback for safety
        plot_time_comp_safe = np.arange(np.sum(valid_comp_idx))
    else:
        plot_time_comp_safe = plot_time_comp


    plt.plot(plot_time_comp_safe, original_disq4_comp[valid_comp_idx], label='Original DISQ4 (Observed)', linestyle=':', marker='.')
    plt.plot(plot_time_comp_safe, extended_dis_series[valid_comp_idx], label='Calculated DIS (from est. l,f)')

    corr_ext_dis = np.nan
    if np.sum(valid_comp_idx) > 1:
        corr_ext_dis = np.corrcoef(original_disq4_comp[valid_comp_idx], extended_dis_series[valid_comp_idx])[0, 1]
    plt.title(f'Original DISQ4 vs. Calculated DIS from Estimated l,f (Corr: {corr_ext_dis:.3f})')
    plt.xlabel('Time'); plt.ylabel('Disagreement (DIS)'); plt.legend(); plt.grid(True)
    # Save plot to output directory
    plot3_path = os.path.join(output_dir, 'original_vs_calculated_dis.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot3_path}")
    plt.show()
    if not np.isnan(corr_ext_dis): print(f"Correlation (Original DISQ4 vs Calculated DIS from l,f): {corr_ext_dis:.4f}")
else:
    print("Skipping extended DIS calculation as estimated series are empty.")

print(f"\nAnalysis run with fixed BETA_CONSTANT = {BETA_CONSTANT}")
print("--- Script Finished ---")

# Ensure extended_dis_series and original_disq4_comp are numpy arrays or pandas Series
# (Based on your notebook code, they should be numpy arrays)
from statsmodels.tsa.stattools import acf

# Create Pandas Series for easier summary statistics and autocorrelation calculation
extended_dis_series_ps = pd.Series(extended_dis_series, name='Calculated DIS (from est. l,f)')
original_disq4_comp_ps = pd.Series(original_disq4_comp, name='Original DISQ4 (Observed)')

# Get summary statistics for each series
summary_extended = extended_dis_series_ps.describe()
summary_original = original_disq4_comp_ps.describe()

# Calculate autocorrelation at lag 1, handling potential NaNs
autocorr_extended = acf(extended_dis_series_ps.dropna(), nlags=1, fft=False)[1]
autocorr_original = acf(original_disq4_comp_ps.dropna(), nlags=1, fft=False)[1]

# Create a new row for autocorrelation
autocorr_row = pd.Series({'Original DISQ4 (Observed)': autocorr_original,
                          'Calculated DIS (from est. l,f)': autocorr_extended}, name='Autocorrelation (Lag 1)')

# Combine the summaries and the autocorrelation row into a single DataFrame
summary_comparison = pd.concat([summary_original, summary_extended], axis=1)
summary_comparison = pd.concat([summary_comparison, pd.DataFrame(autocorr_row).T])


# Print the comparison
print("Summary Statistics Comparison (including Autocorrelation):")
print(summary_comparison)

"""Here we add the rest of the analysis

"""

def getYield2(f, l, tau, pars, M, dt, device):
    rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars

    NT = round(tau / dt) #Number of time steps

    # Initialize tensors for all paths
    ft = f * torch.ones(M, device=device)
    lt = l * torch.ones(M, device=device)
    DFt = torch.ones(M, device=device)

    # Generate random numbers for all paths and time steps
    dZ = torch.sqrt(torch.tensor(dt)) * torch.randn(NT, M, device=device)

    for i in range(NT):
        theta_alp = DEL * (0.5 - ft)
        alpt = 1 / (1 + torch.exp(-lt))
        phit = ft / (rhoA + nu) + (1 - ft) / (rhoB + nu)
        betAt = (rhoA + nu) * phit
        betBt = (rhoB + nu) * phit

        muft = nu * (alpt * betAt * (1 - ft) - (1 - alpt) * betBt * ft) + (rhoB - rhoA) * ft * (1 - ft) + DEL**2 * (1/2 - ft) * ft * (1 - ft)
        sigft = ft * (1 - ft) * DEL
        dft = muft * dt + sigft * (dZ[i, :] - theta_alp * dt)
        dlt = kap * (lbar - lt) * dt + sig_l * (dZ[i, :] - theta_alp * dt)

        lt =lt + dlt  # update l0
        ft =ft + dft  # update f0
        ft = torch.clamp(ft, 0, 1)  # Ensure f stays within [0, 1]
        rbar = muY - sigY**2
        r = rbar + rhoA * ft + rhoB * (1 - ft)  + nu * (1 - alpt * betAt - (1 - alpt) * betBt)
        DFt *= torch.exp(-r * dt)

    B = torch.mean(DFt)
    return -torch.log(B) / tau

def getYieldandYieldVola(f, l, tau, pars, M, dt, device):
  rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars
  #Since we are doing autodiff we need to make sure we have enabled autograd
  lt = torch.tensor([l], device=device, requires_grad=True)  # Example value
  ft = torch.tensor([f], device=device, requires_grad=True)  # Example value
  #Calculating the yield
  y = getYield2(ft, lt, tau, pars, M, dt, device)
  # Compute gradients
  y.backward()
  sigY = ft.grad.item()*ft*(1-ft)*DEL+lt.grad.item()*sig_l
  return y.item(), sigY.item(), ft.grad.item(), lt.grad.item()

# Function to fit AR(1)-GARCH(1,1) model and extract volatility
def fit_ar1_garch11V2(yield_series):
    # Rescale the data to avoid convergence issues
    yield_series_rescaled = yield_series * 100

    # Fit AR(1)-GARCH(1,1) model
    model = arch_model(yield_series_rescaled, vol='Garch', p=1, q=1, mean='AR', lags=1)
    res = model.fit(disp='off')

    # Extract volatility (conditional standard deviation)
    volatility = res.conditional_volatility / 100  # Scale back the volatility to match original scale
    return volatility

estimated_l = estimated_l_arr
estimated_f = estimated_f_arr
pars = pars_global
M = M_simulation;
DEL = DEL_global
# Maturities to evaluate
maturities = [2, 3, 5, 7, 10]
# Adjust column names based on whether they need a leading zero
columns = [f'TIPSY0{tau}' if tau < 10 else f'TIPSY{tau}' for tau in maturities]

# Extract yield data for the specified maturities
yield_data = merged_df[columns].values[startObs:] / 100

# Fit AR(1)-GARCH(1,1) model and extract GARCH volatilities for each maturity
garch_volas = {}
for i, tau in enumerate(maturities):
    yield_series = yield_data[:, i]  # Get the yield series for the given maturity
    garch_volas[tau] = fit_ar1_garch11V2(yield_series)  # Store the GARCH volatilities

# Prepare to store model-implied volatilities and yields
implied_volas = {tau: [] for tau in maturities}
implied_yields = {tau: [] for tau in maturities}
implied_riskpremia = {tau: [] for tau in maturities}

# Iterate over each time step for estimation and use UKF estimates
for t in range(num_timesteps):
    # Get current state estimates (l and f) from the UKF output
    l = estimated_l[t]
    f = estimated_f[t]

    # Compute the model-implied yield and volatility for each maturity
    for tau in maturities:
        dtsim = 1/12
        y, sigY, _, _ = getYieldandYieldVola(f, l, tau, pars, M, dtsim, device)
        stdy = np.abs(sigY)
        theta = DEL*(0.5-f)
        rp = -tau*sigY*theta
        implied_riskpremia[tau].append(rp)
        # Store the model-implied yield and volatility
        implied_yields[tau].append(y)
        implied_volas[tau].append(stdy)

# Convert the results into pandas DataFrames for easier analysis
garch_volas_df = pd.DataFrame(garch_volas)
implied_volas_df = pd.DataFrame(implied_volas)
implied_yields_df = pd.DataFrame(implied_yields)
implied_riskpremia_df = pd.DataFrame(implied_riskpremia)


# Print the DataFrames for a quick view of the results
print("GARCH Volatilities:")
print(garch_volas_df.head())

print("\nModel-Implied Volatilities:")
print(implied_volas_df.head())

print("\nModel-Implied Yields:")
print(implied_yields_df.head())

print(garch_volas_df.isna().sum())
print(implied_volas_df.isna().sum())

for tau in maturities:
    print(f"Maturity {tau}: Length of GARCH volatilities = {len(garch_volas_df[tau])}, Length of model-implied volatilities = {len(implied_volas_df[tau])}")

# Calculate correlations for yields and volatilities
yield_correlations = {}
volatility_correlations = {}

for tau in maturities:
    # Correlation between observed and model-implied yields
    observed_yield_series = yield_data[:, maturities.index(tau)]
    model_yield_series = implied_yields_df[tau]
    yield_corr = np.corrcoef(observed_yield_series, model_yield_series)[0, 1]
    yield_correlations[tau] = yield_corr

    # Handle NaN in the first value of GARCH volatility by excluding it
    garch_vola_series = garch_volas_df[tau].iloc[1:]  # Skip the first value
    model_vola_series = implied_volas_df[tau].iloc[1:]  # Align with the GARCH data

    # Ensure that both series are of the same length after slicing
    min_length = min(len(garch_vola_series), len(model_vola_series))
    garch_vola_series = garch_vola_series[:min_length]
    model_vola_series = model_vola_series[:min_length]

    # Calculate the correlation between GARCH and model-implied volatilities
    vola_corr = np.corrcoef(garch_vola_series, model_vola_series)[0, 1]
    volatility_correlations[tau] = vola_corr

# Print the correlation results
print("Yield Correlations:")
for tau, corr in yield_correlations.items():
    print(f"Maturity {tau}: Correlation between observed and model-implied yields: {corr:.4f}")

print("\nVolatility Correlations:")
for tau, corr in volatility_correlations.items():
    print(f"Maturity {tau}: Correlation between GARCH and model-implied volatilities: {corr:.4f}")

# Create a time index for plotting
time = range(startObs, startObs + num_timesteps)

# Handle NaN values in GARCH volatilities by dropping or filling them
garch_volas_df = pd.DataFrame(garch_volas).fillna(method='bfill')  # Backfill NaN values
# Or, you can drop the first value if it's always NaN:
# garch_volas_df = pd.DataFrame(garch_volas).iloc[1:]

# Convert the implied volatilities and yields to DataFrames for easier plotting
implied_volas_df = pd.DataFrame(implied_volas)
implied_yields_df = pd.DataFrame(implied_yields)

# Create a time axis with quarterly dates starting from Q1 1999 or adjust based on startObs
time = pd.date_range(start='1999Q1', periods=num_timesteps, freq='Q')

# Handle NaN values in GARCH volatilities by filling them
garch_volas_df = pd.DataFrame(garch_volas).fillna(method='bfill')  # Backfill NaN values

# Convert the implied volatilities and yields to DataFrames for easier plotting
implied_volas_df = pd.DataFrame(implied_volas)
implied_yields_df = pd.DataFrame(implied_yields)

# Assuming maturities include 1 and 10 years, with corresponding model-implied values
maturities = [1, 2, 3, 5, 7, 10]

# Extract the observed data for the 1-year and 10-year yields
observed_y1 = merged_df['TIPSY02'].values[startObs:] / 100  # Scale to match the other yields
observed_y10 = merged_df['TIPSY10'].values[startObs:] / 100
observed_slope = observed_y10 - observed_y1

# Calculate model-implied slopes
implied_y1 = implied_yields_df[2].values  # Model-implied 1-year yield
implied_y10 = implied_yields_df[10].values  # Model-implied 10-year yield
implied_slope = implied_y10 - implied_y1

# Calculate the correlation between the observed and model-implied slopes
slope_correlation = np.corrcoef(observed_slope, implied_slope)[0, 1]
print(f"Correlation between observed and model-implied slope (y10 - y1): {slope_correlation:.4f}")

# Create a time index for plotting
time = range(startObs, startObs + num_timesteps)

plt.figure(figsize=(10, 5))
plt.plot(time, observed_slope, label='Observed Slope (y10 - y2)', color='orange', linestyle='dotted')
plt.plot(time, implied_slope, label='Model-Implied Slope (y10 - y2)', color='blue', linestyle='solid')
plt.xlabel('Time step')
plt.ylabel('Slope (y10 - y1)')
plt.title(f'Observed vs. Model-Implied Yield Slope\nCorrelation: {slope_correlation:.4f}')
plt.legend()
plt.grid(True)
# Save plot to output directory
plot4_path = os.path.join(output_dir, 'yield_slope_comparison.png')
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot4_path}")
plt.show()

merged_df.head()

merged_df.columns = merged_df.columns.str.strip()

# Shift only the necessary yields by 4 quarters for future values
merged_df['Y1_t_plus_4'] = merged_df['Y1'].shift(-4)
merged_df['Y2_t_plus_4'] = merged_df['Y2'].shift(-4)
merged_df['Y3_t_plus_4'] = merged_df['Y3'].shift(-4)
merged_df['Y4_t_plus_4'] = merged_df['Y4'].shift(-4)

# Calculate excess returns
merged_df['rx_2'] = 2 * merged_df['Y2'] - merged_df['Y1_t_plus_4'] - merged_df['Y1']
merged_df['rx_3'] = 3 * merged_df['Y3'] - 2 * merged_df['Y2_t_plus_4'] - merged_df['Y1']
merged_df['rx_4'] = 4 * merged_df['Y4'] - 3 * merged_df['Y3_t_plus_4'] - merged_df['Y1']
merged_df['rx_5'] = 5 * merged_df['Y5'] - 4 * merged_df['Y4_t_plus_4'] - merged_df['Y1']

# Shift only the necessary yields by 4 quarters for future values
merged_df['Y1_t_plus_4'] = merged_df['Y1'].shift(-4)
merged_df['Y2_t_plus_4'] = merged_df['Y2'].shift(-4)
merged_df['Y3_t_plus_4'] = merged_df['Y3'].shift(-4)
merged_df['Y4_t_plus_4'] = merged_df['Y4'].shift(-4)

# Calculate excess returns
merged_df['rx_2'] = 2 * merged_df['Y2'] - merged_df['Y1_t_plus_4'] - merged_df['Y1']
merged_df['rx_3'] = 3 * merged_df['Y3'] - 2 * merged_df['Y2_t_plus_4'] - merged_df['Y1']
merged_df['rx_4'] = 4 * merged_df['Y4'] - 3 * merged_df['Y3_t_plus_4'] - merged_df['Y1']
merged_df['rx_5'] = 5 * merged_df['Y5'] - 4 * merged_df['Y4_t_plus_4'] - merged_df['Y1']

# Rename columns for garch_volas_df
garch_volas_df.columns = ['garch2', 'garch3', 'garch5', 'garch7', 'garch10']

# Rename columns for implied_volas_df
implied_volas_df.columns = ['iv2', 'iv3', 'iv5', 'iv7', 'iv10']

# Rename columns for implied_yields_df
implied_yields_df.columns = ['iy2', 'iy3', 'iy5', 'iy7', 'iy10']

# Rename columns for implied_riskpremia_df
implied_riskpremia_df.columns = ['rp2', 'rp3', 'rp5', 'rp7', 'rp10']

estimated_df = pd.DataFrame({
    'l': estimated_l,
    'f': estimated_f
})

#Creating a dataframe that starts at startObs
df = merged_df.iloc[startObs:].reset_index(drop=True)
df = pd.concat([df, garch_volas_df, estimated_df, implied_volas_df, implied_yields_df, implied_riskpremia_df], axis=1)
df.head()

import statsmodels.api as sm

filtered_data = df.dropna(subset=['rx_2', 'rx_3','rx_5', 'rp2','rp3','rp5'])

# Define the pairs of dependent and independent variables
rx_cols = ['rx_2', 'rx_3', 'rx_5']
rp_cols = ['rp2', 'rp3', 'rp5']

# Run predictive regressions for each rx_n onto the corresponding rp_n
for rx_col, rp_col in zip(rx_cols, rp_cols):
    X = filtered_data[rp_col]  # Independent variable
    y = filtered_data[rx_col]  # Dependent variable

    # Add a constant to the independent variable (for the intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Print the regression summary
    print(f"Regression results for {rx_col} on {rp_col}:")
    print(results.summary())
    print("\n" + "="*80 + "\n")

filtered_data = df.dropna(subset=['garch2', 'garch3','garch5', 'garch7', 'garch10','iv2','iv3','iv5','iv7', 'iv10'])

# Define the pairs of dependent and independent variables
rx_cols = ['garch2', 'garch3','garch5', 'garch7', 'garch10']
rp_cols = ['iv2','iv3','iv5','iv7', 'iv10']

# Run predictive regressions for each rx_n onto the corresponding rp_n
for rx_col, rp_col in zip(rx_cols, rp_cols):
    X = filtered_data[rp_col]  # Independent variable
    y = filtered_data[rx_col]  # Dependent variable

    # Add a constant to the independent variable (for the intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Print the regression summary
    print(f"Regression results for {rx_col} on {rp_col}:")
    print(results.summary())
    print("\n" + "="*80 + "\n")

filtered_data = df.dropna(subset=['TIPSY02', 'TIPSY03','TIPSY05', 'TIPSY07', 'TIPSY10','iy2','iy3','iy5','iy7', 'iy10'])

# Define the pairs of dependent and independent variables
rx_cols = ['TIPSY02', 'TIPSY03','TIPSY05', 'TIPSY07', 'TIPSY10']
rp_cols = ['iy2','iy3','iy5','iy7', 'iy10']

# Run predictive regressions for each rx_n onto the corresponding rp_n
for rx_col, rp_col in zip(rx_cols, rp_cols):
    X = filtered_data[rp_col]  # Independent variable
    y = filtered_data[rx_col]/100  # Dependent variable

    # Add a constant to the independent variable (for the intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Print the regression summary
    print(f"Regression results for {rx_col} on {rp_col}:")
    print(results.summary())
    print("\n" + "="*80 + "\n")

# Ensure df is available and contains the necessary columns

# Calculate the observed yield spread
df['Observed_Yield_Spread'] = df['TIPSY10'] / 100 - df['TIPSY02'] / 100

# Calculate the model-implied yield spread
df['Implied_Yield_Spread'] = df['iy10'] - df['iy2']

# Filter data to drop rows where either the observed or implied spread is NaN
filtered_data_spread = df.dropna(subset=['Observed_Yield_Spread', 'Implied_Yield_Spread'])

# Define the dependent variable (y) and independent variable (X) for the spread regression
y_spread = filtered_data_spread['Observed_Yield_Spread']
X_spread = filtered_data_spread['Implied_Yield_Spread']

# Add a constant to the independent variable (for the intercept)
X_spread = sm.add_constant(X_spread)

# Fit the OLS model with HAC standard errors
model_spread = sm.OLS(y_spread, X_spread)
results_spread = model_spread.fit(cov_type='HAC', cov_kwds={'maxlags': 4})

# Print the regression summary
print("Regression results for Observed Yield Spread (TIPSY10-TIPSY02) on Model-Implied Yield Spread (iy10-iy2):")
print(results_spread.summary())
print("\n" + "="*80 + "\n")

# Assuming estimated_m_arr, estimated_l_arr, and estimated_f_arr contain your estimated values

# Create a DataFrame from the estimated arrays
estimated_df = pd.DataFrame({
    'estimated_l': estimated_l_arr,
    'estimated_f': estimated_f_arr,
    'estimated_m': estimated_m_arr
})

# Define the filename for the CSV
csv_filename = 'estimated_statesAlternativeFilterJFE.csv'
csv_filepath = os.path.join(output_dir, csv_filename)

# Save the DataFrame to a CSV file
estimated_df.to_csv(csv_filepath, index=False)

print(f"Estimated states saved to {csv_filepath}")