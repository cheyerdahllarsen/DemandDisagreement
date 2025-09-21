# -*- coding: utf-8 -*-
"""SimulateDisandYield - Local Version

Adapted from Colab version for local execution.
"""

import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
import csv
import time
import datetime
import torch

# Set up local file paths
output_dir = 'Data/Model Disagreement'
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, 'TESTsimulation_resultsDISandYields10000yearMonthlyN2.csv')

def getShortRate(f, alp, pars):
  rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars
  phiA = 1/(rhoA + nu)
  phiB = 1/(rhoB + nu)
  phi = f*phiA + (1-f)*phiB
  betA = (rhoA+nu)*phi
  betB = (rhoB+nu)*phi
  bet = alp*betA+(1-alp)*betB
  rhot = f*rhoA+(1-f)*rhoB
  r = rhot + muY-sigY**2 + nu*(1-bet)
  return r

def getYield(f, l, tau, pars, M, dt, device):
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
        ft = torch.clamp(ft, 0, 1)
        rbar = muY - sigY**2
        r = rbar + rhoA * ft + rhoB * (1 - ft)  + nu * (1 - alpt * betAt - (1 - alpt) * betBt)
        DFt *= torch.exp(-r * dt)

    B = torch.mean(DFt)
    return -torch.log(B) / tau

def getYieldandYieldVola(f, alp, tau, pars, M, dt, device):
  rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars
  #Since we are doing autodiff we need to make sure we have enabled autograd
  l = -np.log(1/alp-1)
  lt = torch.tensor([l], device=device, requires_grad=True)  # Example value
  ft = torch.tensor([f], device=device, requires_grad=True)  # Example value
  #Calculating the yield
  y = getYield(ft, lt, tau, pars, M, dt, device)
  # Compute gradients
  y.backward()
  sigY = ft.grad.item()*ft*(1-ft)*DEL+lt.grad.item()*sig_l
  return y.item(), sigY.item(), ft.grad.item(), lt.grad.item()

def getAllMaturities(f, alp, taus, pars, M, dt, device):
  results = []
  for tau in taus:
    partCalc = []
    y, sigY, _, _ = getYieldandYieldVola(f, alp, tau, pars, M, dt, device)
    results.append(y)
    results.append(sigY)
  return results

def getExpectedShortRate(f, l, tau, pars, M, dt, device):
    rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY = pars

    NT = round(tau / dt)  # Number of time steps

    # Initialize tensors for all paths under each of the three beliefs
    ft = torch.tensor(f, dtype=torch.float32, device=device).repeat(M)
    fat = torch.tensor(f, dtype=torch.float32, device=device).repeat(M)
    fbt = torch.tensor(f, dtype=torch.float32, device=device).repeat(M)

    lt = torch.tensor(l, dtype=torch.float32, device=device).repeat(M)
    lat = torch.tensor(l, dtype=torch.float32, device=device).repeat(M)
    lbt = torch.tensor(l, dtype=torch.float32, device=device).repeat(M)

    # Generate random numbers for all paths and time steps. Make sure to check the change of measure here!
    dZ = torch.sqrt(torch.tensor(dt, dtype=torch.float32, device=device)) * torch.randn(NT, M, device=device)
    dZa = dZ - torch.ones(NT, M, device=device) * torch.tensor(0.5 * DEL * dt, dtype=torch.float32, device=device)
    dZb = dZ + torch.ones(NT, M, device=device) * torch.tensor(0.5 * DEL * dt, dtype=torch.float32, device=device)

    rbar = muY - sigY**2  # This should probably be moved outside the loop as a constant

    for i in range(NT):
        # Under objective
        alpt = 1 / (1 + torch.exp(-lt))
        phit = ft / (rhoA + nu) + (1 - ft) / (rhoB + nu)
        betAt = (rhoA + nu) * phit
        betBt = (rhoB + nu) * phit

        muft = nu * (alpt * betAt * (1 - ft) - (1 - alpt) * betBt * ft) + (rhoB - rhoA) * ft * (1 - ft) + DEL**2 * (1/2 - ft) * ft * (1 - ft)
        sigft = ft * (1 - ft) * DEL
        dft = muft * dt + sigft * dZ[i, :]
        dlt = kap * (lbar - lt) * dt + sig_l * dZ[i, :]

        lt = lt + dlt  # update lt
        ft = ft + dft  # update ft
        ft = torch.clamp(ft, 0, 1)
        r = rbar + rhoA * ft + rhoB * (1 - ft) + nu * (1 - alpt * betAt - (1 - alpt) * betBt)

        # Under agent a
        alpt = 1 / (1 + torch.exp(-lat))
        phit = fat / (rhoA + nu) + (1 - fat) / (rhoB + nu)
        betAt = (rhoA + nu) * phit
        betBt = (rhoB + nu) * phit

        mufat = nu * (alpt * betAt * (1 - fat) - (1 - alpt) * betBt * fat) + (rhoB - rhoA) * fat * (1 - fat) + DEL**2 * (1/2 - fat) * fat * (1 - fat)
        sigfat = fat * (1 - fat) * DEL
        dfat = mufat * dt + sigfat * dZa[i, :]
        dlat = kap * (lbar - lat) * dt + sig_l * dZa[i, :]

        lat = lat + dlat  # update lat
        fat = fat + dfat  # update fat
        ra = rbar + rhoA * fat + rhoB * (1 - fat) + nu * (1 - alpt * betAt - (1 - alpt) * betBt)

        # Under agent b
        alpt = 1 / (1 + torch.exp(-lbt))
        phit = fbt / (rhoA + nu) + (1 - fbt) / (rhoB + nu)
        betAt = (rhoA + nu) * phit
        betBt = (rhoB + nu) * phit

        mufbt = nu * (alpt * betAt * (1 - fbt) - (1 - alpt) * betBt * fbt) + (rhoB - rhoA) * fbt * (1 - fbt) + DEL**2 * (1/2 - fbt) * fbt * (1 - fbt)
        sigfbt = fbt * (1 - fbt) * DEL
        dfbt = mufbt * dt + sigfbt * dZb[i, :]
        dlbt = kap * (lbar - lbt) * dt + sig_l * dZb[i, :]

        lbt = lbt + dlbt  # update lbt
        fbt = fbt + dfbt  # update fbt
        rb = rbar + rhoA * fbt + rhoB * (1 - fbt) + nu * (1 - alpt * betAt - (1 - alpt) * betBt)

        DIS = 0.5 * np.abs(torch.mean(rb).item() - torch.mean(ra).item())

    return torch.mean(r).item(), torch.mean(ra).item(), torch.mean(rb).item(), DIS

# Model parameters
DEL = 0.8
rhoA = -0.015
rhoB = 0.025
nu = 0.02
kap = 0.01
lbar = 0
sig_l = 0.1
muY = 0.02
sigY = 0.033
pars = [rhoA , rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY]

# Set up device - will use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Uncomment the line below to force CPU usage if needed
# device = torch.device("cpu")
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

dt = 1 / 12
M = 100000  # Number of paths. Note that when we do automatic diff it "eats memory" so Used 100K usually
# If you run into memory issues, try reducing M to 50000 or 25000

taus = [1, 2, 3, 4, 5]

def generate_normalized_random_data(n_samples, dt):
    # Generate random data (from a normal distribution with mean=0, std=1)
    data = np.random.randn(n_samples)

    # Calculate the mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    # Normalize the data to have exactly mean 0
    data_normalized = data - mean

    # Scale the data to have standard deviation sqrt(1/dt)
    desired_std_dev = np.sqrt(dt)
    data_scaled = data_normalized * (desired_std_dev / std_dev)

    # Return the final normalized and scaled data
    return data_scaled

def save_results(results, columns, file_path):
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)  # 'a' for append

def main():
    """Main simulation function"""
    print("Starting simulation...")
    print(f"Output file: {file_path}")
    
    save_interval = 100  # Every 100 iterations
    
    start_time = time.time()
    #Simulating unconditional yield properties
    T = 100 #10000 in the paper, but here we use 100 for testing
    dt = 1 / 12
    NT = int(T / dt)
    alpt = 0.5 #starting value for alpha
    ft = 0.5 #starting value for f
    
    # Generate random data
    dZ = generate_normalized_random_data(NT, dt)
    results = []
    lt = np.log(1 / alpt - 1)
    
    for i in range(NT):
        phiA = 1 / (rhoA + nu)
        phiB = 1 / (rhoB + nu)
        phi = ft * phiA + (1 - ft) * phiB
        betAt= (rhoA + nu) * phi
        betBt = (rhoB + nu) * phi
        
        muft = nu * (alpt * betAt * (1 - ft) - (1 - alpt) * betBt * ft) + (rhoB - rhoA) * ft * (1 - ft) + DEL**2 * (1/2 - ft) * ft * (1 - ft)
        sigft = ft * (1 - ft) * DEL
        dft = muft * dt + sigft * dZ[i]
        dlt = kap * (lbar - lt) * dt + sig_l * dZ[i]
        lt =lt + dlt  # update lt
        ft =ft + dft  # update ft
        
        # Clamp ft to valid range
        if ft< 0.001:
            ft =  0.001 + 0.000001
        if ft>1- 0.001:
            ft = 1-0.000001- 0.001
            
        alpt = 1 / (1+np.exp(-lt))
        
        # Ensure ft and alpt are finite before calling the interpolation function
        if np.isfinite(ft) and np.isfinite(alpt):
            resultsYields = getAllMaturities(ft, alpt, taus, pars, M, dt, device)
            rObj, ra, rb, DIS = getExpectedShortRate(ft, lt, 1, pars, M, dt, device)
            r = getShortRate(ft, alpt,  pars)
            results.append([ft, alpt, DIS, r, *resultsYields, *pars])
        else:
            # Handle the non-finite case, e.g., by skipping or using default values
            print("Non-finite values encountered: ft={}, alpt={}".format(ft, alpt))
            # Optionally, add your logic here to handle this situation
            
        if i % save_interval == 0:
            print(f"Iteration {i}/{NT}, Progress: {i / NT * 100:.2f}%")
            columns = ['ft', 'alpt', 'DIS','r'] + [f'yield_{i}' for i in range(len(resultsYields))] + [f'param_{i}' for i in range(len(pars))]
            save_results(results, columns, file_path)  # Save intermediate results
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Convert results to a pandas DataFrame
    columns = ['ft', 'alpt', 'DIS','r'] + [f'yield_{i}' for i in range(len(resultsYields))] + [f'param_{i}' for i in range(len(pars))]
    df = pd.DataFrame(results, columns=columns)

    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

    print(f"Results written to '{file_path}'")
    print(f"Data shape: {df.shape}")
    print(f"dZ mean: {dZ.mean()}")
    print(f"dZ std: {dZ.std()}")

if __name__ == "__main__":
    main()
