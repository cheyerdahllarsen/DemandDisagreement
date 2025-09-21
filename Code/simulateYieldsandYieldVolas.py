# -*- coding: utf-8 -*-
"""
Simulate Yields and Yield Volatilities

This script uses pre-computed grid data to simulate unconditional yield properties 
through Monte Carlo simulation and saves results to HDF5 format.

Note: This script has been modified to run locally instead of on Google Colab.
"""

import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import hashlib
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
import csv
import time
import datetime

# Function to generate a hash from parameters
def generate_filename_hash(pars):
    # Convert parameters to a string and then to a hash
    pars_str = '_'.join(map(str, pars))
    return hashlib.md5(pars_str.encode()).hexdigest()[:8]

# Function to check if the results file for given parameters exists
def find_results_file(pars, reverseFalseConsensus=False):
    filename_hash = generate_filename_hash(pars)
    
    if reverseFalseConsensus:
        file_path = os.path.join('Data', 'Model Disagreement', f'results_{filename_hash}.csv')
    else:
        file_path = os.path.join('Data', 'Model Disagreement', f'resultsLong_{filename_hash}.csv')

    # Check if file exists
    if os.path.exists(file_path):
        return file_path
    else:
        return None

# Function to load data from file
def load_data_for_interpolation(pars, reverseFalseConsensus=False):
    file_path = find_results_file(pars, reverseFalseConsensus)
    if file_path:
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    else:
        print("File not found. Please check the parameters or generate the data.")
        return None

def interpolate_nearest_two(fhat, alphahat, df, tree):
    """
    Interpolates 14-dimensional results at the point (fhat, alphahat) based on the nearest and second nearest points in the dataframe, using a pre-built k-d tree.

    Parameters:
    fhat (float): 'f' value of the target point.
    alphahat (float): 'alpha' value of the target point.
    df (pd.DataFrame): DataFrame containing 'f', 'alpha', and 'result_{i}' (i=0,1,...,13) columns.
    tree (cKDTree): Pre-built k-d tree for the 'f' and 'alpha' coordinates.

    Returns:
    np.array: Interpolated values of result_0, result_1, ..., result_13 at (fhat, alphahat).
    """
    _, indices = tree.query([fhat, alphahat], k=2)

    nearest_point = df.iloc[indices[0]]
    second_nearest_point = df.iloc[indices[1]]

    dist_nearest = np.linalg.norm([nearest_point['f'] - fhat, nearest_point['alpha'] - alphahat])
    dist_second_nearest = np.linalg.norm([second_nearest_point['f'] - fhat, second_nearest_point['alpha'] - alphahat])

    # Weighted average interpolation
    if dist_nearest + dist_second_nearest == 0:  # Check to avoid division by zero
        return nearest_point[['result_{}'.format(i) for i in range(14)]].values

    weight_nearest = 1 / dist_nearest
    weight_second_nearest = 1 / dist_second_nearest
    total_weight = weight_nearest + weight_second_nearest

    interpolated_results = np.zeros(14)
    for i in range(14):
        result_key = f'result_{i}'
        interpolated_results[i] = (nearest_point[result_key] * weight_nearest + second_nearest_point[result_key] * weight_second_nearest) / total_weight

    return interpolated_results

def main():
    # Model parameters
    DEL = 0.1
    rhoA = -0.015
    rhoB = 0.025
    nu = 0.02
    kap = 0.01
    lbar = 0
    sig_l = 0.1
    muY = 0.02
    sigY = 0.033
    pars = [rhoA, rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY]
    
    # Automatically set reverseFalseConsensus based on DEL value
    # If DEL >= 0: reverseFalseConsensus = False (use resultsLong_{hash}.csv)
    # If DEL < 0: reverseFalseConsensus = True (use results_{hash}.csv)
    reverseFalseConsensus = DEL < 0
    
    print(f"DEL = {DEL}")
    print(f"reverseFalseConsensus = {reverseFalseConsensus}")
    
    # Load grid data
    df = load_data_for_interpolation(pars, reverseFalseConsensus)
    if df is None:
        print("Cannot proceed without grid data. Please run the grid generation script first.")
        return

    # Pre-built tree for interpolation
    tree = cKDTree(df[['f', 'alpha']].values)

    # Simulating unconditional yield properties
    start_time = time.time()
    T = 1000 # We use 500000 in the paper
    dt = 1 / 12
    NT = int(T / dt)
    alpt = 0.5  # starting value for alpha
    ft = 0.5    # starting value for f
    dZ = np.sqrt(dt) * np.random.normal(size=NT)
    yield_values = []
    sigY_values = []
    alp_values = []
    f_values = []
    lt = np.log(1 / alpt - 1)
    
    print(f"Running Monte Carlo simulation with {NT} time steps...")
    
    for i in range(NT):
        if i % 10000 == 0:  # Progress indicator
            print(f"Progress: {i}/{NT} ({i/NT*100:.1f}%)")
            
        phiA = 1 / (rhoA + nu)
        phiB = 1 / (rhoB + nu)
        phi = ft * phiA + (1 - ft) * phiB
        betAt = (rhoA + nu) * phi
        betBt = (rhoB + nu) * phi
        
        muft = nu * (alpt * betAt * (1 - ft) - (1 - alpt) * betBt * ft) + (rhoB - rhoA) * ft * (1 - ft) + DEL**2 * (1/2 - ft) * ft * (1 - ft)
        sigft = ft * (1 - ft) * DEL
        dft = muft * dt + sigft * dZ[i]
        dlt = kap * (lbar - lt) * dt + sig_l * dZ[i]
        lt = lt + dlt  # update lt
        ft = ft + dft  # update ft
        
        # Boundary conditions with improved handling
        if ft < 0.001:
            ft = 0.0015 + 0.000001
        if ft > 1 - 0.001:
            ft = 1 - 0.000001 - 0.0015
        if alpt < 0.001:
            alpt = 0.0015 + 0.000001
        if alpt > 1 - 0.001:
            alpt = 1 - 0.000001 - 0.0015
            
        # Ensure ft and alpt are finite before calling the interpolation function
        if np.isfinite(ft) and np.isfinite(alpt):
            interpolated_results = interpolate_nearest_two(ft, alpt, df, tree)
            yield_values.append(interpolated_results[::2])
            sigY_values.append(interpolated_results[1::2])
        else:
            # Handle the non-finite case
            print("Non-finite values encountered: ft={}, alpt={}".format(ft, alpt))
            # Skip this iteration
            continue

        alpt = 1 / (1 + np.exp(-lt))
        f_values.append(ft)
        alp_values.append(alpt)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation elapsed time: {elapsed_time} seconds")

    # Save results to local HDF5 file
    start_time = time.time()
    
    # Create the output directory if it doesn't exist
    output_dir = 'Data/Model Disagreement'
    os.makedirs(output_dir, exist_ok=True)
    
    # Replicate the pars vector to match the length of other columns
    pars_replicated = np.tile(pars, int(np.ceil(len(f_values) / len(pars))))[:len(f_values)]

    # Prepare the data for saving
    results_df = pd.DataFrame({
        'f': f_values,
        'alpha': alp_values,
        'yield': yield_values,
        'sigY': sigY_values,
        'pars': pars_replicated
    })

    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_h5 = os.path.join(output_dir, f"TESTresultsLong_{timestamp}.h5") #TEST to avoid overwriting the resultsLong_{timestamp}.h5 file

    # Save the data to an HDF5 file
    results_df.to_hdf(filename_h5, key='df', mode='w')
    print(f"Results saved to {filename_h5}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Save elapsed time: {elapsed_time} seconds")

   
if __name__ == "__main__":
    main()