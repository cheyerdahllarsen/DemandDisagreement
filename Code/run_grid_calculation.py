import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import gc
import pandas as pd
import hashlib
import os

# Function to generate a hash from parameters
def generate_filename_hash(pars):
    # Convert parameters to a string and then to a hash
    pars_str = '_'.join(map(str, pars))
    return hashlib.md5(pars_str.encode()).hexdigest()[:8]  # Use first 8 characters

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

# Create the Grid and Calculate Output
def calculate_and_save_results(f_values, alpha_values, pars):
    # Create the output directory if it doesn't exist
    output_dir = 'Data/Model Disagreement'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for f in f_values:
      for alp in alpha_values:
        output = getAllMaturities(f, alp, taus, pars, M, dt, device) # assuming pars is defined
        results.append([f, alp, *output, *pars])
        print(f'the f value is {f} and the alpha is {alp}')
    # Check the length of the output and adjust columns accordingly
    if len(results) > 0:
        num_output_elements = len(results[0]) - 2 - len(pars)  # f, alpha, and pars are excluded

    # Columns for DataFrame
    columns = ['f', 'alpha'] + [f'result_{i}' for i in range(num_output_elements)] + [f'par_{i}' for i in range(len(pars))]

    # Ensure the number of columns matches the data
    assert len(columns) == len(results[0]), "Column count does not match data length."

    df = pd.DataFrame(results, columns=columns)

    # Save to local directory
    filename_hash = generate_filename_hash(pars)
    file_path = os.path.join(output_dir, f'resultsLong_{filename_hash}.csv')
    df.to_csv(file_path, index=False)

    print(f"Data saved to local directory with filename: resultsLong_{filename_hash}.csv")
    print(f"Full path: {file_path}")

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Set up hyperparameters
    dt = 1 / 12
    M = 500000  # Number of paths. Note that when we do automatic diff it "eats memory"

    # Set up model parameters
    DEL = 0.1
    rhoA = 0.001
    rhoB = 0.05
    nu = 0.02
    kap = 0.01
    lbar = 0
    sig_l = 0.1
    muY = 0.02
    sigY = 0.033
    pars = [rhoA , rhoB, nu, DEL, kap, lbar, sig_l, muY, sigY]

    # Create the grid
    taus = [1, 2, 3, 4, 5, 7, 10]

    f_min = 0.001
    f_max = 1 - f_min
    alp_min = 0.001
    alp_max = 1 - alp_min
    f_steps = 100
    alp_steps = 100

    # Step 1: Create the Grid
    f_values = np.linspace(f_min, f_max, num=f_steps) # Define f_min, f_max, f_steps
    alpha_values = np.linspace(alp_min, alp_max, num=alp_steps) # Define alpha_min, alpha_max, alpha_steps

    # Run the calculation
    calculate_and_save_results(f_values, alpha_values, pars)