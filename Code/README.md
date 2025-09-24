# Replication Package for "Demand Disagreement"

## Overview

This replication package contains the code and data necessary to reproduce the results in the paper "Demand Disagreement" by Christian Heyerdahl-Larsen and Philipp Illeditsch. The main analysis is conducted in the Jupyter notebook `ReplicationTablesAndFigurePaper.ipynb`, which generates all tables and figures presented in the paper. The code constructs analysis files from multiple data sources including Survey of Professional Forecasters (SPF) data, model-generated disagreement measures, and financial market data. The replication should take approximately 2-4 hours to complete on a standard desktop machine.

## Data Availability and Provenance Statements

### Statement about Rights

☑ I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript.

☑ I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. 

### Summary of Availability

☑ All data are publicly available.

## Details on each Data Source

| Data Name | Data Files | Location | Provided | Citation |
|:--:|:--:|:--:|:--:|:--:|
| Survey of Professional Forecasters (SPF) | Individual_*.xlsx files | Data/SPF/2024/ | TRUE | Federal Reserve Bank of Philadelphia (2024) |
| Model-Generated Disagreement | *.npy, *.h5, *.csv files | Data/Model Disagreement/ | TRUE | Author-generated |
| NYSE Volume Data | StockVolumeNYSE.mat | Data/ | TRUE | NYSE (2024) |
| Shiller Annual Data | ShillerAnnualData.mat | Data/ | TRUE | Shiller (2024) |
| Nominal Yields | NominalYields.xlsx | Data/ | TRUE | Federal Reserve Economic Data (FRED) |
| Disagreement Time Series | DisagreementTS2024ForRegressions*.xlsx | Data/ | TRUE | Author-constructed |

### Survey of Professional Forecasters (SPF)
The SPF data contains individual forecaster predictions for various macroeconomic variables including CPI, GDP, employment, and interest rates. Data are provided in Excel format (.xlsx) with individual forecaster responses for each variable. The data spans multiple quarters and includes both point forecasts and uncertainty measures. Source: Federal Reserve Bank of Philadelphia, Survey of Professional Forecasters, accessed 2024.

### Model-Generated Disagreement Data
The model disagreement data consists of simulated results from the theoretical model presented in the paper. Files include:
- `DIS02V2.npy`, `DIS04V2.npy`, `DIS06V2.npy`, `DIS08V2.npy`: Disagreement measures for different parameter values
- `resultsLongV2.xlsx`: Long-run simulation results
- `resultsLong_*.h5`: HDF5 files containing detailed simulation outputs
- `model_data_for_f_plotsV4.npz`: Data for plotting model predictions

### Financial Market Data
- **NYSE Volume Data**: Daily trading volume data for NYSE from 2004-2017, provided in MATLAB format (.mat)
- **Shiller Annual Data**: Annual stock market and economic data including real returns, dividends, and consumption data
- **Nominal Yields**: Treasury yield data for various maturities, sourced from FRED

### Disagreement Time Series
Author-constructed time series of disagreement measures based on SPF data, provided in multiple versions (V0, V1, and main version) for robustness checks.

## Data File Downloads

The data is provided in three separate zip files due to size constraints:

### **Part A: Main Dataset** (`DemandDisagreementData.zip`)
- **DOI**: 10.17632/5vpb5rzv7s.1
- **Size**: ~2-3GB
- **Contents**: Complete data folder structure with all files except HDF5 (.h5) files and consumption share distribution files
- **Includes**: SPF data, financial market data, MATLAB files, Excel files, NumPy arrays

### **Part B: HDF5 Simulation Data** (`DemandDisagreementH5files.zip`)
- **DOI**: 10.17632/md24pnhz67.1
- **Size**: ~8-9GB  
- **Contents**: All HDF5 (.h5) files from model simulations
- **Setup**: Extract and place all .h5 files in `Data/Model Disagreement/` folder

### **Part C: Consumption Share Distribution** (`ConsumptionShareDistribution.zip`)
- **DOI**: 10.17632/rk343v5jk4.1
- **Size**: ~100-200MB
- **Contents**: Consumption share distribution files for the Online Appendix
- **Setup**: Extract and place all files in `Data/Model Disagreement/` folder

### **Complete Setup Instructions**
1. Download all three parts from their respective Mendeley Data DOIs
2. Extract `DemandDisagreementData.zip` to get the main data structure
3. Extract `DemandDisagreementH5files.zip` and copy all .h5 files to `Data/Model Disagreement/` folder
4. Extract `ConsumptionShareDistribution.zip` and copy all files to `Data/Model Disagreement/` folder
5. The dataset is now complete and ready for replication

## Computational Requirements

### Software Requirements
- Python 3.8 or higher
- MATLAB R2019b or higher
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - statsmodels
  - scipy
  - scikit-learn
  - seaborn
  - arch
  - stargazer
  - openpyxl
  - h5py
- Required MATLAB toolboxes:
  - Statistics and Machine Learning Toolbox

### Hardware Requirements
☑ Feasible to run on a desktop machine

**Details**: The code was last run on a Windows 11 machine with 128GB RAM and Intel i9 processor with 
a NVIDIA RTX 4090 GPU. The main notebook as it is now should complete in a few minutes. 
Recreating every file from scratch would take 1-2 weeks (rough estimate).

## Description of programs/code

### Main Analysis Files
- **`ReplicationTablesAndFigurePaper.ipynb`**: Main Jupyter notebook that reproduces all tables and figures in the paper. This notebook:
  - Loads and processes SPF data
  - Generates disagreement measures
  - Runs empirical regressions
  - Creates model predictions
  - Produces all figures and tables

### Supporting Analysis Files
- **`Table1.m`**: MATLAB script that generates Table 1 correlation matrix
- **`simulateYieldsandYieldVolas.py`**: Generates yield and volatility simulations
- **`simulatedisandyieldForRegressions.py`**: Creates simulated data for regression analysis
- **`CreateConditionalYieldVolasAndRiskPremia.ipynb`**: Creates conditional yield volatility and risk premium measures
- **`CreateGridYieldsandYieldsVola.ipynb`**: Generates yield grid calculations
- **`UnconditionalYieldCurveFileCreation.ipynb`**: Creates unconditional yield curve data

### Output Files
The main notebook generates the following output files:
- **Tables**: `DIS_regression_level_summary.tex`, `DIS_regression_change_summary.tex`, `DIS_regression_log_summary.tex`, `r2_summary.tex`
- **Figures**: `UnconditionalYieldsJFE.png`, `UnconditionalYieldVolatilitiesJFE.png`, `APUnconditionalStockMarketVolatilityJFE.png`, `APUnconditionalExcessReturnsJFE.png`, `BondVola5yearJFE.png`, `ExRBond5yearJFE.png`

## Instructions to Replicators

1. **Setup Environment**:
   - Install Python 3.8+ and required packages listed above
   - Install MATLAB R2019b+ with Statistics and Machine Learning Toolbox
   - Ensure all data files are in the correct directories as specified in the file structure

2. **Run MATLAB Code** (if needed):
   - Execute `Table1.m` to generate Table 1 (optional, as Table1.tex is already provided)

3. **Run Main Analysis**:
   - Open `ReplicationTablesAndFigurePaper.ipynb` in Jupyter
   - Execute all cells in order
   - The notebook will generate all tables and figures automatically

### Details
- The main notebook should be run from the `Code/` directory
- All data files are already processed and ready for analysis
- Output files will be saved in the `Code/` directory
- The notebook includes clear section headers indicating which parts generate which tables/figures

## List of tables and programs

The provided code reproduces:
☑ All numbers provided in text in the paper
☑ All tables and figures in the paper

| Figure/Table # | Program | Output File | Note |
|:--:|:--:|:--:|:--:|
| Table 1 | Table1.m | Data/Table1.tex | Correlation matrix |
| Tables 1-7 (Internet Appendix) | ReplicationTablesAndFigurePaper.ipynb | DIS_regression_*.tex | Regression results |
| R² Summary | ReplicationTablesAndFigurePaper.ipynb | r2_summary.tex | Model fit statistics |
| Figure 1 | ReplicationTablesAndFigurePaper.ipynb | - | Disagreement time series |
| Figure 2 | ReplicationTablesAndFigurePaper.ipynb | - | Model vs data comparison |
| Figure 3 | ReplicationTablesAndFigurePaper.ipynb | UnconditionalYieldsJFE.png | Unconditional yields |
| Figure 3 | ReplicationTablesAndFigurePaper.ipynb | UnconditionalYieldVolatilitiesJFE.png | Yield volatilities |
| Figure 3 | ReplicationTablesAndFigurePaper.ipynb | APUnconditionalStockMarketVolatilityJFE.png | Stock market volatility |
| Figure 3 | ReplicationTablesAndFigurePaper.ipynb | APUnconditionalExcessReturnsJFE.png | Excess returns |
| Figure 3 | ReplicationTablesAndFigurePaper.ipynb | BondVola5yearJFE.png | Bond volatility |
| Figure 3 | ReplicationTablesAndFigurePaper.ipynb | ExRBond5yearJFE.png | Bond excess returns |
| Figure 4 | ReplicationTablesAndFigurePaper.ipynb | - | Conditional analysis |
| Figure 5 | ReplicationTablesAndFigurePaper.ipynb | - | Model predictions |
| Figure 6 | ReplicationTablesAndFigurePaper.ipynb | - | Robustness analysis |
| Tables 3-4 | ReplicationTablesAndFigurePaper.ipynb | - | Main empirical results |
| Tables 8-15 (Internet Appendix) | ReplicationTablesAndFigurePaper.ipynb | - | Additional regressions |
| Table 5 | ReplicationTablesAndFigurePaper.ipynb | - | Main results table |
| Section III (Online Appendix) | alternativefilterjfeOnlineAppendix.py | estimated_statesAlternativeFilterJFE.csv, *.png | Alternative filtering analysis |
| Section VI (Online Appendix) | FigureLearning_SEP2025.m | FigLearning_*.eps | Learning figures and results |
| Section VII (Online Appendix) | FigProduction_SEP2025.m | FigProd*.eps | Production figures and results |
| Sections VIII-XII (Online Appendix) | OnlineAppendixExtraMatlabFiguresEXE.m | StateVariables2Dis.mat, PDhist.mat | Additional model analysis and figures |

### Helper Functions for Online Appendix
The following MATLAB functions are required for the online appendix analysis:
- **`OnlineAppendixConsumptionSharesDynamics.m`**: Calculates consumption share dynamics (drift and volatility)
- **`OnlineAppendixStockMarket.m`**: Computes stock market related quantities and risk measures
- **`OnlineAppendixStockMarketCriticalValues.m`**: Calculates critical values for stock market analysis
- **`OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean.m`**: Simulates consumption share dynamics for long-run analysis

## References

Federal Reserve Bank of Philadelphia. 2024. "Survey of Professional Forecasters." Philadelphia, PA: Federal Reserve Bank of Philadelphia. https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters

NYSE. 2024. "NYSE Volume Data." New York, NY: New York Stock Exchange.

Shiller, Robert J. 2024. "Irrational Exuberance Data." New Haven, CT: Yale University. http://www.econ.yale.edu/~shiller/data.htm

U.S. Federal Reserve Economic Data (FRED). 2024. "Treasury Constant Maturity Rate." St. Louis, MO: Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/

## License for Code

The code is licensed under a MIT license. 

## License for Data

The data are provided under appropriate licenses as specified by the original data providers.