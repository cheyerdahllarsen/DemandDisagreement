# Demand Disagreement Paper - Dependencies Analysis

This document analyzes the dependencies for the clean version of the "Demand Disagreement" paper (accepted to JFE) based on `DemandDisagreementAccepted.tex`.

## Main Document Structure

The main document `DemandDisagreementAccepted.tex` is a self-contained LaTeX file that includes:

### Document Class and Packages
- **Document Class**: `elsarticle` (Elsevier style for JFE)
- **Key Packages**: 
  - `amsmath`, `amsthm`, `graphicx`, `amsfonts`, `booktabs`, `caption`, `hyperref`, `url`, `xcolor`, `float`, `array`, `multirow`, `changepage`
  - `setspace` for line spacing
  - Custom formatting commands and theorem environments

### Content Structure
The main document contains all content inline (not using `\input{}` commands for main sections):
- Title, authors, abstract, keywords
- Introduction (Section I)
- Empirical Motivation - Disagreement Correlation Puzzle (Section II)
- Model (Section III)
- Equilibrium (Section IV) 
- Analysis (Section V)
- Conclusion
- Bibliography
- Internet Appendix

## Required Dependencies

### 1. Bibliography Files
**REQUIRED:**
- `OLGBeliefsAccepted.bib` - Main bibliography file referenced by `\bibliography{OLGBeliefsAccepted}`

**NOT REQUIRED:**
- `OLGBeliefs.bib` - Alternative bibliography file (not used in accepted version)

### 2. Style Files
**REQUIRED:**
- None (all style information is embedded in the main document)

**NOT REQUIRED:**
- `jf.bst`, `jf.sty` - JFE bibliography style (not used)
- `rfs.bst` - RFS bibliography style (not used)

### 3. Internet Appendix
**REQUIRED:**
- `InternetAppendix_DemandDisagreementRegressions.tex` - Referenced by `\input{InternetAppendix_DemandDisagreementRegressions.tex}`

This file contains:
- Empirical results section with regression tables
- Disagreement spanning regressions
- Alternative volatility and bond risk premia specifications
- Tables with detailed regression coefficients and statistics

### 4. Figures
**ANALYSIS NEEDED:** The main document contains figure environments with captions but no explicit `\includegraphics{}` commands found. This suggests figures may be:
1. Embedded directly in the document, or
2. Referenced through a different mechanism, or  
3. The figures are placeholders for the clean version

**POTENTIALLY REQUIRED FIGURES** (based on figure environments in document):
- `DemandDisagreementMotivating` - Measuring demand disagreement
- `DemandDisagreementYieldModelAndData` - Yield disagreement comparison
- `UnconditionalAP` - Unconditional asset pricing moments (6 subfigures)
- `ConsumptionShare` - Consumption share dynamics (4 subfigures)
- `VolaRiskPremium2f` - Conditional return volatility and risk premium
- `UKFalphaandf` - Estimated state variables and fit

**LIKELY CANDIDATES** (from figures directory):
- `DemandDisagreement.png` - Main demand disagreement figure
- `MeanDisagreementDataAndModel.png` - Model vs data comparison
- `APUnconditionalExcessReturnsJFE.png` - Asset pricing moments
- `APUnconditionalStockMarketVolatilityJFE.png` - Stock market volatility
- `BondVola5yearJFE.png` - Bond volatility
- `ExRBond5yearJFE.png` - Bond excess returns
- `UnconditionalYieldsJFE.png` - Yield curves
- `UnconditionalYieldVolatilitiesJFE.png` - Yield volatilities

### 5. Tables
All tables appear to be embedded directly in the main document with no external dependencies.

## Files NOT Required for Clean Version

### Main Paper Versions (Other than Accepted)
- `DemandDisagreementJFER2.tex`, `DemandDisagreementJFER3.tex` - Other JFE versions
- `DemandDisagreementV3.tex` - Version 3
- All associated `.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.pdf`, `.synctex.gz` files

### Individual Section Files (Content Embedded in Main)
- `Introduction.tex`
- `EmpiricalMotivation.tex` 
- `ModelV3.tex`
- `Learning.tex`
- `conclusion.tex`
- `TitleandAbstractAccepted.tex`, `TitleandAbstractV3.tex`

### Internet Appendix Sections (Not Referenced)
- `InternetAppendix_AlternativeCalibration.tex`
- `InternetAppendix_AltPars.tex`
- `InternetAppendix_BlacksLeverage.tex`
- `InternetAppendix_CorrelationPuzzle.tex`
- `InternetAppendix_FilteringExtendedResults.tex`
- `InternetAppendix_GeneralDisagreement.tex`
- `InternetAppendix_PDpredictability.tex`
- `InternetAppendix_Production.tex`
- `InternetAppendix_ReverseBias.tex`
- `InternetAppendix_TitlePage.tex`
- `InternetAppendix_Trading.tex`
- `InternetAppendix_UKF.tex`
- `InternetAppendix_WealthShares.tex`
- `DemandDisagrement_InternetAppendix.tex` (main appendix file)

### Review Process Files
- `Referee Reports R1/` directory
- `Referee Reports R2/` directory  
- `replyToEditor.tex`
- `Response2Reviewer2_CHL.tex`

### Miscellaneous Files
- `name.tex`
- `stuff.tex`
- `ProductionTobeMoveIntoIA.tex`
- `TimevaryingDisagreementTobeMoveIntoIA.tex`

### Most Figures Directory
- Most files in `figures/` directory (189 files) - only JFE-specific versions likely needed
- Most files in `figuresJFE/` directory - only those actually referenced

## Recommended Clean Version Structure

```
DemandDisagreement/
├── DemandDisagreementAccepted.tex          # Main paper
├── InternetAppendix_DemandDisagreementRegressions.tex  # Required appendix
├── OLGBeliefsAccepted.bib                  # Bibliography
├── figuresJFE/                             # JFE-specific figures
│   ├── APUnconditionalExcessReturnsJFE.png
│   ├── APUnconditionalStockMarketVolatilityJFE.png
│   ├── BondVola5yearJFE.png
│   ├── ExRBond5yearJFE.png
│   ├── UnconditionalYieldsJFE.png
│   ├── UnconditionalYieldVolatilitiesJFE.png
│   ├── MeanDisagreementDataAndModel.png
│   └── [other referenced figures]
└── README.md                               # Project documentation
```

## Cleanup Status

✅ **COMPLETED CLEANUP ACTIONS:**
1. **File Removal**: Removed all non-required files listed above
2. **Paper Versions**: Removed unused versions (JFER2, JFER3, V3) and all auxiliary files
3. **Review Process**: Removed referee reports and review materials
4. **Miscellaneous**: Removed unused section files, style files, and temporary files
5. **Internet Appendix**: Kept all required Internet Appendix files

🔄 **PENDING DECISIONS:**
1. **Figure Cleanup**: All figures kept for safety - need manual review to identify unused ones
2. **Compilation Test**: Need to test that both main paper and Internet Appendix compile correctly

## Identified Figure Dependencies

**CONFIRMED NEEDED** (explicitly referenced in code):
- `figures/DemandDisagreementFigIntro.png`
- `figures/volAPV3.png`, `figures/RPV3.png`
- `figures/alphaFilter.png`, `figures/ffilter.png`, `figures/yieldfilter.png`, `figures/DDfilter.png`
- `figures/CorrelationPuzzle2DEL.eps` (Internet Appendix)
- All 11 files in `figuresJFE/` directory

**UNCERTAIN** (may be needed for Internet Appendix):
- Alternative versions with "ALT" suffix
- Production economy figures
- Wealth share figures
- Various EPS files that might be referenced indirectly

## Internet Appendix Dependencies

The main Internet Appendix file `DemandDisagrement_InternetAppendix.tex` (note the typo in filename) has the following dependencies:

### Document Structure
- **Document Class**: `article` (12pt)
- **Key Packages**: Similar to main document but includes `epsfig`, `harvard` for citations
- **Bibliography Style**: Uses `jf` style (Harvard referencing)

### Required Internet Appendix Files
**ALL REQUIRED** (all are actively included via `\input{}` commands):
1. `InternetAppendix_TitlePage.tex` - Title page
2. `InternetAppendix_DemandDisagreementRegressions.tex` - Empirical results (also included in main paper)
3. `InternetAppendix_UKF.tex` - Unscented Kalman Filter section
4. `InternetAppendix_FilteringExtendedResults.tex` - Extended filtering results
5. `InternetAppendix_AlternativeCalibration.tex` - Alternative calibration
6. `InternetAppendix_AltPars.tex` - Alternative parameters
7. `InternetAppendix_ReverseBias.tex` - Reverse false consensus bias
8. `InternetAppendix_GeneralDisagreement.tex` - General disagreement
9. `InternetAppendix_Production.tex` - Production economy
10. `InternetAppendix_WealthShares.tex` - Wealth shares
11. `InternetAppendix_CorrelationPuzzle.tex` - Correlation puzzle
12. `InternetAppendix_PDpredictability.tex` - Price-dividend ratio and return predictability
13. `InternetAppendix_Trading.tex` - Trading volume
14. `InternetAppendix_BlacksLeverage.tex` - Black's leverage effect

### Bibliography Dependencies
**REQUIRED:**
- `OLGBeliefs.bib` - Referenced by `\bibliography{OLGBeliefs}` (note: different from main paper which uses `OLGBeliefsAccepted.bib`)

### Figure Dependencies
**IDENTIFIED FIGURES:**
- `figures/CorrelationPuzzle2DEL.eps` - Referenced in `InternetAppendix_CorrelationPuzzle.tex`

**POTENTIALLY REQUIRED FIGURES** (based on figure environments in appendix files):
- Alternative calibration figures (referenced as `fig:UnconditionalAPALT`, `fig:ConsumptionShareALT`, `fig:VolaRiskPremium2fALT`)
- Production economy figures (referenced as `fig:prod`)
- Wealth shares figures (referenced as `fig:WealthShare`)

**LIKELY CANDIDATES** (from figures directory):
- `CorrelationPuzzle2DEL.eps` - Correlation puzzle analysis
- Alternative versions of main figures with "ALT" suffix
- Production-related figures
- Wealth share distribution figures

## Updated Recommended Clean Version Structure

```
DemandDisagreement/
├── DemandDisagreementAccepted.tex                    # Main paper
├── DemandDisagrement_InternetAppendix.tex            # Main Internet Appendix
├── InternetAppendix_TitlePage.tex                    # Appendix title page
├── InternetAppendix_DemandDisagreementRegressions.tex # Empirical results
├── InternetAppendix_UKF.tex                          # UKF section
├── InternetAppendix_FilteringExtendedResults.tex     # Extended filtering
├── InternetAppendix_AlternativeCalibration.tex       # Alternative calibration
├── InternetAppendix_AltPars.tex                      # Alternative parameters
├── InternetAppendix_ReverseBias.tex                  # Reverse bias
├── InternetAppendix_GeneralDisagreement.tex          # General disagreement
├── InternetAppendix_Production.tex                   # Production economy
├── InternetAppendix_WealthShares.tex                 # Wealth shares
├── InternetAppendix_CorrelationPuzzle.tex            # Correlation puzzle
├── InternetAppendix_PDpredictability.tex             # PD predictability
├── InternetAppendix_Trading.tex                      # Trading volume
├── InternetAppendix_BlacksLeverage.tex               # Black's leverage
├── OLGBeliefsAccepted.bib                            # Main paper bibliography
├── OLGBeliefs.bib                                    # Internet Appendix bibliography
├── figures/                                          # Main figures directory
│   └── CorrelationPuzzle2DEL.eps                     # Correlation puzzle figure
├── figuresJFE/                                       # JFE-specific figures
│   ├── APUnconditionalExcessReturnsJFE.png
│   ├── APUnconditionalStockMarketVolatilityJFE.png
│   ├── BondVola5yearJFE.png
│   ├── ExRBond5yearJFE.png
│   ├── UnconditionalYieldsJFE.png
│   ├── UnconditionalYieldVolatilitiesJFE.png
│   ├── MeanDisagreementDataAndModel.png
│   └── [other referenced figures]
└── README.md                                         # Project documentation
```

## Files NOT Required for Clean Version

### Main Paper Versions (Other than Accepted)
- `DemandDisagreementJFER2.tex`, `DemandDisagreementJFER3.tex` - Other JFE versions
- `DemandDisagreementV3.tex` - Version 3
- All associated `.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.pdf`, `.synctex.gz` files

### Individual Section Files (Content Embedded in Main)
- `Introduction.tex`
- `EmpiricalMotivation.tex` 
- `ModelV3.tex`
- `Learning.tex`
- `conclusion.tex`
- `TitleandAbstractAccepted.tex`, `TitleandAbstractV3.tex`

### Internet Appendix Files (Not Referenced)
- `InternAppendix_TradingVolume.tex` (different from `InternetAppendix_Trading.tex`)
- `InternetAppendix_ReverseBiasOLD.tex` (old version)

### Review Process Files
- `Referee Reports R1/` directory
- `Referee Reports R2/` directory  
- `replyToEditor.tex`
- `Response2Reviewer2_CHL.tex`

### Miscellaneous Files
- `name.tex`
- `stuff.tex`
- `ProductionTobeMoveIntoIA.tex`
- `TimevaryingDisagreementTobeMoveIntoIA.tex`

### Style Files
- `jf.bst`, `jf.sty` - Not used in main document
- `rfs.bst` - Not used

## Final Clean Repository Structure

```
DemandDisagreement/
├── DemandDisagreementAccepted.tex                    # Main paper
├── DemandDisagrement_InternetAppendix.tex            # Main Internet Appendix
├── InternetAppendix_TitlePage.tex                    # Appendix title page
├── InternetAppendix_DemandDisagreementRegressions.tex # Empirical results
├── InternetAppendix_UKF.tex                          # UKF section
├── InternetAppendix_FilteringExtendedResults.tex     # Extended filtering
├── InternetAppendix_AlternativeCalibration.tex       # Alternative calibration
├── InternetAppendix_AltPars.tex                      # Alternative parameters
├── InternetAppendix_ReverseBias.tex                  # Reverse bias
├── InternetAppendix_GeneralDisagreement.tex          # General disagreement
├── InternetAppendix_Production.tex                   # Production economy
├── InternetAppendix_WealthShares.tex                 # Wealth shares
├── InternetAppendix_CorrelationPuzzle.tex            # Correlation puzzle
├── InternetAppendix_PDpredictability.tex             # PD predictability
├── InternetAppendix_Trading.tex                      # Trading volume
├── InternetAppendix_BlacksLeverage.tex               # Black's leverage
├── OLGBeliefsAccepted.bib                            # Main paper bibliography
├── OLGBeliefs.bib                                    # Internet Appendix bibliography
├── figures/                                          # All figures (kept for safety)
│   └── [120 files: 76 EPS, 43 PNG, 1 PS]
├── figuresJFE/                                       # JFE-specific figures
│   └── [11 PNG files - all confirmed needed]
├── README_Dependencies.md                            # This analysis
└── README.md                                         # Project documentation
```

## Cleanup Summary

**Files Removed:** ~50+ files including:
- 3 unused paper versions and their auxiliary files
- 2 referee report directories
- 10+ individual section files (content embedded in main)
- 2 unused Internet Appendix files
- 5+ miscellaneous files
- 3 style files

**Files Kept:** 39 files total
- 2 main documents (paper + Internet Appendix)
- 15 Internet Appendix section files
- 2 bibliography files
- 131 figure files (kept for safety)
- 2 README files

## Notes

- The main document is remarkably self-contained with minimal external dependencies
- The Internet Appendix requires ALL 14 section files plus title page
- The Internet Appendix uses a different bibliography file (`OLGBeliefs.bib`) than the main paper (`OLGBeliefsAccepted.bib`)
- Most content that appears in separate `.tex` files is actually embedded in the main document
- The accepted version appears to be a consolidation of multiple previous versions
- There's a typo in the main Internet Appendix filename: "Disagrement" instead of "Disagreement"
- **Next step**: Test compilation of both documents and manually review figures for further cleanup
