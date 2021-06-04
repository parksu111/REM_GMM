# REM_GMM

﻿This folder contains the raw data and code used for analysis in the paper ‘A probabilistic model for the ultradian timing of REM sleep in mice’. 
 
 Code:
* Not all files in this folder contain code. However, they were placed in the same folder because having all these files in one directory makes it easier to run the analyses
* Code for main figures
   * ‘fig1.py’
   * ‘fig2.py’
   * ‘fig3.py’
   * ‘fig4.py’
   * ‘fig5.py’
   * ‘fig6.py’
   * ‘fig7.py’
* Code for various analyses
   * ‘rempropensity.py’ - Contains helper functions used in all the scripts for main figures
   * ‘gmmparameters.py’ - Code used to estimate GMM parameters for light phase
   * ‘lftest.py’ - Code used to perform Lilliefors-corrected KS test for light phase
   * ‘darkgmm.py’ - Code used to estimate GMM parameters for dark phase
   * ‘darklftest.py’ - Code used to perform Lilliefors-corrected KS test for dark phase
* Code for Supporting Information
   * ‘s1table_light.py’
   * ‘s1table_dark.py’
   * ‘s3table.py’
   * ‘s5figure.py’
   * ‘s6figure.py’
   * ‘s7figure.py’
   * ‘s8fig_s10fig_10.py’ - 10 second MA threshold
   * ‘s8fig_s10fig_30.py’ - 30 second MA threshold
   * ‘s8fig_s10fig_0.py’ - No MA
   * ‘s9figure.py’
   * ‘s11figure.py’
   * ‘s12figure.py’
   * ‘s13figure.py’
* Data files containing parameters, model coefficients, etc.
   * ‘2gmmDF.csv’ - GMM parameters for light phase
   * ‘2linfitDF.csv’ - Linear fit coefficients relating GMM parameters to REMpre for the light phase
   * ‘2logfitDF.csv’ - Log fit coefficients relating GMM parameters to REMpre for the light phase
   * ‘darkgmmDF.csv’ - GMM parameters for the dark phase
   * ‘darklinfitDF.csv’ - Linear fit coefficients relating GMM parameters to REMpre for the dark phase
   * ‘darklogfitDF.csv’ - Log fit coefficients relating GMM parameters to REMpre for the dark phase
   * ‘gmm0DF.csv’ - GMM parameters with no MA
   * ‘gmm10DF.csv’ - GMM parameters with 10 second MA threshold
   * ‘gmm30DF.csv’ - GMM parameters with 30 second MA threshold
