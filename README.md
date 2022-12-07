# A probabilistic model for the ultradian timing of REM sleep in mice

This repository contains the code used for analysis in the paper [A probabilistic model for the ultradian timing of REM sleep in mice](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009316). 

**Citation**: Park S-H, Baik J, Hong J, Antila H, Kurland B, Chung S, et al. (2021) A probabilistic model for the ultradian timing of REM sleep in mice. PLoS Comput Biol 17(8): e1009316. https://doi.org/10.1371/journal.pcbi.1009316
 
 ## Abstract
 A salient feature of mammalian sleep is the alternation between rapid eye movement (REM) and non-REM (NREM) sleep. However, how these two sleep stages influence each other and thereby regulate the timing of REM sleep episodes is still largely unresolved. Here, we developed a statistical model that specifies the relationship between REM and subsequent NREM sleep to quantify how REM sleep affects the following NREM sleep duration and its electrophysiological features in mice. We show that a lognormal mixture model well describes how the preceding REM sleep duration influences the amount of NREM sleep till the next REM sleep episode. The model supports the existence of two different types of sleep cycles: Short cycles form closely interspaced sequences of REM sleep episodes, whereas during long cycles, REM sleep is first followed by an interval of NREM sleep during which transitions to REM sleep are extremely unlikely. This refractory period is characterized by low power in the theta and sigma range of the electroencephalogram (EEG), low spindle rate and frequent microarousals, and its duration proportionally increases with the preceding REM sleep duration. Using our model, we estimated the propensity for REM sleep at the transition from NREM to REM sleep and found that entering REM sleep with higher propensity resulted in longer REM sleep episodes with reduced EEG power. Compared with the light phase, the buildup of REM sleep propensity was slower during the dark phase. Our data-driven modeling approach uncovered basic principles underlying the timing and duration of REM sleep episodes in mice and provides a flexible framework to describe the ultradian regulation of REM sleep in health and disease.
 
 
 **Code**:
* Not all files in this folder contain code. However, they were placed in the same folder because having all these files in one directory makes it easier to run the analyses.
* Code for main figures:
   * ‘fig1.py’
   * ‘fig2.py’
   * ‘fig3.py’
   * ‘fig4.py’
   * ‘fig5.py’
   * ‘fig6.py’
   * ‘fig7.py’
* Code for various analyses:
   * ‘rempropensity.py’ - Contains helper functions used in all the scripts for main figures
   * ‘gmmparameters.py’ - Code used to estimate GMM parameters for light phase
   * ‘lftest.py’ - Code used to perform Lilliefors-corrected KS test for light phase
   * ‘darkgmm.py’ - Code used to estimate GMM parameters for dark phase
   * ‘darklftest.py’ - Code used to perform Lilliefors-corrected KS test for dark phase
* Code for Supporting Information:
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
* Data files containing parameters, model coefficients, etc... :
   * ‘2gmmDF.csv’ - GMM parameters for light phase
   * ‘2linfitDF.csv’ - Linear fit coefficients relating GMM parameters to REMpre for the light phase
   * ‘2logfitDF.csv’ - Log fit coefficients relating GMM parameters to REMpre for the light phase
   * ‘darkgmmDF.csv’ - GMM parameters for the dark phase
   * ‘darklinfitDF.csv’ - Linear fit coefficients relating GMM parameters to REMpre for the dark phase
   * ‘darklogfitDF.csv’ - Log fit coefficients relating GMM parameters to REMpre for the dark phase
   * ‘gmm0DF.csv’ - GMM parameters with no MA
   * ‘gmm10DF.csv’ - GMM parameters with 10 second MA threshold
   * ‘gmm30DF.csv’ - GMM parameters with 30 second MA threshold

All the data used for this project can be downloaded [here](https://upenn.box.com/s/3zcesr4a7l7hgb9andmq4di4t6zvaoql).
