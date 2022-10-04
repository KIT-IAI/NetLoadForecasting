# Disclaimer

In the code and in the documentation of the code there is mostly the term "mismatch" used. This is equivalent to the
net load described in the paper. Here, it is meant as the mismatch between the load and the renewable energies.

Furthermore, we deleted the urlib package from our requirements.txt as it has security issues. This package should not influence the reproducibility of the code since we only used it for notification purposes, however, you are free to install this package yourself at your own risk.

# Reproducing the results

In order to reproduce the results, it is important to perform each step individually and to observe the requirements
used. Here, different results were generated on different systems with different requirements. These information can be taken from
the requirements.txt files contained in the folders and the different readmes in the folders.

If the feature selection should also to be reproduced, a two-step procedure is recommended. The first step is used to
produce the data with the forecasting framework. Afterwards, the code in the feature selections folder is to be executed
on the resulting time series. This results in the model configurations given in the folder Modelfiles Forecasting
Framework. These are compatible to the given Forecasting Framework and can be used in step two to produce the model
results.
After that, the code in the "Evaluation Code" folder is used to calculate the results, the copulas and also to perform
the significance tests. In each of the folders is described again in more detail how to use them.