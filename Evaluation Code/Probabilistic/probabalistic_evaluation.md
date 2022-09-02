# Copula Code and Evaluation Code

This code was used for the copulas and the resulting evaluation.
The folder contains files where the path must be adjusted to the given folder structure and sometimes needs a folder
structure to work.

## Use the Copula Files

Therefore, the files "main_disaggregated.py" and "main_supp_demand.py" are used.
To use the copula files different folders on the level with the files are required.
It needs a folder data which contains the mismatch timeseries as "Data/mismatch_0.pkl".

It is possible to use the <model_path_parameter> to differ between multiple sets of subresults and save the
corresponding copula result there.
The results of the different components need to be placed in "<model_path_parameter>/subresults/demand/*.pkl" for
example for demand and in the same way for every other supplier and the supply_forecast. In addition, a result folder need to be there (<model_path_parameter>/results_demand_supply/" and" <
model_path_parameter>/results_disaggregated").

It was run on python 3.8.6 with intel 19.1 (Red Hat Enterprise Linux (RHEL) 7.7) within a virtual environment based on the requirements_copula.txt provided in
the folder.

## Evaluation

The results of the individual predictions can be determined by the scripts in the "calc_crps_subresults" folder.
The results of the copulas can then be evaluated by the two files calculate_aggregated_crps.py and
calculate_partial_aggregated_and_disaggregated_crps.py. Here the paths need to be set. For the significance test, the
diebold mariano test in "dm_test_prob.py" located here was used.




