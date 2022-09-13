# How to run this framework

## Data Creation

Data creation was run under Python 3.8. with an environment which is defined in requirements_data_creation.txt.
Therefore, in the main.py file only the data creation part needs to be active. It combines the data of the future 
energysytem with weather and calendar data. Therefore, it was run under Ubuntu 18.04.5 LTS. 

The base data is available at "Forecasting
Framework/forecasting_framework/datamodifiers/data/elec_s_100_ec_lv1.0_Co2L0.2-1H_cDE_y2010-2019_time.csv"

Also the weather data need to be downloaded and the three aggregate scripts in forecasting_framework/datamodifier need 
to be executed. As described in weatherdata.md.

Feature List:

| feature name             | Description                        |
|--------------------------|------------------------------------|
| "ssr_south"              |                                    |
| "ssr_middle"             | Surface net solar radiation        |
| "ssr_north"              |                                    |
| "str_south"              |                                    |
| "str_middle"             | Surface net thermal radiation      |
| "str_north"              |                                    |
| "t2m_north"              |                                    |
| "t2m_middle"             | Temperature 2m                     |
| "t2m_south"              |                                    |
| "public_holiday"         | Public Holiday                     |
| "partial_public_holiday" | Partial public holiday             |
| "summertime"             | Summertime                         |
| "u100_east"              | eastward Wind Velocity Baltic Sea  |
| "u100_north"             | eastward Wind Velocity North Sea   |
| "v100_east"              | northward Wind Velocity Baltic Sea |
| "v100_north"             | northward Wind Velocity North Sea  |
| "length_north"           | Wind Velocity Baltic Sea           |
| "length_east"            | Wind Velocity North Sea            |
| "onwind_u100_north"      |                                    |
| "onwind_u100_middle"     | eastward Wind Velocity             |
| "onwind_u100_south"      |                                    |
| "onwind_v100_north"      |                                    |
| "onwind_v100_middle"     | northward Wind Velocity            |
| "onwind_v100_south"      |                                    |
| "onwind_length_north"    |                                    |
| "onwind_length_middle"   | Wind Velocity                      |
| "onwind_length_south"    |                                    |
| "weekday",               | Weekdays One hot encoded           |
| "hour"                   | Hours One hot encoded              |
| "lag_0"                  | y 24 hours ago                     |
| "lag_1"                  | y 48 hours ago                     |
| "lag_2"                  | y 72 hours ago                     |
| "lag_3"                  | y 96 hours ago                     |
| "lag_4"                  | y 120 hours ago                    |
| "lag_5"                  | y 144 hours ago                    |
| "lag_6"                  | y 168 hours ago                    |
| "lag_7"                  | y 172 hours ago                    |



## Forcasting and Bayesian Optimization

Forecasting was done using the support of BwUniCluster (Red Hat Enterprise Linux (RHEL) 7.7). Therefore, we use 3.8.6 with intel 19.1 compiler with the
requirements given in "requirements_mismatch_forecast.txt".

## Usage of modelfiles

To forecast the given modelfiles in the folder "Modelfiles Forcasting Framework" were used.
They need to be put in the model's folder of the framework. For example, "models/solar_mlp".

They can be executed one by one or all together. Therefore, the modelfiles given the train parameter needs to be true
or need to be manually executed by the parameter --datapath .
Depending on the usage of parameters when executing the main.py .

The results will be saved into the folder "models/model_results". Here the results of the single folds as well as the
hyperparameters evaluated for the best model of the bayesian optimisation and the forecast result for the test data are
saved.
