# Net Load Forecasting using Different Aggregation Levels

This repository contains code and data to replicate the results in "Net Load Forecasting using Different Aggregation
Levels".

> Maximilian Beichter, Kaleb Phipps, Martha Maria Frysztacki, Ralf Mikut, Veit Hagenmeyer and Nicole Ludwig (2022). “Net
> Load Forecasting using Different Aggregation
> Levels”. In: <to_insert>. doi: [](<doi>)

## Acknowledgements

The authors acknowledge support by the state of Baden-Württemberg through bwHPC.

We acknowledge the use of Diebold Mariano test from John Tsang (https://github.com/johntwk/Diebold-Mariano-Test).

## Funding

This work is funded by the German Research Foundation (DFG) as part of the Research Training Group 2153
“Energy Status Data – Informatics Methods for its Collection, Analysis and Exploitation”, by the Helmholtz
Association’s Initiative and Networking Fund through Helmholtz AI, and the Helmholtz Association under the
Program “Energy System Design”. Nicole Ludwig acknowledges funding by the DFG under Germany’s Excellence
Strategy – EXC number 2064/1 – Project number 390727645.

## Data

The weather data used in this study are ERA5
reanalysis data and openly available via the Copernicus Climate Data Store (CDS)
https://cds.climate.copernicus.eu/home .

The PyPSA-Eur Data is available direct in the folder "PyPSA Data". To combine this data with our feature generation you need to launch the forecasting framework with the use of the ERA5 weather data.

<h2>License</h2>

This code is licensed under the [MIT License](LICENSE).