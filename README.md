# MSc Research: Predicting Flash Flood Events using Hydrological Modeling and Machine Learning

This repository contains selected Python code developed as part of my MSc research on
event-based flash flood detection using simulated discharge time series.

## Objective
To detect flash flood events from simulated discharge derived using a simplified
unit hydrograph approach, combined with machine-learning classification methods.

## Methods
- Extraction of precipitation data from ERA5 and ECMWF HRES forecasts
- Development of a unit hydrograph–based rainfall–runoff model to simulate discharge
  from observed and forecast precipitation
- Machine learning classification for event detection:
  Logistic Regression and Random Forest (scikit-learn)
- Model evaluation using precision–recall, false alarm ratio (FAR),
  and event/spell-based verification metrics

## Files
- `ECMWF_data_process.ipynb`  
  Extraction and preprocessing of ERA5 and ECMWF HRES precipitation data

- `hydrological_response.ipynb`  
  Simulation of discharge from observed and forecast precipitation using a
  unit hydrograph function

- `machine_learning_dynamic.ipynb`  
  Training and validation of machine-learning classifiers for flash flood
  event detection

- `all_year_metrics.ipynb`  
  Performance evaluation of event predictions using precision–recall,
  FAR, and event-based verification

## Notes
- Code is shared for demonstration purposes.
- Input datasets are excluded due to size and data confidentiality.
