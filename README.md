# Bachelor Thesis Appendix

This repository contains all supplementary materials for the thesis on stochastic choice behavior.

## Contents

- `DecisionMakingData32.csv`: Full dataset comprising raw responses from all 32 participants as exported from Google Forms, prior to any data cleaning or exclusion.  
- `DecisionMakingDataOhne3_28.csv`: Refined dataset after applying exclusion criteria, where responses from Participant 3 and Participant 28 were removed due to concerns such as incomplete data, irregular response patterns, or violation of experimental requirements.  
- `clean_beta_output.csv`: Cleaned behavioral data used for model estimation.  
- `Simulation_Results.csv`: Output of model simulations.  
- `all_model_results.csv`: Summary of model fit results across conditions.

## Code Scripts

- `simulated_data_thesis.py`: Main analysis script using simulated data to fit models, evaluate empirical performance, and generate figures.
- `bachelor_thesis_analysis_code.py`: Main analysis script using real experimental data to fit models, evaluate empirical performance, and generate figures.

## Requirements

Python 3.8+ with the following packages:
- numpy  
- pandas  
- matplotlib  
- scipy
- sklearn
## Notes

The original questionnaire was implemented using Google Forms and contained randomized binary choice tasks across three behavioral conditions (indifference, indecisiveness, and experimentation). The raw responses were exported in CSV format and anonymized prior to analysis.

## Questionnaire Access

The original survey was created using Google Forms and designed to test stochastic choice behavior under three conditions: indifference, indecisiveness, and experimentation.

You can view the questionnaire form via the following link:  
(https://docs.google.com/forms/d/e/1FAIpQLSfB1xFI8e2gakB4NFKe3MpBM2pIZWIr0IkDkH4JfWt-Tn6-8Q/viewform?usp=header)

Note: The form has been closed for responses and is now provided solely for reference.
