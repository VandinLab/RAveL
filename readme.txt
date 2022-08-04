The code runs on python 3 and assumes there is a virtual environment called ci_venv installed.

IT_utils.py contains a set of utility functions to perform and store (conditional) independence test results, as well as the dep() function in use in PCMB.
rad_utils.py contains a set of utility functions to perform SD estimation with Rademacher averages, and it should be performed before running RAveL algorithms
dependency_infos.py is a container class with information on the statistical independence test to perform, on the dataset file, and the file containing the results of the independence test
ids_creator.py is a script to create the DAG used in the synthetic experiment
lcd_methods.py contains RAveL and the other set of methods used for Local Causal Discovery 
main.py contains the script for running both synthetic and real world experiments

You can use run_experiments.sh to create the datasets and run the experiments.
Datasets are stored in a folder "dataset", and independence tests results onto one called "IT_results"

The real world dataset is the famous Boston housing dataset presented in
    Harrison Jr, D. and Rubinfeld, D.L., 1978. Hedonic housing prices and the demand for clean air. Journal of environmental economics and management, 5(1), pp.81-102

The code comes under a MIT license. Feel free to contact the authors at:
 - Dario Simionato <dario.simionato@phd.dei.unipd.it>
 - Fabio Vandin <fabio.vandin@unipd.it> [corresponding author]


The packages that must be installed in the environment are in "requirements.txt" file, despite not all of them may be strictly necessary.