The code runs on python 3 and exploits R libraries for conditional independence tests with pearson coefficient + t student test, as well for navigating the DAG structure.
Those functions may be easily adapted using other python packages, if required.

IT_utils.py contains a set of utility functions to perform and store (conditional) independence test results, as well as the dep() function in use in PCMB.
rad_utils.py contains a set of utility functions to perform SD estimation with Rademacher averages, and it should be performed before running RAveL algorithms
dependency_infos.py is a container class with information on the statistical independence test to perform, on the dataset file, and the file containing the results of the independence test
ids_creator.py is a script to create the DAG used in the synthetic experiment
main.py contains the script for running both synthetic and real world experiments

You can use run_experiments.sh to create the datasets and run the experiments. They assume there is a virtual environment called ci_venv installed.
Datasets are stored in a folder "dataset", and independence tests results onto one called "IT_results"

The real world dataset is the famous Boston housing dataset presented in
    Harrison Jr, D. and Rubinfeld, D.L., 1978. Hedonic housing prices and the demand for clean air. Journal of environmental economics and management, 5(1), pp.81-102

The code comes under a MIT license. Feel free to contact the authors at:
 - Dario Simionato <dario.simionato@phd.dei.unipd.it>
 - Fabio Vandin <fabio.vandin@unipd.it> [corresponding author]


The following packages were installed on virtual environment that run the code, despite not all those packages are strictly required (python 3):


argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
asttokens==2.0.5
attrs==21.4.0
backcall==0.2.0
backports.zoneinfo==0.2.1
beautifulsoup4==4.11.1
bleach==5.0.0
cffi==1.15.0
cycler==0.11.0
debugpy==1.6.0
decorator==5.1.1
defusedxml==0.7.1
dill==0.3.4
entrypoints==0.4
executing==0.8.3
fastjsonschema==2.15.3
fcit==1.2.0
fonttools==4.31.2
importlib-resources==5.7.1
ipykernel==6.13.0
ipython==8.3.0
ipython-genutils==0.2.0
ipywidgets==7.7.0
jedi==0.18.1
Jinja2==3.0.3
joblib==1.1.0
jsonschema==4.5.1
jupyter==1.0.0
jupyter-client==7.3.1
jupyter-console==6.4.3
jupyter-core==4.10.0
jupyterlab-pygments==0.2.2
jupyterlab-widgets==1.1.0
kiwisolver==1.4.2
MarkupSafe==2.0.1
matplotlib==3.5.1
matplotlib-inline==0.1.3
mistune==0.8.4
multiprocess==0.70.12.2
nbclient==0.6.3
nbconvert==6.5.0
nbformat==5.4.0
nest-asyncio==1.5.5
notebook==6.4.11
numpy==1.21.4
packaging==21.3
pandas==1.3.4
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.1.0
prometheus-client==0.14.1
prompt-toolkit==3.0.29
psutil==5.9.0
ptyprocess==0.7.0
pure-eval==0.2.2
pycparser==2.21
Pygments==2.12.0
pyparsing==3.0.7
pyrsistent==0.18.1
python-dateutil==2.8.2
pytz==2021.3
pytz-deprecation-shim==0.1.0.post0
pyzmq==22.3.0
qtconsole==5.3.0
QtPy==2.1.0
rpy2==3.4.5
scikit-learn==1.0.1
scipy==1.7.3
seaborn==0.11.2
Send2Trash==1.8.0
six==1.16.0
sklearn==0.0
soupsieve==2.3.2.post1
stack-data==0.2.0
terminado==0.13.3
threadpoolctl==3.0.0
tinycss2==1.1.1
tornado==6.1
tqdm==4.62.3
traitlets==5.2.0
tzdata==2021.5
tzlocal==4.1
wcwidth==0.2.5
webencodings==0.5.1
widgetsnbextension==3.6.0
zipp==3.8.0


Packages requested (R):
- bnlearn
- readr
- Rcpp
- RcppZiggurat
- Rfast