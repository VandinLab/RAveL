!/bin/env bash
source ci_venv/bin/activate

python ids_creator.py

## Part 1 : RAveL vs SoA algorithms

# run PCMB
python main.py -t 1 -v 13 -e 15 -r 5 -s 10 -c 50
# run IAMB
python main.py -t 2 -v 13 -e 15 -r 5 -s 10 -c 50
# run RAveL with Bonferroni correction
python main.py -t 3 -v 13 -e 15 -r 5 -s 10 -c 50
# calculate Rademacher independence tests and run RAveL (std version)
python main.py -t 4 -v 13 -e 15 -r 5 -s 10 -c 50
python main.py -t 5 -v 13 -e 15 -r 5 -s 10 -c 50
# merge results
python main.py -t 6 -v 13 -e 15 -r 5 -s 10 
# plot RAveL vs SoA algorithms
python main.py -t 7 -v 13 -e 15 -r 5 -s 10 

## Part 2: RAvel comparison between Bonferroni and Rademacher by increasing number of variables and with our statistic
python main.py -t 3 -v 13 -e 0 -r 5 -s 10 -c 50
python main.py -t 3 -v 13 -e 5 -r 5 -s 10 -c 50
python main.py -t 3 -v 13 -e 10 -r 5 -s 10 -c 50
python main.py -t 4 -v 13 -e 0 -r 5 -s 10 -c 50
python main.py -t 4 -v 13 -e 5 -r 5 -s 10 -c 50
python main.py -t 4 -v 13 -e 10 -r 5 -s 10 -c 50
python main.py -t 5 -v 13 -e 0 -r 5 -s 10 -c 50
python main.py -t 5 -v 13 -e 5 -r 5 -s 10 -c 50
python main.py -t 5 -v 13 -e 10 -r 5 -s 10 -c 50
# repeat calculus with Rademacer on our statistic
python main.py -t 44 -v 13 -e 0 -r 5 -s 10 -c 50
python main.py -t 44 -v 13 -e 5 -r 5 -s 10 -c 50
python main.py -t 44 -v 13 -e 10 -r 5 -s 10 -c 50
python main.py -t 44 -v 13 -e 15 -r 5 -s 10 -c 50
python main.py -t 55 -v 13 -e 0 -r 5 -s 10 -c 50
python main.py -t 55 -v 13 -e 5 -r 5 -s 10 -c 50
python main.py -t 55 -v 13 -e 10 -r 5 -s 10 -c 50
python main.py -t 55 -v 13 -e 15 -r 5 -s 10 -c 50
# merge results
python main.py -t 66 -v 13 -e 0 -r 5 -s 10 -c 50
python main.py -t 66 -v 13 -e 5 -r 5 -s 10 -c 50
python main.py -t 66 -v 13 -e 10 -r 5 -s 10 -c 50
python main.py -t 66 -v 13 -e 15 -r 5 -s 10 -c 50
# plot RAvel comparison between Bonferroni and Rademacher by increasing number of variables
python main.py -t 77 -v 13 -e 15 -r 5 -s 10 
# plot RAvel comparison between Bonferroni, Rademacher with std statistic and Rademacher with our proposed stat
python main.py -t 777 -v 13 -e 15 -r 5 -s 10 


# Empirical test
python main.py -t 8