# creates a dataset of given size
import numpy as np
from numpy.lib import math
import pandas as pd
import argparse
import os
from tqdm import tqdm
import shutil
import math


parser = argparse.ArgumentParser('main')
parser.add_argument('-a', '--train_size', type=int, default= 10000)
parser.add_argument('-b', '--valid_size', type=int, default= 100000)
parser.add_argument('-n', '--number_rep_structure', type=int, default= 1)
parser.add_argument('-t', '--test_type', type=int, default= 0)
parser.add_argument('-v', '--variables', type=int, default= 0)
parser.add_argument('-x', '--is_test', type=bool, default= False)
parser.add_argument('-z', '--zero_init_t', type=bool, default= False)

args = parser.parse_args()

train_size = args.train_size
valid_size = args.valid_size
number_rep_structure = args.number_rep_structure
n_vars = args.variables
test_type = args.test_type
is_test = args.is_test
zero_init_t = args.zero_init_t

print("Parameters =",args)


test_type = 10
path = os.path.join(os.getcwd(), "datasets","multiple"+str(test_type))

n_rep = 4

for n_test in tqdm(range(100)):
    for tot_size in [100,250,500,1000,5000,10000,25000,50000,100000,250000,500000]:
        if not os.path.exists(os.path.join(path,"v"+str(test_type)+"_S"+str(tot_size)+"_t"+str(n_test)+"_r"+str(n_rep)+".csv")):
            # creation of the training set
            a = []
            b = []
            c = []
            t = []

            a = np.random.normal(scale=1.0, size=(tot_size, 1))
            b.append(3 + 3 * a + np.random.normal(scale=1.0, size=(tot_size, 1)))
            c.append(-1 - 2 * a + np.random.normal(scale=1.0, size=(tot_size, 1)))
            t.append((-3 + 2 * b[-1] + 3 * c[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))/math.sqrt(14))
            for iii in range(1,n_rep):
                b.append(3 + 3 * t[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))
                c.append(-1 - 2 * t[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))
                t.append((-3 + 2 * b[-1] + 3 * c[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))/math.sqrt(14))

            df = pd.DataFrame()
            A_s = ["A0" for i in range(1)]
            df_A = pd.DataFrame(a,columns = A_s)
            B_s = ["B"+str(i+1) for i in range(n_rep)]
            df_B = pd.DataFrame(np.reshape(np.array(b),(n_rep,-1)).T,columns = B_s)
            C_s = ["C"+str(i+1) for i in range(n_rep)]
            df_C = pd.DataFrame(np.reshape(np.array(c),(n_rep,-1)).T,columns = C_s)
            T_s = ["A"+str(i+1) for i in range(n_rep)]  # in the original version, the DAG was composed by A0,Bs,Cs,Ts but then A0 and Ts have been named as As 
            df_T = pd.DataFrame(np.reshape(np.array(t),(n_rep,-1)).T,columns = T_s)
            
            df = pd.concat([df_A,df_B,df_C,df_T], axis = 1)
            df.to_csv(os.path.join(path,"v"+str(test_type)+"_S"+str(tot_size)+"_t"+str(n_test)+"_r"+str(n_rep)+".csv"), index=None)

for n_test in tqdm(range(5)):
    for tot_size in [100000,250000,500000]:
        if not os.path.exists(os.path.join(path,"v"+str(test_type)+"_S"+str(tot_size)+"_t"+str(n_test)+"_r"+str(n_rep)+".csv")):
            # creation of the training set
            a = []
            b = []
            c = []
            t = []

            a = np.random.normal(scale=1.0, size=(tot_size, 1))
            b.append(3 + 3 * a + np.random.normal(scale=1.0, size=(tot_size, 1)))
            c.append(-1 - 2 * a + np.random.normal(scale=1.0, size=(tot_size, 1)))
            t.append((-3 + 2 * b[-1] + 3 * c[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))/math.sqrt(14))
            for iii in range(1,n_rep):
                b.append(3 + 3 * t[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))
                c.append(-1 - 2 * t[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))
                t.append((-3 + 2 * b[-1] + 3 * c[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))/math.sqrt(14))

            df = pd.DataFrame()
            A_s = ["A0" for i in range(1)]
            df_A = pd.DataFrame(a,columns = A_s)
            B_s = ["B"+str(i+1) for i in range(n_rep)]
            df_B = pd.DataFrame(np.reshape(np.array(b),(n_rep,-1)).T,columns = B_s)
            C_s = ["C"+str(i+1) for i in range(n_rep)]
            df_C = pd.DataFrame(np.reshape(np.array(c),(n_rep,-1)).T,columns = C_s)
            T_s = ["A"+str(i+1) for i in range(n_rep)]  # in the original version, the DAG was composed by A0,Bs,Cs,Ts but then A0 and Ts have been named as As
            df_T = pd.DataFrame(np.reshape(np.array(t),(n_rep,-1)).T,columns = T_s)
            
            df = pd.concat([df_A,df_B,df_C,df_T], axis = 1)
            df.to_csv(os.path.join(path,"v"+str(test_type)+"_S"+str(tot_size)+"_t"+str(n_test)+"_r"+str(n_rep)+".csv"), index=None)
