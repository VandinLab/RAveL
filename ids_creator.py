# creates a dataset of given size
import numpy as np
from numpy.lib import math
import pandas as pd
import argparse
import os
from tqdm import tqdm
import math


test_type = 13
n_rep = 5
test_numb = 100
ds_sizes = [100,250,500,1000,5000,10000,25000,50000,100000,250000]
# test_numb = 10
# ds_sizes = [100,250]

path = os.path.join(os.getcwd(), "datasets","multiple"+str(test_type))

for n_test in tqdm(range(test_numb)):
    for ext_vars in [0,5,10,15]:
        for tot_size in ds_sizes:
            if not os.path.exists(os.path.join(path,"v"+str(test_type)+"_S"+str(tot_size)+"_t"+str(n_test)+"_r"+str(n_rep)+"_e"+str(ext_vars)+".csv")):
                # creation of the training set
                a = []
                b = []
                c = []
                e = []

                a.append(np.random.normal(scale=1.0, size=(tot_size, 1)))
                b.append(np.random.normal(scale=1.0, size=(tot_size, 1)))
                c.append((-3 + 2 * a[-1] + 3 * b[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))/math.sqrt(14))
                for iii in range(1,n_rep):
                    a.append(3 + 3 * c[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))
                    b.append(-1 - 2 * c[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))
                    c.append((-3 + 2 * a[-1] + 3 * b[-1] + np.random.normal(scale=1.0, size=(tot_size, 1)))/math.sqrt(14))
                for iii in range(ext_vars):
                    e.append(np.random.normal(scale=1.0, size=(tot_size, 1)))

                df = pd.DataFrame()
                A_s = ["A"+str(i+1) for i in range(n_rep)]
                df_A = pd.DataFrame(np.reshape(np.array(a),(n_rep,-1)).T,columns = A_s)
                B_s = ["B"+str(i+1) for i in range(n_rep)]
                df_B = pd.DataFrame(np.reshape(np.array(b),(n_rep,-1)).T,columns = B_s)
                C_s = ["C"+str(i+1) for i in range(n_rep)]
                df_C = pd.DataFrame(np.reshape(np.array(c),(n_rep,-1)).T,columns = C_s)
                if ext_vars!=0:
                    E_s = ["E"+str(i+1) for i in range(ext_vars)]
                    df_E = pd.DataFrame(np.reshape(np.array(e),(ext_vars,-1)).T,columns = E_s)
                    df = pd.concat([df_A,df_B,df_C,df_E], axis = 1)
                else:
                    df = pd.concat([df_A,df_B,df_C], axis = 1)
                df.to_csv(os.path.join(path,"v"+str(test_type)+"_S"+str(tot_size)+"_t"+str(n_test)+"_r"+str(n_rep)+"_e"+str(ext_vars)+".csv"), index=None)