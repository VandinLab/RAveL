from dependency_infos import dependencyInfos
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import itertools
import argparse
from datetime import datetime
from threadpoolctl import threadpool_limits
import IT_utils
import pickle
import rad_utils
from lcd_methods import *
from pgmpy.base import DAG


# python main.py -t 2 -v 13 -e 0 -r 5 -s 5 -c 50

def init_res_dicts(versions, deltas_tot, test_t, ds_sizes):
    pcds = dict([(v,{}) for v in versions])
    pcs = dict([(v,{}) for v in versions])
    mbs = dict([(v,{}) for v in versions])
    for v in versions:
        pcds[v] =  dict([(d,{}) for d in deltas_tot])
        pcs[v] =  dict([(d,{}) for d in deltas_tot])
        mbs[v] =  dict([(d,{}) for d in deltas_tot])
    for v in versions:
        for d in deltas_tot:
            pcds[v][d] =  dict([(ty,{}) for ty in test_t])
            pcs[v][d] =  dict([(ty,{}) for ty in test_t])
            mbs[v][d] =  dict([(ty,{}) for ty in test_t])
    for v in versions:
        for d in deltas_tot:
            for size in ds_sizes:
                for ty in test_t:
                    pcds[v][d][ty][size]   = {}
                    pcs[v][d][ty][size]    = {}
                    mbs[v][d][ty][size]    = {}
    return pcds, pcs, mbs

def save_current_exp(versions, deltas_tot, test_t, ds_sizes, var_names, file_folder, n_rep, stop_iteration, n_ext, method_name, pcds, pcs, mbs):
    test_numbers = [i for i in range(stop_iteration)]
    for ver in versions:
        for test_type in test_t:
            #result resume creation
            for d in deltas_tot:
                file_name = "res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)
                if n_ext >= 0:
                    file_name += "_e"+str(n_ext)
                if d < 0.0001:
                    file_name +="_delta"+str(d)[:3]+str(d)[-4:]+"_"+method_name+".csv"
                else:
                    file_name +="_delta"+str(d)+"_"+method_name+".csv"
                complete_name = os.path.join(file_folder, file_name)
                cc = [[str(v)+"_t"+str(t) for t in ts] for v in var_names]
                cc = ["type"] + [item for sublist in cc for item in sublist]
                data = []
                for s in ds_sizes:
                    row = []
                    row.append("PCD_"+str(s))
                    dd = pcds.get(ver).get(d).get(test_type).get(s,{})
                    for v in var_names:
                        for t in test_numbers:
                            dt = dd.get(t,{})
                            row.append(dt.get(v,"-"))
                    data.append(row)
                    row = []
                    row.append("PC_"+str(s))
                    dd = pcs.get(ver).get(d).get(test_type).get(s,{})
                    for v in var_names:
                        for t in test_numbers:
                            dt = dd.get(t,{})
                            row.append(dt.get(v,"-"))
                    data.append(row)
                    row = []
                    row.append("MB_"+str(s))
                    dd = mbs.get(ver).get(d).get(test_type).get(s,{})
                    # if s not in mbs.get(ver).get(d).get(test_type):
                    #     print(s,"not in ",mbs.get(ver).get(d).get(test_type).keys())
                    # else:
                    #     print("found", mbs.get(ver).get(d).get(test_type).get(s))
                    #     print("tot", mbs.get(ver).get(d).get(test_type))
                    for v in vars:
                        for t in test_numbers:   
                            dt = dd.get(t,{})
                            # if t not in dd:
                            #     print(t, "not in ", dd.keys())
                            #     input()
                            row.append(dt.get(v,"-"))
                    data.append(row)
                res_df = pd.DataFrame(data,columns=cc)
                res_df.to_csv(complete_name)

def merge_saved_results(versions, deltas_tot, test_t, file_folder, n_rep, stop_iteration, n_ext, method_name, exp_version):
    for ver in versions:
        for tt in test_t:
                
            in_files = []
            for d in deltas_tot:

                file_name = "res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)
                if n_ext >= 0:
                    file_name += "_e"+str(n_ext)
                if d < 0.0001:
                    file_name +="_delta"+str(d)[:3]+str(d)[-4:]+"_"+method_name+".csv"
                else:
                    file_name +="_delta"+str(d)+"_"+method_name+".csv"
                complete_name = os.path.join(file_folder, file_name)

                in_files.append(pd.read_csv(complete_name,sep = ",") )
            for i in range(len(in_files)):
                in_files[i]["key"] = [j+0.1*i for j in range(in_files[i].shape[0])]
                in_files[i]["delta"] = [deltas_tot[i] for j in range(in_files[i].shape[0])]

            out_df = pd.DataFrame(columns=in_files[0].columns)
            for f in in_files:
                out_df=out_df.append(f)

            cols = list(out_df.columns)
            cols = cols[-2:] + cols[:-2]
            out_df = out_df[cols]
            out_df = out_df.sort_values("key")
            file_name = get_merged_file_name(stop_iteration,exp_version,n_rep,n_ext,ver,tt,method_name)
            out_df.to_csv(os.path.join("IT_results","total_res",file_name))

def get_merged_file_name(stop_iteration,exp_version,n_rep,n_ext,ver,tt,method_name):
    file_name = "tot_on"+str(stop_iteration)+"_v"+str(exp_version)+"_r"+str(n_rep)
    if n_ext >= 0:
        file_name += "_e"+str(n_ext)
    file_name += "_v"+str(ver)+"_"+tt+"_"+method_name+".csv"
    return file_name

# np.random.seed(120395)
parser = argparse.ArgumentParser('main') 
parser.add_argument('-c', '--n_cores', type=int, default= 72)
parser.add_argument('-e', '--ext_vars', type=int, default= -1)
parser.add_argument('-r', '--n_repetitions', type=int, default= 100)
parser.add_argument('-s', '--stop_iteration', type=int, default= -1)
parser.add_argument('-t', '--test_number', type=int, default= 0)
parser.add_argument('-v', '--exp_version', type=int, default= -1)

args = parser.parse_args()
 
n_cores = args.n_cores
ext_vars = args.ext_vars
stop_iteration = args.stop_iteration
n_repetitions = args.n_repetitions
test_number = args.test_number
exp_version = args.exp_version

print("Parameters =",args)

n_rep = n_repetitions
if exp_version == 13:
    n_extra = ext_vars
    vars = ["A"+str(i+1) for i in range(n_rep)] + ["B"+str(i+1) for i in range(n_rep)] + ["C"+str(i+1) for i in range(n_rep)] + ["E"+str(i+1) for i in range(n_extra)]
    sv = len(vars)
    N = sv*(sv-1)*(2**(sv-3))  # diffrently from the notes, T is in vars

    d_struct = "[A1][B1][C1|A1:B1]" + "".join(["[A"+str(i+1)+"|C"+str(i)+"][B"+str(i+1)+"|C"+str(i)+"][C"+str(i+1)+"|A"+str(i+1)+":B"+str(i+1)+"]" for i in range(1,n_rep)]) + "".join(["[E"+str(i+1)+"]" for i in range(n_extra)])
    dag = DAG()
    dag.add_nodes_from(vars)
    dag.add_edges_from([("A"+str(i),"C"+str(i)) for i in range(1,n_rep+1)])
    dag.add_edges_from([("B"+str(i),"C"+str(i)) for i in range(1,n_rep+1)])
    dag.add_edges_from([("C"+str(i),"A"+str(i+1)) for i in range(1,n_rep)])
    dag.add_edges_from([("C"+str(i),"B"+str(i+1)) for i in range(1,n_rep)])


ds_sizes = [100,250,500,1000,5000,10000,25000,50000,100000,250000]
ds_sizes = [100,250]  # for debugging
versions = [1]
ts = [i for i in range(stop_iteration)]


# TEST 1: test PCMB with and without Bonferroni
if test_number == 1:
    def perform_test(*args):
        # Given a set of parameters, it loads the corrispondent dictionary and then it returns the informativeness
        # stats for each variable
        params = args[0]
        test_type = params[0]
        size = params[1]
        test_n = params[2]
        vars = params[3]
        delta = params[4]
        N = params[5]
        version = params[6]
        n_rep = params[7]
        n_extra = params[8]
        exp_v = params[9]

        gen_fn = "v"+str(exp_v)+"_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_e"+str(n_extra)
        data_file = os.path.join("datasets","multiple"+str(exp_v),gen_fn+".csv")

        indep_file = os.path.join("IT_results","multiple"+str(exp_v),test_type+"_"+gen_fn+".csv") 
               
        if os.path.exists(indep_file):
            try:
                indep_df = pd.read_csv(indep_file,sep = ";")
            except:
                print("!!!",indep_file)
                indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
                indep_df.to_csv(indep_file, sep = ";")
        else:
            indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
            indep_df.to_csv(indep_file, sep = ";")
        dep_infos = dependencyInfos(indep_df, method="p-value", independence_method=test_type, data_file=data_file, 
                        indep_file=indep_file, save_every_update=False, independence_language="python")
        dep_infos.look_dsep = {}
        dep_infos.n_rep = n_rep


        # perform experiment
        pcds = {}
        pcs = {}
        mbs = {}
        if version == 1:
            for v in vars:  
                pcds[v] = get_pcd(dep_infos,v,vars,delta/N)
                pcs[v]  = get_pc(dep_infos,v,vars,delta/N)
                mbs[v]  = pcmb(dep_infos,v,vars,delta/N)

        IT_utils.save_IT_dataframe(dep_infos)
        return test_n, delta/N, test_type, pcds, pcs, mbs, version
    
    deltas = [0.05]
    deltas_tot = [0.05, 0.05/N]
        
    test_t = ["cor"]
    tot_els = len(list(itertools.product(test_t, ts, [vars],deltas,[1,N],versions)))
    
    pcds, pcs, mbs = init_res_dicts(versions, deltas_tot, test_t, ds_sizes)
        
    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j in list(itertools.product(test_t, [size], ts, [vars],deltas,[1,N],versions, [n_rep],[n_extra],[exp_version])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, pcd,pc,mb,version in sorted(results,key= lambda x: x[0]):
                pcds[version][d][test_type][size][test_n] = pcd
                pcs[version][d][test_type][size][test_n] = pc
                mbs[version][d][test_type][size][test_n] = mb

            save_current_exp(versions, deltas_tot, test_t, ds_sizes, var_names = vars, file_folder = "IT_results/multiple13", n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "PCMB", pcds = pcds, pcs = pcs, mbs=mbs)
            merge_saved_results(versions, deltas_tot, test_t, file_folder = "IT_results/multiple13", n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "PCMB", exp_version = exp_version)
                     
# TEST 2: test IAMB with and without Bonferroni
if test_number == 2:
    def perform_test(*args):
        # Given a set of parameters, it loads the corrispondent dictionary and then it returns the informativeness
        # stats for each variable
        params = args[0]
        test_type = params[0]
        size = params[1]
        test_n = params[2]
        vars = params[3]
        delta = params[4]
        N = params[5]
        version = params[6]
        n_rep = params[7]
        n_extra = params[8]
        exp_v = params[9]

        gen_fn = "v"+str(exp_v)+"_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_e"+str(n_extra)
        data_file = os.path.join("datasets","multiple"+str(exp_v),gen_fn+".csv")

        indep_file = os.path.join("IT_results","multiple"+str(exp_v),test_type+"_"+gen_fn+".csv") 
               
        if os.path.exists(indep_file):
            try:
                indep_df = pd.read_csv(indep_file,sep = ";")
            except:
                print("!!!",indep_file)
                indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
                indep_df.to_csv(indep_file, sep = ";")
        else:
            indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
            indep_df.to_csv(indep_file, sep = ";")
        dep_infos = dependencyInfos(indep_df, method="p-value", independence_method=test_type, data_file=data_file, 
                        indep_file=indep_file, save_every_update=False, independence_language="python")
        dep_infos.look_dsep = {}
        dep_infos.n_rep = n_rep


        # perform experiment
        pcds = {}
        pcs = {}
        mbs = {}
        for v in vars:  
            mbs[v]  = iamb(dep_infos,v,vars,delta/N)

        IT_utils.save_IT_dataframe(dep_infos)
        return test_n, delta/N, test_type, mbs,  version

    deltas = [0.05]
    deltas_tot = [0.05, 0.05/N]
    test_t = ["cor"]
    tot_els = 2*len(list(itertools.product(test_t, ts, [vars],deltas,[1],versions)))
    
    pcds, pcs, mbs = init_res_dicts(versions, deltas_tot, test_t, ds_sizes)        

    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j in list(itertools.product(test_t, [size], ts, [vars],deltas,[1,N],versions, [n_rep],[n_extra],[exp_version])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, mb,version in sorted(results,key= lambda x: x[0]):
                mbs[version][d][test_type][size][test_n] = mb
            
            save_current_exp(versions, deltas_tot, test_t, ds_sizes, var_names = vars, file_folder = "IT_results/multiple13", n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "IAMB", pcds = pcds, pcs = pcs, mbs=mbs)
            merge_saved_results(versions, deltas_tot, test_t, file_folder = "IT_results/multiple13", n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "IAMB", exp_version = exp_version)

# TEST 3: test RAveL-PC and RAveL-MB with Bonferroni
if test_number == 3:
    def perform_test(*args):
        # Given a set of parameters, it loads the corrispondent dictionary and then it returns the informativeness
        # stats for each variable
        params = args[0]
        test_type = params[0]
        size = params[1]
        test_n = params[2]
        vars = params[3]
        delta = params[4]
        N = params[5]
        version = params[6]
        n_rep = params[7]
        n_extra = params[8]
        exp_v = params[9]

        gen_fn = "v"+str(exp_v)+"_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_e"+str(n_extra)
        data_file = os.path.join("datasets","multiple"+str(exp_v),gen_fn+".csv")

        indep_file = os.path.join("IT_results","multiple"+str(exp_v),test_type+"_"+gen_fn+".csv") 
               
        if os.path.exists(indep_file):
            try:
                indep_df = pd.read_csv(indep_file,sep = ";")
            except:
                print("!!!",indep_file)
                indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
                indep_df.to_csv(indep_file, sep = ";")
        else:
            indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
            indep_df.to_csv(indep_file, sep = ";")
        dep_infos = dependencyInfos(indep_df, method="p-value", independence_method=test_type, data_file=data_file, 
                        indep_file=indep_file, save_every_update=False, independence_language="python")
        dep_infos.look_dsep = {}
        dep_infos.n_rep = n_rep


        # perform experiment
        pcds = {}
        pcs = {}
        mbs = {}
        for v in vars:  
            pcs[v]  = RAveL_PC(dep_infos,v,vars,delta,N, max_z_size = 2)
            mbs[v]  = RAveL_MB(dep_infos,v,vars,delta,N, max_z_size = 2)

        IT_utils.save_IT_dataframe(dep_infos)
        return test_n, delta/N, test_type, pcds, pcs, mbs, version
    
    deltas = [0.05]
    deltas_tot = [0.05/N]
        
    test_t = ["cor"]
    tot_els = len(list(itertools.product(test_t, ts, [vars],deltas,[N],versions)))
    
    pcds, pcs, mbs = init_res_dicts(versions, deltas_tot, test_t, ds_sizes)
        
    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j in list(itertools.product(test_t, [size], ts, [vars],deltas,[N],versions, [n_rep],[n_extra],[exp_version])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, pcd,pc,mb,version in sorted(results,key= lambda x: x[0]):
                pcds[version][d][test_type][size][test_n] = pcd
                pcs[version][d][test_type][size][test_n] = pc
                mbs[version][d][test_type][size][test_n] = mb
            
            save_current_exp(versions, deltas_tot, test_t, ds_sizes, var_names = vars, file_folder = "IT_results/multiple13", n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "RAveL_Bonferroni", pcds = pcds, pcs = pcs, mbs=mbs)
            merge_saved_results(versions, deltas_tot, test_t, file_folder = "IT_results/multiple13", n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "RAveL_Bonferroni", exp_version = exp_version)

# TEST 4: Rademacher indep test calculator
if test_number == 4:
    for n_rep in [n_repetitions]:
        n_extra = ext_vars
        couples_dict = rad_utils.select_candidates(vars)

        for t in range(stop_iteration):
            for s in ds_sizes:
                filename = "v"+str(exp_version)+"_S"+str(s)+"_t"+str(t)+"_r"+str(n_rep)+"_e"+str(n_extra)+".csv"
                prefix = rad_utils.get_file_prefixes(computation_type= "total", corrected_c = True, normalize_using_maxes= True, my_stat=False)
                if not os.path.exists("IT_results/multiple"+str(exp_version)+"/"+prefix+filename):
                    r_vals, bound, tot_couples, tot_triples = rad_utils.calculate_stat_SD_bounds("datasets/multiple"+str(exp_version)+"/"+filename, vars, delta = 0.05, couples_dict= couples_dict, n_proc = n_cores, corrected_bound = True, corrected_c = True, normalize_using_maxes= True, max_z_size = 2)
                    rad_utils.write_on_csv("IT_results/multiple"+str(exp_version)+"/"+prefix+filename,r_vals, bound[0], tot_couples, tot_triples)

# TEST 44: MY_STAT Rademacher indep test calculator
if test_number == 44:
    for n_rep in [n_repetitions]:
        n_extra = ext_vars
        couples_dict = rad_utils.select_candidates(vars)

        for t in range(stop_iteration):
            for s in ds_sizes:
                filename = "v"+str(exp_version)+"_S"+str(s)+"_t"+str(t)+"_r"+str(n_rep)+"_e"+str(n_extra)+".csv"                
                prefix = rad_utils.get_file_prefixes(computation_type= "total", corrected_c = True, normalize_using_maxes= True, my_stat=True)
                if not os.path.exists("IT_results/multiple"+str(exp_version)+"/"+prefix+filename):
                    r_vals, bound, tot_couples, tot_triples = rad_utils.calculate_stat_SD_bounds("datasets/multiple"+str(exp_version)+"/"+filename, vars, delta = 0.05, couples_dict= couples_dict, n_proc = n_cores, corrected_bound = True, my_stat=True, corrected_c = False, normalize_using_maxes= True, max_z_size = 2)  # Corrected c is removed since I'm using my_stat variant
                    rad_utils.write_on_csv("IT_results/multiple"+str(exp_version)+"/"+prefix+filename,r_vals, bound[0], tot_couples, tot_triples)


# TEST 5: test RAveL-PC and RAveL-MB with Rademacher
if test_number == 5:
    def perform_test(*args):
        # Given a set of parameters, it loads the corrispondent dictionary and then it returns the informativeness
        # stats for each variable
        params = args[0]
        test_type = params[0]
        size = params[1]
        test_n = params[2]
        vars = params[3]
        delta = params[4]
        N = params[5]
        version = params[6]
        n_rep = params[7]
        n_extra = params[8]
        exp_v = params[9]
        
        filename = "v"+str(exp_v)+"_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_e"+str(n_extra)
        data_file = os.path.join("datasets","multiple"+str(exp_v),filename+".csv")
        prefix = rad_utils.get_file_prefixes(computation_type= "total", corrected_c = True, normalize_using_maxes= True, my_stat=False)
        indep_file = os.path.join("IT_results","multiple"+str(exp_v),prefix+filename+".csv")        
        
        if os.path.exists(indep_file):
            indep_df = pd.read_csv(indep_file,sep = ";")
            if len(indep_df.columns.values) < 2:
                indep_df = pd.read_csv(indep_file)

        dep_infos = dependencyInfos(indep_df, method="Rademacher", independence_method=test_type, data_file=data_file, 
                        indep_file=indep_file, save_every_update=False, independence_language="-")        
        pcds = {}
        pcs = {}
        mbs = {}
        for v in vars:  
            pcs[v]  = RAveL_PC(dep_infos,v,vars,delta,N, max_z_size = 2)
            mbs[v]  = RAveL_MB(dep_infos,v,vars,delta,N, max_z_size = 2)
        return test_n, delta/N, test_type, pcds, pcs, mbs, version
    
    N = 1 ##### no need of correction for MHT in Rademacher examples since Rademacher calculus already takes care of MHT
    deltas = [0.05]
    deltas_tot = [0.05]
        
    test_t = ["Rademacher"]
    tot_els = 2*len(list(itertools.product(test_t, ts, [vars],deltas,[1],versions)))
    
    pcds, pcs, mbs = init_res_dicts(versions, deltas_tot, test_t, ds_sizes)
        
    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j in list(itertools.product(test_t, [size], ts, [vars],deltas,[N],versions, [n_rep],[n_extra],[exp_version])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, pcd,pc,mb,version in sorted(results,key= lambda x: x[0]):
                pcds[version][d][test_type][size][test_n] = pcd
                pcs[version][d][test_type][size][test_n] = pc
                mbs[version][d][test_type][size][test_n] = mb

            save_current_exp(versions, deltas_tot, test_t, ds_sizes, var_names = vars, file_folder = "IT_results/multiple"+str(exp_version), n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "RAveL", pcds = pcds, pcs = pcs, mbs=mbs)
            merge_saved_results(versions, deltas_tot, test_t, file_folder = "IT_results/multiple"+str(exp_version), n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "RAveL", exp_version = exp_version)

# TEST 55: test RAveL-PC and RAveL-MB with Rademacher WITH MY STAT
if test_number == 55:
    def perform_test(*args):
        # Given a set of parameters, it loads the corrispondent dictionary and then it returns the informativeness
        # stats for each variable
        params = args[0]
        test_type = params[0]
        size = params[1]
        test_n = params[2]
        vars = params[3]
        delta = params[4]
        N = params[5]
        version = params[6]
        n_rep = params[7]
        n_extra = params[8]
        exp_v = params[9]
        
        filename = "v"+str(exp_v)+"_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_e"+str(n_extra)
        data_file = os.path.join("datasets","multiple"+str(exp_v),filename+".csv")
        prefix = rad_utils.get_file_prefixes(computation_type= "total", corrected_c = True, normalize_using_maxes= True, my_stat=True)
        indep_file = os.path.join("IT_results","multiple"+str(exp_v),prefix+filename+".csv")       
        
        if os.path.exists(indep_file):
            indep_df = pd.read_csv(indep_file,sep = ";")
            if len(indep_df.columns.values) < 2:
                indep_df = pd.read_csv(indep_file)

        dep_infos = dependencyInfos(indep_df, method="Rademacher", independence_method=test_type, data_file=data_file, 
                        indep_file=indep_file, save_every_update=False, independence_language="-")        
        pcds = {}
        pcs = {}
        mbs = {}
        for v in vars:  
            pcs[v]  = RAveL_PC(dep_infos,v,vars,delta,N, max_z_size = 2)
            mbs[v]  = RAveL_MB(dep_infos,v,vars,delta,N, max_z_size = 2)
        return test_n, delta/N, test_type, pcds, pcs, mbs, version
    
    deltas = [0.05]
    deltas_tot = [0.05]
    N = 1 ##### no need of correction for MHT in Rademacher examples since Rademacher calculus already takes care of MHT

    test_t = ["my_stat"]
    tot_els = 2*len(list(itertools.product(test_t, ts, [vars],deltas,[1],versions)))
    
    pcds, pcs, mbs = init_res_dicts(versions, deltas_tot, test_t, ds_sizes)

    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j in list(itertools.product(test_t, [size], ts, [vars],deltas,[N],versions, [n_rep],[n_extra],[exp_version])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, pcd,pc,mb,version in sorted(results,key= lambda x: x[0]):
                pcds[version][d][test_type][size][test_n] = pcd
                pcs[version][d][test_type][size][test_n] = pc
                mbs[version][d][test_type][size][test_n] = mb

            save_current_exp(versions, deltas_tot, test_t, ds_sizes, var_names = vars, file_folder = "IT_results/multiple"+str(exp_version), n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "RAveL", pcds = pcds, pcs = pcs, mbs=mbs)
            merge_saved_results(versions, deltas_tot, test_t, file_folder = "IT_results/multiple"+str(exp_version), n_rep=n_rep, stop_iteration=stop_iteration, n_ext = ext_vars, method_name = "RAveL", exp_version = exp_version)


# TEST 6: result analysis on synthetic data for RAveL vs SoA methods
if test_number == 6:
    n_exps = stop_iteration

    deltas = [0.05]
    deltas_tot = [0.05, 0.05/N]
    res_PCMB = (vars, ds_sizes, [1],deltas_tot,"PCMB",dag,"PCMB", "cor")

    res_IAMB = (vars, ds_sizes, [1],deltas_tot,"IAMB",dag,"IAMB", "cor")

    deltas = [0.05]
    deltas_tot = [0.05/N]
    
    res_RAveL_Bon = (vars, ds_sizes, [1],deltas_tot,"RAveL_Bon",dag,"RAveL_Bonferroni","cor")

    deltas = [0.05]
    deltas_tot = [0.05]
    res_RAveL_std = (vars, ds_sizes, [1],deltas_tot,"RAveL_std",dag,"RAveL","Rademacher")

    counter_pc_per_test = dict()
    counter_PC_name = 'counter_pc_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(n_exps)+'.pickle'
    with open(os.path.join('IT_results',counter_PC_name), 'wb') as handle:
        pickle.dump(counter_pc_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counter_mb_per_test = dict()
    counter_MB_name = 'counter_mb_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(n_exps)+'.pickle'
    with open(os.path.join('IT_results',counter_MB_name), 'wb') as handle:
        pickle.dump(counter_mb_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counter_pc_per_test = dict()
    counter_mb_per_test = dict()

    parameter_list = [res_PCMB, res_IAMB, res_RAveL_Bon, res_RAveL_std]
    for pl in parameter_list:
        vars_selected = pl[0]
        sizes_selected = pl[1]
        versions_selected = pl[2]
        deltas_selected = pl[3]
        method_id = pl[4]
        dag_struc = pl[5]
        method_name = pl[6]
        test_type = pl[7]
        
        for ver in versions_selected:
            f_name_selected = os.path.join("IT_results","total_res",get_merged_file_name(stop_iteration,exp_version,n_rep,ext_vars,ver,test_type,method_name))

            in_df = pd.read_csv(f_name_selected)
            dic_key = method_id + "_version_" + str(ver)
            counter_pc_per_test[dic_key] = dict()
            counter_mb_per_test[dic_key] = dict()
            counter_pc_per_test[dic_key]["FP"]   = dict([(d,{}) for d in deltas_selected])
            counter_mb_per_test[dic_key]["FP"] = dict([(d,{}) for d in deltas_selected])
            counter_pc_per_test[dic_key]["FN"]   = dict([(d,{}) for d in deltas_selected])
            counter_mb_per_test[dic_key]["FN"] = dict([(d,{}) for d in deltas_selected])

            cc = [[str(v)+"_t"+str(t) for t in range(n_exps)] for v in vars_selected]
            cc = [item for sublist in cc for item in sublist]
            
            for d in deltas_selected:
                
                in_file = in_df[in_df['delta'] == d]
                if in_file.shape[0] == 0:
                    for dd in set(in_df['delta']):
                        if np.abs(dd-d) < d*1e-7:
                            in_file = in_df[in_df['delta'] == dd]

                for size in sizes_selected:
                    df_pc = in_file[in_file["type"]=="PC_"+str(size)]
                    df_pc = df_pc[cc].values
                    df_pc = np.reshape(df_pc,(-1,n_exps))
                    df_mb = in_file[in_file["type"]=="MB_"+str(size)]
                    df_mb = df_mb[cc].values
                    df_mb = np.reshape(df_mb,(-1,n_exps))
                    look_pc = {}
                    look_mb = {}
                    look_el2 = {}
                    
                    for i in range(len(vars_selected)):
                        v = vars_selected[i]
                        c_pcd = 0
                        c_pc = 0
                        pc = IT_utils.get_pc(v,dag_struc)
                        look_pc[v] = pc
                        c_mb = 0
                        mb = IT_utils.get_mb(v,dag_struc)
                        look_mb[v] = mb

                    counter_pc_per_test[dic_key]["FP"][d][size] = []
                    counter_mb_per_test[dic_key]["FP"][d][size] = []
                    counter_pc_per_test[dic_key]["FN"][d][size] = []
                    counter_mb_per_test[dic_key]["FN"][d][size] = []
                    
                    
                    for j in range(n_exps):
                        c_pc = -1
                        c_mb = -1
                        if df_pc.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_pc[i,j] != "-":
                                c_pc = np.sum([not element in look_pc[vars_selected[i]] for i in range(len(vars_selected)) for element in eval(df_pc[i,j])])
                        if df_mb.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_mb[i,j] != "-":
                                c_mb = np.sum([not element in look_mb[vars_selected[i]] for i in range(len(vars_selected)) for element in eval(df_mb[i,j])])
                        counter_pc_per_test[dic_key]["FP"][d][size].append(c_pc)
                        counter_mb_per_test[dic_key]["FP"][d][size].append(c_mb)

                    for j in range(n_exps):
                        c_pc = -1
                        c_mb = -1
                        if df_pc.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_pc[i,j] != "-":
                                c_pc = np.sum([not element in eval(df_pc[i,j]) for i in range(len(vars_selected)) for element in look_pc[vars_selected[i]]])
                        if df_mb.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_mb[i,j] != "-":
                                c_mb = np.sum([not element in eval(df_mb[i,j]) for i in range(len(vars_selected)) for element in look_mb[vars_selected[i]]])
                        counter_pc_per_test[dic_key]["FN"][d][size].append(c_pc)
                        counter_mb_per_test[dic_key]["FN"][d][size].append(c_mb)
        
            with open(os.path.join('IT_results',counter_PC_name), 'wb') as handle:
                pickle.dump(counter_pc_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join('IT_results',counter_MB_name), 'wb') as handle:
                pickle.dump(counter_mb_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

# TEST 66: result analysis on synthetic data only for RAveL algos
if test_number == 66:
    n_exps = stop_iteration

    deltas = [0.05]
    deltas_tot = [0.05/N]
    
    res_RAveL_Bon = (vars, ds_sizes, [1],deltas_tot,"RAveL_Bon",dag,"RAveL_Bonferroni","cor")

    deltas = [0.05]
    deltas_tot = [0.05]
    res_RAveL_std = (vars, ds_sizes, [1],deltas_tot,"RAveL_std",dag,"RAveL","Rademacher")

    deltas = [0.05]
    deltas_tot = [0.05]
    res_RAveL_my_stat = (vars, ds_sizes, [1],deltas_tot, "RAveL_my_stat",dag,"RAveL","my_stat")

    counter_pc_per_test = dict()
    counter_PC_name = 'counter_pc_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(n_exps)+'_only_RAveL.pickle'
    with open(os.path.join('IT_results',counter_PC_name), 'wb') as handle:
        pickle.dump(counter_pc_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counter_mb_per_test = dict()
    counter_MB_name = 'counter_mb_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(n_exps)+'_only_RAveL.pickle'
    with open(os.path.join('IT_results',counter_MB_name), 'wb') as handle:
        pickle.dump(counter_mb_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counter_pc_per_test = dict()
    counter_mb_per_test = dict()

    parameter_list = [res_RAveL_Bon, res_RAveL_std, res_RAveL_my_stat]
    for pl in parameter_list:
        vars_selected = pl[0]
        sizes_selected = pl[1]
        versions_selected = pl[2]
        deltas_selected = pl[3]
        method_id = pl[4]
        dag_struc = pl[5]
        method_name = pl[6]
        test_type = pl[7]
        
        for ver in versions_selected:
            f_name_selected = os.path.join("IT_results","total_res",get_merged_file_name(stop_iteration,exp_version,n_rep,ext_vars,ver,test_type,method_name))

            in_df = pd.read_csv(f_name_selected)
            dic_key = method_id + "_version_" + str(ver)
            counter_pc_per_test[dic_key] = dict()
            counter_mb_per_test[dic_key] = dict()
            counter_pc_per_test[dic_key]["FP"]   = dict([(d,{}) for d in deltas_selected])
            counter_mb_per_test[dic_key]["FP"] = dict([(d,{}) for d in deltas_selected])
            counter_pc_per_test[dic_key]["FN"]   = dict([(d,{}) for d in deltas_selected])
            counter_mb_per_test[dic_key]["FN"] = dict([(d,{}) for d in deltas_selected])

            cc = [[str(v)+"_t"+str(t) for t in range(n_exps)] for v in vars_selected]
            cc = [item for sublist in cc for item in sublist]
            
            for d in deltas_selected:
                
                in_file = in_df[in_df['delta'] == d]
                if in_file.shape[0] == 0:
                    for dd in set(in_df['delta']):
                        if np.abs(dd-d) < d*1e-7:
                            in_file = in_df[in_df['delta'] == dd]

                for size in sizes_selected:
                    df_pc = in_file[in_file["type"]=="PC_"+str(size)]
                    df_pc = df_pc[cc].values
                    df_pc = np.reshape(df_pc,(-1,n_exps))
                    df_mb = in_file[in_file["type"]=="MB_"+str(size)]
                    df_mb = df_mb[cc].values
                    df_mb = np.reshape(df_mb,(-1,n_exps))
                    look_pc = {}
                    look_mb = {}
                    look_el2 = {}
                    
                    for i in range(len(vars_selected)):
                        v = vars_selected[i]
                        c_pcd = 0
                        c_pc = 0
                        pc = IT_utils.get_pc(v,dag_struc)
                        look_pc[v] = pc
                        c_mb = 0
                        mb = IT_utils.get_mb(v,dag_struc)
                        look_mb[v] = mb

                    counter_pc_per_test[dic_key]["FP"][d][size] = []
                    counter_mb_per_test[dic_key]["FP"][d][size] = []
                    counter_pc_per_test[dic_key]["FN"][d][size] = []
                    counter_mb_per_test[dic_key]["FN"][d][size] = []
                    
                    
                    for j in range(n_exps):
                        c_pc = -1
                        c_mb = -1
                        if df_pc.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_pc[i,j] != "-":
                                c_pc = np.sum([not element in look_pc[vars_selected[i]] for i in range(len(vars_selected)) for element in eval(df_pc[i,j])])
                        if df_mb.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_mb[i,j] != "-":
                                c_mb = np.sum([not element in look_mb[vars_selected[i]] for i in range(len(vars_selected)) for element in eval(df_mb[i,j])])

                        counter_pc_per_test[dic_key]["FP"][d][size].append(c_pc)
                        counter_mb_per_test[dic_key]["FP"][d][size].append(c_mb)

                    for j in range(n_exps):
                        c_pc = -1
                        c_mb = -1
                        if df_pc.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_pc[i,j] != "-":
                                    c_pc = np.sum([not element in eval(df_pc[i,j]) for i in range(len(vars_selected)) for element in look_pc[vars_selected[i]]])
                        if df_mb.shape[0] == len(vars_selected):  # may be that I don't have any measure for these vals
                            if df_mb[i,j] != "-":
                                    c_mb = np.sum([not element in eval(df_mb[i,j]) for i in range(len(vars_selected)) for element in look_mb[vars_selected[i]]])
                                
                        counter_pc_per_test[dic_key]["FN"][d][size].append(c_pc)
                        counter_mb_per_test[dic_key]["FN"][d][size].append(c_mb)
        
            with open(os.path.join('IT_results',counter_PC_name), 'wb') as handle:
                pickle.dump(counter_pc_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join('IT_results',counter_MB_name), 'wb') as handle:
                pickle.dump(counter_mb_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TEST 7: plots for RAveL vs SoA methods
if test_number == 7:

    c_list = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    dim = 2.25
    mms = 8

    # control FWER
    fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)

    titles = ["FWER PC","FWER MB"]

    data = [pickle.load(open(os.path.join('IT_results','counter_pc_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'.pickle'),"rb")),
            pickle.load(open(os.path.join('IT_results','counter_mb_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'.pickle'),"rb"))]
    corr_delta = [k for k in data[0]["RAveL_Bon_version_1"]["FP"].keys() if k != 0.05][0]
    for i in range(2):
        xs = sorted(data[i]["RAveL_std_version_1"]["FP"][0.05].keys())
        ys = np.mean([np.array(data[i]["PCMB_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
        if i == 0:
            ax[i].plot(xs,ys, label = "GetPC", marker = "^",ms = mms)
        else:
            ax[i].plot(xs,ys, label = " PCMB", marker = "^",ms = mms)

        ys = np.mean([np.array(data[i]["PCMB_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
        if i == 0:
            ax[i].plot(xs,ys, label = "GetPC - Bonferroni corr.", marker = "^",ms = mms)
        else:
            ax[i].plot(xs,ys, label = " PCMB - Bonferroni corr.", marker = "^",ms = mms)

        if i != 0: # IAMB is ony for MB discovery
            ys = np.mean([np.array(data[i]["IAMB_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = " IAMB", marker = "^",ms = mms)
            ys = np.mean([np.array(data[i]["IAMB_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = "IAMB - Bonferroni corr.", marker = "^",ms = mms)

        ys = np.mean([np.array(data[i]["RAveL_std_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
        ax[i].plot(xs,ys, label = "RAveL", marker = "^",ms = mms)
        ys = np.mean([np.array(data[i]["RAveL_Bon_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
        ax[i].plot(xs,ys, label = "RAveL - Bonferroni corr.", marker = "v",ms = mms)

    for i in range(2):               
        ax[i].axhline(0.05, ls='--', color='r')
        ax[i].set_ylabel(titles[i])
        ax[i].set_xlabel("Sample size")
        ax[i].set_xscale('log')
        ax[i].set_xlim(90)
        ax[i].set_ylim(-0.05,1.05)
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(xs)
        ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("pdfs/FWER_method.pdf")

    fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)
    titles = ["%FN PC","%FN MB"]

    legends = ["RAveL-PC", "RAveL-MB"]

    correction_PC = 4 + 8 * (5 -1) 
    correction_MB = 6 + 10 * (5 -1)     
    corrections = [correction_PC, correction_MB]
    for i in range(2):
        xs = sorted(data[i]["RAveL_std_version_1"]["FN"][0.05].keys())
        ys = np.mean([np.array(data[i]["RAveL_std_version_1"]["FN"][0.05][size]) for size in xs],axis = 1)/corrections[i]
        ax[i].plot(xs,ys, label = "RAveL", marker = "^",ms = mms)
        ys = np.mean([np.array(data[i]["RAveL_Bon_version_1"]["FN"][corr_delta][size]) for size in xs],axis = 1)/corrections[i]
        ax[i].plot(xs,ys, label = "RAveL - Bonferroni corr.", marker = "v",ms = mms)
    for i in range(2):   
        ax[i].set_ylabel(titles[i])
        ax[i].set_xlabel("Sample size")
        ax[i].set_xscale('log')
        ax[i].set_xlim(90)
        ax[i].set_ylim(-0.05,1.05)
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(xs)
        ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("pdfs/FN_method_with_guarantees.pdf")

# TEST 77: plots for RAveL algos by varying number of parameters
if test_number == 77:

    c_list = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    dim = 2.25
    mms = 8

    fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)
    titles = ["%FN PC","%FN MB"]

    legends = ["RAveL-PC", "RAveL-MB"]

    correction_PC = 4 + 8 * (5 -1) 
    correction_MB = 6 + 10 * (5 -1)     
    corrections = [correction_PC, correction_MB]  # corrections are the same because true positives are always the same despite the number of ext vars
    
    for e in [0,5,10,15]:
        n_extra = e
        n_vars_tot = 15+e
        data = [pickle.load(open(os.path.join('IT_results','counter_pc_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'_only_RAveL.pickle'),"rb")),
                pickle.load(open(os.path.join('IT_results','counter_mb_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'_only_RAveL.pickle'),"rb"))]
        corr_delta = [k for k in data[0]["RAveL_Bon_version_1"]["FP"].keys() if k != 0.05][0]
        for i in range(2):
            xs = sorted(data[i]["RAveL_std_version_1"]["FN"][0.05].keys())
            ys = np.mean([np.array(data[i]["RAveL_std_version_1"]["FN"][0.05][size]) for size in xs],axis = 1)/corrections[i]
            ax[i].plot(xs,ys, label = legends[i] + " - "+str(n_vars_tot)+" vars", marker = "^",ms = mms)
            ys = np.mean([np.array(data[i]["RAveL_my_stat_version_1"]["FN"][0.05][size]) for size in xs],axis = 1)/corrections[i]
            ax[i].plot(xs,ys, label = legends[i] + " - "+str(n_vars_tot)+" vars, our stat", marker = "*",ms = mms)
            ys = np.mean([np.array(data[i]["RAveL_Bon_version_1"]["FN"][corr_delta][size]) for size in xs],axis = 1)/corrections[i]
            ax[i].plot(xs,ys, label = legends[i]+" - Bonferroni corr." + " "+str(n_vars_tot)+" vars", marker = "v",ms = mms)

    for i in range(2):   
        ax[i].set_ylabel(titles[i])
        ax[i].set_xlabel("Sample size")
        ax[i].set_xscale('log')
        ax[i].set_xlim(90)
        ax[i].set_ylim(-0.05,1.05)
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(xs)
        ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("pdfs/FN_per_variables_number.pdf")

    # control FWER
    fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)

    titles = ["FWER PC","FWER MB"]

    for e in [0,5,10,15]:
        n_extra = e
        n_vars_tot = 15+e
        data = [pickle.load(open(os.path.join('IT_results','counter_pc_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'_only_RAveL.pickle'),"rb")),
                pickle.load(open(os.path.join('IT_results','counter_mb_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'_only_RAveL.pickle'),"rb"))]
        corr_delta = [k for k in data[0]["RAveL_Bon_version_1"]["FP"].keys() if k != 0.05][0]
        for i in range(2):
            xs = sorted(data[i]["RAveL_std_version_1"]["FP"][0.05].keys())
            ys = np.mean([np.array(data[i]["RAveL_std_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = legends[i] + " - "+str(n_vars_tot)+" vars", marker = "^",ms = mms)
            ys = np.mean([np.array(data[i]["RAveL_my_stat_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = legends[i] + " - "+str(n_vars_tot)+" vars, our stat", marker = "^",ms = mms)
            ys = np.mean([np.array(data[i]["RAveL_Bon_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = legends[i]+" - Bonferroni corr." + " "+str(n_vars_tot)+" vars", marker = "v",ms = mms)

    for i in range(2):   
        ax[i].axhline(0.05, ls='--', color='r')
        ax[i].set_ylabel(titles[i])
        ax[i].set_xlabel("Sample size")
        ax[i].set_xscale('log')
        ax[i].set_xlim(90)
        ax[i].set_ylim(-0.05,1.05)
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(xs)
        ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("pdfs/FWER_per_variables_number_[control].pdf")

# TEST 777: plots for RAveL algos and our statistic
if test_number == 777:

    c_list = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    dim = 2.25
    mms = 8

    fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)
    titles = ["%FN PC","%FN MB"]

    legends = ["RAveL-PC", "RAveL-MB"]

    correction_PC = 4 + 8 * (5 -1) 
    correction_MB = 6 + 10 * (5 -1)     
    corrections = [correction_PC, correction_MB]
    
    data = [pickle.load(open(os.path.join('IT_results','counter_pc_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'_only_RAveL.pickle'),"rb")), 
            pickle.load(open(os.path.join('IT_results','counter_mb_per_test_v'+str(exp_version)+'_e'+str(n_extra)+"_r"+str(n_rep)+'_s'+str(stop_iteration)+'_only_RAveL.pickle'),"rb"))] 
    corr_delta = [k for k in data[0]["RAveL_Bon_version_1"]["FP"].keys() if k != 0.05][0]
    
    for i in range(2):
        xs = sorted(data[i]["RAveL_std_version_1"]["FN"][0.05].keys())
        ys = np.mean([np.array(data[i]["RAveL_std_version_1"]["FN"][0.05][size]) for size in xs],axis = 1)/corrections[i]
        ax[i].plot(xs,ys, label = legends[i] + " - 30 vars", marker = "^", color = "tab:pink", ms = mms)
        ys = np.mean([np.array(data[i]["RAveL_my_stat_version_1"]["FN"][0.05][size]) for size in xs],axis = 1)/corrections[i]
        ax[i].plot(xs,ys, label = legends[i] + " - 30 vars, our stat", marker = "*",ms = mms)
        ys = np.mean([np.array(data[i]["RAveL_Bon_version_1"]["FN"][corr_delta][size]) for size in xs],axis = 1)/corrections[i]
        ax[i].plot(xs,ys, label = legends[i]+" - Bonferroni corr." + " 30 vars", color = "tab:gray", marker = "v",ms = mms)

    for i in range(2):   
        ax[i].set_ylabel(titles[i])
        ax[i].set_xlabel("Sample size")
        ax[i].set_xscale('log')
        ax[i].set_xlim(90)
        ax[i].set_ylim(-0.05,1.05)
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(xs)
        ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("pdfs/FN_w_our_stat.pdf")

    # control FWER
    fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)

    titles = ["FWER PC","FWER MB"]
    corr_delta = [k for k in data[0]["RAveL_Bon_version_1"]["FP"].keys() if k != 0.05][0]
    for i in range(2):
        xs = sorted(data[i]["RAveL_std_version_1"]["FP"][0.05].keys())
        ys = np.mean([np.array(data[i]["RAveL_std_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
        ax[i].plot(xs,ys, label = legends[i] + " - 30 vars", marker = "^", color = "tab:pink", ms = mms)
        ys = np.mean([np.array(data[i]["RAveL_my_stat_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
        ax[i].plot(xs,ys, label = legends[i] + " - 30 vars, our stat", marker = "^",ms = mms)
        ys = np.mean([np.array(data[i]["RAveL_Bon_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
        ax[i].plot(xs,ys, label = legends[i]+" - Bonferroni corr." + " 30 vars", color = "tab:gray", marker = "v",ms = mms)

    for i in range(2):   
        ax[i].axhline(0.05, ls='--', color='r')
        ax[i].set_ylabel(titles[i])
        ax[i].set_xlabel("Sample size")
        ax[i].set_xscale('log')
        ax[i].set_xlim(90)
        ax[i].set_ylim(-0.05,1.05)
        ax[i].set_xticks(xs)
        ax[i].set_xticklabels(xs)
        ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("pdfs/FWER_w_our_stat_[control].pdf")


# TEST 8: RADEMACHER calculus on Boston housing standard
if test_number == 8:
    input_test_number = 75

    ############ boston ds 

    # CRIM per capita crime rate by town 
    # ZN proportion of residential land zoned for lots over 25,000 sq.ft. 
    # INDUS proportion of non-retail business acres per town 
    # CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
    # NOX nitric oxides concentration (parts per 10 million) 
    # RM average number of rooms per dwelling 
    # AGE proportion of owner-occupied units built prior to 1940 
    # DIS weighted distances to five Boston employment centres 
    # RAD index of accessibility to radial highways 
    # TAX full-value property-tax rate per $10,000 
    # PTRATIO pupil-teacher ratio by town 
    # B 1000(Bk 0.63)^2 where Bk is the proportion of blacks by town 
    # LSTAT % lower status of the population 
    # MEDV Median value of owner-occupied homes in $1000's

    delta_input = 0.01
    vvs = ["CRIM" ,"ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

    df = pd.read_csv(os.path.join("datasets","real_data","housing.csv"), header=None, delimiter=r"\s+", names = vvs)
    df = df.sample(frac=1, random_state = 120395)
    df.astype("float64").to_csv(os.path.join("datasets","real_data","housing_shuffled.csv"))    

    vvs = ["CRIM" ,"ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
    t = "MEDV"
    sv = len(vvs)
    N = sv*(sv-1)*(2**(sv-3))      
    delta = 0.01

    data_file = os.path.join("datasets","real_data","housing_shuffled.csv")

    filename = data_file
    indep_file = os.path.join("IT_results","real_data","py_housing_shuffled.csv")
    
    if os.path.exists(indep_file):
        try:
            indep_df = pd.read_csv(indep_file,sep = ";")
        except:
            indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
            indep_df.to_csv(indep_file, sep = ";")
    else:
        indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
        indep_df.to_csv(indep_file, sep = ";")
    dep_infos = dependencyInfos(indep_df, method="p-value", independence_method="cor", data_file=data_file, 
                    indep_file=indep_file, save_every_update=False, independence_language="python")
    
    
    res = RAveL_MB(dep_infos, t, vvs,delta,N,max_z_size = len(vvs)-2 )

    IT_utils.save_IT_dataframe(dep_infos)
    print()
    print(str(delta)+"MB "+" "+str(res)+"\n")
    print(str(delta)+"PC "+" "+str(RAveL_PC(dep_infos, t, vvs, delta, 1))+"\n")