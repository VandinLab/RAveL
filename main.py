from operator import index

from pandas.io.parsers import read_csv
from dependency_infos import dependencyInfos
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import itertools
from itertools import chain, combinations, repeat
import argparse
from datetime import datetime
import warnings
from tqdm import tqdm
import math
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
import subprocess
import IT_utils
import pickle
import rad_utils

def get_pcd(dep_infos,t,v, delta, track_ITs = False, ordered_ITs = False, verbose = False, max_z_size = -1):
    if t in dep_infos.lookout_PCD:
        return dep_infos.lookout_PCD[t]
    # V set of variables, T target
    pcd = set()
    can_pcd = v.copy()
    can_pcd.remove(t)

    if track_ITs:
        ITs = set()
        
    if verbose:
        print("+++ GetPCD(",t,")","V=",v)
        print()

    pcd_changed = True

    while pcd_changed:
        old_pcd = pcd.copy()
        if verbose:
            print("        Begin of iteration", "canPCD:",can_pcd,"PCD:", pcd)

        # remove false positives from can_pcd
        sep = dict()
        for x in can_pcd:
            # using a function to avoid computation of p values for superset of conditioning sets with not enoguh elements in it
            sep[x] = IT_utils.get_argmin_dep(dep_infos, t,x, pcd, delta, max_z_size = max_z_size)

        to_remove = []
        for x in can_pcd:
            if IT_utils.independence(dep_infos,t,x,sep[x], delta = delta, verbose = bool(int(verbose/10)), level = 2):
                if verbose:
                    print("        Removing",x)
                to_remove.append(x)
                dep_infos.lookout_independence[(t,x)] = sep[x]
            if track_ITs:
                ITs.add(IT_utils.independence(dep_infos,t,x,sep[x], delta = delta, return_string = True, ordered = ordered_ITs)[1])

        for x in to_remove: #cannot do it all in once since there are runtime errors
            can_pcd.remove(x)

        if verbose:
            print("        Adding an element to PCD", can_pcd, pcd)

        if len(can_pcd) > 0:  # I might remove all elements from can_pcd
            # add the best candidate to pcd
            dependencies = [(x, IT_utils.association(dep_infos,t,x,sep[x], delta)) for x in can_pcd]
            dependencies = sorted(dependencies, key = lambda element : element[1], reverse = True)
            y = dependencies[0][0]  # take the argmax
            # print("- added to PCD",y)
            pcd.add(y)
            can_pcd.remove(y)

        if verbose:
            print("        Removing false positives from PCD", can_pcd, pcd)    
        
        # remove false positives from pcd
        for x in pcd:
            sep[x] = IT_utils.get_argmin_dep(dep_infos, t,x, pcd, delta, max_z_size = max_z_size)
            
        remove_el = []
        for x in pcd:
            if IT_utils.independence(dep_infos,t,x,sep[x], delta = delta, verbose = bool(int(verbose/10)), level = 2):
                if verbose:
                    print("        Removing",x)
                remove_el.append(x) # to solve runtime errors
                dep_infos.lookout_independence[(t,x)] = sep[x]
            if track_ITs:
                ITs.add(IT_utils.independence(dep_infos,t,x,sep[x], delta = delta, return_string = True, ordered = ordered_ITs)[1])

        for x in remove_el:
            pcd.remove(x)
        
        if verbose:
            print("        End of iteration", "canPCD:",can_pcd,"PCD:", pcd)
            print()
        pcd_changed = not old_pcd == pcd

    if verbose:
        print("--- GetPCD(",t,")","Result=",pcd)
    dep_infos.lookout_PCD[t] = pcd
    if track_ITs:
        return pcd, ITs
    return pcd

def get_pc(dep_infos,t, v, delta, track_ITs = False, ordered_ITs = False, verbose = False, max_z_size = -1):
    if t in dep_infos.lookout_PC:
        return dep_infos.lookout_PC[t]
    pc = set()
    ITs = set()
    if track_ITs:
        ITs = ITs.union(get_pcd(dep_infos,t,v, delta, track_ITs, ordered_ITs, max_z_size = max_z_size)[1])
    if verbose:
        print("+++ GetPC(",t,")","V=",v)
        print()

    for x in get_pcd(dep_infos,t,v, delta, verbose = int(verbose/10) * True, max_z_size = max_z_size):
        if t in get_pcd(dep_infos,x,v, delta, verbose = int(verbose/10) * True, max_z_size = max_z_size):
            pc.add(x)
            if verbose:
                print("   ",x,"added to PC(",t,")")

        if track_ITs:
            ITs = ITs.union(get_pcd(dep_infos,x,v, delta, track_ITs, ordered_ITs, max_z_size = max_z_size)[1])

    if verbose:
        print()
        print("--- GetPC(",t,")","Result=",pc)

    dep_infos.lookout_PC[t] = pc
    if track_ITs:
        return pc, ITs
    return pc

def pcmb(dep_infos,t, v, delta, track_ITs = False, ordered_ITs = False, verbose = False, max_z_size = -1):

    if t in dep_infos.lookout_MB:
        return dep_infos.lookout_MB[t]

    if verbose:
        print("+++ PCMB(",t,")","V=",v)
        print()

    # add true positives to MB
    pc = get_pc(dep_infos,t, v,  delta, verbose = int(verbose/10) * True, max_z_size = max_z_size)
    mb = pc.copy()

    ITs = set()
    if track_ITs:
        ITs = ITs.union(get_pc(dep_infos,t,v, delta, track_ITs, ordered_ITs)[1])

    # add more true positives to MB
    for y in pc:
        if track_ITs:
            ITs = ITs.union(get_pc(dep_infos,y,v, delta, track_ITs, ordered_ITs)[1])

        for x in get_pc(dep_infos,y,v, delta, verbose = int(verbose/10) * True, max_z_size = max_z_size):
            if not x in pc and x!=t:

                if (t,x) in dep_infos.lookout_independence:
                    z = dep_infos.lookout_independence[(t,x)]
                else:
                    z = dep_infos.lookout_independence[(x,t)]

                z_and_y = set(list(z))
                z_and_y.add(y)
                if not IT_utils.independence(dep_infos,t,x,z_and_y,   delta = delta):
                    mb.add(x)
                    if verbose*True:
                        print(x,"added to MB (I am considering element",y," of PC)")
                else:
                    if verbose*True:
                        print(x,"NOT added to MB (I am considering element",y," of PC)")
                if track_ITs:
                    ITs.add(IT_utils.independence(dep_infos,t,x,z_and_y,   delta = delta, return_string = True, ordered = ordered_ITs)[1])
            
    
    if verbose*True:
        print()
        print("--- PCMB(",t,")","Result=",mb)
    
    dep_infos.lookout_MB[t] = mb
    if track_ITs:
        return mb, ITs
    return mb

def iamb(dep_infos,t, v, delta, track_ITs = False, ordered_ITs = False, test_with_all_els = False, verbose = False, max_z_size = -1):
    mb = []
    no_changes = False
    while not no_changes:
        mb_old = mb.copy()
        y = sorted([(IT_utils.association(dep_infos, t, vv, set(mb)),vv) for vv in v if t != vv and vv not in mb], key = lambda x: x[0], reverse= True)
        if len(y)== 0:
            print("ERROR len(y) == 0","T",t,"V",v,"MB",mb)
        elif len(y[0]) < 2:
            print("ERROR len(y[0])<2","T",t,"V",v,"MB",mb)
        y = y[0][1] 
        if not IT_utils.independence(dep_infos,t,y,set(mb),delta):
            mb.append(y)
        no_changes = mb_old == mb
    updated_mb = mb.copy() # cannot modify mb since it is on cycle term
    for x in mb:
        mb_setminus_x = set([vv for vv in updated_mb if vv!=x])
        if IT_utils.independence(dep_infos, t,x,mb_setminus_x, delta):
            updated_mb = [vv for vv in updated_mb if vv!=x]
    return updated_mb

def RAveL_PC(dep_infos,t, v, delta, N, max_z_size = -2):
    if max_z_size == -2:
        max_z_size = len(v)-2       
    if t in dep_infos.lookout_RAveL_PC:
        return dep_infos.lookout_RAveL_PC[t]
    pc = []
    for x in v:
        if x != t:
            els_cond_set = [z for z in v if z not in [t,x]]
            if (t,x) in dep_infos.lookout_independence:
                z =  dep_infos.lookout_independence[(t,x)]
            elif (x,t) in dep_infos.lookout_independence:
                z =  dep_infos.lookout_independence[(x,t)]
            else:
                z = IT_utils.get_argmin_dep(dep_infos, t,x, els_cond_set, delta/N, max_z_size = max_z_size)

            if z == "-":
                pc.append(x)
            else:
                if not IT_utils.independence(dep_infos,t,x,z, delta/N):
                    pc.append(x)
                    dep_infos.lookout_independence[(t,x)] = "-"
                else:
                    dep_infos.lookout_independence[(t,x)] = z
    dep_infos.lookout_RAveL_PC[t] = pc
    return pc

def RAveL_MB(dep_infos,t, v, delta, N, max_z_size):
    if t in dep_infos.lookout_RAveL_MB:
        return dep_infos.lookout_RAveL_MB[t]

    pc = RAveL_PC(dep_infos,t, v,delta,N, max_z_size)
    mb = pc.copy()

    for y in pc:
        for x in RAveL_PC(dep_infos,y,v, delta, N, max_z_size):
            if not x in mb and x!=t:
                z_and_y = set([va for va in v if va!=x and va!=t])
                if not IT_utils.independence(dep_infos,t,x,z_and_y, delta = delta/N):
                    mb.append(x)
    dep_infos.lookout_RAveL_MB[t] = mb
    return mb


parser = argparse.ArgumentParser('main') 
parser.add_argument('-b', '--valid_size', type=int, default= 100000)
parser.add_argument('-c', '--n_cores', type=int, default= 72)
parser.add_argument('-d', '--delta_input', type=float, default= 0.05)
parser.add_argument('-t', '--test_number', type=int, default= 0)
parser.add_argument('-i', '--input_test_number', type=int, default= 0)
parser.add_argument('-n', '--n_variables', type=int, default= 0)
parser.add_argument('-f', '--file', type=str, default= "")
parser.add_argument('-v', '--verbose', type=int, default= 0)
parser.add_argument('-s', '--stop_iteration', type=int, default= 100)
parser.add_argument('-z', '--zero_init_t', type=bool, default= False)

args = parser.parse_args()
 
valid_size = args.valid_size
input_test_number = args.input_test_number
test_number = args.test_number
n_vars = args.n_variables
verbose = args.verbose
file = args.file
zero_init_t = args.zero_init_t
n_cores = args.n_cores
delta_input = args.delta_input
stop_iteration = args.stop_iteration

print("Parameters =",args)

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

        data_file = os.path.join("datasets","multiple10","v10_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+".csv")
        if delta/N < 0.0001:
            indep_file = os.path.join("IT_results","multiple10",test_type+"_multiple_s"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_delta"+str(delta/N)[:3]+str(delta/N)[-4:]+".csv")
        else:
            indep_file = os.path.join("IT_results","multiple10",test_type+"_multiple_s"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_delta"+str(delta/N)+".csv")
        
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
    
    n_rep=4
    vars = ["A0"] + ["B"+str(i+1) for i in range(n_rep)] + ["C"+str(i+1) for i in range(n_rep)] + ["A"+str(i+1) for i in range(n_rep)]
    ds_sizes = [100,250,500,1000,5000,10000,25000,50000,100000,250000]
    sv = len(vars)
    N = sv*(sv-1)*(2**(sv-3))  # diffrently from the notes, T is in vars
    deltas = [0.05]
    deltas_tot = [0.05, 0.05/N]
        
    versions = [1]
    ts = [i for i in range(stop_iteration)]
    test_t = ["cor"]
    tot_els = len(list(itertools.product(test_t, ts, [vars],deltas,[1,N],versions)))
    

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
        

    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product(test_t, [size], ts, [vars],deltas,[1,N],versions, [n_rep])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, pcd,pc,mb,version in sorted(results,key= lambda x: x[0]):
                pcds[version][d][test_type][size][test_n] = pcd
                pcs[version][d][test_type][size][test_n] = pc
                mbs[version][d][test_type][size][test_n] = mb

            if (True): #data saving part
                for ver in versions:
                    for test_type in test_t:
                        #result resume creation
                        for d in deltas_tot:
                            if d < 0.0001:
                                name = os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"_PCMB.csv")
                            else:
                                name = os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"_PCMB.csv")
                            cc = [[str(v)+"_t"+str(t) for t in ts] for v in vars]
                            cc = ["type"] + [item for sublist in cc for item in sublist]
                            data = []
                            for s in ds_sizes:
                                row = []
                                row.append("PCD_"+str(s))
                                dd = pcds.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                                row = []
                                row.append("PC_"+str(s))
                                dd = pcs.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                                row = []
                                row.append("MB_"+str(s))
                                dd = mbs.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                            res_df = pd.DataFrame(data,columns=cc)
                            res_df.to_csv(name)
                        
            if (True): #data merging part
                for ver in versions:
                    for tt in test_t:
                            
                        in_files = []
                        for d in deltas_tot:
                            if d < 0.0001:
                                in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"_PCMB.csv"),sep = ",") )
                            else:
                                in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"_PCMB.csv"),sep = ",") )
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
                        out_df.to_csv(os.path.join("IT_results","total_res","tot_on"+str(stop_iteration)+"_multiple_cont10_r"+str(n_rep)+"_v"+str(ver)+"_"+tt+"_PCMB.csv"))
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

        data_file = os.path.join("datasets","multiple10","v10_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+".csv")
        if delta/N < 0.0001:
            indep_file = os.path.join("IT_results","multiple10",test_type+"_multiple_s"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_delta"+str(delta/N)[:3]+str(delta/N)[-4:]+".csv")
        else:
            indep_file = os.path.join("IT_results","multiple10",test_type+"_multiple_s"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_delta"+str(delta/N)+".csv")

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
        mods_mbs = {}
        for v in vars:  
            mbs[v]  = iamb(dep_infos,v,vars,delta/N)

        IT_utils.save_IT_dataframe(dep_infos)
        return test_n, delta/N, test_type, mbs,  version
    
    n_rep=4
    vars = ["A0"] + ["B"+str(i+1) for i in range(n_rep)] + ["C"+str(i+1) for i in range(n_rep)] + ["A"+str(i+1) for i in range(n_rep)]
    ds_sizes = [100,250,500,1000,5000,10000,25000,50000,100000,250000]
    sv = len(vars)
    N = sv*(sv-1)*(2**(sv-3))  # diffrently from the notes, T is in vars
    deltas = [0.05]
    deltas_tot = [0.05, 0.05/N]
        
    versions = [1]
    ts = [i for i in range(stop_iteration)]
    test_t = ["cor"]
    tot_els = 2*len(list(itertools.product(test_t, ts, [vars],deltas,[1],versions)))
    

    mbs = dict([(v,{}) for v in versions])
    for v in versions:
        mbs[v] =  dict([(d,{}) for d in deltas_tot])
    for v in versions:
        for d in deltas_tot:
            mbs[v][d] =  dict([(ty,{}) for ty in test_t])
    for v in versions:
        for d in deltas_tot:
            for size in ds_sizes:
                for ty in test_t:
                    mbs[v][d][ty][size]    = {}
        

    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product(test_t, [size], ts, [vars],deltas,[1,N],versions, [n_rep])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type,mb,version in sorted(results,key= lambda x: x[0]):
                mbs[version][d][test_type][size][test_n] = {}

            for test_n, d, test_type, mb,version in sorted(results,key= lambda x: x[0]):
                mbs[version][d][test_type][size][test_n][False] = mb

            if (True): #data saving part
                for mod_PCMB in [False]:
                    for ver in versions:
                        for test_type in test_t:
                            #result resume creation
                            for d in deltas_tot:
                                if d < 0.0001:
                                    name = os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"_IAMB.csv")
                                else:
                                    name = os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"_IAMB.csv")
                                cc = [[str(v)+"_t"+str(t) for t in ts] for v in vars]
                                cc = ["type"] + [item for sublist in cc for item in sublist]
                                data = []
                                for s in ds_sizes:
                                    row = []
                                    row.append("MB_"+str(s))
                                    dd = mbs.get(ver)
                                    dd = dd.get(d)
                                    dd = dd.get(test_type)
                                    dd = dd.get(s,{})
                                    for v in vars:
                                        for t in ts:
                                            dt = dd.get(t,{})
                                            dt = dt.get(mod_PCMB,{})
                                            row.append(dt.get(v,"-"))
                                    data.append(row)
                                res_df = pd.DataFrame(data,columns=cc)
                                res_df.to_csv(name)
                        
            if (True): #data merging part
                for mod_PCMB in [False]:
                    for ver in versions:
                        for tt in test_t:
                                
                            in_files = []
                            for d in deltas_tot:
                                if d < 0.0001:
                                    in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"_IAMB.csv"),sep = ",") )
                                else:
                                    in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"_IAMB.csv"),sep = ",") )
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
                            out_df.to_csv(os.path.join("IT_results","total_res","tot_on"+str(stop_iteration)+"_multiple_cont10_r"+str(n_rep)+"_v"+str(ver)+"_"+tt+"_IAMB.csv"))
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

        data_file = os.path.join("datasets","multiple10","v10_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+".csv")
        if delta/N < 0.0001:
            indep_file = os.path.join("IT_results","multiple10",test_type+"_multiple_s"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_delta"+str(delta/N)[:3]+str(delta/N)[-4:]+".csv")
        else:
            indep_file = os.path.join("IT_results","multiple10",test_type+"_multiple_s"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+"_delta"+str(delta/N)+".csv")
        
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
            pcs[v]  = RAveL_PC(dep_infos,v,vars,delta,N, max_z_size = len(vars)-2)
            mbs[v]  = RAveL_MB(dep_infos,v,vars,delta,N, max_z_size = len(vars)-2)

        IT_utils.save_IT_dataframe(dep_infos)
        return test_n, delta/N, test_type, pcds, pcs, mbs, version
    
    n_rep=4
    vars = ["A0"] + ["B"+str(i+1) for i in range(n_rep)] + ["C"+str(i+1) for i in range(n_rep)] + ["A"+str(i+1) for i in range(n_rep)]
    ds_sizes = [100,250,500,1000,5000,10000,25000,50000,100000,250000]
    sv = len(vars)
    N = sv*(sv-1)*(2**(sv-3))  # diffrently from the notes, T is in vars
    deltas = [0.05]
    deltas_tot = [0.05/N]
        
    versions = [1]
    ts = [i for i in range(stop_iteration)]
    test_t = ["cor"]
    tot_els = len(list(itertools.product(test_t, ts, [vars],deltas,[N],versions)))  

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
        

    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            # input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product(test_t, [size], ts, [vars],deltas,[1,N],versions, [n_rep])))
            input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product(test_t, [size], ts, [vars],deltas,[N],versions, [n_rep])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, pcd,pc,mb,version in sorted(results,key= lambda x: x[0]):
                pcds[version][d][test_type][size][test_n] = pcd
                pcs[version][d][test_type][size][test_n] = pc
                mbs[version][d][test_type][size][test_n] = mb

            if (True): #data saving part
                for ver in versions:
                    for test_type in test_t:
                        #result resume creation
                        for d in deltas_tot:
                            if d < 0.0001:
                                name = os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"RAveL_Bonferroni.csv")
                            else:
                                name = os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"RAveL_Bonferroni.csv")
                            cc = [[str(v)+"_t"+str(t) for t in ts] for v in vars]
                            cc = ["type"] + [item for sublist in cc for item in sublist]
                            data = []
                            for s in ds_sizes:
                                row = []
                                row.append("PCD_"+str(s))
                                dd = pcds.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                                row = []
                                row.append("PC_"+str(s))
                                dd = pcs.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                                row = []
                                row.append("MB_"+str(s))
                                dd = mbs.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                            res_df = pd.DataFrame(data,columns=cc)
                            res_df.to_csv(name)
                        
            if (True): #data merging part
                for ver in versions:
                    for tt in test_t:
                            
                        in_files = []
                        for d in deltas_tot:
                            if d < 0.0001:
                                in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"RAveL_Bonferroni.csv"),sep = ",") )
                            else:
                                in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","res_on"+str(stop_iteration)+"_"+str(ver)+"_"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"RAveL_Bonferroni.csv"),sep = ",") )
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
                        out_df.to_csv(os.path.join("IT_results","total_res","tot_on"+str(stop_iteration)+"_multiple_cont10_r"+str(n_rep)+"_v"+str(ver)+"_"+tt+"_RAveL.csv"))
# TEST 4: RADEMACHER independencies calculator for synthetic data
if test_number == 4:
    ds_sizes = [100,250,500,1000,5000,10000,25000,50000,100000,250000]
    if stop_iteration == 5:
        ds_sizes = ds_sizes + [50000,100000,250000,500000]

    for n_rep in [4]:
        n_extra = 0
        vvs = ["A0"] + ["B"+str(i+1) for i in range(n_rep)] + ["C"+str(i+1) for i in range(n_rep)] + ["A"+str(i+1) for i in range(n_rep)]
        for s in ds_sizes:
            print("starting size",s)
            for t in range(stop_iteration):
                filename = "v10_S"+str(s)+"_t"+str(t)+"_r"+str(n_rep)+".csv"
                if not os.path.exists("IT_results/multiple10/"+"maxN_corrC_RAD_total_"+filename):
                    couples_dict = rad_utils.select_candidates(vvs)
                    r_vals, bound, tot_couples, tot_triples = rad_utils.calculate_stat_SD_bounds("datasets/multiple10/"+filename, vvs, delta = 0.05, couples_dict= couples_dict, n_proc = 50, corrected_bound = True, corrected_c = True, normalize_using_maxes= True, max_z_size = len(vvs)-2)
                    rad_utils.write_on_csv("IT_results/multiple10/"+"maxN_corrC_RAD_total_"+filename,r_vals, bound[0], tot_couples, tot_triples)
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
        
        filename = "v10_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+".csv"
                
        data_file = os.path.join("datasets","multiple10","v10_S"+str(size)+"_t"+str(test_n)+"_r"+str(n_rep)+".csv")
        if delta/N < 0.0001:
            indep_file = os.path.join("IT_results","multiple10","maxN_corrC_RAD_total_"+filename)
        else:
            indep_file = os.path.join("IT_results","multiple10","maxN_corrC_RAD_total_"+filename)
        
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
            pcs[v]  = RAveL_PC(dep_infos,v,vars,delta,N, max_z_size = len(vars)-2)
            mbs[v]  = RAveL_MB(dep_infos,v,vars,delta,N, max_z_size = len(vars)-2)
        return test_n, delta/N, test_type, pcds, pcs, mbs, version
    
    n_rep = 4
    vars = ["A0"] + ["B"+str(i+1) for i in range(n_rep)] + ["C"+str(i+1) for i in range(n_rep)] + ["A"+str(i+1) for i in range(n_rep)]
    ds_sizes = [100,250,500,1000,5000,10000,25000,50000,100000,250000]
    if stop_iteration == 5:
        ds_sizes = ds_sizes + [50000,100000,250000,500000]

    sv = len(vars)
    N = sv*(sv-1)*(2**(sv-3))  # diffrently from the notes, T is in vars
    deltas = [0.05]
    deltas_tot = [0.05]
        
    versions = [1]
    ts = [i for i in range(stop_iteration)]
    test_t = ["Rademacher"]
    tot_els = 2*len(list(itertools.product(test_t, ts, [vars],deltas,[1],versions)))
    
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
        

    for size in ds_sizes:
        print(datetime.now(), "Starting cycle with",size,"elements")
        with threadpool_limits(limits=1, user_api='blas'):
            p = mp.Pool(min(n_cores,72,tot_els))
            input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product(test_t, [size], ts, [vars],deltas,[1],versions, [n_rep])))
            results = p.map(perform_test, input)   
            p.close()
            p.join()

            for test_n, d, test_type, pcd,pc,mb,version in sorted(results,key= lambda x: x[0]):
                pcds[version][d][test_type][size][test_n] = pcd
                pcs[version][d][test_type][size][test_n] = pc
                mbs[version][d][test_type][size][test_n] = mb

            if (True): #data saving part
                for ver in versions:
                    for test_type in test_t:
                        #result resume creation
                        for d in deltas_tot:
                            if d < 0.0001:
                                name = os.path.join("IT_results","multiple10","total_res_on"+str(stop_iteration)+"_"+str(ver)+"_c"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"RAveL.csv")
                            else:
                                name = os.path.join("IT_results","multiple10","total_res_on"+str(stop_iteration)+"_"+str(ver)+"_c"+test_type+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"RAveL.csv")
                            cc = [[str(v)+"_t"+str(t) for t in ts] for v in vars]
                            cc = ["type"] + [item for sublist in cc for item in sublist]
                            data = []
                            for s in ds_sizes:
                                row = []
                                row.append("PCD_"+str(s))
                                dd = pcds.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                                row = []
                                row.append("PC_"+str(s))
                                dd = pcs.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                                row = []
                                row.append("MB_"+str(s))
                                dd = mbs.get(ver)
                                dd = dd.get(d)
                                dd = dd.get(test_type)
                                dd = dd.get(s,{})
                                for v in vars:
                                    for t in ts:
                                        dt = dd.get(t,{})
                                        row.append(dt.get(v,"-"))
                                data.append(row)
                            res_df = pd.DataFrame(data,columns=cc)
                            res_df.to_csv(name)
                        
            if (True): #data merging part
                for ver in versions:
                    for tt in test_t:
                            
                        in_files = []
                        for d in deltas_tot:
                            if d < 0.0001:
                                in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","total_res_on"+str(stop_iteration)+"_"+str(ver)+"_c"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)[:3]+str(d)[-4:]+"RAveL.csv"),sep = ",") )
                            else:
                                in_files.append(pd.read_csv(os.path.join("IT_results","multiple10","total_res_on"+str(stop_iteration)+"_"+str(ver)+"_c"+tt+"_multiple_r"+str(n_rep)+"_delta"+str(d)+"RAveL.csv"),sep = ",") )
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
                        out_df.to_csv(os.path.join("IT_results","total_res","total_tot_on"+str(stop_iteration)+"_multiple_cont10_r"+str(n_rep)+"_v"+str(ver)+"_c"+tt+"_RAveL.csv"))

# TEST 6: result analysis on synthetic data
if test_number == 6:
    n_exps = stop_iteration

    n_rep = 4
    vars = ["A0"] + ["B"+str(i+1) for i in range(n_rep)] + ["C"+str(i+1) for i in range(n_rep)] + ["A"+str(i+1) for i in range(n_rep)]
    d_struc = "[A0][B1|A0][C1|A0][A1|B1:C1]" + "".join(["[B"+str(i+1)+"|A"+str(i)+"][C"+str(i+1)+"|A"+str(i)+"][A"+str(i+1)+"|B"+str(i+1)+":C"+str(i+1)+"]" for i in range(1,n_rep)])
    ds_sizes = [100,250,500,1000,5000,10000,25000]
    sv = len(vars)
    N = sv*(sv-1)*(2**(sv-3)) 
    deltas = [0.05]
    deltas_tot = [0.05, 0.05/N]
    multiple10_r4_PCMB = (vars, ds_sizes, [1],deltas_tot,"multiple10_r4_PCMB",d_struc)
    multiple10_r4_IAMB = (vars, ds_sizes, [1],deltas_tot,"multiple10_r4_IAMB",d_struc)

    deltas = [0.05/N]
    deltas_tot = [0.05/N]
    multiple10_r4_RAveL_Bon = (vars, ds_sizes, [1],deltas_tot,"multiple10_r4_RAveL_Bon",d_struc)

    deltas = [0.05]
    deltas_tot = [0.05]
    total_multiple10_r4_RAveL_Rad_c = (vars, ds_sizes+[50000,100000,250000,500000], [1],deltas_tot,"total_multiple10_r4_RAveL_Rad_c",d_struc)

    counter_pc_per_test = dict()
    with open(os.path.join('IT_results','counter_pc_per_test.pickle'), 'wb') as handle:
        pickle.dump(counter_pc_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counter_mb_per_test = dict()
    with open(os.path.join('IT_results','counter_mb_per_test.pickle'), 'wb') as handle:
        pickle.dump(counter_mb_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counter_pc_per_test = dict()
    counter_mb_per_test = dict()

    parameter_list = [multiple10_r4_PCMB, multiple10_r4_IAMB, multiple10_r4_RAveL_Bon, total_multiple10_r4_RAveL_Rad_c]
    for pl in parameter_list:
        vars_selected = pl[0]
        sizes_selected = pl[1]
        versions_selected = pl[2]
        deltas_selected = pl[3]
        method_name = pl[4]
        dag_struc = pl[5]
        
        for ver in versions_selected:
            if "multiple10_r4_PCMB" == method_name:
                f_name_selected = os.path.join("IT_results","total_res","tot_on"+str(stop_iteration)+"_multiple_cont10_r4_v1_cor_PCMB.csv")
            if "multiple10_r4_IAMB" == method_name:
                f_name_selected = os.path.join("IT_results","total_res","tot_on"+str(stop_iteration)+"_multiple_cont10_r4_v1_cor_IAMB.csv")
            if "multiple10_r4_RAveL_Bon" == method_name:
                f_name_selected = os.path.join("IT_results","total_res","tot_on"+str(stop_iteration)+"_multiple_cont10_r4_v1_cor_RAveL.csv")
            if "total_multiple10_r4_RAveL_Rad_c" == method_name:
                f_name_selected = os.path.join("IT_results","total_res","total_tot_on"+str(stop_iteration)+"_multiple_cont10_r4_v1_cRademacher_RAveL.csv")

            in_df = pd.read_csv(f_name_selected)
            dic_key = method_name + "_version_" + str(ver)
            counter_pc_per_test[dic_key] = dict()
            counter_mb_per_test[dic_key] = dict()
            counter_pc_per_test[dic_key]["FP"]   = dict([(d,{}) for d in deltas_selected])
            counter_mb_per_test[dic_key]["FP"] = dict([(d,{}) for d in deltas_selected])
            counter_pc_per_test[dic_key]["FN"]   = dict([(d,{}) for d in deltas_selected])
            counter_mb_per_test[dic_key]["FN"] = dict([(d,{}) for d in deltas_selected])
            
            for d in deltas_selected:
                
                in_file = in_df[in_df['delta'] == d]
                if in_file.shape[0] == 0:
                    for dd in set(in_df['delta']):
                        if np.abs(dd-d) < d*1e-7:
                            in_file = in_df[in_df['delta'] == dd]

                for size in sizes_selected:
                    if size > 25000 or size == 300:
                        n_exps = 5
                    cc = [[str(v)+"_t"+str(t) for t in range(n_exps)] for v in vars_selected]
                    cc = [item for sublist in cc for item in sublist]
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
        
            with open(os.path.join('IT_results','counter_pc_per_test.pickle'), 'wb') as handle:
                pickle.dump(counter_pc_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join('IT_results','counter_mb_per_test.pickle'), 'wb') as handle:
                pickle.dump(counter_mb_per_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
# TEST 7: plots
if test_number == 7:
    couples = [(10,"v10_multiple_rep4")]

    data = [
            pickle.load(open(os.path.join('IT_results',"counter_pc_per_test.pickle"),"rb")),
            pickle.load(open(os.path.join('IT_results',"counter_mb_per_test.pickle"),"rb"))]

    corr_delta = [k for k in data[0]["multiple10_r4_PCMB_version_1"]["FP"].keys() if k != 0.05][0]
    c_list = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    dim = 2.25
    mms = 8
    for test_n, dic_k in couples:
        test = test_n
        dic_key = dic_k
        fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)
        titles = ["FWER PC","FWER MB"]

        corr_delta = [k for k in data[0]["multiple10_r4_PCMB_version_1"]["FP"].keys() if k != 0.05][0]
        legends_ravel = [["RAveL-PC","RAveL-PC - Bonferroni correction"],
                ["RAveL-MB","RAveL-MB - Bonferroni correction"]]
        legends_others = [["GetPC - no correction","GetPC - Bonferroni correction","IAMB - no correction","IAMB - Bonferroni correction"],
            ["PCMB - no correction","PCMB - Bonferroni correction","IAMB - no correction","IAMB - Bonferroni correction"]]

        for i in range(2):
            ax[i].axhline(0.05, ls='--', color='r')
            xs = sorted(data[i]["multiple10_r4_PCMB_version_1"]["FP"][0.05].keys())

            ys = np.mean([np.array(data[i]["total_multiple10_r4_RAveL_Rad_c_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = legends_ravel[i][0], marker = "^",ms = mms, c = c_list[2])
            ys = np.mean([np.array(data[i]["multiple10_r4_RAveL_Bon_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = legends_ravel[i][1], marker = "v",ms = mms, c = c_list[1])
            ys = np.mean([np.array(data[i]["multiple10_r4_PCMB_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = legends_others[i][0], marker = "<",ms = mms, c = c_list[3])
            ys = np.mean([np.array(data[i]["multiple10_r4_PCMB_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
            ax[i].plot(xs,ys, label = legends_others[i][1], marker = ">",ms = mms, c = c_list[4])
            if i == 1:
                ys = np.mean([np.array(data[i]["multiple10_r4_IAMB_version_1"]["FP"][0.05][size])>0 for size in xs],axis = 1)
                ax[i].plot(xs,ys, label = legends_others[i][2], marker = "D",ms = mms, c = c_list[5])
                ys = np.mean([np.array(data[i]["multiple10_r4_IAMB_version_1"]["FP"][corr_delta][size])>0 for size in xs],axis = 1)
                ax[i].plot(xs,ys, label = legends_others[i][3], marker = "s",ms = mms, c = c_list[6])

            ax[i].set_ylabel(titles[i])
            ax[i].set_xlabel("Sample size")
            ax[i].set_xscale('log')
            ax[i].set_xlim(90)
            ax[i].set_ylim(-0.05,1.05)
            ax[i].set_xticks(xs)
            ax[i].set_xticklabels(xs)
            ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("FWER.pdf")
            
    
    for test_n, dic_k in couples:
        test = test_n
        dic_key = dic_k
        fig, ax = plt.subplots(1,2, figsize = (int(50/dim),int(16/dim)), sharey = True)
        titles = ["%FN PC","%FN MB"]

        corr_delta = [k for k in data[0]["multiple10_r4_RAveL_Bon_version_1"]["FP"].keys() if k != 0.05][0]
        legends = [["RAveL-PC","RAveL-PC - Bonferroni correction"],
                ["RAveL-MB","RAveL-MB - Bonferroni correction"]]
        
        correction_PC = correction_MB = 1
        
        correction_PC = 8 * int(dic_k[-1:]) 
        correction_MB = 10 * int(dic_k[-1:]) 
        correction_el2 = 12 + 20 * (int(dic_k[-1:]) -1)
        
        corrections = [correction_PC, correction_MB]
        
        for i in range(2):
            xs = sorted(data[i]["total_multiple10_r4_RAveL_Rad_c_version_1"]["FN"][0.05].keys())

            ys = np.mean([np.array(data[i]["total_multiple10_r4_RAveL_Rad_c_version_1"]["FN"][0.05][size][:5]) for size in xs],axis = 1)/corrections[i]
            ax[i].plot(xs,ys, label = legends[i][0], marker = "^",ms = mms, c = c_list[2])

            ax[i].set_ylabel(titles[i])
            ax[i].set_xlabel("Sample size")
            ax[i].set_xscale('log')
            ax[i].set_xlim(90)
            ax[i].set_ylim(-0.05,1.05)
            ax[i].set_xticks(xs)
            ax[i].set_xticklabels(xs)
            ax[i].legend()

    plt.subplots_adjust(wspace=0.1)
    plt.savefig("FN_percentage.pdf")

# TEST 8: RADEMACHER calculus on Boston housing standard
if test_number == 8:
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

    vvs = ["CRIM" ,"ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    df = pd.read_csv(os.path.join("datasets","real_data","housing.csv"), header=None, delimiter=r"\s+", names = vvs)
    df.astype("float64").to_csv(os.path.join("datasets","real_data","housing_commas.csv"))
    t = "MEDV"
    sv = len(vvs)
    N = sv*(sv-1)*(2**(sv-3)) 
    delta = 0.01/N
            
    data_file = os.path.join("datasets","real_data","housing_commas.csv")
    indep_file = os.path.join("IT_results","real_data","py_housing.csv")
    
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
    
    res = RAveL_MB(dep_infos, t, vvs,delta,1,max_z_size = len(vvs)-2)

    IT_utils.save_IT_dataframe(dep_infos)
    print()
    print(str(delta)+"MB "+str(res)+"\n")
    print(str(delta)+"PC "+str(RAveL_PC(dep_infos, t, vvs, delta, 1))+"\n")

