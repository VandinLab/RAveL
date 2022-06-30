# -*- coding: utf-8 -*-
from dependency_infos import dependencyInfos
from math import exp
import os
import pandas as pd
import numpy as np
from itertools import chain, combinations, repeat
import warnings
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import IT_utils
import itertools
import multiprocess as mp
from threadpoolctl import threadpool_limits
import math

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def calculate_sup_dev_bound(m, n, R_tilde, delta, z, c, get_splitted_values):
    # calculating the supreme deviation bound and splitting each contribution into summands
    s1 = 2*R_tilde
    s2 = math.sqrt(c*(4*m*R_tilde+c*math.log(4/delta))*math.log(4/delta))/m
    s3 = c*math.log(4/delta)/m
    s4 = c*math.sqrt(math.log(4/delta)/(2*m))
    if get_splitted_values:
        return s1 + s2 + s3 + s4, (s1,s2,s3,s4)
    return s1 + s2 + s3 + s4

def compose_dataset(X, col_names, selection):
    # given a selection of variables, create a dataset using only the respective variables
    X_ret = []
    for var in selection:
        X_ret.append(X[:,col_names.index(var)])
    return np.array(X_ret).transpose()

def couple_divider(filename, tot_vars, delta, max_z_size, ds_folder, it_folder):
    couples = [e for e in combinations(tot_vars, 2)]
    deps = []
    indeps = []
    data_file = ds_folder+filename
    indep_file = it_folder + "cor_"+filename
    
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

    dep_infos = dependencyInfos(indep_df, method="p-value", independence_method="cor", data_file=data_file, 
                    indep_file=indep_file, save_every_update=False, independence_language="python")
    for v1,v2 in couples:
        other_vars = [v for v in tot_vars if v!= v1 and v!=v2]
        z = IT_utils.get_argmin_dep(dep_infos,v1,v2,other_vars,delta,max_z_size)
        if not IT_utils.independence(dep_infos, v1,v2,z, delta):
            deps.append((v1,v2))
        else:
            indeps.append((v1,v2))
    IT_utils.save_IT_dataframe(dep_infos)
    return deps, indeps

def divide_into_datasets(total_ds, n_vars):
    assert total_ds.shape[0]%n_vars == 0
    return np.array([total_ds[i*n_vars:i*n_vars+n_vars] for i in range(int(total_ds.shape[0]/n_vars))])

def select_candidates(tot_vars):
    candidates = []
    for i in range(len(tot_vars)):
        for j in range(len(tot_vars)):
            if i<j:
                candidates.append((tot_vars[i],tot_vars[j]))
    n_els_max = math.floor(len(candidates)/(len(tot_vars)-1))
    dic_els = {}
    for i in range(len(tot_vars)):
        cnt = 0
        dic_els[tot_vars[i]] = []
        for c in candidates:
            if tot_vars[i] in c and cnt != n_els_max:
                dic_els[tot_vars[i]].append(c)
                cnt += 1
        for c in dic_els[tot_vars[i]]:
            candidates.remove(c)
    # print(candidates)
    return dic_els


def calculate_residuals(target,data):
    reg = LinearRegression().fit(data, target)
    residuals = target - reg.predict(data)
    # print(residuals.shape)
    return residuals

def calculate_summands(xs,ys,zs = np.array([]), TT ="", corrected_bound = False, my_stat = False, normalize_using_maxes = False):
    if not (type(zs)==type(()) and len(zs)==0):
        if zs.shape[0] != 0:
            # better to have normalized data before doing the regression
            xs -= np.mean(xs)
            xs /= np.std(xs)
            ys -= np.mean(ys)
            ys /= np.std(ys)
            zs -= np.mean(zs)
            zs /= np.std(zs)
            e_x = calculate_residuals(xs,zs) 
            e_y = calculate_residuals(ys,zs)
            xs = e_x
            ys = e_y
    n = xs.shape[0]
    xs -= np.mean(xs)
    # print("xs;",max(np.abs(xs)),";",np.std(xs),";",max(np.abs(xs))/np.std(xs))
    if normalize_using_maxes:
        if len(set(xs)) > 1:
            # xs -= (max(xs) - min(xs))
            xs /= max(np.abs(xs))
            assert max(np.abs(xs)) < 1.01
    elif not np.std(xs)==0:
        xs /= np.std(xs)
    ys -= np.mean(ys)
    # print("ys;",max(np.abs(ys)),";",np.std(ys),";",max(np.abs(ys))/np.std(ys))
    if normalize_using_maxes:
        if len(set(ys)) > 1:
            # ys -= (max(ys) - min(ys))
            ys /= max(np.abs(ys))
            assert max(np.abs(ys)) < 1.01
    elif not np.std(ys)==0:
        ys /= np.std(ys)
    if not my_stat:
        if not corrected_bound:
            r_summs = np.multiply(xs,ys)/ (n-1)
        else:
            r_summs = np.multiply(xs,ys)/ (n-1) * n
    else:
        # r_summs = np.sign(np.multiply(xs,ys))
        r_summs = np.multiply(xs,ys) / np.power(np.maximum(np.abs(xs),np.abs(ys)),2)
        # r_summs = np.multiply(xs,ys) / (np.power(xs,2)+np.power(ys,2))
        if max(r_summs) >1:
            print(np.max(r_summs),np.mean(r_summs))
    return r_summs

def calculate_r_estimates_per_ds(xs,ys, n_vars, zs = np.array([]), my_stat = False, normalize_using_maxes = False):
    # selecting multiples of n_vars
    xs = xs[:int(xs.shape[0]/n_vars)*n_vars]
    ys = ys[:int(ys.shape[0]/n_vars)*n_vars]
    xs = divide_into_datasets(xs,n_vars)
    ys = divide_into_datasets(ys,n_vars)
    if not (type(zs)==type(()) and len(zs)==0):
        if zs.shape[0] != 0:
            if len(zs.shape)>1:
                zs = zs[:int(zs.shape[0]/n_vars)*n_vars,:]
            else:
                zs = zs[:int(zs.shape[0]/n_vars)*n_vars]
            zs = divide_into_datasets(zs,n_vars)
            # better to have normalized data before doing the regression
            xs = [x-np.mean(x) for x in xs]
            xs = [x/np.std(x) for x in xs]
            ys = [y-np.mean(y) for y in ys]
            ys = [y/np.std(y) for y in ys]
            zs = [z-np.mean(z) for z in zs]
            zs = [z/np.std(z) for z in zs]
            e_x = [calculate_residuals(xs[i],zs[i]) for i in range(len(xs))] 
            e_y = [calculate_residuals(ys[i],zs[i]) for i in range(len(xs))]
            xs = np.array(e_x)
            ys = np.array(e_y)
    xs = [x-np.mean(x) for x in xs]
    if normalize_using_maxes:
        xs = [x/max(np.abs(x)) for x in xs]
        assert max(xs) < 1.01
    else:
        xs = [x/np.std(x) for x in xs]
    ys = [y-np.mean(y) for y in ys]
    if normalize_using_maxes:
        ys = [y/max(np.abs(y)) for y in ys]
        assert max(ys) < 1.01
    else:
        ys = [y/np.std(y) for y in ys]
    if not my_stat:
        r_summs = [np.multiply(xs[i],ys[i])/ (n_vars-1) for i in range(len(xs))]
    else:
        r_summs = [np.multiply(xs[i],ys[i]) / np.power(np.maximum(xs[i],ys[i]),2) for i in range(len(xs))]
    return r_summs

                
def calculate_stat_ERA_per_var(*args):
    # variable, filename, sigma, couples_dict
    params = args[0]
    variable = params[0]
    filename = params[1]
    sigma = params[2]
    couples_dict = params[3]
    corrected_bound = params[4]
    max_z_size = params[5]
    my_stat = params[6]
    normalize_using_maxes = params[7]
    sigma_cp = sigma.view()
    n_sigma = sigma_cp.shape[1]
    data = pd.read_csv(filename) 
    col = [c for c in list(data.columns.values) if not "Unnamed" in c]
    n_vars = len(col)
    #  calculate test on var1 and var2 conditioning on emptyset
    couples = np.array(couples_dict[variable])
    cond_sets = np.array([z for z in IT_utils.powerset(col, max_z_size = max_z_size, get_emptyset= False)])
    triplets = []
    for c in couples:
        for s in cond_sets:
            if not c[0] in s and not c[1] in s:
                triplets.append((c,s))
        if max_z_size != n_vars -2: # adding the last conditioning set with all the vars but v1 and v2
            s = [v for v in col if v!= c[0] and v!= c[1]]
            triplets.append((c,tuple(s)))

    if len(triplets) > 0:
        r_mat_uncond = np.array([calculate_summands(data[t[0]].values,data[t[1]].values, corrected_bound = corrected_bound, my_stat=my_stat, normalize_using_maxes = normalize_using_maxes) for t in couples]).T
        if not corrected_bound:
            r_vals = np.sum(r_mat_uncond,axis=0).tolist()
        else:
            r_vals = np.mean(r_mat_uncond,axis=0).tolist()

        m = r_mat_uncond.shape[0]
        sums = np.matmul(sigma_cp.T,r_mat_uncond)  # n x m * m x k = n x k matrix
        sups_vector = np.max(sums,axis = 1)
        step_size = 1000
        for i in range(math.ceil(len(triplets)/step_size)):
            # print("A")
            # if i!= 0 and i%10==0:
            #     print("B")
            #     break
            #calculate stats for the block
            r_mat_cond = np.array([calculate_summands(data[t[0][0]].values,data[t[0][1]].values, compose_dataset(data.values,col, selection=t[1]),t, corrected_bound = corrected_bound, my_stat=my_stat, normalize_using_maxes=normalize_using_maxes) 
                for t in triplets[i*step_size:(i+1)*step_size]]).T
            # print("pt.0")
            sums_cond = np.matmul(sigma_cp.T,r_mat_cond)
            sups_vector = np.maximum(sups_vector, np.max(sums_cond,axis = 1))
            #update stats
            # print("pt.1")
            if not corrected_bound:
                r_vals.append(np.sum(r_mat_cond,axis=0).tolist())
            else:
                r_vals.append(np.mean(r_mat_cond,axis=0).tolist())
    return variable, r_vals, sups_vector, couples, triplets

def calculate_stat_ERA_per_couple(*args):
    # variable, filename, sigma, couples_dict
    params = args[0]
    tot_vars = params[0]
    filename = params[1]
    sigma = params[2]
    selected_couple = params[3]
    v1 = selected_couple[0]
    v2 = selected_couple[1]
    corrected_bound = params[4]
    max_z_size = params[5]
    my_stat = params[6]
    normalize_using_maxes = params[7]

    debug = False
    if debug:
        print("performing couple", selected_couple)
    sigma_cp = sigma.view()
    n_sigma = sigma_cp.shape[1]
    data = pd.read_csv(filename)[tot_vars]  # select only variables, not indexes
    n_vars = len(tot_vars)
    #  calculate test on var1 and var2 conditioning on emptyset
    cond_sets = [z for z in IT_utils.powerset(tot_vars, max_z_size = max_z_size, get_emptyset= False) if v1 not in z and v2 not in z]
    if max_z_size != len(tot_vars)-2: # adding the last conditioning set with all the vars but v1 and v2
        s = [v for v in tot_vars if v!= v1 and v!= v2]
        cond_sets.append(tuple(s))
    if debug:
        print("cond sets", cond_sets)
    #unconditional case
    r_vals = np.array([calculate_summands(data[v1].values,data[v2].values, corrected_bound = corrected_bound, my_stat=my_stat, normalize_using_maxes = normalize_using_maxes)]).T
    m = r_vals.shape[0]
    sums = np.matmul(sigma_cp.T,r_vals)  # n x m * m x k = n x k matrix
    sups_vector = np.max(sums,axis = 1)
    r_vals = [np.mean(r_vals,axis=0)[0]]
    if debug:
        print("sups_vector shape", sups_vector.shape)
        print("r_vals", r_vals)
    #calulating conditional case in blocks
    step_size = 1000
    for i in range(math.ceil(len(cond_sets)/step_size)):
        #calculate stats for the block
        r_mat_cond = np.array([calculate_summands(data[v1].values,data[v2].values, compose_dataset(data.values,tot_vars, selection=cs), corrected_bound = corrected_bound, my_stat=my_stat, normalize_using_maxes = normalize_using_maxes) 
                                for cs in cond_sets[i*step_size:(i+1)*step_size]]).T
        sums_cond = np.matmul(sigma_cp.T,r_mat_cond)
        sups_vector = np.maximum(sups_vector, np.max(sums_cond,axis = 1))
        #update stats
        if not corrected_bound:
            r_vals+=np.sum(r_mat_cond,axis=0).tolist()
        else:
            r_vals+=np.mean(r_mat_cond,axis=0).tolist()
    if debug:
        print("sups_vector shape", sups_vector.shape)
        print("r_vals len", len(r_vals))
        print("r_vals", r_vals)
        print("cond sets len", len([()]+cond_sets))
        print()

    return selected_couple, r_vals, sups_vector, [()]+cond_sets
  
def calculate_stat_ERA_per_dataset_couple(*args):
    # variable, filename, sigma, couples_dict
    params = args[0]
    tot_vars = params[0]
    filename = params[1]
    sigma = params[2]
    selected_couple = params[3]
    v1 = selected_couple[0]
    v2 = selected_couple[1]
    corrected_bound = params[4]
    max_z_size = params[5]
    my_stat = params[6]
    normalize_using_maxes = params[7]

    debug = False
    sigma_cp = sigma.view()
    n_sigma = sigma_cp.shape[1]
    data = pd.read_csv(filename)[tot_vars]  # select only variables, not indexes
    n_vars = len(tot_vars)
    #  calculate test on var1 and var2 conditioning on emptyset
    cond_sets = [z for z in IT_utils.powerset(tot_vars, max_z_size = max_z_size, get_emptyset= False) if v1 not in z and v2 not in z]
    if max_z_size != len(tot_vars)-2: # adding the last conditioning set with all the vars but v1 and v2
        s = [v for v in tot_vars if v!= v1 and v!= v2]
        cond_sets.append(tuple(s))
    #unconditional case
    r_vals = np.array([calculate_r_estimates_per_ds(data[v1].values,data[v2].values, n_vars, my_stat=my_stat, normalize_using_maxes=normalize_using_maxes)]).T
    m = r_vals.shape[0]
    sums = np.matmul(sigma_cp.T,r_vals)  # n x m * m x k = n x k matrix
    sups_vector = np.max(sums,axis = 1)
    r_vals = [np.mean(r_vals,axis=0)[0]]
    #calulating conditional case in blocks
    step_size = 1000
    for i in range(math.ceil(len(cond_sets)/step_size)):
        #calculate stats for the block
        r_mat_cond = np.array([calculate_r_estimates_per_ds(data[v1].values,data[v2].values, n_vars, compose_dataset(data.values,tot_vars, selection=cs), my_stat=my_stat, normalize_using_maxes=normalize_using_maxes) 
                                for cs in cond_sets[i*step_size:(i+1)*step_size]]).T
        sums_cond = np.matmul(sigma_cp.T,r_mat_cond)
        sups_vector = np.maximum(sups_vector, np.max(sums_cond,axis = 1))
        #update stats
        r_vals+=np.mean(r_mat_cond,axis=0).tolist()

    return selected_couple, r_vals, sups_vector, [()]+cond_sets


def calculate_stat_SD_bounds(filename, tot_vars, delta, couples_dict, n_sigma = 1000, corrected_bound = False, n_proc = 1, debug = False, max_z_size = 2, normalize_using_maxes = False, corrected_c = False, my_stat = False):
    data = pd.read_csv(filename) 
    m = data.shape[0]
    # interval of size 2 with between -1 and 1
    if not corrected_c:
        z = 1
        c = 2
    else:
        if not normalize_using_maxes:
            z = m
            c = 2*m
        else:
            z = m/(m-1)
            c = 2*m/(m-1)
    sigma = np.random.rand(m,n_sigma) - 0.5
    sigma[sigma>=0] = 1
    sigma[sigma<0] = -1 
    tot_couples = []
    tot_triples = []
    r_vals = []
    tot_sups = None
    with threadpool_limits(limits=1, user_api='blas'):
        p = mp.Pool(min(len(tot_vars), n_proc))
        # variable, filename, sigma, couples_dict
        input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product(tot_vars,[filename],[sigma],[couples_dict],[corrected_bound],[max_z_size],[my_stat],[normalize_using_maxes])))
        results = p.map(calculate_stat_ERA_per_var, input)   
        p.close()
        p.join()
        for v, r_vs, sups, couples, triples in sorted(results,key= lambda x: x[0]):
            tot_couples.append([couples])
            tot_triples.append([triples]) 
            r_vals.append(r_vs)
            if tot_sups is None:
                tot_sups = sups
            else:
                tot_sups = np.maximum(tot_sups,sups)
    R_hat = np.sum(tot_sups)/m/n_sigma
    # r_vals, R_hat = calculate_stat_ERA_per_var(variable, filename, sigma, couples_dict)
    R_tilde = R_hat + 2*z*math.sqrt(math.log(4/delta)/(2*n_sigma*m))
    if debug:
        print("R_hat",R_hat)
        print("R_tilde",R_tilde)
        print("scomposed bound",calculate_sup_dev_bound(m, n_sigma, R_tilde, delta, z = z, c = c, get_splitted_values=True))
    return r_vals, calculate_sup_dev_bound(m, n_sigma, R_tilde, delta, z = z, c = c, get_splitted_values=True), tot_couples, tot_triples

def calculate_stat_SD_bounds_per_couple(filename, tot_vars, delta, n_sigma = 1000, corrected_bound = True, n_proc = 1, debug = False, max_z_size = 2, normalize_using_maxes = False, corrected_c = False, my_stat = False):
    data = pd.read_csv(filename) 
    m = data.shape[0]
    # interval of size 2 with between -1 and 1
    if not corrected_c:
        z = 1
        c = 2
    else:
        if not normalize_using_maxes:
            z = m
            c = 2*m    
        else:
            z = m/(m-1)
            c = 2*m/(m-1)
    sigma = np.random.rand(m,n_sigma) - 0.5
    sigma[sigma>=0] = 1
    sigma[sigma<0] = -1 
    r_vals = []
    results_dict = {}
    with threadpool_limits(limits=1, user_api='blas'):
        p = mp.Pool(min(int(len(tot_vars)*(len(tot_vars)-1)/2),n_proc))
        couples = [e for e in combinations(tot_vars, 2)]
        # variable, filename, sigma, couples_dict
        input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product([tot_vars],[filename],[sigma],couples,[corrected_bound],[max_z_size],[my_stat], [normalize_using_maxes])))
        results = p.map(calculate_stat_ERA_per_couple, input)   
        p.close()
        p.join()

        for couple, r_vals, sups_vector, cond_sets in sorted(results,key= lambda x: x[0]):
            R_hat = np.sum(sups_vector)/m/n_sigma
            R_tilde = max(R_hat + 2*z*math.sqrt(math.log(4/delta)/(2*n_sigma*m)),0)
            decomposed_bound = calculate_sup_dev_bound(m, n_sigma, R_tilde, delta, z = z, c = c, get_splitted_values=True)
            results_dict[couple] = (r_vals, R_hat, R_tilde, decomposed_bound, cond_sets)
            
    return results_dict
  
def calculate_stat_SD_bounds_per_group(filename, tot_vars, delta_c, delta_split = 0.05, n_sigma = 1000, max_z_size = 2, normalize_using_maxes = False, corrected_c = False, my_stat = False, thread_limits = 1, corrected_bound = True, n_proc = 1, debug = False, ds_folder = "datasets/multiple13/", it_folder = "IT_results/multiple13/"):
    data = pd.read_csv(ds_folder+filename) 
    m = data.shape[0]
    # interval of size 2 with between -1 and 1
    if not corrected_c:
        z = 1
        c = 2
    else:
        if not normalize_using_maxes:
            z = m
            c = 2*m    
        else:
            z = m/(m-1)
            c = 2*m/(m-1)
    sigma = np.random.rand(m,n_sigma) - 0.5
    sigma[sigma>=0] = 1
    sigma[sigma<0] = -1 
    r_vals = []

    results_dict = {}
    with threadpool_limits(limits=10, user_api='blas'):
        deps, indeps = couple_divider(filename, tot_vars, delta = delta_split, max_z_size = max_z_size, ds_folder = ds_folder, it_folder = it_folder)

    with threadpool_limits(limits=thread_limits, user_api='blas'):
        p = mp.Pool(min(max(len(deps),len(indeps)),n_proc))
        # variable, filename, sigma, couples_dict
        input_f = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product([tot_vars],[ds_folder+filename],[sigma],deps,[corrected_bound],[max_z_size],[my_stat], [normalize_using_maxes])))
        results = p.map(calculate_stat_ERA_per_couple, input_f)   
        p.close()
        p.join()

        sups_vector = np.zeros_like(results[0][2])
        for couple, r_vals, sups_v_couple, cond_sets in sorted(results,key= lambda x: x[0]):
            sups_vector = np.maximum(sups_vector,sups_v_couple)

        R_hat = np.sum(sups_vector)/m/n_sigma
        R_tilde = max(R_hat + 2*z*math.sqrt(math.log(4/delta_c)/(2*n_sigma*m)),0)
        decomposed_bound = calculate_sup_dev_bound(m, n_sigma, R_tilde, delta_c, z = z, c = c, get_splitted_values=True)
        
        for couple, r_vals, sups_v_couple, cond_sets in sorted(results,key= lambda x: x[0]):
            results_dict[couple] = (r_vals, R_hat, R_tilde, decomposed_bound, cond_sets)

        p = mp.Pool(min(max(len(deps),len(indeps)),n_proc))
        # variable, filename, sigma, couples_dict
        input_f = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product([tot_vars],[ds_folder+filename],[sigma],indeps,[corrected_bound],[max_z_size], [my_stat], [normalize_using_maxes])))
        results = p.map(calculate_stat_ERA_per_couple, input_f)   
        p.close()
        p.join()

        sups_vector = np.zeros_like(results[0][2])
        for couple, r_vals, sups_v_couple, cond_sets in sorted(results,key= lambda x: x[0]):
            sups_vector = np.maximum(sups_vector,sups_v_couple)

        R_hat = np.sum(sups_vector)/m/n_sigma
        R_tilde = max(R_hat + 2*z*math.sqrt(math.log(4/delta_c)/(2*n_sigma*m)),0)
        decomposed_bound = calculate_sup_dev_bound(m, n_sigma, R_tilde, delta_c, z = z, c = c, get_splitted_values=True)
        
        for couple, r_vals, sups_v_couple, cond_sets in sorted(results,key= lambda x: x[0]):
            results_dict[couple] = (r_vals, R_hat, R_tilde, decomposed_bound, cond_sets)
            
    return results_dict

def calculate_stat_SD_bounds_per_dataset_couple(filename, tot_vars, delta, n_sigma = 1000, corrected_bound = True, n_proc = 1, debug = False, max_z_size = 2, normalize_using_maxes = False, corrected_c = False, my_stat = False):
    data = pd.read_csv(filename) 
    m = int(data.shape[0]/len(tot_vars))
    # interval of size 2 with between -1 and 1
    if not corrected_c:
        z = 1
        c = 2
    else:
        if not normalize_using_maxes:
            z = m
            c = 2*m    
        else:
            z = m/(m-1)
            c = 2*m/(m-1)
    sigma = np.random.rand(m,n_sigma) - 0.5
    sigma[sigma>=0] = 1
    sigma[sigma<0] = -1 
    r_vals = []
    results_dict = {}
    with threadpool_limits(limits=1, user_api='blas'):
        p = mp.Pool(min(int(len(tot_vars)*(len(tot_vars)-1)/2),n_proc))
        couples = [e for e in combinations(tot_vars, 2)]
        # variable, filename, sigma, couples_dict
        input = ((a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in list(itertools.product([tot_vars],[filename],[sigma],couples,[corrected_bound],[max_z_size],[my_stat], [normalize_using_maxes])))
        results = p.map(calculate_stat_ERA_per_dataset_couple, input)   
        p.close()
        p.join()

        for couple, r_vals, sups_vector, cond_sets in sorted(results,key= lambda x: x[0]):
            R_hat = np.sum(sups_vector)/m/n_sigma
            R_tilde = max(R_hat + 2*z*math.sqrt(math.log(4/delta)/(2*n_sigma*m)),0)
            decomposed_bound = calculate_sup_dev_bound(m, n_sigma, R_tilde, delta, z = z, c = c, get_splitted_values=True)
            results_dict[couple] = (r_vals, R_hat, R_tilde, decomposed_bound, cond_sets)
            
    return results_dict


def write_on_csv(filename, r_vals, bound_tot_val, tot_couples, tot_triples):
    df = pd.DataFrame(columns = ["first_term", "second_term", "cond_set", "lb","stat","up"])
    for i in range(len(tot_couples)):
        firsts = []
        seconds = []
        conds = []
        lbs = []
        stats = []
        ups = []
        for j in range(len(tot_couples[i][0])):  # unconditional
            firsts.append(tot_couples[i][0][j][0])
            seconds.append(tot_couples[i][0][j][1])
            conds.append("()")
            lbs.append(r_vals[i][j]-bound_tot_val )
            stats.append(r_vals[i][j])
            ups.append(r_vals[i][j]+bound_tot_val)
        df2 = pd.DataFrame.from_dict(
            {"first_term": firsts,
            "second_term": seconds,
            "cond_set": conds ,
            "lb": lbs ,
            "stat": stats ,
            "up": ups }
        )
        cumulative = 0
        df = pd.concat([df,df2])
        df.to_csv(filename)
        triplets_idx = len(tot_couples[i][0])
        # the first #couples elements are single values (one per couple)
        # but then there are len(triplets)/step_size arrays of maximum step_size elements
        # that are related to the couples
        for j in range(triplets_idx,len(r_vals[i])):  # unconditional
            firsts = []
            seconds = []
            conds = []
            lbs = []
            stats = []
            ups = []
            for k in range(len(r_vals[i][j])):
                # dat = {"first_term":tot_triples[i][0][j][0][0],
                #     "second_term":tot_triples[i][0][j][0][1],
                #     "cond_set": tot_triples[i][0][j][1],
                #     "lb":r_vals[i][j][k]-bound_tot_val ,
                #     "stat": r_vals[i][j][k],
                #     "up": r_vals[i][j][k]+bound_tot_val}
                # df.append(dat,ignore_index=True)
                firsts.append(tot_triples[i][0][cumulative][0][0])
                seconds.append(tot_triples[i][0][cumulative][0][1])
                conds.append(IT_utils.convert_to_string(tot_triples[i][0][cumulative][1]))
                lbs.append(r_vals[i][j][k]-bound_tot_val)
                stats.append(r_vals[i][j][k])
                ups.append(r_vals[i][j][k]+bound_tot_val)
                cumulative+=1
            df2 = pd.DataFrame.from_dict(
                {"first_term": firsts,
                "second_term": seconds,
                "cond_set": conds ,
                "lb": lbs ,
                "stat": stats ,
                "up": ups }
            )
            df = pd.concat([df,df2])
        df.to_csv(filename)

def write_on_csv_per_couple(filename, results_dict):
    df = pd.DataFrame(columns = ["first_term", "second_term", "cond_set", "lb","stat","up"])

    for k in results_dict:
        v1 = k[0]
        v2 = k[1]

        r_vals, R_hat, R_tilde, decomposed_bound, cond_sets = results_dict[k]
        firsts = [v1 for i in range(len(r_vals))]
        seconds = [v2 for i in range(len(r_vals))]
        conds = [IT_utils.convert_to_string(c) for c in cond_sets]
        lbs = (np.array(r_vals) - decomposed_bound[0]).tolist()
        stats = r_vals
        ups = (np.array(r_vals) + decomposed_bound[0]).tolist()
        df2 = pd.DataFrame.from_dict(
            {"first_term": firsts,
            "second_term": seconds,
            "cond_set": conds ,
            "lb": lbs ,
            "stat": stats ,
            "up": ups }
        )
        df = pd.concat([df,df2])
    df.to_csv(filename)


