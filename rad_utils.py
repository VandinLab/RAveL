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

# def couple_divider(filename, tot_vars, delta, max_z_size, ds_folder, it_folder):
#     couples = [e for e in combinations(tot_vars, 2)]
#     deps = []
#     indeps = []
#     oracle = False
#     if oracle:
#         n_rep = 4
#         d_struc = "[A1][B1][C1|A1:B1]" + "".join(["[A"+str(i+1)+"|C"+str(i)+"][B"+str(i+1)+"|C"+str(i)+"][C"+str(i+1)+"|A"+str(i+1)+":B"+str(i+1)+"]" for i in range(1,n_rep+1)])
#         for v1,v2 in couples:
#             if v2 in IT_utils.get_pc(v1, d_struc):
#                 deps.append((v1,v2))
#             else:
#                 indeps.append((v1,v2))
#     else:
#         data_file = ds_folder+filename
#         indep_file = it_folder + "cor_"+filename
        
#         if os.path.exists(indep_file):
#             try:
#                 indep_df = pd.read_csv(indep_file,sep = ";")
#             except:
#                 print("!!!",indep_file)
#                 indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
#                 indep_df.to_csv(indep_file, sep = ";")
#         else:
#             indep_df = pd.DataFrame(columns=["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
#             indep_df.to_csv(indep_file, sep = ";")

#         dep_infos = dependencyInfos(indep_df, method="p-value", independence_method="cor", data_file=data_file, 
#                         indep_file=indep_file, save_every_update=False, independence_language="python")
#         for v1,v2 in tqdm(couples):
#         # for v1,v2 in couples:
#             other_vars = [v for v in tot_vars if v!= v1 and v!=v2]
#             z = IT_utils.get_argmin_dep(dep_infos,v1,v2,other_vars,delta,max_z_size)
#             if not IT_utils.independence(dep_infos, v1,v2,z, delta):
#                 deps.append((v1,v2))
#             else:
#                 indeps.append((v1,v2))
#         IT_utils.save_IT_dataframe(dep_infos)
#     return deps, indeps

# def divide_into_datasets(total_ds, n_vars):
#     assert total_ds.shape[0]%n_vars == 0
#     return np.array([total_ds[i*n_vars:i*n_vars+n_vars] for i in range(int(total_ds.shape[0]/n_vars))])

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

def calculate_summands(xs,ys,zs = np.array([]), corrected_bound = False, my_stat = False, normalize_using_maxes = False):
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
        r_summs = np.multiply(xs,ys) / np.power(np.maximum(np.abs(xs),np.abs(ys)),2)
        if max(r_summs) >1:
            print(np.max(r_summs),np.mean(r_summs))
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
        r_mat_uncond = np.array([calculate_summands(xs = data[t[0]].values, ys = data[t[1]].values, corrected_bound = corrected_bound, my_stat=my_stat, normalize_using_maxes = normalize_using_maxes) for t in couples]).T
        if not corrected_bound:
            r_vals = np.sum(r_mat_uncond,axis=0).tolist()
        else:
            r_vals = np.mean(r_mat_uncond,axis=0).tolist()

        m = r_mat_uncond.shape[0]
        sums = np.matmul(sigma_cp.T,r_mat_uncond)  # n x m * m x k = n x k matrix
        sups_vector = np.max(sums,axis = 1)
        step_size = 1000
        for i in range(math.ceil(len(triplets)/step_size)):
            #calculate stats for the block
            # r_mat_cond = np.array([calculate_summands(xs = data[t[0][0]].values, ys = data[t[0][1]].values, zs = compose_dataset(data.values,col, selection=t[1]), t, corrected_bound = corrected_bound, my_stat=my_stat, normalize_using_maxes=normalize_using_maxes) 
            #     for t in triplets[i*step_size:(i+1)*step_size]]).T
            r_mat_cond = np.array([calculate_summands(xs = data[t[0][0]].values, ys = data[t[0][1]].values, zs = compose_dataset(data.values,col, selection=t[1]), corrected_bound = corrected_bound, my_stat=my_stat, normalize_using_maxes=normalize_using_maxes) 
                for t in triplets[i*step_size:(i+1)*step_size]]).T
            sums_cond = np.matmul(sigma_cp.T,r_mat_cond)
            sups_vector = np.maximum(sups_vector, np.max(sums_cond,axis = 1))
            #update stats
            if not corrected_bound:
                r_vals.append(np.sum(r_mat_cond,axis=0).tolist())
            else:
                r_vals.append(np.mean(r_mat_cond,axis=0).tolist())
    return variable, r_vals, sups_vector, couples, triplets


def calculate_stat_SD_bounds(filename, tot_vars, delta, couples_dict, n_sigma = 1000, corrected_bound = False, n_proc = 1, debug = False, max_z_size = 2, normalize_using_maxes = False, corrected_c = False, my_stat = False):
    if not corrected_c and not my_stat:
        print("If you're using the Pearson coefficient you must activate the correction for c and z since\
            each element of the sum iss not in [-1,1]")
        raise Exception
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


def get_file_prefixes(computation_type, normalize_using_maxes = False, corrected_c = False, my_stat = False):
    assert computation_type == "total"
    if my_stat:
        return "my_stat_total_"
    r = ""
    if normalize_using_maxes:
        r+="maxN"
    if corrected_c:
        r+="corrC"
    r += "RAD_total_"
    return r