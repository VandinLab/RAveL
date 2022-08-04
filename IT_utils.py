# -*- coding: utf-8 -*-
import traceback
from dependency_infos import dependencyInfos
from math import exp
import os
import pandas as pd
import numpy as np
from itertools import chain, combinations, repeat, groupby, product
from datetime import datetime
import warnings
from tqdm import tqdm
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import math
from scipy import stats
from sklearn.linear_model import LinearRegression
from fcit import fcit
import rad_utils
from threadpoolctl import threadpool_limits
import multiprocess as mp
from pgmpy.base import DAG

def count_elements(seq, keys = []) -> dict:
    """Utility function to count element of each histogram bar."""
    hist = {}
    for k in keys:
        hist.setdefault(k,0)
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist 

def convert_to_string(z):
    if len(z) > 1:
        z = sorted(z)
    t_present = "T" in z
    if t_present:
        z = list(z)
        z.remove("T")
        z = set(z)
    digit_only = ''.join(x for x in str(z) if x.isdigit())
    if len(digit_only) > 0: #it is an experiment with A1, A2, ...
        z = ''.join(x for x in str(sorted(z, key= lambda el: (el[1],el[0]))) if x.isalpha() or x.isdigit())
    else: # it is an experiment with A, B, ...
        z = ''.join(x for x in str(sorted(z)) if x.isalpha())
    if t_present:
        z += "T" 
    if len(z) == 0:
        z = "()" 

    return z

def powerset(iterable, get_emptyset = True, max_z_size = -1):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if max_z_size == -1:
        if len(s)>10:
            max_z_size = 4
        else:
            max_z_size = len(s)
    if get_emptyset:
        return chain.from_iterable(combinations(s, r) for r in range(max_z_size+1))
    return chain.from_iterable(combinations(s, r) for r in range(1,max_z_size+1))

def parse_IT(t,x,z, indep, ordered):
    if t<x or not ordered:
        return t+" "*indep+"!"*(not(indep))+"⊥ "+x+" | "+convert_to_string(z)
    return x+" "*indep+"!"*(not(indep))+"⊥ "+t+" | "+convert_to_string(z)

def get_argmin_dep(dep_infos, t,x, pcd, delta, max_z_size = -1):
    ps = sorted([el for el in powerset(pcd,max_z_size = max_z_size) if t not in el and x not in el], key = lambda element: (len(element),str(element)))
    # print(len(ps))
    not_enough_els = set()
    deps = []
    max_p_val = 0

    for z in ps:
        if not z in not_enough_els:
            assoc = association(dep_infos,t,x,z, delta)  
            deps.append((z,assoc))    # must add z and association
            pval = -assoc  # association returns the negative p value
            max_p_val = max(max_p_val, pval)
            if dep_infos.method != "Rademacher" and dep_infos.method != "my_stat":
                if pval == -1 and len(z)==0:  #conditioning on the emptyset creates p val = -1 => all the p vals will be -1
                    not_enough_els = ps
                    # print("preempt 1")
                    break
                elif pval == -1:
                    bad_list = [bad_z for bad_z in ps if set(z).issubset(set(bad_z))]
                    [not_enough_els.add(bad_z) for bad_z in bad_list]
                    # print("preempt 2")
                elif max_p_val == 1: # no need to get other subsets if I get a p-value of sure independence (=lowest association)
                    # print("preempt 3")
                    break
                elif max_p_val > delta and dep_infos.method: # in our example, that's a sure independence
                    # print("preempt 4")
                    break
            else:
                lb, assoc, ub = association(dep_infos,t,x,z, delta, get_rademacher_bounds=True)
                if lb<=0<=ub:   # independent since it contains 0
                    break
    dependencies = sorted(deps, key = lambda element : element[1])
    return dependencies[0][0]  # take the argmin

##
# .csv file specification for conditional dependencies
#   fields
#  first_term, second_term, cond_set, p_value, statistic, lb
#  where each term and set contains only letters and the empty set is identified by '()'
#  and the p_value is the p value of the test with independence as null hypothesis.
#  In methods that do not use this independence test (e.g. summand independence methods)
#  The field lb (=lower bound) contains the lower bound of the summand under study
##

def association(dep_infos,t,x,z, delta = -1, get_rademacher_bounds = False, smart_mode = True):
    # returns conditional association of T and X given Z

    # preprocessing of z keeping only letters or numbers
    # z = ''.join(x for x in str(sorted(z)) if x.isalpha() or x.isnumeric())
    z_str = convert_to_string(z)
    df = dep_infos.df
    method = dep_infos.method
    lookup_table = df[df["first_term"] == t]
    lookup_table = lookup_table[lookup_table["second_term"] == x]
    if len(z_str) == 0:
        z_str = "()"
    lookup_table = lookup_table[lookup_table["cond_set"] == z_str]
    if len(lookup_table)>0:
        if method == "p-value":
            return -lookup_table["p_value"].values[0]  # the stronger the association, the lower the p-value for independence
        if method == "pearson":
            return -lookup_table["p_value"].values[0]  # the stronger the association, the lower the p-value for independence
        if method == "dsep":
            return -lookup_table["p_value"].values[0]  # the stronger the association, the lower the p-value for independence
        if method == "permutation":
            return -lookup_table["p_value"].values[0]  # the stronger the association, the lower the p-value for independence
        if method == "statistic":
            return lookup_table["statistic"].values[0]
        if method == "summands":
            return lookup_table["lb"].values[0]   #no need of "-"" sign since the higher the lb, the higher the stronger the dependence (opposite of p values)
        if method == "summands_naive":
            return lookup_table["naive_lb"].values[0]
        if method == "summands_mb":
            return lookup_table["multiple_lb"].values[0]
        if method == "summands_lb":
            return lookup_table["single_lb"].values[0]
        if method == "Rademacher" or method == "my_stat":
            if get_rademacher_bounds:
                return lookup_table["lb"].values[0],lookup_table["stat"].values[0],lookup_table["up"].values[0]
            return abs(lookup_table["stat"].values[0])
        print("METHOD NOT FOUND")
    else:
        if method == "Rademacher" or method == "my_stat" or smart_mode:  # I MUST have it already calculated or I suppose I already have (for smart_mode)
            if method == "Rademacher" and smart_mode == False:
                print("RAD not calculated","x",x,"t",t,"z",z)
            return association(dep_infos,x,t,z, delta = delta, get_rademacher_bounds=get_rademacher_bounds, smart_mode = False)
        else:
            calculate_dependency_python(dep_infos, t, x, z_str, z)
        return association(dep_infos,t,x,z)

def independence(dep_infos,t,x,z, delta, verbose = False, return_string = False, ordered = False, level = 0):
    # returns True if X and T are conditionally independent given Z, meaning I cannot reject the null hypothesis
    method = dep_infos.method
    
    if not "summands" in method:
        if "Rademacher" == method or "my_stat" == method:
            lb, stat, up = association(dep_infos,t,x,z, delta, get_rademacher_bounds=True)
            res = lb <= 0 <= up # independent if bound contains 0
        else:
            res = -association(dep_infos,t,x,z, delta) > delta
    else:
        res = association(dep_infos,t,x,z, delta) <= 0   # if the lower bound is <= 0 then the vars are indep
    if verbose:
        if level == 2:
            print("        ",end = '')
        if level == 1:
            print("    ",end = '')
        if not "summands" in method:
            print(parse_IT(t,x,z, res, ordered), "p-value",-association(dep_infos,t,x,z,delta))
        else:
            print(parse_IT(t,x,z, res, ordered), method, association(dep_infos,t,x,z,delta))
    if return_string:
        return res, parse_IT(t,x,z, res, ordered)
    return res

def calculate_dependency_python(dep_infos, t, x, z_str, z):
    method = dep_infos.method
    independence_method = dep_infos.independence_method
    data_file = dep_infos.data_file
    df = dep_infos.df
    indep_file = dep_infos.indep_file
    save_every_update = dep_infos.save_every_update
    calculus_type = dep_infos.calculus_type

    if len(z) == 0:
        z = ["()"]
    else:
        z = [''.join(x for x in str(zz) if x.isdigit() or x.isalpha()) for zz in z]  # selecting only letters and numbers
    if not "summand" in method and method!= "permutation":
        if data_file != "" and independence_method!="": 
            selection_cols = [t,x]
            conditioning_cols = z

            df_data = pd.read_csv(data_file)
            p_val = -1
            cor = -1
            lb = -1

            if independence_method == "cor":   
                # t test on pearson coefficient
                z_data = []
                if z !=["()"]:
                    z_data = df_data[z]
                cor, p_val = calculate_t_test(df_data[t].values,df_data[x].values,z_data)
            else:
                dof_tx = (len(set(df_data[t]))-1)*(len(set(df_data[x]))-1)
                dof = dof_tx
                if z != ["()"]:
                    for cc in conditioning_cols:
                        dof *= len(set(df_data[cc]))
                    uniques = [row[1].values for row in df_data[conditioning_cols].iterrows()]
                    uniques = set([tuple(row.tolist()) for row in uniques])
                    df_dict = {single_val: pd.merge(pd.DataFrame([single_val],columns=conditioning_cols), df_data, on=conditioning_cols,how='left') for single_val in uniques}
                else:
                    df_dict = {"()": df_data}

                statistic = 0
                not_enough_cells = False
                for k,v in df_dict.items():
                    els_c0 = list(set(v[selection_cols[0]]))
                    els_c1 = list(set(v[selection_cols[1]]))
                    if min(len(els_c0),len(els_c1)) > 1:
                        data = [[v[(v[selection_cols[0]]==els_c0[i]) & (v[selection_cols[1]]==els_c1[j])].shape[0]
                                        for i in range(len(els_c0))] for j in range(len(els_c1))]
                        
                        lam = "undefined"
                        if independence_method == "g2":
                            lam = "log-likelihood" 
                        if independence_method == "x2":
                            lam = "pearson"
                        stat_lib, pv_lib, dof_lib, expected = chi2_contingency(data, correction=False,lambda_ = lam)  # expected measurements
                        
                        if calculus_type == "all_>=5":
                            not_enough_cells = not_enough_cells or np.min(expected) < 5
                        elif calculus_type == "only_>=5":
                            dof = dof - dof_tx * (np.min(expected)<5)

                        if (not not_enough_cells and calculus_type != "only_>=5") or (calculus_type == "only_>=5" and np.min(expected)>=5):
                            expected = expected.ravel()
                            data = np.array(data).ravel()
                            if independence_method == "g2":
                                log_mat = np.log(np.divide(data,expected))
                                log_mat[data == 0] = 0
                                statistic += 2 * np.dot(data,log_mat)

                            if independence_method == "x2":
                                statistic += np.sum(np.divide(np.power(data-expected,2),expected))
                
                if calculus_type == "only_>=5":
                    not_enough_cells = dof == 0

                if not_enough_cells:
                    p_val = -1
                    cor = -1
                    lb = -1
                else:
                    p_val = 1 - chi2.cdf(x=statistic, df = dof ) # degrees of freedom = (n_uniques T -1) * (n_uniques X -1) * product(n_uniques var_in_Z)
                    cor = statistic
                    lb = dof

            a_row = pd.DataFrame([[t,x,z_str, p_val,cor,lb]], columns = ["first_term", "second_term", "cond_set", "p_value", "statistic", "lb"])
            dep_infos.df = pd.concat([df, a_row])
            if indep_file != "" and (save_every_update or dep_infos.save_counter==1000):
                save_IT_dataframe(dep_infos)
                dep_infos.save_counter = 0
                
            dep_infos.save_counter += 1
    elif method == "permutation":
        raise Exception("permutation_based_IT not implemented yet")
        # permutation_based_IT(dep_infos, t, x, z, z_str)
    else:
        raise Exception("CALCULUS OF SUMMAND AD HOC NOT IMPLEMENTED YET")

def calculate_residuals(target,data):
    reg = LinearRegression().fit(data, target)
    residuals = target - reg.predict(data)
    # print(residuals.shape)
    return residuals

def calculate_t_test(x,y,z=[]):
    # returns stat and pvalue
    if len(z) == 0:
        return stats.pearsonr(x,y)
    
    xs = np.reshape(x,(-1,1))
    ys = np.reshape(y,(-1,1))
    if len(z.shape) == 1:
        zs = np.reshape(z,(-1,1))
    else:
        zs = z
    xs -= np.mean(xs)
    xs /= np.std(xs)
    ys -= np.mean(ys)
    ys /= np.std(ys)
    e_x = np.reshape(calculate_residuals(xs,zs), (1,-1))[0]
    e_y = np.reshape(calculate_residuals(ys,zs), (1,-1))[0]
    dof = len(xs) - 2 - zs.shape[1]
    r_stat = stats.pearsonr(e_x,e_y)[0]
    stat = r_stat/math.sqrt((1-r_stat**2)/dof)
    return (r_stat, (1-stats.t.cdf(abs(stat),dof))*2)

def save_IT_dataframe(dep_infos):
    dep_infos.df.to_csv(dep_infos.indep_file,sep = ";")

def get_mb(v, dag):
    return dag.get_markov_blanket(v)
    
def get_pc(v, dag):
    res = dag.get_parents(v) + dag.get_children(v)
    res = [e for e in set(res)]
    return res

def get_parents(v, dag):
    return dag.get_parents(v)

