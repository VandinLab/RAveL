import IT_utils

## Library for Local Causal Discovery methods 

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
