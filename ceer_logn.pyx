# cython:
cimport cython
import bisect

import numpy as np
cimport numpy as np
from libc.math cimport ceil

bisect_left = bisect.bisect_left
bisect_right = bisect.bisect_right

from cython.view cimport array
import random

def stuff():
    pass


beta = np.random.beta

cdef packed struct LinkedNode:
    int next
    int val
    int round

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef ceer_logn_bootstrap(np.ndarray[np.float64_t,ndim=1] impostors,np.ndarray[np.float64_t,ndim=1] genuines, is_sorted=False,int n_iterations=10):
    cdef int genlen,implen,d_implen,d_genlen,i,c_iter
    cdef int gen_min,gen_max, bs_gen_min, bs_gen_max
    cdef int imp_min,imp_max, bs_imp_min, bs_imp_max
    cdef int rmax,rmin,rmid,imax,imin,imid
    genlen = genuines.shape[0]
    implen = impostors.shape[0]
    cdef LinkedNode[:] gen_bs = array((genlen,),itemsize=sizeof(LinkedNode),format='iii')
    for i in range(genlen):
        gen_bs[i].next = -1
        gen_bs[i].round = -1
    cdef LinkedNode[:] imp_bs = array((implen,),itemsize=sizeof(LinkedNode),format='iii')
    for i in range(implen):
        imp_bs[i].next = -1
        imp_bs[i].round = -1
    cdef int kg1,kg2,kp1,kp2,ip1,ip2,ig1,ig2
    cdef int imp_ll_min,gen_ll_min,imp_ll_max,gen_ll_max,centre_offset
    cdef float frr_1,frr_2,far_1, vmid, sp1,sp2,sg1,sg2
    cdef float[:] eers = array((n_iterations,), itemsize=sizeof(float),format='f')
    if not is_sorted:
        impostors[::-1].sort()
        genuines[::-1].sort()
    d_genlen = genlen - 1
    d_implen = implen - 1
    # np.random.seed(seed)

    for c_iter in range(n_iterations):

        # Initialize interest ranges. One for the sampled set, one for the bootstrapped set.
        gen_min, gen_max = bs_gen_min, bs_gen_max = (0, d_genlen)
        imp_min, imp_max = bs_imp_min, bs_imp_max = (0, d_implen)

        # distribution draw
        gen_ll_min = gen_ll_max = imp_ll_min = imp_ll_max = -1
        centre_offset = 0
        head_gen_ll = -1
        head_imp_ll = -1
        while bs_gen_max - bs_gen_min > 1 or bs_imp_max - bs_imp_min > 1:
            kg1 = (bs_gen_max + bs_gen_min) // 2
            kp1 = (bs_imp_max + bs_imp_min) // 2
            kg2 = kg1 + 1
            kp2 = kp1 + 1
            if gen_bs[kg1].round != c_iter:
                gen_bs[kg1].next = kg2
                gen_bs[kg1].val = round(beta(kg1 - bs_gen_min - centre_offset +1,bs_gen_max- kg1 +1) * (gen_max - gen_min) + gen_min)
                gen_bs[kg1].round = c_iter
                if gen_ll_min != -1:
                    gen_bs[gen_ll_min].next = kg1
            else:
                gen_bs[kg1].next = kg2

            if gen_bs[kg2].round != c_iter:
                gen_bs[kg2].next = gen_ll_max
                gen_bs[kg2].val = round(beta(1,bs_gen_max - kg1+1) * (gen_max -  gen_bs[kg1].val) +  gen_bs[kg1].val)
                gen_bs[kg2].round = c_iter

            if imp_bs[kp1].round != c_iter:
                imp_bs[kp1].next = kp2
                imp_bs[kp1].val = round(beta(kp1 - bs_imp_min - centre_offset +1,bs_imp_max- kp1 +1) * (imp_max - imp_min) + imp_min)
                imp_bs[kp1].round = c_iter
                if imp_ll_min != -1:
                    imp_bs[imp_ll_min].next = kp1
            else:
                imp_bs[kp1].next = kp2

            if imp_bs[kp2].round != c_iter:
                imp_bs[kp2].next = imp_ll_max
                imp_bs[kp2].val = round(beta(1,bs_imp_max - kp1+1) * (imp_max -  imp_bs[kp1].val) +  imp_bs[kp1].val)
                imp_bs[kp2].round = c_iter

            ig1 = gen_bs[kg1].val
            ip2 = d_implen - imp_bs[kp1].val
            ig2 = gen_bs[kg2].val
            ip1 = d_implen - imp_bs[kp2].val

            if head_gen_ll == -1:
                head_gen_ll = kg1
            if head_imp_ll == -1:
                head_imp_ll = kp1

            if genuines[ig1] > impostors[ip2]:
                bs_gen_max = gen_ll_max = kg1
                bs_imp_max = imp_ll_max = kp1
                gen_max = ig1
                imp_max = imp_bs[kp2].val
                continue

            if impostors[ip1] > genuines[ig2]:
                bs_gen_min = kg1
                bs_imp_min = kp1
                gen_min = ig2
                imp_min = imp_bs[kp1].val
                gen_ll_min = kg2
                imp_ll_min = kp2
                if kg1 < head_gen_ll:
                    head_gen_ll = kg1
                if kp1 < head_imp_ll:
                    head_imp_ll = kp1
                centre_offset = 1
                continue
            # print("Fell through")
            break
        sg1, sg2, sp1, sp2 = genuines[ig1], genuines[ig2], impostors[ip1], impostors[ip2]

        # Find FRR and FARs closest to the EER line.
        if sg1 >= sp1:
            # print("iggi")
            # ig1 = find_and_fill_gen(sp1)
            rmin = head_gen_ll
            rmax = gen_bs[rmin].next
            imin = gen_bs[rmin].val

            # Minimize the range across which to search to for the
            while rmax != -1:
                imax = gen_bs[rmax].val
                if genuines[imin] < sp1 < genuines[imax]:
                    break
                rmin = rmax
                rmax = gen_bs[rmin].next
                imin = gen_bs[rmin].val
            # Use binary search to find the actual point
            if rmax == -1:
                rmax = d_genlen
                imax = d_genlen
            imid = (rmin + rmax) // 2
            while rmax - rmin > 1:
                rmid = (rmin + rmax) // 2
                imid = round(beta(rmid - rmin+1,rmax - rmid +1) * (imax - imin) + imin)
                # print((rmin,rmid,rmax),imid)
                vmid = genuines[imid]
                gen_bs[rmid].next = rmax if gen_bs[rmax].round == c_iter else -1
                gen_bs[rmid].val = imid
                gen_bs[rmin].next = rmid
                if vmid == sp1:
                    break
                    pass
                elif vmid > sp1:
                    rmax = rmid
                    imax = imid
                elif vmid < sp1:
                    rmin = rmid
                    imin = imid
            ig1_ = imid

            rmin = head_gen_ll
            rmax = gen_bs[rmin].next
            imin = gen_bs[rmin].val
            # Minimize the range across which to search to for the
            while rmax != -1:
                imax = gen_bs[rmax].val
                if genuines[imin] < sp2 < genuines[imax]:
                    break
                rmin = rmax
                rmax = gen_bs[rmin].next
                imin = gen_bs[rmin].val
            # Use binary search to find the actual point
            if rmax == -1:
                rmax = d_genlen
                imax = d_genlen
            imid = (rmin + rmax) // 2
            while rmax - rmin > 1:
                rmid = (rmin + rmax) // 2
                imid = round(beta(rmid - rmin+1,rmax - rmid +1) * (imax - imin) + imin)
                vmid = genuines[imid]
                gen_bs[rmid].next = rmax if gen_bs[rmax].round == c_iter else -1
                gen_bs[rmid].val = imid
                gen_bs[rmin].next = rmid
                if vmid == sp2:
                    break
                    pass
                elif vmid > sp2:
                    rmax = rmid
                    imax = imid
                elif vmid < sp2:
                    rmin = rmid
                    imin = imid
            ig2_ = imid
            frr_1 = <float>ig1_ / <float>d_genlen
            far_1 = 1. - (<float>ip1 / <float>d_implen)
            frr_2 = <float>ig2_ / <float>d_genlen
        else:
            rmin = head_imp_ll
            rmax = imp_bs[rmin].next
            imin = imp_bs[rmin].val
            # Minimize the range across which to search to for the
            while rmax != -1:
                imax = imp_bs[rmax].val
                if impostors[d_implen - imin] > sg1 > impostors[d_implen - imax]:
                    break
                rmin = rmax
                rmax = imp_bs[rmin].next
                imin = imp_bs[rmin].val
            # Use binary search to find the actual point
            if rmax == -1:
                rmax = d_implen
                imax = d_implen
            imid = (rmin + rmax) // 2
            while rmax - rmin > 1:
                rmid = (rmin + rmax) // 2
                imid = round(beta(rmid - rmin+1,rmax - rmid +1) * (imax - imin) + imin)
                vmid = impostors[d_implen - imid]
                imp_bs[rmid].next = rmax if imp_bs[rmin].round == c_iter else -1
                imp_bs[rmid].val = imid
                imp_bs[rmin].next = rmid
                if vmid == sg1:
                    break
                    pass
                elif vmid < sg1:
                    rmax = rmid
                    imax = imid
                elif vmid > sg1:
                    rmin = rmid
                    imin = imid
            ip1_ = imid
            frr_1 = <float>ig1 / <float>d_genlen
            far_1 = <float>ip1_ / <float>d_implen
            frr_2 = <float>ig2 / <float>d_genlen
        if far_1 - frr_2 == 0:
            eers[c_iter] = frr_2
        elif (far_1 - frr_1) / (far_1 - frr_2) <= 0:
            eers[c_iter] = far_1
        else:
            eers[c_iter] = frr_2

    return eers

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef ceer_logn(np.ndarray[np.float64_t,ndim=1] impostors,np.ndarray[np.float64_t,ndim=1] genuines, is_sorted=False):
    """
    O(log n) function for calculating EERs. Consists of two phases.
    1. Determine the general location of the intersection between the EER line and the 
    :param impostors: 
    :param genuines: 
    :param is_sorted: 
    :return: 
    """

    """Initialize required params"""
    cdef int implen, genlen, ig1_,ig2_,ip1_,ip2_, ig1,ip2,ig2,ip1
    cdef float sg1,sg2,sp1,sp2,pos, dep
    cdef float far_1,frr_1,frr_2,
    cdef float d_implen,d_genlen, imp_pos, gen_pos

    """Sort both impostor and genuine lists in ascending order, if not already done"""
    if not is_sorted:
        impostors[::-1].sort(axis=0)
        genuines[::-1].sort(axis=0)

    """Get """
    genlen = genuines.shape[0]
    implen = impostors.shape[0]
    d_genlen = genlen - 1.
    d_implen = implen - 1.
    pos = .5
    dep = 1.
    if impostors[implen - 1] < genuines[0]:
        return 0.0
    elif genuines[genlen - 1] < impostors[0]:
        return 1.0

    while True:
        ig1 = <int>((1. - pos) * d_genlen)
        ip2 = <int>ceil(pos * d_implen)
        if genuines[ig1] > impostors[ip2]:
            dep *= 2
            pos += 1. / dep
            continue
        ig2 = <int>ceil((1. - pos) * d_genlen)
        ip1 = <int>(pos * d_implen)
        if impostors[ip1] > genuines[ig2]:
            dep *= 2
            pos -= 1. / dep
            continue

        sg1, sg2, sp1, sp2 = genuines[ig1], genuines[ig2], impostors[ip1], impostors[ip2]
        break
    # Find FRR and FARs closest to the EER line.

    if sg1 >=  sp1:
        ig1_ = bisect_right(genuines, sp1)
        ig2_ = bisect_left(genuines, sp2) - 1
        frr_1 = (ig1_ / d_genlen)
        far_1 = 1. - (ip1 / d_implen)
        frr_2 = ig2_ / d_genlen

    else:
        ip1_ = bisect_right(impostors, sg1)
        frr_1 = (ig1 / d_genlen)
        far_1 = 1. - (ip1_ / d_implen)
        frr_2 = ig2 / d_genlen

    if far_1 - frr_2 == 0:
        return frr_2
    elif (far_1 - frr_1) / (far_1 - frr_2) <= 0:
        return far_1
    else:
        return frr_2

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef bootstrap_draw_sorted(np.ndarray[np.float64_t,ndim=1] a,int samples=-1):
    cdef int i, j, d, r, c
    c = 0
    if samples == -1:
        samples = len(a)
    cdef int[:] counts = array((samples,),itemsize=sizeof(int),format='i')
    for i in range(samples):
        counts[i] = 0
    cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(samples)
    ran = random.randint
    cdef np.ndarray[np.long_t,ndim=1] rands = np.random.randint(0,samples,samples)
    for i in range(samples):
        counts[rands[i]] += 1
    for i in range(samples):
        d = counts[i]
        for j in range(d):
            out[c] = a[i]
            c += 1
    return out




