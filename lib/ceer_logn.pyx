# %%cython
import bisect

import cython
import numpy as np
cimport numpy as np
from libc.math cimport floor,ceil
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import pairwise_distances

# ceil = math.ceil
# floor = math.floor

bisect_left = bisect.bisect_left
bisect_right = bisect.bisect_right

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
        impostors[::-1].sort()
        genuines[::-1].sort()

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
cpdef shift_logn(np.ndarray[np.float64_t,ndim=1] impostors,np.ndarray[np.float64_t,ndim=1] genuines, is_sorted=False):
    """
    O(log n) function for calculating EERs. Consists of two phases.
    1. Determine the general location of the intersection between the EER line and the 
    :param impostors: 
    :param genuines: 
    :param is_sorted: 
    :return: 
    """

    """Initialize required params"""
    cdef int implen, genlen, ig1_,ig2_,ip1_,ip2_, ig1,ip2,ig2,ip1,kp1
    cdef float sg1,sg2,sp1,sp2
    cdef float far_1,frr_1,frr_2,
    cdef float d_implen,d_genlen

    """Sort both impostor and genuine lists in ascending order, if not already done"""
    if not is_sorted:
        impostors[::-1].sort()
        genuines[::-1].sort()

    """Get """
    genlen = genuines.shape[0]
    implen = impostors.shape[0]
    d_genlen = genlen - 1.
    d_implen = implen - 1.

    if impostors[implen - 1] < genuines[0]:
        return 0.0
    elif genuines[genlen - 1] < impostors[0]:
        return 1.0

    gen_min = 0
    imp_min = 0
    gen_max = genlen -1
    imp_max = implen -1

    while gen_max - gen_min != 1 and imp_max - imp_min != 1:
        kp1 = (imp_max - imp_min) / 2 + imp_min

        ig1 = (gen_max - gen_min) / 2 + gen_min
        ip1 = implen - 2 - kp1
        ig2 = ig1 + 1
        ip2 = ip1 + 1

        if genuines[ig1] > impostors[ip2]:
            gen_max = ig1
            imp_max = kp1
        elif impostors[ip1] > genuines[ig2]:
            # dep *= 2
            # pos -= 1. / dep # -> .25
            gen_min = ig1
            imp_min = kp1
        else:
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