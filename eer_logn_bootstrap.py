from bisect import bisect_right, bisect_left
from math import ceil
import numpy as np


def eer_logn_bootstrap(impostors, genuines, is_sorted=False):
    if not is_sorted:
        impostors[::-1].sort()
        genuines[::-1].sort()
    genlen = len(genuines)
    implen = len(impostors)
    d_genlen = genlen - 1
    d_implen = implen - 1
    # np.random.seed(seed)

    # Initialize interest ranges. One for the sampled set, one for the bootstrapped set.
    gen_min, gen_max = bs_gen_min, bs_gen_max = (0, d_genlen)
    imp_min, imp_max = bs_imp_min, bs_imp_max = (0, d_implen)

    # distribution draw
    def emp_dist(k_stat, k_size, range_min, range_max):
        return int(round(np.random.beta(k_stat+1,k_size-k_stat+1) * (range_max - range_min) + range_min))

    gen_bs = [None] * genlen
    imp_bs = [None] * implen

    gen_ll_min = gen_ll_max = imp_ll_min = imp_ll_max = None
    centre_offset = 0
    head_gen_ll = None
    head_imp_ll = None
    while bs_gen_max - bs_gen_min > 1 or bs_imp_max - bs_imp_min > 1:
        kg1 = (bs_gen_max + bs_gen_min) // 2
        kp1 = (bs_imp_max + bs_imp_min) // 2
        kg2 = kg1 + 1
        kp2 = kp1 + 1
        if gen_bs[kg1] is None:
            gen_bs[kg1] = (kg2,
                           emp_dist(kg1 - bs_gen_min - centre_offset, bs_gen_max - bs_gen_min - centre_offset,
                                    gen_min, gen_max))
            if gen_ll_min is not None:
                gen_bs[gen_ll_min] = (kg1, gen_bs[gen_ll_min][1])
        else:
            gen_bs[kg1] = (kg2, gen_bs[kg1][1])

        if gen_bs[kg2] is None:
            gen_bs[kg2] = (gen_ll_max, emp_dist(0, bs_gen_max - kg1, gen_bs[kg1][1], gen_max))

        if imp_bs[kp1] is None:
            imp_bs[kp1] = (kp2,
                           emp_dist(kp1 - bs_imp_min - centre_offset, bs_imp_max - bs_imp_min - centre_offset,
                                    imp_min, imp_max))
            if imp_ll_min is not None:
                imp_bs[imp_ll_min] = (kp1, imp_bs[imp_ll_min][1])
        else:
            imp_bs[kp1] = (kp2, imp_bs[kp1][1])

        if imp_bs[kp2] is None:
            imp_bs[kp2] = (imp_ll_max, emp_dist(0, bs_imp_max - kp1, imp_bs[kp1][1], imp_max))

        ig1 = gen_bs[kg1][1]
        ip2 = d_implen - imp_bs[kp1][1]
        ig2 = gen_bs[kg2][1]
        ip1 = d_implen - imp_bs[kp2][1]

        if head_gen_ll is None:
            head_gen_ll = kg1
        if head_imp_ll is None:
            head_imp_ll = kp1

        if genuines[ig1] > impostors[ip2]:
            bs_gen_max = gen_ll_max = kg1
            bs_imp_max = imp_ll_max = kp1
            gen_max = ig1
            imp_max = imp_bs[kp2][1]
            continue

        if impostors[ip1] > genuines[ig2]:
            bs_gen_min = kg1
            bs_imp_min = kp1
            gen_min = ig2
            imp_min = imp_bs[kp1][1]
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
        rmax, imin = gen_bs[rmin]
        # Minimize the range across which to search to for the
        while rmax is not None:
            imax = gen_bs[rmax][1]
            if genuines[imin] < sp1 < genuines[imax]:
                break
            rmin = rmax
            rmax, imin = gen_bs[rmin]
        # Use binary search to find the actual point
        if rmax is None:
            rmax = d_genlen
            imax = d_genlen
        imid = (rmin + rmax) // 2
        while rmax - rmin > 1:
            rmid = (rmin + rmax) // 2
            imid = emp_dist(rmid - rmin, rmax - rmin, imin, imax)
            # print((rmin,rmid,rmax),imid)
            vmid = genuines[imid]
            gen_bs[rmid] = (rmax if gen_bs[rmax] is not None else None, imid)
            gen_bs[rmin] = (rmid, gen_bs[rmin][1])
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
        rmax, imin = gen_bs[rmin]
        # Minimize the range across which to search to for the
        while rmax is not None:
            imax = gen_bs[rmax][1]
            if genuines[imin] < sp2 < genuines[imax]:
                break
            rmin = rmax
            rmax, imin = gen_bs[rmin]
        # Use binary search to find the actual point
        if rmax is None:
            rmax = d_genlen
            imax = d_genlen
        imid = (rmin + rmax) // 2
        while rmax - rmin > 1:
            rmid = (rmin + rmax) // 2
            imid = emp_dist(rmid - rmin, rmax - rmin, imin, imax)
            vmid = genuines[imid]
            gen_bs[rmid] = (rmax if gen_bs[rmax] is not None else None, imid)
            gen_bs[rmin] = (rmid, gen_bs[rmin][1])
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
        frr_1 = (ig1_ / d_genlen)
        far_1 = 1 - (ip1 / d_implen)
        frr_2 = ig2_ / d_genlen
    else:
        rmin = head_imp_ll
        rmax, imin = imp_bs[head_imp_ll]
        # Minimize the range across which to search to for the
        while rmax is not None:
            imax = imp_bs[rmax][1]
            if impostors[d_implen - imin] > sg1 > impostors[d_implen - imax]:
                break
            rmin = rmax
            rmax, imin = imp_bs[rmin]
        # Use binary search to find the actual point
        if rmax is None:
            rmax = d_implen
            imax = d_implen
        imid = (rmin + rmax) // 2
        while rmax - rmin > 1:
            rmid = (rmin + rmax) // 2
            imid = emp_dist(rmid - rmin, rmax - rmin, imin, imax)
            # print((rmin,rmid,rmax),imid)
            vmid = impostors[d_implen - imid]
            imp_bs[rmid] = (rmax if imp_bs[rmax] is not None else None, imid)
            imp_bs[rmin] = (rmid, imp_bs[rmin][1])
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
        frr_1 = (ig1 / d_genlen)
        far_1 = (ip1_ / d_implen)
        frr_2 = ig2 / d_genlen
    if far_1 - frr_2 == 0:
        return frr_2
    elif (far_1 - frr_1) / (far_1 - frr_2) <= 0:
        return far_1
    else:
        return frr_2