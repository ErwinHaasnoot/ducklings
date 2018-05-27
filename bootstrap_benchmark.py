import time

import numpy as np

from ceer_logn import ceer_logn, ceer_logn_bootstrap, bootstrap_draw_sorted
from eer_logn_bootstrap import eer_logn_bootstrap
from bob.measure import eer_rocch

np.random.seed(0)
m = 10000


def bootstrap_naive_rocch(impostors, genuines):
    imps = bootstrap_draw_sorted(impostors)
    gens = bootstrap_draw_sorted(genuines)
    return eer_rocch(imps, gens)


def bootstrap_sorted_single(impostors, genuines):
    imps = bootstrap_draw_sorted(impostors)
    gens = bootstrap_draw_sorted(genuines)
    eer = ceer_logn(imps, gens, is_sorted=True)
    return eer


def bootstrap_naive_single(impostors, genuines):
    imps = np.random.choice(impostors, len(impostors))
    gens = np.random.choice(genuines, len(genuines))
    imps.sort()
    gens.sort()
    eer = ceer_logn(imps, gens, is_sorted=True)
    return eer


with open('results.csv', 'w') as f:
    for n in [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8][:12]:
        print("Sample size: %s, bootstrap iterations: %s" % (n, m))
        n_imp = n
        impostors = np.array(sorted(np.random.rand(int(n_imp)) - .375))
        genuines = np.array(sorted(np.random.rand(int(n)) + .375))

        eer1 = ceer_logn(np.array(impostors), np.array(genuines), is_sorted=True)
        print("actual eer:", eer1)

        start = time.time()
        eers_smart_cython = ceer_logn_bootstrap(impostors, genuines, is_sorted=True, n_iterations=m)
        stop = time.time() - start
        eers_smart_cython = sorted(eers_smart_cython)
        print("bootstrap smart-cython", stop,
              "seconds(%s,%s)" % (eers_smart_cython[int(m * 0.025)], eers_smart_cython[int(m * 0.975)]))
        f.write("%s,%s\n" % ("smart_cython", stop))

        start = time.time()
        eers_smart = [eer_logn_bootstrap(impostors, genuines, is_sorted=True) for _ in range(m)]
        stop = time.time() - start
        eers_smart = sorted(eers_smart)
        print("bootstrap smart pure-python", time.time() - start,
              "seconds (%s,%s)" % (eers_smart[int(m * 0.025)], eers_smart[int(m * 0.975)]))
        f.write("%s,%s\n" % ("smart_pure", stop))

        start = time.time()
        eers_naive = [bootstrap_sorted_single(impostors, genuines) for _ in range(m)]
        stop = time.time() - start
        eers_naive = sorted(eers_naive)
        print("bootstrap sorted cython FEER", stop,
              "seconds(%s,%s)" % (eers_naive[int(m * 0.025)], eers_naive[int(m * 0.975)]))
        f.write("%s,%s\n" % ("sorted_feer", stop))

        start = time.time()
        eers_naive = [bootstrap_naive_single(impostors, genuines) for _ in range(m)]
        stop = time.time() - start
        eers_naive = sorted(eers_naive)
        print("bootstrap naive cython FEER", stop,
              "seconds(%s,%s)" % (eers_naive[int(m * 0.025)], eers_naive[int(m * 0.975)]))
        f.write("%s,%s\n" % ("naive_feer", stop))

        # start = time.time()
        # eers_bob_rocch = [bootstrap_naive_rocch(impostors, genuines) for _ in range(m)]
        # stop = time.time() - start
        # eers_bob_rocch = sorted(eers_bob_rocch)
        # print("bootstrap naive bob eer_rocch", stop,
        #       "seconds(%s,%s)" % (eers_bob_rocch[int(m * 0.025)], eers_bob_rocch[int(m * 0.975)]))
        # f.write("%s,%s\n" % ("eer_rocch", stop))
