# %%
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from ceer_logn import ceer_logn
from ceer_logn_bootstrap import eer_logn_bootstrap

n = 121
n_imp = 2 * n
tries = 20000
for s in range(tries):
    np.random.seed(s)
    print("Seeded with : %s" % s)
    rands_imp = np.random.rand(1, int(n_imp)) - .375
    rands_gen = np.random.rand(1, int(n)) + .375
    rands_imp[::-1].sort(axis=1)
    rands_gen[::-1].sort(axis=1)
    errors = [0] * 1
    for i in range(1):
        impostors = rands_imp[i, :]
        genuines = rands_gen[i, :]
        eer1 = ceer_logn(impostors, genuines, is_sorted=True)
        eer2 = eer_logn_bootstrap(impostors, genuines, is_sorted=True,seed=s)
        if eer1 - eer2 > 1 / (2 * n):
            print(eer1 - eer2)
            print(eer1)
            print(eer2)
            # print("impostors,genuines = (%s, %s)" % (list(impostors),list(genuines)))

            raise Exception("Difference too high for seed %s" % s)
        # print(ceer_logn(impostors,genuines,is_sorted=True))

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# n = 50000
# k = int(n/2)
# m = int(1e3)
# S = list(range(n))
#
# j = [0] * m
# j_ = [0] * m
# S_ = np.random.choice(S,(n,m))
# S_.sort(axis=0)
# j = S_[k,:]
# j_ = (np.random.beta(k,n+1-k,(m,1)) * (n+1)).round()
#
# plt.title('n=%s, k=%s, m=%s, ordered' % (n,k,m))
# plt.hist(j,alpha=.6,label='Bootstrapped')
# plt.hist(j_,alpha=.6,label='Beta dist drawn')
# plt.xlabel('k')
# plt.ylabel('count')
# plt.legend()
# plt.show()
#
#
