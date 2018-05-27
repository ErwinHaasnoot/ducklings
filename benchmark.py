# %%
import math
import time

from scipy import interpolate
from scipy.optimize import brentq
# from sklearn.metrics import roc_curve
from bob.measure import eer_rocch
import numpy as np
import matplotlib.pyplot as plt

from ceer_logn import ceer_logn, shift_logn


def eer_brentq(impostors, genuines, is_sorted=False):
    """
    This EER function uses the roc_curve function in sklearn to calculate all important combinations of TPR and FPR values and then uses a form of linear interpolation to estimate the EER
    :param impostors: An iterable of impostor scores
    :param genuines: An iterable of genuine scores
    :param is_sorted: Bool indicating if iterable is sorted
    :return: EER
    """
    scores = list(genuines) + list(impostors)
    labels = [1] * len(genuines) + [-1] * len(impostors)
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)


def eer_farfrr(impostors, genuines, is_sorted=False):
    """
    This EER function uses the roc_curve function in sklearn to calculate all important combinations of TPR and FPR values and then search for the point where the function FAR-FRR change sign.
    :param impostors: An iterable of impostor scores
    :param genuines: An iterable of genuine scores
    :param is_sorted: Bool indicating if iterable is sorted
    :return: EER
    """
    scores = list(genuines) + list(impostors)
    labels = [1] * len(genuines) + [-1] * len(impostors)
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=True)
    dists = (1 - tpr) - fpr
    idx1 = np.where(dists == dists[dists >= 0].min())
    idx2 = np.where(dists == dists[dists < 0].max())

    x = ((1 - tpr)[idx1], fpr[idx1])
    y = ((1 - tpr)[idx2], fpr[idx2])
    a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
    return x[0] + a * (y[0] - x[0])


def eer_bob_rocch(impostors, genuines, is_sorted=False):
    """
    EER function that wraps LDIAP's bob.measure toolkit EER function.
    :param impostors: An iterable of impostor scores
    :param genuines: An iterable of genuine scores
    :param is_sorted: Bool indicating if iterable is sorted
    :return: EER
    """
    return eer_rocch(impostors, genuines)


def eer_logn(impostors, genuines, is_sorted=False):
    """
    EER function that wraps Haasnoot2018's O(log n) EER function
    :param impostors: An iterable of impostor scores
    :param genuines: An iterable of genuine scores
    :param is_sorted: Bool indicating if iterable is sorted
    :return: EER
    """
    return ceer_logn(impostors, genuines, is_sorted=is_sorted)


def eer_bootstrap(impostors, genuines, is_sorted=False):
    return shift_logn(impostors, genuines, is_sorted=is_sorted)

try:
    import line_profiler
except ImportError:
    print("No line profiler, skipping test.")
    import sys
    sys.exit(0)


def assert_stats(profile, name):
    profile.print_stats()
    stats = profile.get_stats()
    assert len(stats.timings) > 0, "No profile stats."
    for key, timings in stats.timings.items():
        if key[-1] == name:
            assert len(timings) > 0
            break
    else:
        raise ValueError("No stats for %s." % name)

# %%
n_rounds = 100
start_total = time.time()

functions = [eer_logn]
sizes = [1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8][:15]
sorteds = [True]
out = np.zeros((len(sizes), len(sorteds), len(functions), n_rounds))



for i, size in enumerate(sizes):
    size_start = time.time()
    for j, is_sorted in enumerate(sorteds):
        for k in range(n_rounds):
            rands = np.random.rand(2, int(size))
            if is_sorted:
                rands[::-1].sort(axis=1)
            impostors = rands[0, :] - .375
            genuines = rands[1, :] + .375
            eers = [0]*len(functions)
            for l, fn in enumerate(functions):
                start = time.time()
                eers[l] = fn(impostors, genuines, is_sorted=is_sorted)
                out[i, j, l, k] = time.time() - start

            if abs(eers[0] - eers[1]) > 1e-4:
                print(functions[0],eers[0])
                print(functions[1],eers[1])
                print("Eers are not the same!")
                raise Exception()
    print("Finished size %s in %s seconds" % (size, time.time() - size_start))
# %%
means = out.mean(axis=3)
# # Sorted
# plt.clf()
# plt.title('Performance on sorted lists')
# plt.plot(sizes, means[:, 1, 0], label='brentq')
# plt.plot(sizes, means[:, 1, 1], label='bob_rocch')
# # plt.plot(sizes, means[:, 1, 2], label='logn')
# plt.xlabel('Size')
# plt.ylabel('Avg Speed (s)')
# plt.xscale('log')
# plt.ylim((-1, 10))
# plt.legend()
# plt.show()
# # Unsorted
#%%
plt.title('Performance on unsorted lists')
plt.plot(sizes, means[:, 0, 0], label='shift_logn')
plt.plot(sizes, means[:, 0, 1], label='logn')
# plt.plot(sizes, means[:, 0, 2], label='logn')
plt.xlabel('Size')
plt.ylabel('Avg Speed (s)')
plt.xscale('log')
# plt.ylim((-1, 10))
plt.legend()
plt.show()
