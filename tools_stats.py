import numpy as np
from scipy import stats



def confidence_interval(test_stat, confidence=0.95):
    a = 1.0 * np.array(test_stat)
    n = len(a)
    m, std = np.mean(a), np.std(a)
    h = std * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, std = np.mean(a), stats.sem(a)
    h = std * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def bootstrap(x, n_bt, func, size=None, reg=False, sigma=None):
    """

    :param x: 1D array
    :param n_bt:
    :param func:
    :param size:
    :param reg:
    :return:
    """
    stats_bt = np.zeros((n_bt,))
    size = len(x) if size is None else size
    sigma = 1 / np.sqrt(size) if sigma is None else sigma
    for i in range(n_bt):
        x1 = np.random.choice(x, size=size)
        if reg:
            noise = np.random.randn(len(x)) * sigma
            x1 += noise
        stats_bt[i] = func(x1)
    return stats_bt


def bootstrap_hypothesis_test(x, y, n_boot, reg=False, sigma=None):
    n_x = len(x)
    n_y = len(y)
    sigma_x = 1 / np.sqrt(n_x) if sigma is None else sigma[0]
    sigma_y = 1 / np.sqrt(n_y) if sigma is None else sigma[1]
    xy = np.append(x, y)
    ind_xy = np.arange(n_x+n_y)
    med_diff = np.zeros((n_boot,))
    for i in range(n_boot):
        ind_x_boot = np.random.choice(ind_xy, size=n_x, replace=False)
        ind_y_boot = np.delete(ind_xy, ind_x_boot)
        x_boot = xy[ind_x_boot]
        y_boot = xy[ind_y_boot]
        if reg:
            x_boot += np.random.randn(n_x) * sigma_x
            y_boot += np.random.randn(n_y) * sigma_y
        med_x_boot = np.median(x_boot)
        med_y_boot = np.median(y_boot)
        med_diff[i] = med_x_boot - med_y_boot
    return med_diff


def kruskal_stats(x, y):
    stat = np.empty((x.shape[1],))
    pval = np.empty((x.shape[1],))
    for i in range(x.shape[1]):
        out = stats.kruskal(x[:, i], y[:, i])
        stat[i] = out[0]
        pval[i] = out[1]
    return stat, pval
