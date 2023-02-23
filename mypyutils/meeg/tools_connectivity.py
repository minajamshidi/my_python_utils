import numpy as np
import scipy.signal as sp
from ..tools_signal import hilbert_
from scipy.stats import pearsonr
import itertools
import multiprocessing
from functools import partial




def connectivity_null_dist(b1, a1, b2, a2, n, m, duration, fs, n_total_iter=10000):
    from scipy.signal import filtfilt
    n_sample = duration * fs
    plvs = np.zeros((n_total_iter,))
    for i_iter in range(n_total_iter):
        x1 = np.random.randn(1, n_sample)
        x1 = filtfilt(b1, a1, x1, axis=1)

        y1 = np.random.randn(1, n_sample)
        y1 = filtfilt(b2, a2, y1, axis=1)

        plv = compute_phase_connectivity(x1, y1, n, m, 'coh', type1='abs')[0]
        plvs[i_iter] = plv[0][0]
    return plvs


#  --------------------------------  --------------------------------  --------------------------------
# connectivity matrix
#  --------------------------------  --------------------------------  --------------------------------

def graph_area_under_roc(groundtruth_graph, test_graph, return_all=True):
    from tools_general import threshold_matrix, blur_matrix
    G = threshold_matrix(groundtruth_graph, perc=None, threshold=np.max(groundtruth_graph), binary=True)
    G = blur_matrix(G, 5)
    G = (G > 0) + 0
    Gnot = np.mod(G + 1, 2)

    perc_range = np.arange(0, 101, 1)[::-1]
    tp = np.zeros(perc_range.shape)
    fp = np.zeros(perc_range.shape)
    for n_perc, perc in enumerate(perc_range):
        test_graph_th_1 = threshold_conn_matrix(test_graph, perc=perc, binary=False)
        tp[n_perc] = np.sum(test_graph_th_1 * G)
        fp[n_perc] = np.sum(test_graph_th_1 * Gnot)

    max_auc = np.max(fp) * np.max(tp)
    auc = np.sum(np.diff(fp) * tp[:-1]) / max_auc

    auc_fp = np.sum(0.01 * fp[:-1]) / np.max(fp)
    auc_tp = np.sum(0.01 * tp[:-1]) / np.max(fp)
    if return_all:
        return auc, auc_fp, auc_tp
    else:
        return auc


def graph_roc000(groundtruth_graph, test_graph):
    from tools_general import threshold_matrix, blur_matrix
    G = threshold_matrix(groundtruth_graph, perc=None, threshold=np.max(groundtruth_graph), binary=True)
    # G = blur_matrix(G, 2)
    G = (G > 0) + 0
    Gnot = np.mod(G + 1, 2)

    test_graph = test_graph / np.max(test_graph)

    thresh_range = np.arange(0.01, 1.01, .01)[::-1]
    tpr = np.zeros(thresh_range.shape)
    fpr = np.zeros(thresh_range.shape)
    for n_thresh, thresh in enumerate(thresh_range):
        # test_graph_th_1 = threshold_matrix(test_graph, perc=None, threshold=thresh, binary=False)
        test_graph_th_b = threshold_matrix(test_graph, perc=None, threshold=thresh, binary=True)
        test_graph_th_b_not = np.mod(test_graph_th_b + 1, 2)
        TP = np.sum(test_graph_th_b * G)
        FP = np.sum(test_graph_th_b * Gnot)
        TN = np.sum(test_graph_th_b_not * Gnot)
        FN = np.sum(test_graph_th_b_not * G)
        P = TP + FN
        N = FP + TN
        tpr[n_thresh] = TP / P
        fpr[n_thresh] = FP / N
    return fpr, tpr


def graph_roc(groundtruth_graph, test_graph):
    from tools_general import threshold_matrix, blur_matrix
    G = threshold_matrix(groundtruth_graph, perc=None, threshold=np.max(groundtruth_graph), binary=True)
    # G = blur_matrix(G, 2)
    G = (G > 0) + 0
    Gnot = np.mod(G + 1, 2)

    thresh_range = np.arange(0, 1.01, .01)[::-1]
    tp = np.zeros(thresh_range.shape)
    fp = np.zeros(thresh_range.shape)
    for n_thresh, thresh in enumerate(thresh_range):
        test_graph_th_1 = threshold_matrix(test_graph, perc=None, threshold=thresh, binary=False)
        tp[n_thresh] = np.sum(test_graph_th_1 * G) / np.sum(G * test_graph)
        fp[n_thresh] = np.sum(test_graph_th_1 * Gnot) / np.sum(Gnot * test_graph)
    return fp, tp


def graph_roc_2(groundtruth_graph, test_graph):
    from tools_general import threshold_conn_matrix, blur_matrix
    G = threshold_conn_matrix(groundtruth_graph, perc=None, threshold=np.max(groundtruth_graph), binary=True)
    G = blur_matrix(G, 2)
    G = (G > 0) + 0
    g = G.ravel()
    gnot = np.mod(g + 1, 2)

    testg = test_graph.ravel()
    ind_srt = np.argsort(testg)[::-1]
    testg_s = np.sort(testg)[::-1]
    g_srt = g[ind_srt]
    gnot_srt = gnot[ind_srt]
    testg_n = len(testg)
    tp = np.zeros((testg_n + 1,))
    fp = np.zeros((testg_n + 1,))
    for n in range(testg_n + 1):
        testg_s2 = testg_s.copy()
        testg_s2[n:] = 0
        tp[n] = np.sum(testg_s2 * g_srt) / np.sum(testg_s * g_srt)
        fp[n] = np.sum(testg_s2 * gnot_srt) / np.sum(testg_s * gnot_srt)
    return fp, tp


def compute_conn(n, m, fs, plv_type, signals):
    x = signals[0].T
    y = signals[1].T

    conn = np.mean(compute_plv_with_permtest(x, y, n, m, fs, plv_type=plv_type))
    return conn


def compute_conn_2D_parallel(ts_list1, ts_list2, n, m, fs, plv_type, mp=False):
    if mp:
        list_prod = list(itertools.product(ts_list1, ts_list2))
        pool = multiprocessing.Pool()
        func = partial(compute_conn, n, m, fs, plv_type)
        conn_mat_beta_corr_list = pool.map(func, list_prod)
        pool.close()
        # pool.join()
        conn_mat = np.asarray(conn_mat_beta_corr_list).reshape((len(ts_list1), len(ts_list2)))
    else:
        pass
        # n1 = len(ts_list1)
        # n2 = len(ts_list2)
        # con = np.zeros((n1, n2))
        # for i1 in range(n1):
        #     for i2 in range(n2):
    return conn_mat


def compute_conn_phase_roi2roi(n, m, measure, type1, sfreq, m2s, permtest, signals):
    ts_roi1, ts_roi2 = signals[0], signals[1]
    if permtest:
        conn_func = partial(_compute_phase_connectivity, n, m, measure, 1, type1)

    conn = np.zeros((ts_roi1.shape[0] * ts_roi2.shape[0],))
    ii = -1
    for ts1 in ts_roi1:
        for ts2 in ts_roi2:
            ii += 1
            conn1 = compute_phase_connectivity(ts1, ts2, n, m, measure=measure, type1=type1)[0]
            if permtest:
                conns_perm = connectivity_permutation(ts1, ts2, conn_func, iter_num=500,
                                                      perm_type='seg_shuffle', seg_len=int(sfreq))
                if np.abs(conn1) / np.mean(np.abs(conns_perm)) >= 2.42:
                    conn[ii] = np.abs(conn1)
                else:
                    conn[ii] = np.nan
            else:
                conn[ii] = np.abs(conn1)
    conn_return = 0 if np.isnan(m2s(conn)) else m2s(conn)
    return conn_return


def compute_conn_whole(ts_list1, ts_list2, n, m, sfreq, measure='plv',
                       type1='imag', m2s=np.nanmean, permtest=True, n_jobs=3):
    from mne.parallel import parallel_func
    import itertools
    from functools import partial

    list_prod = list(itertools.product(ts_list1, ts_list2))
    if n == m:
        ind1 = np.triu_indices(100, k=1)
        ind2 = np.ravel_multi_index(ind1, (len(ts_list1), len(ts_list1)), order='C')
        list_prod_ts = [list_prod[ii] for ii in ind2]
    else:
        list_prod_ts = list_prod

    func = partial(compute_conn_phase_roi2roi, n, m, measure, type1, sfreq, m2s, permtest)
    parallel, conn_phase_roi2roi_prl, _ = parallel_func(func, n_jobs=n_jobs)
    conns = parallel(conn_phase_roi2roi_prl(signals) for signals in list_prod_ts)
    if n == m:
        conns_ = np.zeros((len(list_prod),))
        conns_[ind2] = np.asarray(conns)
        conn_mat = conns_.reshape((len(ts_list1), len(ts_list2)))
        conn_mat = conn_mat + conn_mat.T
    else:
        conns_ = np.asarray(conns)
        conn_mat = conns_.reshape((len(ts_list1), len(ts_list2)))
    return conn_mat


#  --------------------------------  --------------------------------  --------------------------------
# amplitude-based measures
#  --------------------------------  --------------------------------  --------------------------------

def sensor_series_ampcorr(sensor_series_low_, sensor_series_high_):
    n_chan = len(sensor_series_low_)
    ampcorr = np.zeros((n_chan,))
    for chan1 in range(n_chan):
        data1_abs = np.abs(sensor_series_low_[chan1])
        data2_abs = np.abs(sensor_series_high_[chan1])
        ampcorr[chan1] = pearsonr(data1_abs[0, :], data2_abs[0, :])[0]
    return ampcorr


#  --------------------------------  --------------------------------  --------------------------------
# phase-based measures
#  --------------------------------  --------------------------------  --------------------------------

def compute_coherency1(x, y, n1, n2, coh_type='abs'):
    """
    computes complex coherency from two signals x and y with n1:n2 coupling
    :param x: array_like
              real or complex.
    :param y: array_like
              real or complex
    :param n1: the ratio of coupling of x and y is n1:n2
    :param n2: the ratio of coupling of x and y is n1:n2
    :param coh_type: 'complex', 'abs' (default), 'imag'
    :return: coh: coherency
    """
    if not np.iscomplexobj(x):
        x = hilbert_(x)
    if not np.iscomplexobj(y):
        y = hilbert_(y)

    x_ph = np.angle(x)
    x_abs = np.abs(x)
    y_ph = np.angle(y)
    y_abs = np.abs(y)
    coh_cmplx = np.mean(x_abs * y_abs * np.exp(1j * n2 * x_ph - 1j * n1 * y_ph) /
                        np.sqrt(np.mean(x_abs**2) * np.mean(y_abs**2)))
    if coh_type == 'complex':
        coh = coh_cmplx
    elif coh_type == 'abs':
        coh = np.abs(coh_cmplx)
    elif coh_type == 'imag':
        coh = np.imag(coh_cmplx)
    return coh


def compute_coupling_from_cs(cs1, cs2, cs12):
    cs2m, cs1m = np.meshgrid(cs2, cs1)
    coh = cs12 / np.sqrt(cs1m * cs2m)
    lagcoh = np.imag(coh) / np.sqrt(1 - np.real(coh) ** 2)
    wpli = 2 * lagcoh / (1 + lagcoh ** 2)
    return coh, lagcoh, wpli


def compute_wpli(cs):  # where cs = repetitions x channel x channel
    """
    computed the weighted-phase lag index
    cs is the cross-sepectrum in frequency f
    """
    csi = np.imag(cs)
    outsum = np.nansum(csi, 0)
    outsumW = np.nansum(abs(csi), 0)
    wpli = outsum / outsumW
    return wpli


"""
def compute_plv(ts1, ts2, n, m, stat_test=False):

    computes PLV
    :param ts1: [chan1 x time]
    :param ts2: [chan2 x time]
    :param n: ratio of the frequencies
    :param m: ratio of the frequencies
    :return: plv: [chan1 x chan2] phase locking values

    ts1_zp, n_zp = zero_pad_to_pow2(ts1, axis=1)
    ts1_angle = np.angle(sp.hilbert(ts1_zp[:, :-n_zp]))
    ts2_zp, n_zp = zero_pad_to_pow2(ts2, axis=1)
    ts2_angle = np.angle(sp.hilbert(ts2_zp[:, :-n_zp]))

    ts1_exp = np.exp(1j * m * ts1_angle)
    ts2_exp = np.exp(-1j * n * ts2_angle)
    plv = np.dot(ts1_exp, ts2_exp.T) / ts1_exp.shape[1]
    return plv
"""


def compute_phase_connectivity(ts1, ts2, n, m, measure='coh', axis=1, type1='complex'):
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]
    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    ts1_abs = np.abs(ts1)
    ts2_abs = np.abs(ts2)
    ts1_ph = np.angle(ts1)
    ts2_ph = np.angle(ts2)
    if measure == 'coh':
        nom = np.mean(ts1_abs * ts2_abs * np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
        denom = np.sqrt(np.mean(ts1_abs ** 2, axis=axis) * np.mean(ts2_abs ** 2, axis=axis))
        coh = nom / denom
        if type1 == 'abs':
            coh = np.abs(coh)
        elif type1 == 'imag':
            coh = np.imag(coh)
        return coh
    elif measure == 'plv':
        plv = np.mean(np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
        plv = np.abs(plv) if type1 == 'abs' else plv
        if type1 == 'abs':
            plv = np.abs(plv)
        elif type1 == 'imag':
            plv = np.imag(plv)
        return plv


def compute_coherency(ts1, ts2, n, m, axis=1):
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]
    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    ts1_abs = np.abs(ts1)
    ts2_abs = np.abs(ts2)
    ts1_ph = np.angle(ts1)
    ts2_ph = np.angle(ts2)
    nom = np.mean(ts1_abs * ts2_abs * np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
    denom = np.sqrt(np.mean(ts1_abs ** 2, axis=axis) * np.mean(ts2_abs ** 2, axis=axis))
    coh = nom / denom
    return coh


def compute_plv(ts1, ts2, n, m, plv_type='abs', coh=False):
    """
    computes complex phase locking value.
    :param ts1: array_like [channel x time]
                [channel x time] multi-channel time-series (real or complex)
    :param ts2: array_like
                [channel x time] multi-channel time-series (real or complex)
    :param n: the ratio of coupling of x and y is n:m
    :param m: the ratio of coupling of x and y is n:m
    :param plv_type: 'complex', 'abs' (default), 'imag'
    :param coh: Bool
    :return: plv: phase locking value
    """
    # TODO: test for multichannel
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    nchan1, n_samples = ts1.shape
    nchan2 = ts2.shape[0]

    ts1_ph = np.angle(ts1)
    ts2_ph = np.angle(ts2)
    ts1_abs = np.abs(ts1)
    ts2_abs = np.abs(ts2)
    cplv = np.zeros((nchan1, nchan2), dtype='complex')
    for chan1, ts1_ph_chan in enumerate(ts1_ph):
        for chan2, ts2_ph_chan in enumerate(ts2_ph):
            if coh:
                cplv[chan1, chan2] = np.mean(
                    ts1_abs[chan1, :] * ts2_abs[chan2, :] * np.exp(1j * m * ts1_ph_chan - 1j * n * ts2_ph_chan)) / \
                                     np.sqrt(np.mean(ts1_abs[chan1, :] ** 2)) / np.sqrt(np.mean(ts2_abs[chan2, :] ** 2))
            else:
                cplv[chan1, chan2] = np.mean(np.exp(1j * m * ts1_ph_chan - 1j * n * ts2_ph_chan))

    if plv_type == 'complex':
        return cplv
    elif plv_type == 'abs':
        return np.abs(cplv)
    elif plv_type == 'imag':
        return np.imag(cplv)


def compute_plv_windowed(ts1, ts2, n, m, winlen='whole', overlap_perc=0.5, plv_type='abs', coh=False):
    # TODO: test for multichannel
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    nchan1, n_samples = ts1.shape
    nchan2 = ts2.shape[0]
    if winlen == 'whole':
        winlen = n_samples
    winlen_start = int(np.floor(winlen * (1 - overlap_perc)))

    stop_flag = 0
    n_loop = 0
    plv = np.zeros((nchan1, nchan2), dtype='complex')
    while not stop_flag:
        n_start = n_loop * winlen_start
        n_stop = n_start + winlen
        if n_stop <= n_samples:
            ts1_ = ts1[:, n_start:n_stop]
            ts2_ = ts2[:, n_start:n_stop]
            plv += compute_plv(ts1_, ts2_, n, m, plv_type=plv_type, coh=coh)
            n_loop += 1
        else:
            stop_flag = 1
    plv /= n_loop
    if not plv_type == 'complex':
        plv = np.real(plv)
    return plv


def _synch_perm(n, m, seg_len, measure, type1, sig):
    x = sig[0]  # x.ndim = 1
    y = sig[1]
    seed1 = sig[2]

    if seed1 == -1:
        x_perm = x
    else:
        n_samples = x.shape[0]
        n_seg = int(n_samples // seg_len)
        n_omit = int(n_samples % seg_len)
        x_rest = x[-n_omit:] if n_omit > 0 else np.empty((0,))
        x_truncated = x[:-n_omit] if n_omit > 0 else x
        x_truncated = x_truncated.reshape((1, seg_len, n_seg), order='F')
        np.random.seed(seed=seed1)
        perm1 = np.random.permutation(n_seg)
        x_perm = x_truncated[:, :, perm1].reshape((1, n_samples - n_omit), order='F')
        x_perm = np.concatenate((x_perm.ravel(), x_rest))

    return compute_phase_connectivity(x_perm, y, n, m, measure=measure, type1=type1)[0]


def compute_synch_permtest_parallel(ts1, ts2, n, m, sfreq, ts1_ts2_eq=False, type1='abs', measure='coh',
                                    seg_len=None, iter_num=1000):
    """
    parallelized permutation test for multiple channels

    :param ts1: [nchan1 x time]
    :param ts2: [nchan2 x time]
    :param n:
    :param m:
    :param sfreq:
    :param ts1_ts2_eq: if True then only the upper triangle is computed. Put True only if ts1 and ts2 are identical and n = m = 1
    :param type1: {'abs', 'imag'}.
    :param measure: {'coh', 'plv'}
    :param seg_len: length of the segment for permutation testing
    :param iter_num: number of iterations for the perm test
    :return:
    conn_orig [nchan1 x nchan2]: the original true values of the conenctivity
    pvalue [nchan1 x nchan2]: the pvalues of the connections
    """
    if ts1.ndim == 1:
        ts1 = ts1[np.newaxis, :]
    if ts2.ndim == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    seg_len = int(sfreq) if seg_len is None else seg_len
    nchan1 = ts1.shape[0]
    nchan2 = ts2.shape[0]

    seeds = np.random.randint(low=0, high=2 ** 32, size=(iter_num,))
    seeds = np.append(-np.ones((1,), dtype='int64'), seeds)
    list_prod = list(itertools.product(ts1, ts2, seeds))

    if ts1_ts2_eq:
        ind_triu = np.triu_indices(nchan1, k=1)
        l1 = np.reshape(np.arange(len(list_prod)), (nchan1, nchan2, iter_num + 1))
        list_prod_ind = list(l1[ind_triu].ravel())
        list_prod = [list_prod[i] for i in list_prod_ind]

    pool = multiprocessing.Pool()
    func = partial(_synch_perm, n, m, seg_len, measure, type1)
    synch = pool.map(func, list_prod)
    pool.close()

    if ts1_ts2_eq:
        c1 = np.asarray(synch).reshape((-1, iter_num + 1))
        conn_mat = np.zeros((nchan1, nchan2, iter_num + 1))
        conn_mat[ind_triu] = c1
        conn_mat = conn_mat + np.transpose(conn_mat, axes=(1, 0, 2))
    else:
        conn_mat = np.asarray(synch).reshape((nchan1, nchan2, iter_num + 1))
    conn_mat = np.abs(conn_mat)
    conn_orig = conn_mat[:, :, 0]
    conn_perm = conn_mat[:, :, 1:]

    pvalue = np.zeros((nchan1, nchan2))
    pvalue_rayleigh = np.zeros((nchan1, nchan2))
    if ts1_ts2_eq:
        for i1 in range(nchan1):
            for i2 in range(i1 + 1, nchan2):
                pvalue[i1, i2] = np.mean(conn_perm[i1, i2, :] >= conn_orig[i1, i2])

                plv_perm1 = np.squeeze(conn_perm[i1, i2, :])
                plv_perm1_mean = np.mean(plv_perm1)
                plv_stat = conn_orig[i1, i2] / plv_perm1_mean
                pvalue_rayleigh[i1, i2] = np.exp(-np.pi * plv_stat ** 2 / 4)

        pvalue = pvalue + pvalue.T
        pvalue_rayleigh = pvalue_rayleigh + pvalue_rayleigh.T
    else:
        for i1 in range(nchan1):
            for i2 in range(nchan2):
                pvalue[i1, i2] = np.mean(conn_perm[i1, i2, :] >= conn_orig[i1, i2])

                plv_perm1 = np.squeeze(conn_perm[i1, i2, :])
                plv_perm1_mean = np.mean(plv_perm1)
                plv_stat = conn_orig[i1, i2] / plv_perm1_mean
                pvalue_rayleigh[i1, i2] = np.exp(-np.pi * plv_stat ** 2 / 4)
    return conn_orig, pvalue, pvalue_rayleigh


def compute_plv_with_permtest(ts1, ts2, n, m, sfreq, seg_len=None, plv_type='abs', coh=False,
                              iter_num=0, plv_win='whole', verbose=False):
    """
        do permutation test for testing the significance of the PLV between ts1 and ts2
        :param ts1: [chan1 x time]
        :param ts2: [chan2 x time]
        :param n:
        :param m:
        :param sfreq:
        :param seg_len:
        :param plv_type:
        :param coh:
        :param iter_num:
        :param plv_win:
        :param verbose:
        :return:
        plv_true: the true PLV
        plv_sig: the pvalue of the permutation test
        plv_stat: the statistics: the ratio of the observed plv to the mean of the permutatin PLVs
        plv_perm: plv of the iterations of the permutation test
        """
    # TODO: test for multichannel
    # TODO: verbose
    # ToDo: add axis
    # setting ------------------------------
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]

    if not np.iscomplexobj(ts1):
        ts1 = hilbert_(ts1)
    if not np.iscomplexobj(ts2):
        ts2 = hilbert_(ts2)

    seg_len = int(sfreq) if seg_len is None else int(seg_len)
    nchan1, n_samples = ts1.shape
    nchan2 = ts2.shape[0]

    if plv_win is None:
        plv_winlen = sfreq
    elif plv_win == 'whole':
        plv_winlen = n_samples

    plv_true = compute_plv_windowed(ts1, ts2, n, m, winlen=plv_winlen, plv_type=plv_type, coh=coh)
    if nchan1 == 1 and nchan2 == 1:
        plv_true = np.reshape(plv_true, (1, 1))

    n_seg = int(n_samples // seg_len)
    n_omit = int(n_samples % seg_len)
    ts1_rest = ts1[:, -n_omit:] if n_omit > 0 else np.empty((nchan1, 0))
    ts1_truncated = ts1[:, :-n_omit] if n_omit > 0 else ts1
    ts1_truncated = ts1_truncated.reshape((nchan1, seg_len, n_seg), order='F')

    if not plv_type == 'complex':
        plv_true = np.real(plv_true)

    plv_perm = -1
    pvalue = -1
    if iter_num:
        # TODO: seeds should be corrected
        seeds = np.random.choice(range(10000), size=iter_num, replace=False)
        plv_perm = np.zeros((nchan1, nchan2, iter_num), dtype='complex')
        for n_iter in range(iter_num):
            if verbose:
                print('iteration ' + str(n_iter))
            np.random.seed(seed=seeds[n_iter])
            perm1 = np.random.permutation(n_seg)
            ts1_perm = ts1_truncated[:, :, perm1].reshape((nchan1, n_samples - n_omit), order='F')
            ts1_perm = np.concatenate((ts1_perm, ts1_rest), axis=1)
            plv_perm[:, :, n_iter] = compute_plv_windowed(ts1_perm, ts2, n, m,
                                                          winlen=plv_winlen, plv_type=plv_type, coh=coh)

        plv_perm = np.abs(plv_perm)
        pvalue = np.zeros((nchan1, nchan2))
        for c1 in range(nchan1):
            for c2 in range(nchan2):
                plv_perm1 = np.squeeze(plv_perm[c1, c2, :])
                pvalue[c1, c2] = np.mean(plv_perm1 >= np.abs(plv_true)[c1, c2])

    if iter_num:
        return plv_true, pvalue, plv_perm
    else:
        return plv_true


def sensor_series_plv(sensor_series_low_, sensor_series_high_):
    n_chan = len(sensor_series_low_)
    cplv = np.zeros((n_chan,), dtype='complex')
    for chan1 in range(n_chan):
        cplv[chan1] = \
            np.mean(np.abs(sensor_series_low_[chan1]) * np.abs(sensor_series_high_[chan1]) *
                    np.exp(1j * 2 * np.angle(sensor_series_low_[chan1]) - 1j *
                           np.angle(sensor_series_high_[chan1])), axis=1) / \
            np.sqrt(np.mean(np.abs(sensor_series_low_[chan1]) ** 2, axis=1)) / \
            np.sqrt(np.mean(np.abs(sensor_series_high_[chan1]) ** 2, axis=1))
    return cplv


def _compute_phase_connectivity(n, m, measure, axis, type1, signals):
    ts1, ts2 = signals[0], signals[1]
    if len(ts1.shape) == 1:
        ts1 = ts1[np.newaxis, :]
    if len(ts2.shape) == 1:
        ts2 = ts2[np.newaxis, :]
    ts1 = hilbert_(ts1)
    ts2 = hilbert_(ts2)

    ts1_abs = np.abs(ts1)
    ts2_abs = np.abs(ts2)
    ts1_ph = np.angle(ts1)
    ts2_ph = np.angle(ts2)
    if measure == 'coh':
        nom = np.mean(ts1_abs * ts2_abs * np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
        denom = np.sqrt(np.mean(ts1_abs ** 2, axis=axis) * np.mean(ts2_abs ** 2, axis=axis))
        coh = nom / denom
        if type1 == 'abs':
            coh = np.abs(coh)
        elif type1 == 'imag':
            coh = np.imag(coh)
        return coh
    elif measure == 'plv':
        plv = np.mean(np.exp(1j * m * ts1_ph - 1j * n * ts2_ph), axis=axis)
        plv = np.abs(plv) if type1 == 'abs' else plv
        if type1 == 'abs':
            plv = np.abs(plv)
        elif type1 == 'imag':
            plv = np.imag(plv)
        return plv


#  --------------------------------  --------------------------------  --------------------------------
# Permutation
#  --------------------------------  --------------------------------  --------------------------------

def _permtest_segment_shuffle(ts1, ts2, conn_func, seg_len=None, iter_num=500):
    # TODo: joblib
    n_samples = ts1.shape[1]
    n_seg = int(n_samples // seg_len)
    n_omit = int(n_samples % seg_len)
    ts1_rest = ts1[:, -n_omit:] if n_omit > 0 else np.empty((1, 0))
    ts1_truncated = ts1[:, :-n_omit] if n_omit > 0 else ts1
    ts1_truncated = ts1_truncated.reshape((n_seg, seg_len), order='C')

    conns = np.zeros((iter_num,))
    for iter in range(iter_num):
        perm1 = np.random.permutation(n_seg)
        ts1_perm = ts1_truncated[perm1, :].reshape((1, n_samples - n_omit), order='C')
        ts1_perm = np.concatenate((ts1_perm, ts1_rest), axis=1)
        conns[iter] = conn_func([ts1_perm, ts2])
    return conns


def connectivity_permutation(ts1, ts2, conn_func, perm_type='timeshift', seg_len=None, iter_num=500):
    conns = np.zeros((iter_num,))
    ts1 = hilbert_(ts1)
    ts2 = hilbert_(ts2)
    if perm_type == 'timeshift':
        shift_limit = 64
        for iter in range(iter_num):
            shift = int(np.random.random() * shift_limit)  # np.pi
            ts2_ = np.roll(ts2, shift)  # np.abs(ts2) * np.exp(1j * np.angle(ts2) + 1j * shift)
            conns[iter] = conn_func([ts1, ts2_])
    if perm_type == 'seg_shuffle':
        conns = _permtest_segment_shuffle(ts1, ts2, conn_func, seg_len=seg_len, iter_num=iter_num)

    return conns


#  --------------------------------  --------------------------------  --------------------------------
# other
#  --------------------------------  --------------------------------  --------------------------------

def load_network(path_net, type='wf'):
    from tools_general import load_pickle
    con_mat = load_pickle(path_net)
    n_parc = con_mat.shape[0]
    con_mat1 = np.zeros((n_parc, n_parc))
    if type == 'wf':
        for parc1 in range(n_parc):
            for parc2 in range(n_parc):
                if parc2 > parc1:
                    con_mat1[parc1, parc2] = np.real(np.median(con_mat[parc1, parc2]))
                elif parc1 > parc2:
                    con_mat1[parc1, parc2] = con_mat1[parc2, parc1]
    elif type == 'cfc':
        for parc1 in range(n_parc):
            for parc2 in range(n_parc):
                con_mat1[parc1, parc2] = np.real(np.median(con_mat[parc1, parc2]))
    return con_mat1, con_mat



def count_rsnetworks_interactions(conn, labels):
    from tools_connectivity_plot import rearrange_labels_network
    # labels_net_sorted, idx_lbl_sort = rearrange_labels_network(labels)
    n_lbl = len(labels)
    node_network = [label.name[10:-5] for label in labels]
    node_network = np.asarray(node_network)
    ind_tmp = [i for i in range(n_lbl) if 'LH_Default_PFC' in node_network[i]]
    node_network[ind_tmp] = 'LH_Default_PFC'
    ind_tmp = [i for i in range(n_lbl) if 'RH_Default_PFC' in node_network[i]]
    node_network[ind_tmp] = 'RH_Default_PFC'
    networks = np.unique(node_network)
    n_networks = len(networks)
    networks_lh = [net for i, net in enumerate(networks) if 'LH' in net]
    networks_rh = [net for i, net in enumerate(networks) if 'RH' in net]
    networks = np.append(networks_lh, networks_rh[::-1])
    networks_lbl = np.zeros((n_lbl,))
    for i_lbl, lbl in enumerate(labels):
        networks_lbl[i_lbl] = [i for i, net in enumerate(networks) if net in lbl.name][0]

    net_interaction_1 = np.zeros((n_networks, n_networks))
    for n1 in range(n_networks):
        for n2 in range(n_networks):
            ind_n1 = np.where(networks_lbl == n1)[0][:, np.newaxis]
            ind_n2 = np.where(networks_lbl == n2)[0][np.newaxis, :]
            mat1 = np.take_along_axis(conn, ind_n1, axis=0)
            mat1 = np.take_along_axis(mat1, ind_n2, axis=1)
            net_interaction_1[n1, n2] = np.sum(mat1)

    # ~~~~~~~~~~~~~
    node_network2 = [None for i in range(n_lbl)]
    for i in range(n_lbl):
        name1 = node_network[i]
        try:
            ind_ = name1[::-1].index('_')
            ind_ = -ind_ - 1
            node_network2[i] = name1[:ind_]
            node_network2[i].index('_')
        except:
            node_network2[i] = name1

    networks2 = np.unique(np.asarray(node_network2))
    n_networks2 = len(networks2)

    networks2_lh = [net for i, net in enumerate(networks2) if 'LH' in net]
    networks2_rh = [net for i, net in enumerate(networks2) if 'RH' in net]
    networks2 = np.append(networks2_lh, networks2_rh[::-1])

    networks_lbl2 = np.zeros((n_lbl,))
    for i_lbl, lbl in enumerate(labels):
        networks_lbl2[i_lbl] = [i for i, net in enumerate(networks2) if net in lbl.name][0]

    net_interaction_2 = np.zeros((n_networks2, n_networks2))
    for n1 in range(n_networks2):
        for n2 in range(n_networks2):
            ind_n1 = np.where(networks_lbl2 == n1)[0][:, np.newaxis]
            ind_n2 = np.where(networks_lbl2 == n2)[0][np.newaxis, :]
            mat1 = np.take_along_axis(conn, ind_n1, axis=0)
            mat1 = np.take_along_axis(mat1, ind_n2, axis=1)
            net_interaction_2[n1, n2] = np.sum(mat1)
    return (net_interaction_1, networks), (net_interaction_2, networks2)