import numpy as np
import mne
import scipy as sp
from scipy.fftpack import fft
from mne.parallel import parallel_func
from matplotlib import pyplot as plt
import os.path as op


def labels_rearage_lh_rh(labels):
    labels_lh = [label for label in labels if label.hemi == 'lh']
    labels_rh = [label for label in labels if label.hemi == 'rh']
    labels_ = labels_lh + labels_rh
    return labels_


def extract_inv_sol(data_shape, fwd, raw_info, inv_method='eLORETA'):
    from mne.minimum_norm.inverse import _assemble_kernel, prepare_inverse_operator
    from tools_meeg import inverse_operator
    inv = inverse_operator(data_shape, fwd, raw_info)
    inv_op = prepare_inverse_operator(inv, 1, 0.05, inv_method)
    inv_sol = _assemble_kernel(inv=inv_op, label=None, method=inv_method, pick_ori='normal')[0]
    return inv_sol, inv_op, inv
# ======================================================================================================================
class ParcSeries:
    """
    possible values for mode: 'ssd', 'svd'
    """
    def __init__(self, mode, components, label, spatial_filter=None, spatial_pattern=None):
        self.components = components
        self.mode = mode
        self.label = label
        self.spatial_filter = spatial_filter
        self.spatial_pattern = spatial_pattern
        self.n_components = components.shape[0]

    def unify(self, unify_filter):
        if not unify_filter.shape[1] == self.n_components:
            raise Exception('the spatial filter should have the same number of columns as number of components.')
        return np.dot(unify_filter, self.components)
# ======================================================================================================================


class ModifiedLabel:
    def __init__(self, label_base, vertices):
        self.vertices = vertices
        self.label_base = label_base
# ======================================================================================================================
# label = labels[30] #14, 30
#
# plv30_ssd=compute_plv_with_permtest(x1, y1, 1, 2, fs, plv_type='abs')
# plv30_svd=compute_plv_with_permtest(x1, y1, 1, 2, fs, plv_type='abs')
# plv1430_ssd=compute_plv_with_permtest(x1, y1, 1, 1, fs, plv_type='imag')
# plv1430_svd=compute_plv_with_permtest(x1, y1, 1, 1, fs, plv_type='imag')


def _extract_ssd(data_orig_src, label, src, fs, freqs, n=1):
    """

    :param stc:
    :param labels_mod: should be of class ModifiedLabel
    :return:
    """
    from tools_multivariate import ssd_v2
    lbl_idx, _ = label_idx_whole_brain(src, label)
    data_lbl = data_orig_src[lbl_idx, :]
    x_ssd_, _, _, _ = ssd_v2(data_lbl, fs, fc=freqs, sig=None)
    n = x_ssd_.shape[0] if n is None else n
    return x_ssd_[:n, :]


def _extract_svd(data, src, n, label):
    from sklearn.decomposition import TruncatedSVD
    """
    - apply svd on time series of the voxels of the given parcel
    - if n=None, as many as the svd components are selected that they explain >=95% of the variance
    - if n=n_select, n_select components are selected
    :param data: [vertex x time]
    :param n:
    :return:
    """
    lbl_idx, _ = label_idx_whole_brain(src, label)
    data_lbl = data[lbl_idx, :]
    svd = TruncatedSVD(n_components=n)
    svd.fit(data_lbl)
    component = svd.components_ * svd.singular_values_[np.newaxis, :].T
    return component


def _extract_svd_par(data, labels, src, var_perc, n, ind_lbl):
    """
    - apply svd on time series of the voxels of the given parcel
    - if n=None, as many as the svd components are selected that they explain >=95% of the variance
    - if n=n_select, n_select components are selected
    :param data: [vertex x time]
    :param n:
    :return:
    """
    label = labels[ind_lbl]
    lbl_idx, _ = label_idx_whole_brain(src, label)
    data_lbl = data[lbl_idx, :]
    u, s, _ = np.linalg.svd(data_lbl.T, full_matrices=False)
    if n is None:
        var_explained = np.cumsum(s ** 2 / np.sum(s ** 2))
        ind1 = np.where(var_explained >= var_perc)[0]
        if len(ind1):
            n_select = ind1[0] + 1
            component = u[:, 0:n_select] * s[0:n_select]
        else:
            component = np.zeros((data.shape[1], 1))
    else:
        component = u[:, :n] * s[:n]
    return component


def extract_parcel_time_series(data, labels, src, mode='svd', n_select=1, fs=None, freqs=None, n_jobs=1):
    """

    :param stc:
    :param labels: should be of class ModifiedLabel
    :param src:
    :param mode:
    :param n_select:
    :param n_jobs:
    :return:
    """
    from mne.parallel import parallel_func
    import itertools
    from functools import partial

    n_parc = len(labels)
    if mode == 'ssd':
        print('applying ssd on each parcel... it may take some time')
        parcel_series = [None] * n_parc
        for this_parc, label in enumerate(labels):
            this_parc == int(n_parc / 2) and print('... We are half way done! ;-)')
            parcel_series[this_parc] = _extract_ssd(data, label, src, fs, freqs, n=n_select)
    elif mode == 'svd':
        # label_ts = mne.extract_label_time_course(stc, labels, src, mode='pca_flip', return_generator=False)
        # TODO: hack the function to extract the spatial patterns and build ParcSeries
        # parcel_series = label_ts
        # parallel, extract_svd, _ = parallel_func(_extract_svd, n_jobs)
        # parcel_series = parallel(extract_svd(stc.data, label, src, n=n_select) for label in labels[0:2])
        # n_samples = stc.data.shape[1]
        print('applying svd on each parcel... it may take some time')
        parcel_series = [None] * n_parc
        # TODO: n_jobs>1 does not work!!!!
        func = partial(_extract_svd, data, src, n_select)
        parallel, extract_svd_prl, _ = parallel_func(func, n_jobs=n_jobs)
        parcel_series = parallel(extract_svd_prl(label) for label in labels)
        #
        # for this_parc, label in enumerate(labels):
        #     this_parc == int(n_parc/2) and print('... We are half way done! ;-)')
        #     parcel_series[this_parc] = _extract_svd(data, label, src, n=n_select)
    return parcel_series

# ======================================================================================================================


def label_forward_inverse_effect(src, label, fwd_fixed, inv_op, raw_info, inv_method, subjects_dir):
    n_vox = src[0]['vertno'].shape[0] + src[1]['vertno'].shape[0]
    n_samples = 60*256
    label_hemi = 0 if label.hemi == 'lh' else 1
    vert_src_hemi = src[label_hemi]['vertno']
    vert1 = label.center_of_mass(subject=label.subject,
                                 subjects_dir=subjects_dir,
                                 restrict_vertices=vert_src_hemi)
    _, ind1, _ = np.intersect1d(src[label_hemi]['vertno'], vert1, return_indices=True)
    ind1 += label_hemi * src[0]['vertno'].shape[0]
    data_stc = np.zeros((n_vox, n_samples))
    data_stc[ind1, :] = np.random.randn(1, n_samples)
    vertices = [src[0]['vertno'], src[1]['vertno']]
    stc1 = mne.SourceEstimate(data_stc, vertices, tmin=0, tstep=1/256, subject=label.subject)
    stc1.data = stc1.data / np.max(np.abs(stc1.data)) * 1e-7
    raw1 = mne.apply_forward_raw(fwd=fwd_fixed, stc=stc1, info=raw_info)
    raw1.set_eeg_reference(projection=True)
    stc2 = mne.minimum_norm.apply_inverse_raw(raw1, inverse_operator=inv_op,
                                              lambda2=0.05, method=inv_method, pick_ori='normal')
    stc2.data = np.mean(stc2.data ** 2, axis=1)[:, np.newaxis]
    stc2.plot(subject=label.subject, subjects_dir=subjects_dir, time_viewer=True, hemi='both')
    return stc2

# ======================================================================================================================

def parcel_ts_to_power(parcel_series, labels, src, subject, subjects_dir, plot_pow=True, clim='auto'):

    ts_pow0 = np.empty((0, 0))
    ts_pow1 = np.empty((0, 0))
    vert_labels_centroid0 = np.empty((0, 0))
    vert_labels_centroid1 = np.empty((0, 0))
    for n_lbl, lbl in enumerate(labels):
        label_hemi = 0 if lbl.hemi == 'lh' else 1
        vert_src_hemi = np.intersect1d(src[label_hemi]['vertno'], lbl.vertices)
        # vert_src_hemi = src[label_hemi]['vertno']
        vert_cent = lbl.center_of_mass(subject=subject, subjects_dir=subjects_dir, restrict_vertices=vert_src_hemi)
        ts_lbl = parcel_series[n_lbl]
        pow1 = np.sum(np.mean(ts_lbl ** 2, axis=0))
        if label_hemi:
            vert_labels_centroid1 = np.append(vert_labels_centroid1, vert_cent, axis=None)
            ts_pow1 = np.append(ts_pow1, pow1, axis=None)
        else:
            vert_labels_centroid0 = np.append(vert_labels_centroid0, vert_cent, axis=None)
            ts_pow0 = np.append(ts_pow0, pow1, axis=None)

    ind_0 = np.argsort(vert_labels_centroid0)
    ind_1 = np.argsort(vert_labels_centroid1)
    ts_pow = np.append(ts_pow0[ind_0], ts_pow1[ind_1], axis=None)[:, np.newaxis]

    stc_pow_parc_ts = mne.SourceEstimate(data=ts_pow,
                                         vertices=[vert_labels_centroid0[ind_0], vert_labels_centroid1[ind_1]],
                                         tmin=0, tstep=0.01, subject=subject)
    if plot_pow:
        # stc_pow_parc_ts.plot(subjects_dir=subjects_dir, hemi='both', time_viewer=True, clim=clim)
        stc_pow_parc_ts_plt = mne.SourceEstimate(data=np.sqrt(ts_pow),
                                                 vertices=[vert_labels_centroid0[ind_0], vert_labels_centroid1[ind_1]],
                                                 tmin=0, tstep=0.01, subject=subject)
        plot_stc_power(stc_pow_parc_ts_plt, subjects_dir, clim=clim)
        return stc_pow_parc_ts
    else:
        return stc_pow_parc_ts

# ======================================================================================================================


def stc_to_power_parcels(stc, labels, src, fs, subject, subjects_dir, plot_stc=False):
    vert_lh, vert_rh = src[0]['vertno'], src[1]['vertno']
    n_vox = len(vert_lh) + len(vert_rh)
    vertices = [vert_lh, vert_rh]
    stc_data = np.zeros((n_vox, 1))
    for label_no, label in enumerate(labels):
        roi_idx, _ = label_idx_whole_brain(src, label)
        # this_parcel_ts = parcel_series[label_no]
        # this_parcel_power = np.mean(this_parcel_ts**2) if this_parcel_ts.shape[1] else 0
        this_parcel_power = np.mean(stc.data[roi_idx, :]**2)
        center_idx = find_label_center(label, src, subjects_dir)
        stc_data[center_idx, :] = this_parcel_power
    if plot_stc:
        clim = dict(kind='value', pos_lims=[0, 1 * max(stc_data) / 3, max(stc_data)])
        plot_stc_all_voxels(stc_data, src, fs, subject, subjects_dir, clim=clim)

# ======================================================================================================================


def find_label_center(label, src, subjects_dir):
    """
    what does this do?!!!
    :param label:
    :param src:
    :param subjects_dir:
    :return:
    """
    hemi_idx = 0 if label.hemi == 'lh' else 1
    offset = len(src[0]['vertno'])
    src_sel = np.intersect1d(src[hemi_idx]['vertno'], label.vertices)
    center_vert = label.center_of_mass(subjects_dir=subjects_dir, restrict_vertices=src_sel)
    _, center_idx, _ = np.intersect1d(src[hemi_idx]['vertno'], center_vert, return_indices=True)
    center_idx += offset * hemi_idx
    return center_vert, center_idx[0]
# ======================================================================================================================


"""
from multivariate import canonical_coherence

    parc1 = 10
    parc2 = 0
    x = fft(parcel_series[parc1].components, 512)
    y = fft(parcel_series[parc2].components, 512)
    cx = np.cov(x)
    cy = np.cov(y)
    cxy = np.dot(x, y.T)/x.shape[1]

    coh_complex, wa, wb, ta, tb = canonical_coherence(cx, cy, cxy)
    x = np.dot(wa.T, parcel_series[parc1].components)
    y = np.dot(wb.T, parcel_series[parc2].components)

    x = np.dot(wa.T, xs)
    y = np.dot(wb.T, xs1)


    sig1 = sp.signal.hilbert(x)
    sig2 = sp.signal.hilbert(y)
"""


def vertex_index_full(vertex_nr, src):
    n_vox_lh = src[0]['vertno'].shape[0]
    vert_full = np.append(src[0]['vertno'], n_vox_lh + src[1]['vertno'])
    vertices_ = np.concatenate(vertex_nr)
    _, idx1, idx2 = np.intersect1d(vertices_, vert_full, return_indices=True)
    ind = np.argsort(idx1)
    idx2 = idx2[ind]
    n_vert_l = vertex_nr[1].shape[0]
    idx2[-n_vert_l:] = idx2[-n_vert_l:] + n_vox_lh
    return idx2


def one_vertex_index_full(vertex_nr, src, hemi):
    hemi_idx = 0 if hemi == 'lh' else 1
    n_vox_lh = src[0]['vertno'].shape[0]
    vert_full = np.append(src[0]['vertno'], n_vox_lh + src[1]['vertno'])
    _, _, idx2 = np.intersect1d(vertex_nr, vert_full, return_indices=True)
    idx2 += n_vox_lh * hemi_idx
    return idx2


def label_idx_whole_brain(src, label):
    """
    finding the vertex numbers corresponding to vertices in parcel specified in label
    :param src: source spaces
    :param label: label object of the desired parcel
    :return: parc_idx: indexes of the vertices in label,
             parc_hemi_idx: indexes of the vertices in label, but in the corresponding hemisphere
    """
    offset = src[0]['vertno'].shape[0]
    this_hemi = 0 if (label.hemi == 'lh') else 1
    idx, _, parc_hemi_idx = np.intersect1d(label.vertices, src[this_hemi]['vertno'], return_indices=True)
    # parc_hemi_idx = np.searchsorted(src[this_hemi]['vertno'], idx)
    parc_idx = parc_hemi_idx + offset * this_hemi

    return parc_idx, parc_hemi_idx


def vertices_label(src, labels):
    vert_lh, vert_rh = src[0].get('vertno'), src[1].get('vertno')
    src_ident = [np.full(len(vert_lh), -1, dtype='int'), np.full(len(vert_rh), -1, dtype='int')]
    for label_no, label in enumerate(labels):
        this_hemi = 0 if (label.hemi == 'lh') else 1
        idx = np.intersect1d(label.vertices, src[this_hemi]['vertno'])
        ROI_idx = np.searchsorted(src[this_hemi]['vertno'], idx)
        src_ident[this_hemi][ROI_idx] = label_no
    return src_ident


def find_label_of_a_vertex_hemi(labels, vert, hemi):
    label_vert = -1
    for n_lbl, label in enumerate(labels):
        if label.hemi == hemi:
            _, ind1, ind2 = np.intersect1d(label.vertices, vert, return_indices=True)
            if len(ind2):
                label_vert = int(n_lbl)
                break
    return label_vert


def find_label_of_vertex_whole(src, labels, vert):
    # TODO: Caution: Should be checked - didnt work the last check
    flag = np.zeros((len(vert),))
    n_vert_lh = src[0]['vertno'].shape[0]
    label_vert = np.ones((len(vert),), dtype='int') * -1
    for n_lbl, label in enumerate(labels):
        if label.hemi == 'rh':
            vert_lbl = label.vertices + n_vert_lh
        else:
            vert_lbl = label.vertices
        _, ind1, ind2 = np.intersect1d(vert_lbl, vert, return_indices=True)
        if len(ind2):
            flag[ind2] = 1
            label_vert[ind2] = int(n_lbl)
        if np.prod(flag):
            break
    return label_vert


def plot_a_label(src, label, subjects_dir, cmp='auto'):
    parc_idx, _ = label_idx_whole_brain(src, label)
    n_vox = src[0]['vertno'].shape[0] + src[1]['vertno'].shape[0]
    data_parc = np.zeros((n_vox, 1))
    data_parc[parc_idx] = 1
    vertices = [src[0]['vertno'], src[1]['vertno']]
    stc_new = mne.SourceEstimate(data_parc, vertices, tmin=0, tstep=0.01, subject=label.subject)
    return stc_new.plot(subject=label.subject, subjects_dir=subjects_dir, time_viewer=True, hemi='both', colormap=cmp)


def project_label_back(src, label, leadfield, inv_sol, subjects_dir, background='white', hemi='both'):
    parc_idx, _ = label_idx_whole_brain(src, label)
    n_vox = src[0]['vertno'].shape[0] + src[1]['vertno'].shape[0]
    data_parc = np.zeros((n_vox, 1))
    data_parc[parc_idx] = 1
    data_parc2 = inv_sol @ (leadfield @ data_parc)
    data_parc2 = np.abs(data_parc2)
    vertices = [src[0]['vertno'], src[1]['vertno']]
    stc_new = mne.SourceEstimate(data_parc2, vertices, tmin=0, tstep=0.01, subject=label.subject)
    return stc_new.plot(subject=label.subject, subjects_dir=subjects_dir, time_viewer=True, hemi=hemi,
                        background=background)


def plot_labels(src, labels, subjects_dir, cmp='auto', background='white', hemi='both', surface='inflated'):
    n_vox = src[0]['vertno'].shape[0] + src[1]['vertno'].shape[0]
    data_parc = np.zeros((n_vox, 1))
    for n, lbl in enumerate(labels):
        parc_idx, _ = label_idx_whole_brain(src, lbl)
        data_parc[parc_idx] = 1  # (n+1)*10
    vertices = [src[0]['vertno'], src[1]['vertno']]
    stc_new = mne.SourceEstimate(data_parc, vertices, tmin=0, tstep=0.01, subject=labels[0].subject)
    # clim = dict(kind='value', pos_lims=[0, np.round(10*len(labels)/2), 10*len(labels)])
    return stc_new.plot(subject=labels[0].subject, subjects_dir=subjects_dir, surface=surface,
                        time_viewer=True, hemi=hemi, colormap=cmp, background=background)


def sort_vertices_based_on_labels(src,labels):
    src_ident_all = np.full(len(src[0]['vertno'])+len(src[1]['vertno']) ,  -1, dtype='int' )
    label_ident_all = np.full(len(src[0]['vertno']) + len(src[1]['vertno']), -1, dtype='int')
    n_start = 0
    src_ident_all
    for label_no, label in enumerate(labels):
        IDX = label_idx_whole_brain(src, label)
        n_end = n_start + len(IDX)
        src_ident_all[n_start:n_end] = IDX
        label_ident_all[n_start:n_end] = label_no
        n_start = n_end
    return src_ident_all, label_ident_all


def sparse_stc_to_sensor1(stc, src, leadfield, raw_info=None):
    vert_lh, vert_rh = src[0]['vertno'], src[1]['vertno']
    n_vox = vert_lh.shape[0] + vert_rh.shape[0]
    offset = vert_lh.shape[0]
    data = np.zeros((n_vox, stc.data.shape[1]))

    # vertices of left hemi
    vertices_stc = [stc.lh_vertno, stc.rh_vertno]
    offset_stc = len(stc.lh_vertno)
    for hemi in range(2):
        # hemi=0--> left
        _, idx, idx_stc = np.intersect1d(src[hemi]['vertno'], vertices_stc[hemi], return_indices=True)
        if len(idx):
            idx = idx + offset * hemi
            data[idx, :] = stc.data[offset_stc*hemi + idx_stc, :]

    vertices = [src[0]['vertno'], src[1]['vertno']]
    stc_new = mne.SourceEstimate(data, vertices, tmin=0, tstep=0.01, subject=stc.subject)
    # stc_new.plot(subject=subject, subjects_dir=subjects_dir, hemi='both', time_viewer=True)
    data_sensor = np.dot(leadfield, data)
    data_raw = mne.io.RawArray(data_sensor, raw_info) if raw_info is not None else None
    """
    plt.figure()
    mne.viz.plot_topomap(pattern[:, 0], raw.info)
    """

    return data_sensor, stc_new


def plot_sparse_stc_power(stc, src,  subjects_dir, clim='auto', hemi='both'):
    """
    plots the power of the source estimates on the brain.
    :param stc: SourceEstimate instance
    :param subjects_dir: the directory of the MRI
    :param clim: str . possible values "auto", "whole"
                if 'auto' then the colormap is based on data percentiles.
                if 'whole' then tje colormap maps the whole data range.

    :return: figure --> detail: go to mne.SourceEstimate.plot()
    """
    data1 = np.mean(stc.data**2, axis=1)
    stc_new = stc.copy()
    stc_new.data = data1[:, np.newaxis]
    stc_new2 = sparse_stc_to_full_stc(stc_new, src, stc.subject)
    if clim == 'whole':
        clim = dict(kind='value', pos_lims=[0, np.max(data1) / 2, np.max(data1)])
    return stc_new2.plot(subject=stc_new.subject, subjects_dir=subjects_dir, time_viewer=True, hemi=hemi, clim=clim)


def plot_stc_power(stc, subjects_dir, clim='auto', hemi='both'):
    """
    plots the power of the source estimates on the brain.
    :param stc: SourceEstimate instance
    :param subjects_dir: the directory of the MRI
    :param clim: str . possible values "auto", "whole"
                if 'auto' then the colormap is based on data percentiles.
                if 'whole' then tje colormap maps the whole data range.

    :return: figure --> detail: go to mne.SourceEstimate.plot()
    """
    data1 = np.mean(stc.data**2, axis=1)
    stc_new = stc.copy()
    stc_new.data = data1[:, np.newaxis]
    if clim == 'whole':
        clim = dict(kind='value', pos_lims=[0, np.max(data1) / 2, np.max(data1)])
    return stc_new.plot(subject=stc_new.subject, subjects_dir=subjects_dir, time_viewer=True, hemi=hemi, clim=clim, background='white')


def sparse_stc_to_full_stc(stc, src, subject):
    vert_lh, vert_rh = src[0]['vertno'], src[1]['vertno']
    vertices = [vert_lh, vert_rh]
    vertices_np = np.append(vert_lh, vert_rh, axis=None)
    n_vox = len(vert_lh) + len(vert_rh)
    data = np.zeros((n_vox, stc.data.shape[1]))

    vertices_stc = np.append(stc.lh_vertno, stc.rh_vertno, axis=None)
    _, idx1, idx2 = np.intersect1d(vertices_np, vertices_stc, return_indices=True)
    data[idx1, :] = stc.data[idx2, :]
    stc_new = mne.SourceEstimate(data, vertices, tmin=0, tstep=0.01, subject=subject)
    return stc_new


def sparse_stc_to_sensor(stc, src, leadfield, rawinfo, dipole_ind1=[0]):

    """
    PROBLEM
    when we have a sparse stc and we want the spatial pattern and plotting
    :param subject:
    :param subjects_dir:
    :param stc:
    :param src:
    :param fs:
    :param leadfield:
    :param rawinfo:
    :param pattern_title:
    :param dipole_ind:
    :param plot_pattern:
    :param plot_stc:
    :return:
    """
    vert_lh, vert_rh = src[0]['vertno'], src[1]['vertno']
    n_vox = len(vert_lh) + len(vert_rh)
    offset = len(vert_lh)
    vertices_stc = [stc.lh_vertno, stc.rh_vertno]

    pattern = np.zeros((rawinfo['nchan'], len(dipole_ind1)))
    for i_dip, dipole_ind in enumerate(dipole_ind1):
        data = np.zeros((n_vox, 1))
        dipole_hemi = 0 if dipole_ind < len(stc.lh_vertno) else 1
        dipole_no = dipole_ind - dipole_hemi * len(stc.lh_vertno)
        _, idx, _ = np.intersect1d(src[dipole_hemi]['vertno'], vertices_stc[dipole_hemi][dipole_no],
                                   return_indices=True)
        idx = idx + offset * dipole_hemi
        data[idx] = 1
        pattern[:, i_dip:i_dip + 1] = np.dot(leadfield, data)
        pattern[:, i_dip:i_dip + 1] /= np.linalg.norm(pattern[:, i_dip:i_dip + 1])

        # vertices = [src[0]['vertno'], src[1]['vertno']]
        # stc_new = mne.SourceEstimate(data, vertices, tmin=0, tstep=1 / fs, subject=subject)
    # if plot_stc:
    #     stc_new.plot(subject=subject, subjects_dir=subjects_dir, hemi='both', time_viewer=True)
    #
    # if plot_pattern:
    #     plt.figure()
    #     mne.viz.plot_topomap(pattern[:, 0], rawinfo)
    #     plt.title(pattern_title)

    return pattern


def plot_stc_all_voxels(data, src, subject, subjects_dir, clim, colormap, smoothing, hemi, surface, background):
    """
    used when we have an array of voxels values at all voxels (n_vox x T), and want to plot it on the brain
    :param data:
    :param src:
    :param fs:
    :param subject:
    :param subjects_dir:
    :return:
    """
    vertices = [src[0]['vertno'], src[1]['vertno']]
    stc_new = mne.SourceEstimate(data, vertices, tmin=0, tstep=0.01,  subject=subject)
    #clim = dict(kind='value', pos_lims=[0, np.max(data)/2, np.max(data)])
    return stc_new.plot(subject=subject, subjects_dir=subjects_dir, hemi=hemi, clim=clim, time_viewer=True,
                        colormap=colormap, smoothing_steps=smoothing, surface=surface, background=background)


def power_stc_parcellation(stc, src, labels, fs, subjects_dir, clim='auto', plot_stc=False):
    """
    the function to compute the power of stc.data at the centroid of each label
    :param stc:
    :param src:
    :param labels:
    :param fs:
    :param subjects_dir:
    :param clim:
    :param clim:
    :param plot_stc:
    :return:
    """
    subject = labels[0].subject
    data_stc = stc.data
    vertices_stc = [stc.lh_vertno, stc.rh_vertno]
    offset_vert_stc = len(stc.lh_vertno)
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    # vert_src_lh, vert_src_rh = src[0]['vertno'],  src[1]['vertno']

    n_parc = len(labels)
    ind_labels_lh = [n1 for n1, label2 in enumerate(labels) if label2.hemi == 'lh']
    ind_labels_rh = [n1 for n1, label2 in enumerate(labels) if label2.hemi == 'rh']

    pow_labels = np.zeros((n_parc, 1))  # saves the power of stc.data at each parcel
    pow_all_vox_label = np.zeros((n_vox, 1))  # saves the power of stc.data at each voxel
    vert_labels_centroid = np.empty((1, n_parc))

    for n_label, label1 in enumerate(labels):
        label_hemi = 0 if label1.hemi == 'lh' else 1
        vert_src_hemi = src[label_hemi]['vertno']
        vert_labels_centroid[0, n_label] = label1.center_of_mass(subject=subject,
                                                                 subjects_dir=subjects_dir,
                                                                 restrict_vertices=vert_src_hemi)
        label_vertices = np.intersect1d(label1.vertices, src[label_hemi]['vertno'])
        _, idx_vert_stc_label, _ = np.intersect1d(vertices_stc[label_hemi], label_vertices, return_indices=True)

        ind_label_src_vert, _ = label_idx_whole_brain(src, label1)
        if len(idx_vert_stc_label):
            idx_vert_stc_label += offset_vert_stc * label_hemi
            pow_labels[n_label, 0] = np.mean(data_stc[idx_vert_stc_label, :] ** 2)
            pow_all_vox_label[ind_label_src_vert, :] = pow_labels[n_label, 0]

    ind_labels_lh = np.asarray(ind_labels_lh)
    ind_labels_rh = np.asarray(ind_labels_rh)
    vert_lh = vert_labels_centroid[:, ind_labels_lh]
    vert_rh = vert_labels_centroid[:, ind_labels_rh]
    idx_lh = np.argsort(vert_lh)
    vert_lh = np.sort(vert_lh)[0, :]
    idx_rh = np.argsort(vert_rh)
    vert_rh = np.sort(vert_rh)[0, :]
    ind_labels_lh = ind_labels_lh[idx_lh][0, :]
    ind_labels_rh = ind_labels_rh[idx_rh][0, :]

    vertices_stc_pow = [vert_lh, vert_rh]
    data_stc_pow = np.concatenate((pow_labels[ind_labels_lh, :], pow_labels[ind_labels_rh, :]), axis=0)
    stc_pow = mne.SourceEstimate(data_stc_pow, vertices=vertices_stc_pow, tmin=0, tstep=1 / fs, subject=subject)

    if plot_stc:
        if clim == 'whole':
            clim = dict(kind='value', pos_lims=[0, np.max(stc_pow.data) / 2, np.max(stc_pow.data)])
        # elif type(clim) is dict: it is passed to plot function

        stc_pow.plot(subject=subject, subjects_dir=subjects_dir, hemi='both', time_viewer=True, clim=clim)
    return stc_pow, pow_all_vox_label, pow_labels


def select_strongest_sources(stc, perc):
    stc1 = stc.copy()
    data_stc = stc1.data
    pow = np.mean(data_stc ** 2, axis=1)
    th = np.percentile(pow, perc)
    data_stc[pow < th, :] = 0
    stc1.data = data_stc
    return stc1


def labels_centroid(labels, src, subject, subjects_dir):
    n_parc = len(labels)
    ind_labels_lh = [n1 for n1, label2 in enumerate(labels) if label2.hemi == 'lh']
    ind_labels_rh = [n1 for n1, label2 in enumerate(labels) if label2.hemi == 'rh']
    ind_labels_lh = np.asarray(ind_labels_lh)
    ind_labels_rh = np.asarray(ind_labels_rh)

    vert_labels_centroid = np.empty((1, n_parc))
    for n_label, label1 in enumerate(labels):
        label_hemi = 0 if label1.hemi == 'lh' else 1
        vert_src_hemi = src[label_hemi]['vertno']
        vert_labels_centroid[0, n_label] = label1.center_of_mass(subject=subject,
                                                                 subjects_dir=subjects_dir,
                                                                 restrict_vertices=vert_src_hemi)
    vert_lh = vert_labels_centroid[:, ind_labels_lh] if len(ind_labels_lh) else []
    vert_rh = vert_labels_centroid[:, ind_labels_rh] if len(ind_labels_rh) else []
    idx_lh = np.argsort(vert_lh)
    vert_lh = np.sort(vert_lh)[0, :] if len(ind_labels_lh) else []
    idx_rh = np.argsort(vert_rh)
    vert_rh = np.sort(vert_rh)[0, :] if len(ind_labels_rh) else []
    ind_labels_lh = ind_labels_lh[idx_lh][0, :] if len(ind_labels_lh) else []
    ind_labels_rh = ind_labels_rh[idx_rh][0, :] if len(ind_labels_rh) else []

    stc1 = mne.SourceEstimate(data=np.ones((n_parc, 1)), vertices=[vert_lh, vert_rh], tmin=0, tstep=0, subject=subject)
    return stc1, [ind_labels_lh, ind_labels_rh]


def plot_centrality_foci(stc, subject, subjects_dir):
    # from surfer import Brain
    from tools_general import plot_colorbar

    data_cent = stc.data
    cmap = plt.get_cmap('Blues', data_cent.shape[0])
    idx_sort_data = np.argsort(data_cent, axis=0)[:, 0]
    n_vox_lh = stc.lh_vertno.shape[0]
    Brain = mne.viz.get_brain_class()
    brain = Brain(subject, 'both', 'inflated', subjects_dir=subjects_dir,
                  cortex='low_contrast', background='white', size=(800, 600))
    for n, node in enumerate(stc.lh_vertno):
        idx_cmap = np.where(idx_sort_data == n)[0][0]
        brain.add_foci(node, coords_as_verts=True, hemi='lh', color=cmap(idx_cmap)[0:3])  # ,
    for n, node in enumerate(stc.rh_vertno):
        n2 = n_vox_lh + n
        idx_cmap = np.where(idx_sort_data == n2)[0][0]
        brain.add_foci(node, coords_as_verts=True, hemi='rh', color=cmap(idx_cmap)[0:3])
    plot_colorbar(data_cent, cmap, ori='vertical')


def source_space_narrow_band(raw, inv_op, inv_method, f1, f2, lambda2=0.05, pick_ori='normal'):
    iir_params = dict(order=2, ftype='butter')
    raw_nb = raw.copy()
    raw_nb.load_data()
    raw_nb.filter(l_freq=f1, h_freq=f2, method='iir', iir_params=iir_params)
    raw_nb.set_eeg_reference(projection=True)
    stc_nb = mne.minimum_norm.apply_inverse_raw(raw_nb, inverse_operator=inv_op,
                                                lambda2=lambda2, method=inv_method, pick_ori=pick_ori)
    return stc_nb


def source_space_narrow_band_2(raw, f1, f2, inv_sol, ind_raw, ind_inv, vertno, subject):
    sfreq = raw.info['sfreq']
    iir_params = dict(order=2, ftype='butter')
    raw_nb = raw.copy()
    raw_nb.load_data()
    raw_nb.filter(l_freq=f1, h_freq=f2, method='iir', iir_params=iir_params)
    raw_nb.set_eeg_reference(projection=True)
    data = raw_nb.get_data()
    data_src = inv_sol[:, ind_inv] @ data[ind_raw, :]
    stc_nb = mne.SourceEstimate(data_src, vertices=vertno, tmin=0, tstep=1/sfreq, subject=subject)
    return stc_nb


def plot_value_per_label(src, labels, value_vec, subject, subjects_dir):
    lh_vertno = list()
    rh_vertno = list()
    lh_value = list()
    rh_value = list()
    for ind_lbl, label in enumerate(labels):
        if label.hemi == 'lh':
            hemi_idx = 0
            vertno = lh_vertno
            value = lh_value
        else:
            hemi_idx = 1
            vertno = rh_vertno
            value = rh_value
        src_sel = np.intersect1d(src[hemi_idx]['vertno'], label.vertices)
        idx = label.center_of_mass(subject, restrict_vertices=src_sel, subjects_dir=subjects_dir, surf='sphere')
        vertno.append(idx)
        value.append(value_vec[ind_lbl])

    lh_vertno = np.array(lh_vertno)
    rh_vertno = np.array(rh_vertno)
    lh_value = np.array(lh_value)
    rh_value = np.array(rh_value)

    ind_lh = np.argsort(lh_vertno)
    ind_rh = np.argsort(rh_vertno)
    value_whole = np.append(lh_value[ind_lh], rh_value[ind_rh])
    stc = mne.SourceEstimate(value_whole, vertices=[lh_vertno[ind_lh], rh_vertno[ind_rh]], tmin=0,
                             tstep=0.01, subject=subject)
    stc.plot(subject=subject, subjects_dir=subjects_dir, hemi='both')

    # plot_centrality_foci(stc, subject, subjects_dir)


def inverse_solution(inv_op, inv_method):
    from mne.minimum_norm.inverse import _assemble_kernel
    from mne.minimum_norm import prepare_inverse_operator
    inv_op = prepare_inverse_operator(inv_op, nave=1, lambda2=0.05, method=inv_method)
    inv_sol, _, vertno, source_nn = _assemble_kernel(inv=inv_op, label=None, method=inv_method, pick_ori='normal')
    return inv_sol, vertno, source_nn