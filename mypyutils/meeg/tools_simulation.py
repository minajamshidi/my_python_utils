import numpy as np
import mne
# from matplotlib import pyplot as plt
from mne.simulation import simulate_sparse_stc
from tools_signal import zero_pad_to_pow2
from scipy.signal import hilbert
from ..tools_signal import hilbert_
from scipy.signal import filtfilt, butter


def produce_nonsin_sig(x):
    x_h = hilbert_(x)

    n = 2
    sigma2 = np.random.random(1) * 2 * pi - pi
    y2_h = np.abs(x_h) * np.exp(1j * n * np.angle(x_h) + 1j * sigma2)
    y2 = np.real(y2_h)

    n = 3
    sigma3 = np.random.random(1) * 2 * pi - pi
    y3_h = np.abs(x_h) * np.exp(1j * n * np.angle(x_h) + 1j * sigma3)
    y3 = np.real(y3_h)

    n = 4
    sigma4 = np.random.random(1) * 2 * pi - pi
    y4_h = np.abs(x_h) * np.exp(1j * n * np.angle(x_h) + 1j * sigma4)
    y4 = np.real(y4_h)

    x_nonsin = x + 0.25 * y2 + 0.06 * y3 + 0.025 * y4
    return x_nonsin


def kuramoto(n, m, fb,  fs, T, peakfreq_noise, epsilon):
    """
    Kuramoto oscilalators
    :param n: oscillators are n:m coupled
    :param m: -
    :param fb: base-frequency, the central frequencies of the oscillators are nfb and m_fb
    :param T: duration in seconds
    :param n_samp: number of samples
    :param peakfreq_noise:
    :param epsilon:
    :return:
    """
    from numpy import sin, pi
    dt = 1 / fs
    f1 = n * fb
    f2 = m * fb
    n_samp = int(fs * T)
    phase1 = np.zeros((n_samp,))  # pre-allocation of memory
    phase2 = np.zeros((n_samp,))  # pre-allocation of memory
    phase1[0] = 0  # initial theta phase
    phase2[0] = 0  # initial gamma phase

    for j in range(n_samp - 1):
        randn = np.random.randn(1)[0]
        phase1[j + 1] = (phase1[j] + dt * ((f1 + peakfreq_noise * randn) * 2 * pi
                                           + epsilon * sin(n * phase2[j] - m * phase1[j]))) % (2 * pi)
        randn = np.random.randn(1)
        phase2[j + 1] = (phase2[j] + dt * ((f2 + peakfreq_noise * randn) * 2 * pi
                                           + epsilon * sin(m * phase1[j] - n * phase2[j]))) % (2 * pi)
    return phase1, phase2


def kuramuto_oscillators(duration, fs, n, m, f1, f2, sigma_f1, sigma_f2, epsilon):
    from numpy import pi, sin
    _2pi = 2 * pi
    n_samp = int(duration * fs)
    dt = 1 / fs
    phase_1 = np.zeros((n_samp,))  # pre-allocation of memory
    phase_2 = np.zeros((n_samp,))
    for j in range(n_samp - 1):
        freq_dev = sigma_f1 * np.random.randn(1)[0]
        omega1 = (f1 + freq_dev) * _2pi
        phase1 = phase_1[j] + dt * (omega1 + epsilon * sin(n * phase_2[j] - m * phase_1[j]))
        phase_1[j + 1] = phase1 % _2pi
        freq_dev = sigma_f2 * np.random.randn(1)[0]
        omega2 = (f2 + freq_dev) * _2pi
        phase2 = phase_2[j] + dt * (omega2 + epsilon * sin(m * phase_1[j] - n * phase_2[j]))
        phase_2[j + 1] = phase2 % _2pi
    return phase_1, phase_2


def nonsin_sig_kuramoto(duration, fs, epsilon, f1, coeff=0.3):
    from numpy import pi, sin
    phase_1, phase_2 = kuramuto_oscillators(duration, fs, n=1, m=2, f1=f1, f2=2*f1,
                                            sigma_f1=2, sigma_f2=4, epsilon=epsilon)
    phase_lag = np.random.random(1)[0] * pi
    x = sin(phase_1)
    y = sin(phase_2 + phase_lag)
    sig = x + coeff * y
    return sig, (x, y, phase_lag)


def noisy_narrowband(f1, f2, sfreq, snr, n_time_samples, n_sig=1):
    times = np.arange(0, n_time_samples) / sfreq
    noise = _data_fun_pink_noise(times)
    noise = noise[np.newaxis, :]
    s = filtered_randn(f1, f2, sfreq, n_time_samples, n_sig)
    s, factor = _adjust_snr2(s, noise, f1, f2, sfreq, snr)
    sig = s + noise
    return sig, s, noise


def _adjust_snr2(source_data, noise_data, f1, f2, fs, desired_snr=1):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, np.array([f1, f2])/fs * 2, 'bandpass')
    noise_data = filtfilt(b, a, noise_data)
    noise_data = np.real(noise_data)
    noise_var = np.sum(np.var(noise_data, 1))

    source_var = np.sum(np.var(source_data, 1))
    snr_current = source_var/noise_var
    factor = np.sqrt(snr_current/desired_snr)
    source_data /= factor
    # source_raw = mne.io.RawArray(source_data, source_raw.info)
    return source_data, factor


def _data_fun_pink_noise(times):
    import colorednoise as cn
    n_sample = len(times)
    data = cn.powerlaw_psd_gaussian(1, n_sample)
    data /= np.linalg.norm(data)
    return data


def filtered_randn(f1, f2, sfreq, n_time_samples, n_sig=1):
    x1 = np.random.randn(n_sig, n_time_samples)
    b1, a1 = butter(N=2, Wn=np.array([f1, f2])/sfreq*2, btype='bandpass')
    x1 = filtfilt(b1, a1, x1, axis=1)
    x1_h = hilbert_(x1)
    return x1_h


def correlated_ampenv(x, c, fs, f1, f2):
    from scipy.signal import butter, filtfilt
    from tools_general import hilbert_
    from scipy.stats import pearsonr
    if not np.iscomplexobj(x):
        x = hilbert_(x)
    y1 = np.real(x) + c * np.random.randn(x.shape[0], x.shape[1])
    b1, a1 = butter(N=2, Wn=np.array([f1, f2]) / fs * 2, btype='bandpass')
    y1 = filtfilt(b1, a1, y1, axis=1)
    ampenv = np.abs(hilbert_(y1))
    corr = pearsonr(np.abs(hilbert_(y1))[0, :], np.abs(x)[0, :])
    return ampenv, corr


def _produce_random_oscillation(fc, fs, n_osc, times):
    #  _produce_random_oscillation(f_low, f_high, fs, n_osc, times)
    data = np.random.randn(1, n_osc, len(times))
    # iir_params = dict(order=2, ftype='butter')
    # data = mne.filter.filter_data(data, sfreq=fs, l_freq=f_low, h_freq=f_high, n_jobs=2, method='iir',
    #                              phase='zero-double', iir_params=iir_params)
    # data_cmplx, n_zp = zero_pad_to_pow2(data)
    data_cmplx = mne.time_frequency.tfr_array_morlet(data, fs, np.array([fc]), n_cycles=5.0, zero_mean=True,
                                                     use_fft=True, decim=1, output='complex', n_jobs=2)
    # data_cmplx = data_cmplx[:, :-n_zp]
    data_cmplx = data_cmplx.reshape((n_osc, len(times)))
    data_angle = np.angle(data_cmplx)
    data_abs = np.abs(data_cmplx)
    data_real = np.real(data_cmplx)
    return data_real, data_angle, data_abs, data_cmplx


def produce_nonsin_sig(n_sample, fs, dphi_beta, nonsin_mode, ampcorr_coeff=None):
    data_alpha_nonsin = filtered_randn(8, 12, fs, n_sample)
    data_alpha_nonsin = data_alpha_nonsin / np.std(np.real(data_alpha_nonsin), axis=1)[:, np.newaxis]
    data_beta_nonsin = produce_nm_phase_locked_sig(data_alpha_nonsin, dphi_beta, 1, 2, [8, 12], fs)
    phase_beta = np.angle(data_beta_nonsin)
    if nonsin_mode == 1:  # different amplitude envelopes
        pass
    elif nonsin_mode == 2:  # the same amplitude envelopes
        abs_alpha = np.abs(data_alpha_nonsin)
        data_beta_nonsin = abs_alpha * np.exp(1j * phase_beta)
    elif nonsin_mode == 3:  # correlated amplitude envelopes
        abs_beta, _ = correlated_ampenv(data_alpha_nonsin, ampcorr_coeff, fs, 8, 12)
        data_beta_nonsin = abs_beta * np.exp(1j * phase_beta)
    data_beta_nonsin = data_beta_nonsin / np.std(np.real(data_beta_nonsin), axis=1)[:, np.newaxis]
    # data_nonsin = data_alpha_nonsin + 0.2 * data_beta_nonsin
    return data_alpha_nonsin, data_beta_nonsin


def produce_nm_phase_locked_non_sin(sig, phase_lag, n, sfreq, dphi_beta=0, kappa=None):
    if not np.iscomplexobj(sig):
        sig = hilbert_(sig)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]
    data_alpha_nonsin = produce_nm_phase_locked_sig(sig, phase_lag, n, 1, [8, 12], sfreq, kappa)
    data_alpha_nonsin = data_alpha_nonsin / np.std(data_alpha_nonsin, axis=1)[:, np.newaxis]
    data_beta_nonsin = produce_nm_phase_locked_sig(data_alpha_nonsin, dphi_beta, 1, 2, [8, 12], sfreq)
    data_beta_nonsin = data_beta_nonsin / np.std(data_beta_nonsin, axis=1)[:, np.newaxis]
    # data_nonsin = data_alpha_nonsin + 0.2 * data_beta_nonsin
    return data_alpha_nonsin, data_beta_nonsin


def produce_nm_phase_locked_sig(sig, phase_lag, n, m, wn_base, sfreq, nonsin_mode=2, kappa=None):
    """

    :param sig:
    :param phase_lag:
    :param n:
    :param m:
    :param wn_base:
    :param sfreq:
    :param kappa:
                if None, the signals are completely locked to each other
    :return:
    """

    if not np.iscomplexobj(sig):
        sig = hilbert_(sig)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]

    sig_angle = np.angle(sig)
    n_samples = sig.shape[1]
    if nonsin_mode == 2: # the same amplitude envelopes
        sig_abs = np.abs(sig)
    else:
        sig_ = filtered_randn(m*wn_base[0], m*wn_base[1], sfreq, n_samples)
        sig_abs = np.abs(sig_)
    if kappa is None:
        sig_hat = sig_abs * np.exp(1j * m / n * sig_angle + 1j * phase_lag)
    # TODO: kappa von mises
    return sig_hat


def _adjust_snr(source_data, noise_data, fc, fs, desired_snr=1):
    noise_data = noise_data.reshape((1, noise_data.shape[0], noise_data.shape[1]))
    noise_data = mne.time_frequency.tfr_array_morlet(noise_data, fs, np.array([fc]),
                                                     n_cycles=5.0, zero_mean=True, use_fft=True, decim=1,
                                                     output='complex', n_jobs=1, verbose=None)
    noise_data = np.real(np.squeeze(noise_data))
    noise_var = np.sum(np.var(noise_data, 1))

    source_var = np.sum(np.var(source_data, 1))
    snr_current = source_var/noise_var
    factor = np.sqrt(snr_current/desired_snr)
    source_data /= factor
    # source_raw = mne.io.RawArray(source_data, source_raw.info)
    return source_data, factor


def _adjust_snr_osc(fc, fs, subject, fwd, noise_raw, data_source, vertices, snr, raw_info):
    stc = mne.SourceEstimate(data_source, vertices, tmin=0, tstep=1 / fs, subject=subject)
    raw = mne.apply_forward_raw(fwd=fwd, stc=stc, info=raw_info)
    if noise_raw is not None:
        data_sensor, factor = _adjust_snr(source_data=raw.get_data(), noise_data=noise_raw.get_data(),
                                          fc=fc, fs=fs, desired_snr=snr)
    else:
        data_sensor = raw.get_data()
        factor = 1
    data_source /= factor
    return data_source, data_sensor


def exclude_medialwall_vert(labels_med, src):
    from tools_source_space import label_idx_whole_brain
    _, idx_lh = label_idx_whole_brain(src, labels_med[0])
    _, idx_rh = label_idx_whole_brain(src, labels_med[1])
    vert_lh, vert_rh = src[0]['vertno'], src[1]['vertno']
    vert_lh = np.delete(vert_lh, idx_lh)
    vert_rh = np.delete(vert_rh, idx_rh)
    vert = [vert_lh, vert_rh]
    return vert


def simulate_sparse_stc_2(src, data, sfreq, labels_med,
                        labels=None, random_state=None, location='random',
                        subject=None, subjects_dir=None, surf='sphere'):
    """
    adopted from mne.simulation.simulate_sparse_stc
    :param src:
    :param data:
    :param sfreq:
    :param labels:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    from mne.utils import check_random_state, warn
    rng = check_random_state(random_state)
    subject_src = src[0].get('subject_his_id')
    if subject is None:
        subject = subject_src
    elif subject_src is not None and subject != subject_src:
        raise ValueError('subject argument (%s) did not match the source '
                         'space subject_his_id (%s)' % (subject, subject_src))
    n_dipoles = data.shape[0]

    if labels is None:
        # can be vol or surface source space
        offsets = np.linspace(0, n_dipoles, len(src) + 1).astype(int)
        n_dipoles_ss = np.diff(offsets)
        # don't use .choice b/c not on old numpy
        vert_new = exclude_medialwall_vert(labels_med, src)
        vs = [s[np.sort(rng.permutation(np.arange(len(s)))[:n])]
              for n, s in zip(n_dipoles_ss, vert_new)]
        # vs = [s['vertno'][np.sort(rng.permutation(np.arange(s['nuse']))[:n])]
        #      for n, s in zip(n_dipoles_ss, src)]  # \Mina: it takes equal number of vertices from the two hemispheres
        datas = data
    else:  # TODO : should be checked
        from mne.simulation import select_source_in_label

        if n_dipoles != len(labels):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % min(n_dipoles, len(labels)))
        labels = labels[:n_dipoles] if n_dipoles < len(labels) else labels

        vertno = [[], []]
        lh_data = [np.empty((0, data.shape[1]))]
        rh_data = [np.empty((0, data.shape[1]))]
        for i, label in enumerate(labels):
            lh_vertno, rh_vertno = select_source_in_label(
                src, label, rng, location, subject, subjects_dir, surf)
            vertno[0] += lh_vertno
            vertno[1] += rh_vertno
            if len(lh_vertno) != 0:
                lh_data.append(data[i][np.newaxis])
            elif len(rh_vertno) != 0:
                rh_data.append(data[i][np.newaxis])
            else:
                raise ValueError('No vertno found.')
        vs = [np.array(v) for v in vertno]
        datas = [np.concatenate(d) for d in [lh_data, rh_data]]
        # need to sort each hemi by vertex number
        for ii in range(2):
            order = np.argsort(vs[ii])
            vs[ii] = vs[ii][order]
            if len(order) > 0:  # fix for old numpy
                datas[ii] = datas[ii][order]
        datas = np.concatenate(datas)

    tmin, tstep = 0, 1 / sfreq
    stc = mne.SourceEstimate(datas, vertices=vs, tmin=tmin, tstep=tstep, subject=subject)
    # TODO: return the index of the sources corresponding to data
    return stc


def simulate_sparse_stc_vertices(src, n_dipoles, labels_med,
                                 labels_=None, random_state=None, location='center',
                                 subject=None, subjects_dir=None, surf='sphere'):
    """
    adopted from mne.simulation.simulate_sparse_stc

    """
    from mne.utils import check_random_state, warn
    rng = check_random_state(random_state)
    subject_src = src[0].get('subject_his_id')
    if subject is None:
        subject = subject_src
    elif subject_src is not None and subject != subject_src:
        raise ValueError('subject argument (%s) did not match the source '
                         'space subject_his_id (%s)' % (subject, subject_src))
    if labels_ is None:
        # can be vol or surface source space
        offsets = np.linspace(0, n_dipoles, len(src) + 1).astype(int)
        n_dipoles_ss = np.diff(offsets)
        # don't use .choice b/c not on old numpy
        vert_new = exclude_medialwall_vert(labels_med, src)
        vs = [s[np.sort(rng.permutation(np.arange(len(s)))[:n])]
              for n, s in zip(n_dipoles_ss, vert_new)]
        # vs = [s['vertno'][np.sort(rng.permutation(np.arange(s['nuse']))[:n])]
        #      for n, s in zip(n_dipoles_ss, src)]  # \Mina: it takes equal number of vertices from the two hemispheres
    else:  # TODO : should be checked
        from mne.simulation import select_source_in_label

        if n_dipoles != len(labels_):
            warn('The number of labels is different from the number of '
                 'dipoles. %s dipole(s) will be generated.'
                 % min(n_dipoles, len(labels_)))
        labels = labels_[:n_dipoles] if n_dipoles < len(labels_) else labels_

        vertno = [[], []]
        for i, label in enumerate(labels_):
            lh_vertno, rh_vertno = select_source_in_label(
                src, label, rng, location, subject, subjects_dir, surf)
            vertno[0] += lh_vertno
            vertno[1] += rh_vertno

        vs = [np.array(v) for v in vertno]
        # need to sort each hemi by vertex number
        for ii in range(2):
            order = np.argsort(vs[ii])
            vs[ii] = vs[ii][order]
    return vs


def simulate_stc_conn_0(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info,
                      labels_med, labels=None, random_state=None, location='random',
                      subject=None, subjects_dir=None, surf='sphere'):
    """
    doe not have noise.
    :param n_sources:
    :param n_connections:
    :param n_sample:
    :param src:
    :param sfreq:
    :param fwd_fixed:
    :param raw_info:
    :param labels_med:
    :param labels:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn = n_connections['n_alpha_conn']
    n_beta_conn = n_connections['n_beta_conn']

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha

    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_alpha = np.random.permutation(n_alpha + n_nonsin)[:n_alpha_conn * 2].reshape((-1, 2))
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = np.random.random(1) * np.pi  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):
        if np.sum(sig_alpha_nonsin[n_a, :]): # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
            else:  # it is a nonsin-source
                dphi_beta0 = np.random.random(1) * np.pi
                sig_alpha_nonsin[n_a, :], data_alpha_nonsin = produce_nonsin_sig(n_sample, sfreq, dphi_beta0)
            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o][0]
                dphi = np.random.random(1) * np.pi  # phase-shift
                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = np.random.random(1) * np.pi
                    sig_alpha_nonsin[n_a2, :] = produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1,
                                                                                sfreq, dphi_beta=dphi_beta, kappa=None)
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = np.random.random(1) * np.pi
                    sig_alpha_nonsin[n_a2, :] = produce_nm_phase_locked_non_sin(data_alpha_nonsin, dphi, 1,
                                                                                sfreq, dphi_beta=dphi_beta, kappa=None)
    data = np.real(np.concatenate((sig_alpha_nonsin, sig_beta), axis=0))
    source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med, random_state=random_state,
                                       location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)
    return simulated_raw, source_stc


def simulate_stc_conn(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                      labels_med, labels_=None, random_state=None, location='random',
                      subject=None, subjects_dir=None, surf='sphere'):
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']
    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    # vertices_ = np.concatenate(vertices)
    vertices_idx_full = vertex_index_full(vertices, src)
    vert_beta = vertices_idx_full[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    # number of connections ------------------------------------------
    n_alpha_conn = n_connections['n_alpha_conn']
    n_beta_conn = n_connections['n_beta_conn']

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = np.concatenate(noise_stc.vertices)
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_alpha = np.random.permutation(n_alpha + n_nonsin)[:n_alpha_conn * 2].reshape((-1, 2))
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                                             vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 =  pi / 2 * np.random.random(1) + np.pi/4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0)
                factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                data_alpha_nonsin_n_a /= factor
                factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                            vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta =  pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)
    source_stc_orig = mne.SourceEstimate(data, vertices=vertices, tmin=0, tstep=1/sfreq, subject=subject)
    return simulated_raw, source_stc, source_stc_orig


def _adjust_snr_factor(source_data, noise_data, wn, vert_source, leadfield, desired_snr, fs):
    source_data = np.real(source_data)
    noise_data1 = np.real(noise_data)
    b, a = butter(2, np.asarray(wn) / fs * 2, btype='bandpass')
    noise_data1 = filtfilt(b, a, noise_data1, axis=1)
    noise_data1 = np.real(noise_data1)
    noise_var = np.mean(noise_data1 ** 2)
    source_var = np.mean(source_data ** 2)
    leadfield_vert = leadfield[:, vert_source]
    source_var2 = source_var * np.mean(leadfield_vert ** 2)
    snr_current = source_var2 / noise_var
    factor = np.sqrt(snr_current / desired_snr)
    return factor


def _check_snr(source_data, noise_data, wn, vert_source, leadfield):
    source_data = np.real(source_data)
    noise_data = np.real(noise_data)
    b, a = butter(2, np.asarray(wn) / fs * 2, btype='bandpass')
    noise_data = filtfilt(b, a, noise_data, axis=1)
    noise_data = np.real(noise_data)
    noise_var = np.mean(noise_data ** 2)
    source_var = np.mean(source_data ** 2)
    leadfield_vert = leadfield[:, vert_source]
    source_var2 = source_var * np.mean(leadfield_vert ** 2)
    snr = source_var2 / noise_var
    return snr


def simulate_stc_conn_v1(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                         labels_med, nonsin_mode=2, ampcorr_coeff=0.5, labels_=None, random_state=None,
                         location='center', subject=None, subjects_dir=None, surf='sphere'):
    """
    with cfc
    :param n_sources:
    :param n_connections:
    :param n_sample:
    :param src:
    :param sfreq:
    :param fwd_fixed:
    :param raw_info:
    :param leadfield:
    :param snr:
    :param labels_med:
    :param labels_:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    # the final version of simulation scenarios 1 and 2
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']
    n_cfc_conn = n_connections['n_cfc_conn']
    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data / np.max(np.abs(noise_stc.data)) * 1e-1
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    #signals with CFC
    sig_alpha_rem = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_rem, conn_ind_alpha.ravel(), return_indices=True)
    sig_alpha_rem = np.delete(sig_alpha_rem, ind1)

    sig_beta_rem = np.arange(n_beta)
    _, ind1, _ = np.intersect1d(sig_beta_rem, conn_ind_beta.ravel(), return_indices=True)
    sig_beta_rem = np.delete(sig_beta_rem, ind1)

    vert_sigs_ind['conn_ind_cfc'] = np.append(sig_alpha_rem[:, np.newaxis], sig_beta_rem[:, np.newaxis], axis=1)

    # the containers for the signals
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 = pi / 2 * np.random.random(1) + np.pi / 4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0,
                                                                                 nonsin_mode, ampcorr_coeff)
                factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                data_alpha_nonsin_n_a /= factor
                factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                            vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = pi / 2 * np.random.random(1) + np.pi / 4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi / 4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    phase_beta = np.angle(data_beta_nonsin)
                    if nonsin_mode == 1:  # different amplitude envelopes
                        pass
                    elif nonsin_mode == 2:  # the same amplitude envelopes
                        abs_alpha = np.abs(data_alpha_nonsin)
                        data_beta_nonsin = abs_alpha * np.exp(1j * phase_beta)
                    elif nonsin_mode == 3:  # correlated amplitude envelopes
                        abs_beta = correlated_ampenv(data_alpha_nonsin, ampcorr_coeff, 8, 12)
                        data_beta_nonsin = abs_beta * np.exp(1j * phase_beta)
                    data_beta_nonsin = data_beta_nonsin / np.std(data_beta_nonsin, axis=1)[:, np.newaxis]
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin

            if np.sum(np.isin(sig_alpha_rem, n_a)):  # if this source is CFC to any other alpha source
                n_b1 = sig_beta_rem[sig_alpha_rem == n_a]
                dphi = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift
                if n_a < n_alpha:
                    sig_beta[n_b1, :] = produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :],
                                                                    dphi, 1, 2, [8, 12], sfreq)
                else:
                    sig_beta[n_b1, :] = produce_nm_phase_locked_sig(data_alpha_nonsin_n_a,
                                                                    dphi, 1, 2, [8, 12], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b1, :], noise_data, [16, 24],
                                            vert_beta[n_b1], leadfield, snr_beta, sfreq)
                sig_beta[n_b1, :] /= factor

    # Generate beta signals and connections-------------------------
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                        vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-1
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind


def simulate_stc_conn_v1_cfc(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                         labels_med, nonsin_mode=2, ampcorr_coeff=0.5, labels_=None, random_state=None,
                         location='center', subject=None, subjects_dir=None, surf='sphere'):
    """
    with cfc
    :param n_sources:
    :param n_connections:
    :param n_sample:
    :param src:
    :param sfreq:
    :param fwd_fixed:
    :param raw_info:
    :param leadfield:
    :param snr:
    :param labels_med:
    :param labels_:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    # the final version of simulation scenarios 1 and 2
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']
    n_cfc_conn = n_connections['n_cfc_conn']
    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data / np.max(np.abs(noise_stc.data)) * 1e-1
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]

    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    # signals with beta
    beta_perm = np.random.permutation(n_beta)
    conn_ind_beta = beta_perm[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    # signals with nonsin
    nonsin_perm = np.random.permutation(n_nonsin) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = np.array([])
    conn_ind_nonsin = np.array([])
    if n_nonsin:
        conn_ind_nonsin = nonsin_perm[:n_nonsin_conn * 2].reshape((-1, 2))
        vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    # signals with CFC
    vert_sigs_ind['conn_ind_cfc'] = np.array([])
    alpha_perm = np.random.permutation(n_alpha)
    conn_ind_cfc = np.array([])
    conn_ind_cfc_a = np.array([])
    conn_ind_cfc_b = np.array([])
    if n_cfc_conn:
        conn_ind_cfc_b = beta_perm[n_beta_conn*2:n_beta_conn*2+n_cfc_conn]
        conn_ind_cfc_a = alpha_perm[:n_cfc_conn]
        conn_ind_cfc = np.append(conn_ind_cfc_a[:, np.newaxis], conn_ind_cfc_b[:, np.newaxis], axis=1)
        vert_sigs_ind['conn_ind_cfc'] = conn_ind_cfc

    # signals with alpha
    conn_ind_alpha_1 = np.array([])
    vert_sigs_ind['conn_ind_alpha'] = np.array([])
    if n_alpha_conn:
        sig_alpha_rem = np.append(alpha_perm[n_cfc_conn:], nonsin_perm[n_nonsin_conn*2:])
        conn_ind_alpha_1 = np.random.permutation(sig_alpha_rem)[:n_alpha_conn * 2].reshape((-1, 2))
        vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)



    # the containers for the signals
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                if n_noise:
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 = pi / 2 * np.random.random(1) + np.pi / 4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0,
                                                                                 nonsin_mode, ampcorr_coeff)
                if n_noise:
                    factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin_n_a /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    if n_noise:
                        factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                    vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                        sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = pi / 2 * np.random.random(1) + np.pi / 4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    if n_noise:
                        factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                    vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                        data_alpha_nonsin /= factor
                        factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                    vert_alpha_nonsin[n_a2], leadfield, snr_beta, sfreq)
                        data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi / 4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    phase_beta = np.angle(data_beta_nonsin)
                    if nonsin_mode == 1:  # different amplitude envelopes
                        pass
                    elif nonsin_mode == 2:  # the same amplitude envelopes
                        abs_alpha = np.abs(data_alpha_nonsin)
                        data_beta_nonsin = abs_alpha * np.exp(1j * phase_beta)
                    elif nonsin_mode == 3:  # correlated amplitude envelopes
                        abs_beta = correlated_ampenv(data_alpha_nonsin, ampcorr_coeff, 8, 12)
                        data_beta_nonsin = abs_beta * np.exp(1j * phase_beta)
                    data_beta_nonsin = data_beta_nonsin / np.std(data_beta_nonsin, axis=1)[:, np.newaxis]
                    if n_noise:
                        factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                    vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                        data_alpha_nonsin /= factor
                        factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                    vert_alpha_nonsin[n_a2], leadfield, snr_beta, sfreq)
                        data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin

            if np.sum(conn_ind_cfc == n_a):  # if this source is CFC to any other alpha source
                n_b1 = conn_ind_cfc_b[conn_ind_cfc_a == n_a]
                dphi = pi / 2 * np.random.random(1) + np.pi / 4  # phase-shift
                if n_a < n_alpha:
                    sig_beta[n_b1, :] = produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :],
                                                                    dphi, 1, 2, [8, 12], sfreq)
                else:
                    sig_beta[n_b1, :] = produce_nm_phase_locked_sig(data_alpha_nonsin_n_a,
                                                                    dphi, 1, 2, [8, 12], sfreq)
                if n_noise:
                    factor = _adjust_snr_factor(sig_beta[n_b1, :], noise_data, [16, 24],
                                                vert_beta[n_b1], leadfield, snr_beta, sfreq)
                    sig_beta[n_b1, :] /= factor

    # Generate beta signals and connections-------------------------
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            if n_noise:
                factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                            vert_beta[n_b], leadfield, snr_beta, sfreq)
                sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                if n_noise:
                    factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                                vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                    sig_beta[n_b2, :] /= factor

    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    if n_noise:
        data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-1
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind


def simulate_stc_conn_v2(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                      labels_med, nonsin_mode=2, ampcorr_coeff=0.5, labels_=None, random_state=None, location='center',
                      subject=None, subjects_dir=None, surf='sphere'):
    """

    :param n_sources:
    :param n_connections:
    :param n_sample:
    :param src:
    :param sfreq:
    :param fwd_fixed:
    :param raw_info:
    :param leadfield:
    :param snr:
    :param labels_med:
    :param labels_:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    # the final version of simulation scenarios 1 and 2
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']

    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data /  np.max(np.abs(noise_stc.data)) * 1e-1
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                        vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 = pi / 2 * np.random.random(1) + np.pi/4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0,
                                                                                 nonsin_mode, ampcorr_coeff)
                factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                data_alpha_nonsin_n_a /= factor
                factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                            vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    phase_beta = np.angle(data_beta_nonsin)
                    if nonsin_mode == 1:  # different amplitude envelopes
                        pass
                    elif nonsin_mode == 2:  # the same amplitude envelopes
                        abs_alpha = np.abs(data_alpha_nonsin)
                        data_beta_nonsin = abs_alpha * np.exp(1j * phase_beta)
                    elif nonsin_mode == 3:  # correlated amplitude envelopes
                        abs_beta = correlated_ampenv(data_alpha_nonsin, ampcorr_coeff, 8, 12)
                        data_beta_nonsin = abs_beta * np.exp(1j * phase_beta)
                    data_beta_nonsin = data_beta_nonsin / np.std(data_beta_nonsin, axis=1)[:, np.newaxis]
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-1
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind

def simulate_stc_conn_v2_prime0(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                      labels_med, nonsin_mode=2, ampcorr_coeff=0.5, labels_=None, random_state=None, location='center',
                      subject=None, subjects_dir=None, surf='sphere'):
    """

    :param n_sources:
    :param n_connections:
    :param n_sample:
    :param src:
    :param sfreq:
    :param fwd_fixed:
    :param raw_info:
    :param leadfield:
    :param snr:
    :param labels_med:
    :param labels_:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    # the final version of simulation scenarios 1 and 2
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']

    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data / np.max(np.abs(noise_stc.data)) * 1e-1
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                        vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 = pi / 2 * np.random.random(1) + np.pi/4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0,
                                                                                 nonsin_mode, ampcorr_coeff)
                power_alpha = np.mean(np.real(data_alpha_nonsin_n_a) ** 2)
                power_beta = np.mean(np.real(data_beta_nonsin_n_a) ** 2)
                data_alpha_nonsin_n_a = data_alpha_nonsin_n_a / np.sqrt(power_alpha)
                data_beta_nonsin_n_a = data_beta_nonsin_n_a / np.sqrt(power_beta)
                nonsin_sig = data_alpha_nonsin_n_a + 0.3 * data_beta_nonsin_n_a

                factor1 = _adjust_snr_factor(nonsin_sig, noise_data, [8, 12],
                                             vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                nonsin_sig /= factor1

                sig_alpha_nonsin[n_a, :] = nonsin_sig

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)

                    power_alpha = np.mean(np.real(data_alpha_nonsin) ** 2)
                    power_beta = np.mean(np.real(data_beta_nonsin) ** 2)
                    data_alpha_nonsin = data_alpha_nonsin / np.sqrt(power_alpha)
                    data_beta_nonsin = data_beta_nonsin / np.sqrt(power_beta)
                    nonsin_sig = data_alpha_nonsin + 0.3 * data_beta_nonsin

                    factor = _adjust_snr_factor(nonsin_sig, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    nonsin_sig /= factor
                    sig_alpha_nonsin[n_a2, :] = nonsin_sig
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    phase_beta = np.angle(data_beta_nonsin)
                    if nonsin_mode == 1:  # different amplitude envelopes
                        pass
                    elif nonsin_mode == 2:  # the same amplitude envelopes
                        abs_alpha = np.abs(data_alpha_nonsin)
                        data_beta_nonsin = abs_alpha * np.exp(1j * phase_beta)
                    elif nonsin_mode == 3:  # correlated amplitude envelopes
                        abs_beta = correlated_ampenv(data_alpha_nonsin, ampcorr_coeff, 8, 12)
                        data_beta_nonsin = abs_beta * np.exp(1j * phase_beta)
                    data_beta_nonsin = data_beta_nonsin / np.std(data_beta_nonsin, axis=1)[:, np.newaxis]

                    power_alpha = np.mean(np.real(data_alpha_nonsin) ** 2)
                    power_beta = np.mean(np.real(data_beta_nonsin) ** 2)
                    data_alpha_nonsin = data_alpha_nonsin / np.sqrt(power_alpha)
                    data_beta_nonsin = data_beta_nonsin / np.sqrt(power_beta)
                    nonsin_sig = data_alpha_nonsin + 0.3 * data_beta_nonsin

                    factor = _adjust_snr_factor(nonsin_sig, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    nonsin_sig /= factor
                    sig_alpha_nonsin[n_a2, :] = nonsin_sig
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-1
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind

def simulate_stc_conn_v2_prime(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                      labels_med, nonsin_mode=2, ampcorr_coeff=0.5, labels_=None, random_state=None, location='center',
                      subject=None, subjects_dir=None, surf='sphere'):
    """

    :param n_sources:
    :param n_connections:
    :param n_sample:
    :param src:
    :param sfreq:
    :param fwd_fixed:
    :param raw_info:
    :param leadfield:
    :param snr:
    :param labels_med:
    :param labels_:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    # the final version of simulation scenarios 1 and 2
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']

    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data /  np.max(np.abs(noise_stc.data)) * 1e-7
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                        vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                sig_alpha_nonsin_1, sig_alpha_nonsin_all = nonsin_sig_kuramoto(n_sample / sfreq,
                                                                               sfreq, 10, 10, coeff=0.3)
                data_alpha_nonsin_n_a = sig_alpha_nonsin_all[0]
                data_beta_nonsin_n_a = sig_alpha_nonsin_all[1]
                factor = _adjust_snr_factor(sig_alpha_nonsin_1, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin_1 /= factor
                sig_alpha_nonsin[n_a, :] = sig_alpha_nonsin_1

        if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    sig_alpha_nonsin_2 = np.real(np.exp(1j * np.angle(np.hilbert_(data_alpha_nonsin)))) + \
                                         0.3 * np.real(np.exp(1j * np.angle(np.hilbert_(data_beta_nonsin))))
                    factor = _adjust_snr_factor(sig_alpha_nonsin_2, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin_2 /= factor
                    sig_alpha_nonsin[n_a2, :] = sig_alpha_nonsin_2
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_x1x2 = pi / 2 * np.random.random(1) + pi/4
                    dphi_y1y2 = pi / 2 * np.random.random(1) + pi / 4
                    sig_alpha_nonsin_2 = np.real(hilbert_(data_alpha_nonsin_n_a) * np.exp(1j * dphi_x1x2)) + 0.3 * np.real(
                        hilbert_(data_beta_nonsin_n_a) * np.exp(1j * dphi_y1y2))

                    factor = _adjust_snr_factor(sig_alpha_nonsin_2, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    sig_alpha_nonsin_2 /= factor
                    sig_alpha_nonsin[n_a2, :] = sig_alpha_nonsin_2
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-7
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind

def simulate_stc_conn_v22(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                      labels_med, nonsin_mode=2, ampcorr_coeff=0.5, labels_=None, random_state=None, location='center',
                      subject=None, subjects_dir=None, surf='sphere'):
    """

    :param n_sources:
    :param n_connections:
    :param n_sample:
    :param src:
    :param sfreq:
    :param fwd_fixed:
    :param raw_info:
    :param leadfield:
    :param snr:
    :param labels_med:
    :param labels_:
    :param random_state:
    :param location:
    :param subject:
    :param subjects_dir:
    :param surf:
    :return:
    """
    # the final version of simulation scenarios 1 and 2
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']

    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data /  np.max(np.abs(noise_stc.data)) * 1e-7
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                        vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 = pi / 2 * np.random.random(1) + np.pi/4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0,
                                                                                 nonsin_mode, ampcorr_coeff)
                factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                data_alpha_nonsin_n_a /= factor
                factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                            vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    phase_beta = np.angle(data_beta_nonsin)
                    if nonsin_mode == 1:  # different amplitude envelopes
                        pass
                    elif nonsin_mode == 2:  # the same amplitude envelopes
                        abs_alpha = np.abs(data_alpha_nonsin)
                        data_beta_nonsin = abs_alpha * np.exp(1j * phase_beta)
                    elif nonsin_mode == 3:  # correlated amplitude envelopes
                        abs_beta = correlated_ampenv(data_alpha_nonsin, ampcorr_coeff, 8, 12)
                        data_beta_nonsin = abs_beta * np.exp(1j * phase_beta)
                    # data_beta_nonsin = data_beta_nonsin / np.std(data_beta_nonsin, axis=1)[:, np.newaxis]
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-7
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind


def simulate_stc_conn_v3(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                      labels_med, labels_=None, random_state=None, location='center',
                      subject=None, subjects_dir=None, surf='sphere'):
    # the version for event related beta
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin
    n_beta_ERD = n_sources['n_beta_ERD']
    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']

    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data /  np.max(np.abs(noise_stc.data)) * 1e-7
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    # Generate beta signals and connections--------------------------
    # sig_beta_ERD = np.zeros((n_beta_ERD, n_sample), dtype='complex')
    # for n_b in range(n_beta_ERD):



    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                        vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 =  pi / 2 * np.random.random(1) + np.pi/4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0)
                factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                data_alpha_nonsin_n_a /= factor
                factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                            vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta =  pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-7
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind



def simulate_stc_conn_v4(n_sources, n_connections, n_sample, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                      labels_med, labels_=None, random_state=None, location='random',
                      subject=None, subjects_dir=None, surf='sphere'):
    from tools_source_space import vertex_index_full
    from numpy import pi
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']
    n_beta_nonsin_conn = n_connections['n_beta_nonsin_conn']
    n_beta_alpha_conn = n_connections['n_beta_alpha_conn']

    # determine the edges
    vert_sigs_ind = dict()

    # within-freq
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    # cfc ----
    conn_ind_alpha_beta = np.zeros((n_beta_alpha_conn+n_beta_nonsin_conn, 2))
    # alpha-beta
    beta_perm = np.random.permutation(n_beta)[:n_beta_alpha_conn][:, np.newaxis]
    alpha_perm = np.random.permutation(n_alpha)[:n_beta_alpha_conn][:, np.newaxis]
    conn_ind_alpha_beta[:n_beta_alpha_conn, :] = np.append(beta_perm, alpha_perm, axis=1)

    beta_perm = np.random.permutation(n_beta)[:n_beta_alpha_conn][:, np.newaxis]
    nonsin_perm = np.random.permutation(n_nonsin)[:n_beta_nonsin_conn][:, np.newaxis] + n_alpha
    conn_ind_alpha_beta[n_beta_alpha_conn:, :] = np.append(beta_perm, nonsin_perm, axis=1)
    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = np.concatenate(vertices)
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]
    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = np.concatenate(noise_stc.vertices)
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                                             vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 =  pi / 2 * np.random.random(1) + np.pi/4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0)
                factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                data_alpha_nonsin_n_a /= factor
                factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                            vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta =  pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)
    source_stc_orig = mne.SourceEstimate(data, vertices=vertices, tmin=0, tstep=1/sfreq, subject=subject)
    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind


def simulate_sparse_stc_srcsig(source_sig, src, sfreq, fwd_fixed, raw_info, leadfield, snr,
                               labels_med, labels_=None, random_state=None, location='center',
                               subject=None, subjects_dir=None, surf='sphere'):

    # the final version of simulation scenarios 1 and 2
    from tools_source_space import vertex_index_full
    from numpy import pi
    n_sources, n_sample = source_sig.shape
    times = np.arange(0, n_sample) / sfreq
    n_vox = len(src[0]['vertno']) + len(src[1]['vertno'])
    vertices_full = [src[0]['vertno'], src[1]['vertno']]
    snr_alpha = snr['alpha']
    snr_beta = snr['beta']

    # number of sources in each type ----------------------------------
    n_source = n_sources['n_source']
    n_alpha = n_sources['n_alpha']
    n_nonsin = n_sources['n_nonsin']
    n_noise = n_sources['n_noise']
    n_beta = n_source - n_alpha - n_nonsin

    # number of connections ------------------------------------------
    n_alpha_conn_total = n_connections['n_alpha_conn']
    n_nonsin_conn = n_connections['n_nonsin_conn']
    n_alpha_conn = n_alpha_conn_total - n_nonsin_conn
    n_beta_conn = n_connections['n_beta_conn']

    # vertices ----------------------------------
    vertices = simulate_sparse_stc_vertices(src=src, n_dipoles=n_source, labels_med=labels_med,
                                            labels_=labels_, random_state=random_state, location=location,
                                            subject=subject, subjects_dir=subjects_dir, surf=surf)
    vertices_ = vertices.copy()
    vertices_idx_full1 = vertex_index_full(vertices_, src)
    ind_vert_perm = np.random.permutation(np.arange(len(vertices_idx_full1)))
    vertices_idx_full = vertices_idx_full1[ind_vert_perm]
    vert_sigs_ind = dict()
    vert_beta = vertices_idx_full[:n_beta]
    vert_sigs_ind['beta'] = ind_vert_perm[:n_beta]
    vert_alpha_nonsin = vertices_idx_full[n_beta:]
    vert_sigs_ind['alpha'] = ind_vert_perm[n_beta:n_beta+n_alpha]
    vert_sigs_ind['nonsin'] = ind_vert_perm[n_beta + n_alpha:]

    # TODO: Error-check for the comparison n_alpha_conn and n_alpha etc
    # n_alpha_conn * 2 <= n_alpha
    # len(labels_) and total number of dipoles

    if n_noise:
        from mne.simulation import simulate_sparse_stc
        noise_stc = simulate_sparse_stc(src, n_dipoles=n_noise, times=times,
                                        data_fun=_data_fun_pink_noise, subjects_dir=subjects_dir)
        noise_stc.data = noise_stc.data /  np.max(np.abs(noise_stc.data)) * 1e-7
        noise_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=noise_stc, info=raw_info)
        noise_data = noise_raw.get_data()
        vertices_noise = noise_stc.vertices.copy()
        vertices_noise_idx_full = vertex_index_full(vertices_noise, src)
        noise_data_full = np.zeros((n_vox, n_sample))
        noise_data_full[vertices_noise_idx_full, :] = noise_stc.data[:len(vertices_noise_idx_full)]
    # connectivity and data generation---------------------------------------------------------------------------
    """
    Here we select pairs of sources that are connected to each other
    """
    conn_ind_beta = np.random.permutation(n_beta)[:n_beta_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_beta'] = conn_ind_beta

    conn_ind_nonsin = np.random.permutation(n_nonsin)[:n_nonsin_conn * 2].reshape((-1, 2)) + n_alpha
    vert_sigs_ind['conn_ind_nonsin'] = conn_ind_nonsin - n_alpha

    sig_alpha_ind = np.arange(n_alpha + n_nonsin)
    _, ind1, _ = np.intersect1d(sig_alpha_ind, conn_ind_nonsin.ravel(), return_indices=True)
    sig_alpha_ind = np.delete(sig_alpha_ind, ind1)
    conn_ind_alpha_1 = np.random.permutation(sig_alpha_ind)[:n_alpha_conn * 2].reshape((-1, 2))
    vert_sigs_ind['conn_ind_alpha'] = conn_ind_alpha_1

    conn_ind_alpha = np.append(conn_ind_alpha_1, conn_ind_nonsin, axis=0)

    # Generate beta signals and connections--------------------------
    sig_beta = np.zeros((n_beta, n_sample), dtype='complex')
    for n_b in range(n_beta):
        if np.sum(sig_beta[n_b, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            sig_beta[n_b, :] = filtered_randn(16, 24, sfreq, n_sample)
            factor = _adjust_snr_factor(sig_beta[n_b, :], noise_data, [16, 24],
                                        vert_beta[n_b], leadfield, snr_beta, sfreq)
            sig_beta[n_b, :] /= factor
            if np.sum(conn_ind_beta == n_b):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_beta == n_b)
                ind2o = int(not ind2)
                n_b2 = conn_ind_beta[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift
                sig_beta[n_b2, :] = produce_nm_phase_locked_sig(sig_beta[n_b, :], dphi, 1, 1, [16, 24], sfreq)
                factor = _adjust_snr_factor(sig_beta[n_b2, :], noise_data, [16, 24],
                                            vert_beta[n_b2], leadfield,  snr_beta, sfreq)
                sig_beta[n_b2, :] /= factor

    # Generate alpha signals --------------------------
    sig_alpha_nonsin = np.zeros((n_alpha + n_nonsin, n_sample), dtype='complex')
    for n_a in range(n_alpha + n_nonsin):

        if np.sum(sig_alpha_nonsin[n_a, :]):  # if the signal is already generated
            pass
        else:  # if the signal is NOT already generated
            if n_a < n_alpha:  # it is an alpha-source
                sig_alpha_nonsin[n_a, :] = filtered_randn(8, 12, sfreq, n_sample)
                factor = _adjust_snr_factor(sig_alpha_nonsin[n_a, :], noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                sig_alpha_nonsin[n_a, :] /= factor
            else:  # it is a nonsin-source
                dphi_beta0 = pi / 2 * np.random.random(1) + np.pi/4
                data_alpha_nonsin_n_a, data_beta_nonsin_n_a = produce_nonsin_sig(n_sample, sfreq, dphi_beta0,
                                                                                 nonsin_mode, ampcorr_coeff)
                factor = _adjust_snr_factor(data_alpha_nonsin_n_a, noise_data, [8, 12],
                                            vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                data_alpha_nonsin_n_a /= factor
                factor = _adjust_snr_factor(data_beta_nonsin_n_a, noise_data, [16, 24],
                                            vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                data_beta_nonsin_n_a /= factor
                sig_alpha_nonsin[n_a, :] = data_alpha_nonsin_n_a + data_beta_nonsin_n_a

            if np.sum(conn_ind_alpha == n_a):  # if this source is connected to any other source
                # generate that other source
                ind1, ind2 = np.where(conn_ind_alpha == n_a)
                ind2o = int(not ind2)
                n_a2 = conn_ind_alpha[ind1, ind2o]
                dphi = pi / 2 * np.random.random(1) + np.pi/4  # phase-shift

                if n_a < n_alpha and n_a2 < n_alpha:  # if both are alpha
                    sig_alpha_nonsin[n_a2, :] = \
                        produce_nm_phase_locked_sig(sig_alpha_nonsin[n_a, :], dphi, 1, 1, [8, 12], sfreq)
                    factor = _adjust_snr_factor(sig_alpha_nonsin[n_a2, :], noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a2], leadfield, snr_alpha, sfreq)
                    sig_alpha_nonsin[n_a2, :] /= factor

                elif n_a < n_alpha and n_a2 >= n_alpha:  # if n_a is alpha but n_a2 is nonsin
                    dphi_beta = pi / 2 * np.random.random(1) + np.pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(sig_alpha_nonsin[n_a, :], dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
                else:  # if both are nonsin: n_a >= n_alpha and n_a2 >= n_alpha
                    dphi_beta = pi / 2 * np.random.random(1) + pi/4
                    data_alpha_nonsin, data_beta_nonsin = \
                        produce_nm_phase_locked_non_sin(data_alpha_nonsin_n_a, dphi, 1, sfreq,
                                                        dphi_beta=dphi_beta, kappa=None)
                    phase_beta = np.angle(data_beta_nonsin)
                    if nonsin_mode == 1:  # different amplitude envelopes
                        pass
                    elif nonsin_mode == 2:  # the same amplitude envelopes
                        abs_alpha = np.abs(data_alpha_nonsin)
                        data_beta_nonsin = abs_alpha * np.exp(1j * phase_beta)
                    elif nonsin_mode == 3:  # correlated amplitude envelopes
                        abs_beta = correlated_ampenv(data_alpha_nonsin, ampcorr_coeff, 8, 12)
                        data_beta_nonsin = abs_beta * np.exp(1j * phase_beta)
                    data_beta_nonsin = data_beta_nonsin / np.std(data_beta_nonsin, axis=1)[:, np.newaxis]
                    factor = _adjust_snr_factor(data_alpha_nonsin, noise_data, [8, 12],
                                                vert_alpha_nonsin[n_a], leadfield, snr_beta, sfreq)
                    data_alpha_nonsin /= factor
                    factor = _adjust_snr_factor(data_beta_nonsin, noise_data, [16, 24],
                                                vert_alpha_nonsin[n_a], leadfield, snr_alpha, sfreq)
                    data_beta_nonsin /= factor
                    sig_alpha_nonsin[n_a2, :] = data_alpha_nonsin + data_beta_nonsin
    data = np.real(np.concatenate((sig_beta, sig_alpha_nonsin), axis=0))
    data_perm = data.copy()
    data_perm[ind_vert_perm, :] = data
    data_full = np.zeros((n_vox, n_sample))
    data_full[vertices_idx_full, :] = data
    data_full += noise_data_full
    # source_stc = simulate_sparse_stc_2(src, data, sfreq, labels=labels, labels_med=labels_med,
    #                                   random_state=random_state,
    #                                   location=location, subject=subject, subjects_dir=subjects_dir, surf=surf)
    source_stc_orig = mne.SourceEstimate(data_perm, vertices=vertices, tmin=0, tstep=1 / sfreq, subject=subject)
    source_stc = mne.SourceEstimate(data_full, vertices=vertices_full, tmin=0, tstep=1/sfreq, subject=subject)
    source_stc.data = source_stc.data / np.max(np.abs(source_stc.data)) * 1e-7
    simulated_raw = mne.apply_forward_raw(fwd=fwd_fixed, stc=source_stc, info=raw_info)

    return simulated_raw, source_stc, source_stc_orig, vert_sigs_ind


def plv_null_dist(b1, a1, b2, a2, n, m, duration, fs, n_total_iter=10000, plv_type='abs', mirror=False, noise_type='white'):
    from tools_general import filtfilt_mirror
    from scipy.signal import filtfilt
    from tools_connectivity import compute_plv_with_permtest
    n_sample = duration * fs
    plvs = np.zeros((n_total_iter,))
    for i_iter in range(n_total_iter):
        if noise_type == 'white':
            x1 = np.random.randn(1, n_sample)
        else:
            n_samples = int(duration * fs)
            times = np.arange(0, n_samples) / fs
            x1 = _data_fun_pink_noise(times)[np.newaxis, :]
        if mirror:
            x1 = filtfilt_mirror(b1, a1, x1, axis=1)
        else:
            x1 = filtfilt(b1, a1, x1, axis=1)
        if noise_type == 'white':
            y1 = np.random.randn(1, n_sample)
        else:
            n_samples = int(duration * fs)
            times = np.arange(0, n_samples) / fs
            y1 = _data_fun_pink_noise(times)[np.newaxis, :]
        if mirror:
            y1 = filtfilt_mirror(b2, a2, y1, axis=1)
        else:
            y1 = filtfilt(b2, a2, y1, axis=1)

        plv = compute_plv_with_permtest(x1, y1, n, m, fs, plv_type=plv_type)
        plvs[i_iter] = plv[0][0]
    return plvs