import matplotlib.pyplot as plt
import numpy as np


#  --------------------------------  --------------------------------  --------------------------------
# general
#  --------------------------------  --------------------------------  --------------------------------

def dB(data, coeff=10):
    return coeff * np.log10(data)


def dBinv(data, coeff=10):
    return 10 ** (data / 10)
#  --------------------------------  --------------------------------  --------------------------------
# fft and hilbert space
#  --------------------------------  --------------------------------  --------------------------------

def zero_pad_to_pow2(x, axis=1):
    """
    for fast computation of fft, zeros pad the signals to the next power of two
    :param x: [n_signals x n_samples]
    :param axis
    :return: zero-padded signal
    """
    n_samp = x.shape[axis]
    n_sig = x.shape[1-axis]
    n_zp = int(2 ** np.ceil(np.log2(n_samp))) - n_samp
    zp = np.zeros_like(x)
    zp = zp[:n_zp] if axis == 0 else zp[:, :n_zp]
    y = np.append(x, zp, axis=axis)
    return y, n_zp


def hilbert_(x, axis=1):
    """
    computes fast hilbert transform by zero-padding the signal to a length of power of 2.


    :param x: array_like
              Signal data.  Must be real.
    :param axis: the axis along which the hilbert transform is computed, default=1
    :return: x_h : analytic signal of x
    """
    if np.iscomplexobj(x):
        return x
    from scipy.signal import hilbert
    if len(x.shape) == 1:
        x = x[np.newaxis, :] if axis == 1 else x[:, np.newaxis]
    x_zp, n_zp = zero_pad_to_pow2(x, axis=axis)
    x_zp = np.real(x_zp)
    x_h = hilbert(x_zp, axis=axis)
    if n_zp == 0:
        return x_h
    else:
         x_h = x_h[:, :-n_zp] if axis == 1 else x_h[:-n_zp, :]
    return x_h


def fft_(x, fs, axis=1, n_fft=None):
    from scipy.fftpack import fft
    if np.iscomplexobj(x):
        x = np.real(x)
    if x.ndim == 1:
        x = x.reshape((1, x.shape[0])) if axis == 1 else x.reshape((x.shape[0], 1))
    n_sample = x.shape[1]
    n_fft = int(2 ** np.ceil(np.log2(n_sample))) if n_fft is None else n_fft
    x_f = fft(x, n_fft)
    freq = np.arange(0, fs / 2, fs / n_fft)
    n_fft2 = int(n_fft / 2)
    x_f = x_f[0, : n_fft2]
    return freq, x_f


def plot_fft(x, fs, axis=1, n_fft=None):
    freq, x_f = fft_(x, fs, axis=axis, n_fft=n_fft)
    xf_abs = np.abs(x_f)
    # n_fft = np.max(x_f.shape)
    # n_fft2 = int(n_fft / 2)
    # freq = np.arange(0, fs/2, fs/n_fft)
    plt.figure()
    if axis == 1:
        plt.plot(freq, xf_abs.ravel())
    else:
        plt.plot(freq, xf_abs.ravel())
    plt.title('Magnitude of FFT')
    plt.grid()


#  --------------------------------  --------------------------------  --------------------------------
# filtering and filters
#  --------------------------------  --------------------------------  --------------------------------

def morlet_filter(data, sfreq, freq_min, freq_max, freq_res=0.5, n_jobs=1, n_cycles='auto'):
    """
    morlet filtering with linearly spaced frequency bins, for multi-channel data
    :param data: np.ndarray . [channel x time]
    :param sfreq: int . sampling frequency
    :param freq_min:
    :param freq_max:
    :param freq_res: frequency resolution
    :param n_jobs:
    :return: TF of data - complex
    """
    from mne.time_frequency import tfr_array_morlet
    nchan, nsample = data.shape
    data = np.reshape(data, (1, nchan, nsample))
    freq_n = int((freq_max - freq_min) / freq_res) + 1
    freqs, step = np.linspace(freq_min, freq_max, num=freq_n, retstep=True, endpoint=True)
    if n_cycles == 'auto':
        n_cycles = freqs / 2.
    data_tfr = tfr_array_morlet(data, sfreq, freqs, n_cycles=n_cycles, zero_mean=True,
                                use_fft=True, decim=1, output='complex', n_jobs=n_jobs, verbose=None)
    return data_tfr[0, :, :], freqs


def filtfilt_mirror(b, a, data, axis=-1):
    from scipy.signal import filtfilt
    axis = data.ndim - 1 if axis == -1 else axis
    n_sample = data.shape[axis]
    data_flip_neg = -np.flip(data, axis=axis)
    data_mirror = np.concatenate((data_flip_neg, data, data_flip_neg), axis=axis)
    data_filt = filtfilt(b, a, data_mirror, axis=axis)
    indices = {axis: np.arange(n_sample, 2*n_sample, dtype='int')}
    ix = tuple(indices.get(dim, slice(None)) for dim in range(data_filt.ndim))
    data_filt_cut = data_filt[ix]
    return data_filt_cut


def filter_plot_freqz(b, a, fs):
    from scipy.signal import freqz
    w, h = freqz(b, a)
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.plot(w/np.pi*fs/2, 10 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w/np.pi*fs/2, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid(True)
    plt.axis('tight')
    plt.show()


#  --------------------------------  --------------------------------  --------------------------------
# PSD
#  --------------------------------  --------------------------------  --------------------------------

def psd_plot(data, fs, f_max=None, overlap_perc=0.5, freq_res=0.5, axis=1, figax=None):
    """
    plots the spectrum of the input signal

    :param data: ndarray [n_chan x n_samples]
                 data array . can be multi-channel
    :param fs: sampling frequency
    :param f_max: maximum frequency in the plotted spectrum
    :param overlap_perc: overlap percentage of the sliding windows in welch method
    :param freq_res: frequency resolution, in Hz
    :return: no output, plots the spectrum
    """
    from scipy.signal import welch
    if np.iscomplexobj(data):
        data = np.real(data)
    if data.ndim == 1:
        axis = 0
    nfft = 2 ** np.ceil(np.log2(fs / freq_res))
    noverlap = np.floor(overlap_perc * nfft)
    f, pxx = welch(data, fs=fs, nfft=nfft, nperseg=nfft, noverlap=noverlap, axis=axis)
    if f_max is not None:
        indices = {axis: f <= f_max}
        ix = tuple(indices.get(dim, slice(None)) for dim in range(pxx.ndim))
        pxx = pxx[ix]
        f = f[f <= f_max]

    fig, ax, line = figax[0], figax[1], figax[2]
    line.set_data(f, dB(pxx.T))
    plt.ylabel('PSD (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.grid(True)
    return f, pxx, (fig, ax, line)


def psd(data, fs, f_max=None, overlap_perc=0.5, freq_res=0.5, axis=1, plot=True, dB1=True,
        fig='new', interactivePlot=True, clab=None):
    """
    plots the spectrum of the input signal

    :param data: ndarray [n_chan x n_samples]
                 data array . can be multi-channel
    :param fs: sampling frequency
    :param f_max: maximum frequency in the plotted spectrum
    :param overlap_perc: overlap percentage of the sliding windows in welch method
    :param freq_res: frequency resolution, in Hz
    :return: no output, plots the spectrum
    """
    from scipy.signal import welch
    if np.iscomplexobj(data):
        data = np.real(data)
    if data.ndim == 1:
        axis = 0
    nfft = 2 ** np.ceil(np.log2(fs / freq_res))
    noverlap = np.floor(overlap_perc * nfft)
    f, pxx = welch(data, fs=fs, nfft=nfft, nperseg=nfft, noverlap=noverlap, axis=axis)
    if f_max is not None:
        indices = {axis: f <= f_max}
        ix = tuple(indices.get(dim, slice(None)) for dim in range(pxx.ndim))
        pxx = pxx[ix]
        f = f[f <= f_max]
    if plot:
        if fig == 'new':
            fig = plt.figure()
            ax = plt.subplot(111)
        else:
            fig, ax = fig[0], fig[1]
        if dB1:
            line = ax.plot(f, dB(pxx.T), lw=1, picker=1)
        else:
            line = ax.plot(f, pxx.T, lw=1, picker=1)
        if interactivePlot:
            def onpick1(event, clab):
                thisline = event.artist
                n_line = int(str(thisline)[12:-1])
                if clab is not None:
                    print(clab[n_line])
                else:
                    print('channel ' + str(n_line))

            onpick = lambda event: onpick1(event, clab)
            fig.canvas.mpl_connect('pick_event', onpick)
        plt.ylabel('PSD (dB)')
        plt.xlabel('Frequency (Hz)')
        plt.grid(True, ls='dotted')
        return f, pxx, (fig, ax, line)
    return f, pxx


#  --------------------------------  --------------------------------  --------------------------------
# peaks
#  --------------------------------  --------------------------------  --------------------------------

def plot_peaks(x, sig, peaks, peaks_ind, width):
    plt.figure()
    plt.plot(x, sig)
    plt.plot(peaks, sig[peaks_ind], "x")
    plt.hlines(*width, color="C2")
    plt.show()


def my_peak_width(x, sig, peaks, peaks_ind, plot=False):

    n_peaks = len(peaks_ind)
    if plot:
        plt.ioff()
        fig = plt.figure()
        plt.plot(x, sig)
        plt.plot(x[peaks_ind], sig[peaks_ind], "x")

    w1 = np.empty((n_peaks, 1), dtype='int')
    w2 = np.empty((n_peaks, 1), dtype='int')
    for j in range(n_peaks):
        i1 = peaks_ind[j - 1] if j > 0 else 0
        i2 = peaks_ind[j + 1] if j < n_peaks-1 else len(x)-1
        idx_l = np.arange(i1, peaks_ind[j], 1)
        idx_r = np.arange(peaks_ind[j] + 1, i2, 1)
        sig_l = sig[idx_l]
        sig_r = sig[idx_r]

        argmin_l = np.argmin(sig_l)
        argmin_r = np.argmin(sig_r)
        """
        if plot:
            plt.plot(x[idx_l], sig_l)
            plt.plot(x[idx_r], sig_r)
             plt.plot(x[idx_l][argmin_l:], sig_l[argmin_l:])
            plt.plot(x[idx_l[argmin_l]], sig[idx_l[argmin_l]], "x")
            plt.plot(x[idx_r[argmin_r]], sig[idx_r[argmin_r]], "x")
        """
        ind1 = np.argmax([np.array([sig_l[argmin_l], sig_r[argmin_r]])])

        if ind1 == 0:
            w1[j] = idx_l[argmin_l]
            ind_r = np.argmin(np.abs(sig_r[:argmin_r] - sig_l[argmin_l]))
            w2[j] = idx_r[:argmin_r][ind_r]

        elif ind1 == 1:
            w2[j] = idx_r[argmin_r]
            ind_l = np.argmin(np.abs(sig_l[argmin_l:] - sig_r[argmin_r]))
            w1[j] = idx_l[argmin_l:][ind_l]

        if plot:
            plt.plot(x[int(w2[j][0])], sig[int(w2[j][0])], "o")
            plt.plot(x[int(w1[j][0])], sig[int(w1[j][0])], "o")
    width_ind = (w1, w2)
    return width_ind, fig