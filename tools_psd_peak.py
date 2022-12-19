"""
This module provides functions for assessing the peaks of PSD of EEG.

------------------
(C) Mina Jamshidi Idaji, Oct 2019, @ MPI CBS, Leipzig, Germany
https://github.com/minajamshidi/NID
minajamshidi91@gmail.com

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
from matplotlib import pyplot as plt

class Peak:
    def __init__(self, peak, pass_band):
        self.peak = peak
        self.pass_band = pass_band


def my_find_peaks(data, sfreq, f_max=None, overlap_perc=0.5, freq_res=0.5, axis=1):
    from tools_general import psd
    from scipy.signal import find_peaks
    # -----------------------------------------
    # peak detection of the spectrum
    # -----------------------------------------
    n_chan, n_sample = data.shape
    f, pxx = psd(data, fs=sfreq, f_max=f_max, overlap_perc=overlap_perc, freq_res=freq_res, axis=axis, plot=False)
    peaks_tot, peaks_ind_tot, pass_freq_tot = [None] * n_chan, [None] * n_chan, [None] * n_chan
    for i_chan in range(n_chan):
        peaks_ind, _ = find_peaks(pxx[i_chan, :])
        peaks_ind_tot[i_chan] = peaks_ind
        peaks = f[peaks_ind]
        peaks_tot[i_chan] = peaks
        width_ind, fig = my_peak_width(f, pxx[i_chan, :], peaks, peaks_ind, plot=False)
        pass_freq = (f[width_ind[0]], f[width_ind[1]])
        bw = pass_freq[1] - pass_freq[0]
        ind_bw6 = np.where(bw > 6)[0]
        if len(ind_bw6):
            pb1 = pass_freq[0]
            pb2 = pass_freq[1]
            w1 = width_ind[0]
            w2 = width_ind[1]
            for ind in ind_bw6:
                # bw_ex = (bw[ind] - 6)/freq_res + np.mod((bw[ind] - 6)/freq_res, 2)
                # w1[ind] = w1[ind] + int(bw_ex/2)
                # w2[ind] = w2[ind] - int(bw_ex/2)#+ int(6 / freq_res)
                pb1[ind] = peaks[ind] - 3 if (peaks[ind] - 3) >= 0 else 0
                pb2[ind] = peaks[ind] + 3 if (peaks[ind] + 3) <= 45 else 45
            pass_freq_tot[i_chan] = (pb1, pb2)

    return peaks_tot, peaks_ind_tot, pass_freq_tot


def my_peak_width(x, sig, peaks, peaks_ind, plot=False):

    n_peaks = len(peaks_ind)
    if plot:
        # plt.ioff()
        fig = plt.figure()
        plt.plot(x, sig)
        plt.plot(x[peaks_ind], sig[peaks_ind], "x")

    w1 = np.empty((n_peaks, 1), dtype='int')
    w2 = np.empty((n_peaks, 1), dtype='int')
    for j in range(n_peaks):
        i1 = peaks_ind[j - 1] if j > 0 else 0
        i2 = peaks_ind[j + 1] if j < n_peaks-1 else len(x)
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


def _adjust_double_peak(f, pxx_mean, peaks_ind, idx1, idx2):
    """
    combines the peaks of the same frequency band
    (in teh case that there are two peaks in the frequency band of interest)

    procedure:
    -- it puts the mean of the two peaks as the new peak
    -- it finds the new peak prominence of the combined peaks

    :param f: frequency bins
    :param pxx_mean: mean PSD over channels
    :param peaks_ind: index of the peaks on the PSD array
    :param idx1: the index of the first peak
    :param idx2: the index of the second peak
    :return: peak (class Peak), including the peak frequency and pass_band frequencies
    """
    peak = Peak(peak=None, pass_band=None)
    peak.peak = np.array([np.mean(pb)])

    # finds the new peak prominence of the combined peaks ----------------
    i1 = peaks_ind[idx1 - 1] if idx1 > 0 else 0
    i2 = peaks_ind[idx2 + 1] if idx2 < len(peaks_ind) - 1 else len(peaks_ind) - 1
    idx_l = np.arange(i1, peaks_ind[idx1], 1)
    idx_r = np.arange(peaks_ind[idx2] + 1, i2, 1)
    sig_l = pxx_mean[idx_l]  # PSD values before the first peak
    sig_r = pxx_mean[idx_r]  # PSD values after the second peak

    # find the minimums of sig_l and sig_r, and see which one is the less, then decide on the prominence
    argmin_l = np.argmin(sig_l)
    argmin_r = np.argmin(sig_r)
    ind1 = np.argmax([np.array([sig_l[argmin_l], sig_r[argmin_r]])])
    pb = np.zeros((2, 1))
    if ind1 == 0:
        pb[0] = f[idx_l[argmin_l]]
        ind_r = np.argmin(np.abs(sig_r - sig_l[argmin_l]))
        pb[1] = f[idx_r[ind_r]]

    elif ind1 == 1:
        pb[1] = f[idx_r[argmin_r]]
        ind_l = np.argmin(np.abs(sig_l - sig_r[argmin_r]))
        pb[0] = f[idx_l[ind_l]]

    peak.pass_band = pb
    return peak


def find_narrowband_peaks(peaks, peaks_ind, pass_freq, peak_bound, max_bandwidth, f, pxx_mean):
    """
    function to find a narrow-band peak on PSD
    generally, this functions searches among the peaks of the psd for the desired peak. in case there are two peaks in
    the band pf interest, it combines them
    -- it searches in a bandwidth of interest
    -- if the bandwidth of detected peak is more than max_bandwidth, it reduces the bandwidth to max_bandwidth around the peak

    :param peaks: the frequencies of peaks in an array
    :param peaks_ind: index of the peaks on the PSD array
    :param pass_freq: the the prominence of the peaks in a [2xn_peaks] array
    :param peak_bound: the bandwidth of interest, in which we search for the peak
    :param max_bandwidth: the maximum allowed bandwidth for the detected peak
    :param f: frequency bins
    :param pxx_mean: mean PSD over channels
    :return: peak (class Peak), including the peak frequency and pass_band frequencies
    """
    peak = Peak(peak=None, pass_band=None)
    idx_peaks = np.where(np.logical_and(peak_bound[0] <= peaks, peaks <= peak_bound[1]))[0]
    if len(idx_peaks) == 1:
        peak.peak = peaks[idx_peaks]
        peak.pass_band = np.array([pass_freq[0, idx_peaks], pass_freq[1, idx_peaks]])
    elif len(idx_peaks) == 2:
        idx1 = idx_peaks[0]
        idx2 = idx_peaks[1]
        if np.diff(peaks[idx_peaks])[0] <= 3:
            # peak_avg = np.mean(np.array([peaks[idx1], peaks[idx2]]))
            peak = _adjust_double_peak(f, pxx_mean, peaks_ind, idx1, idx2)
        else:
            ind1 = np.argmax(peaks[idx_peaks])
            idx = np.array([idx_peaks[ind1]])
            peak.peak = peaks[idx]
            peak.pass_band = np.array([pass_freq[0, idx], pass_freq[1, idx]])
    if np.diff(peak.pass_band[:, 0])[0] > max_bandwidth:
        peak.pass_band[0, 0] = peak.peak[0] - max_bandwidth/2
        peak.pass_band[1, 0] = peak.peak[0] + max_bandwidth/2
    return peak


def plot_peak(f, pxx_mean, peak_list, title, dB=True, show_plot=False):
    """
    function for plotting peaks on PSD.
    :param f: frequency bins
    :param pxx_mean: mean PSD over channels
    :param peak: a list of class peak including the peak frequency and pass_band frequencies
    :param title: title of the figure
    :param dB: (bool) True if decibel scale is desired
    :param show_plot: (bool) True if it is desired that the plot is shown
    :return: the handle to th figure
    """
    if dB:
        pxx_mean = 10*np.log10(pxx_mean)
    if not show_plot:
        plt.ioff()
    else:
        plt.show()
    fig = plt.figure()
    plt.plot(f, pxx_mean)

    for peak in peak_list:
        idx0 = np.argmin(np.abs(f - peak.peak))
        idx1 = np.argmin(np.abs(f - peak.pass_band[0]))
        idx2 = np.argmin(np.abs(f - peak.pass_band[1]))
        plt.plot(peak.peak, pxx_mean[idx0], "x")
        plt.plot(peak.pass_band[0], pxx_mean[idx1], "o")
        plt.plot(peak.pass_band[1], pxx_mean[idx2], "o")
    plt.xticks(np.arange(min(f), max(f) + 1, 2))
    plt.grid()
    plt.title(title)
    return fig
