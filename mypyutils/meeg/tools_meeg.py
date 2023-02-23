import matplotlib.pyplot as plt
import numpy as np


#  --------------------------------  --------------------------------  --------------------------------
# read, save
#  --------------------------------  --------------------------------  --------------------------------

def read_eeglab_standard_chanloc(raw_name, bads=None, monatage_name='standard_1005'):
    import mne
    raw = mne.io.read_raw_eeglab(raw_name)
    new_names = dict(
        (ch_name,
         ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
        for ch_name in raw.ch_names)
    raw.rename_channels(new_names)
    raw.info['bads'] = bads if bads is not None else []

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
    raw = raw.pick(picks)
    montage = mne.channels.make_standard_montage(monatage_name)
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)  # needed for inverse modeling
    return raw

#  --------------------------------  --------------------------------  --------------------------------
# plot on topomap
#  --------------------------------  --------------------------------  --------------------------------


def plot_topomap_(map, info, title='', vmax=None, vmin=None, ax=None, cmap=None, mask=None):
    from mne.viz import plot_topomap
    map = map.ravel()
    if ax is None:
        fig = plt.figure()
    im, _ = plot_topomap(map, info, vmax=vmax, vmin=vmin, cmap=cmap, mask=mask)
    plt.colorbar(im)
    plt.title(title)
    if ax is None:
        return fig



# def plot_power_topomap(raw, freq_win=None):
#     data = raw.get_data()
#     if freq_win is None:
#         pow = np.mean(data ** 2, axis=-1)
#         plot_topomap_(pow, raw.info, title='', vmax=None, vmin=None, ax=None, cmap=None, mask=None)
#     else:
#         fs = raw.info['freq']
#         raw1 = raw.copy()
#         iir_params = dict(order=2, ftype='butter')
#         b10, a10 = butter(N=2, Wn=np.asarray(freq_win) / fs * 2, btype='bandpass')
#
#         raw1.filter(l_freq=8, h_freq=12, method='iir', iir_params=iir_params)


def plot_psd_topomap_(psds, freqs, raw_info, fmax=None, fmin=None,
                      axis_facecolor='w', fig_facecolor='w', color='k', dB1=True, axes=None):
    # adopted from MNE: raw.plot_psd_topo()
    """
    use this if you wanna plot two PSDs on top of each other
    fig = plt.figure()
    axes = plt.axes([0.015, 0.025, 0.97, 0.95])
    axes.set_facecolor('w')
    """
    from mne.viz.raw import _plot_timeseries_unified, _plot_timeseries, _plot_topo
    from mne.channels.layout import find_layout
    from functools import partial
    fmax = freqs[-1] if fmax is None else fmax
    fmin = freqs[0] if fmin is None else fmin
    f_ind = (freqs <= fmax) & (freqs >= fmin)
    layout = find_layout(raw_info)
    if dB1:
        psds = dB(psds)
    show_func = partial(_plot_timeseries_unified, data=[psds[:, f_ind]], times=[freqs[f_ind]], color=color)
    click_func = partial(_plot_timeseries, data=[psds[:, f_ind]], times=[freqs[f_ind]], color=color)

    fig = _plot_topo(raw_info, times=freqs[f_ind], show_func=show_func,
                     click_func=click_func, x_label='Frequency (Hz)',
                     unified=True, y_label='dB', layout=layout,
                     fig_facecolor=fig_facecolor, axis_facecolor=axis_facecolor, axes=axes)
    return fig


def plot_psd_topomap(data, fs, raw_info, fmax=None, fmin=None, freq_res=0.5, overlap_perc=0.5,
                     axis_facecolor='w', fig_facecolor='w', color='k', dB1=True, axes=None):
    # adopted from MNE: raw.plot_psd_topo()
    """
    use this if you wanna plot two PSDs on top of each other
    fig = plt.figure()
    axes = plt.axes([0.015, 0.025, 0.97, 0.95])
    axes.set_facecolor('w')
    """
    from scipy.signal import welch
    freqs, psds = psd(data, fs, f_max=fmax, overlap_perc=overlap_perc, freq_res=freq_res, axis=1, plot=False)
    fig = plot_psd_topomap_(psds, freqs, raw_info, fmax=fmax, fmin=fmin,
                            axis_facecolor=axis_facecolor, fig_facecolor=fig_facecolor,
                            color=color, dB1=dB1, axes=axes)
    return fig


#  --------------------------------  --------------------------------  --------------------------------
# forward and inverse
#  --------------------------------  --------------------------------  --------------------------------

def inverse_operator(size1, fwd, raw_info):
    import mne
    from mne.minimum_norm import make_inverse_operator
    data = np.random.normal(loc=0.0, scale=1.0, size=size1)
    raw1 = mne.io.RawArray(data, raw_info)
    noise_cov = mne.compute_raw_covariance(raw1, tmin=0, tmax=None)

    inv_op = make_inverse_operator(raw_info, fwd, noise_cov, fixed=False, loose=0.2, depth=0.8)
    return inv_op

