
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

#  --------------------------------  --------------------------------  --------------------------------
# general: strings, saving, loading, directories
#  --------------------------------  --------------------------------  --------------------------------
def strround(x, n=3):
    return str(np.round(x, n))

def combine_names(connector, *nouns):
    word = nouns[0]
    for noun in nouns[1:]:
        word += connector + str(noun)
    return word


def save_pickle(file, var):
    import pickle
    with open(file, "wb") as output_file:
        pickle.dump(var, output_file)


def load_pickle(file):
    import pickle
    with open(file, "rb") as input_file:
        var = pickle.load(input_file)
    return var


def np_parsave(save_name, var):
    np.save(save_name, var)


def write_in_txt(filename, msg):
    """
    the function to write a message in a txt file
    Input arguments:
    ================
    filename: the name of the file, e.g 'file1.txt'
    msg: the message to be writte

    Output arguments:
    =================
    No value returns. The file is modified
    """
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    lines.append(msg+'\n')
    f = open(filename, "w")
    for line in lines:
        f.write(line)
    f.close()


def listdir_restricted(dir_path, string_criterion):
    """
    returns a list of the names of the files in dir_path, whose names contains string_criterion
    :param dir_path: the directory path
    :param string_criterion: the string which should be in the name of the desired files in dir_path
    :return: list of file names
    """
    import os
    IDs_all = os.listdir(dir_path)
    IDs_with_string = [id1 for id1 in IDs_all if string_criterion in id1]
    return IDs_with_string


def listdir_nohidden(dir_path):
    files = os.listdir(dir_path)
    return [f for f in files if not f.startswith('.')]


def save_json_from_numpy(filename, var):
    """
    (c) by Alina Studenova - 2021
    save numpy array as json file
    :param filename:
    :param var:
    :return:
    """
    import json
    with open(filename, "w") as f:
        json.dump(var.tolist(), f)


def load_json_to_numpy(filename):
    """
    (c) by Alina Studenova - 2021

    load json file to a numpy array
    :param filename:
    :return:
    """
    import json
    with open(filename, "r") as f:
        saved_data = json.load(f)

    var = np.array(saved_data)
    return var


def save_json(filename, var):
    """
    (c) by Alina Studenova - 2021

    save list to json
    :param filename:
    :param var:
    :return:
    """
    import json
    with open(filename, "w") as f:
        json.dump(var, f)


def load_json(filename):
    """
    (c) by Alina Studenova - 2021

    load list from json
    :param filename:
    :return:
    """
    import json
    with open(filename, "r") as f:
        saved_data = json.load(f)

    var = saved_data
    return var


#  --------------------------------  --------------------------------  --------------------------------
# stat, probability, information theory
#  --------------------------------  --------------------------------  --------------------------------


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """
    (c) https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
    Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------

    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    import scipy.stats as stats
    import scipy as sp
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        # pc = sp.polyfit(xs, ys + resamp_resid, 1)
        res = stats.linregress(xs, ys + resamp_resid)
        y_hat1 = res[0] * xs + res[1]
        # Plot bootstrap cluster
        ax.plot(xs, y_hat1, "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax


def plot_scatterplot_linearReg_bootstrap(x, y, ax, xlabel='', ylabel='', title=''):
    from scipy.stats import linregress
    x, y = x.ravel(), y.ravel()
    res = linregress(x, y)
    y_hat = res[0] * x + res[1]
    ax.plot(
        x, y, "o", color="#b9cfe7", markersize=8,
        markeredgewidth=1, markeredgecolor="b", markerfacecolor="lightblue"
    )
    ax.plot(x, y_hat, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")
    plot_ci_bootstrap(x, y, y - y_hat, nboot=500, ax=None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)


def kl_divergence(p, q):
    p = p + 1e-5
    q = q + 1e-5
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def data2pdf(data, bins='auto', plot=False):
    h, b = np.histogram(data, bins=bins, density=True)
    b2 = (b[:-1] + b[1:]) / 2
    if plot:
        plt.plot(b2, h),
        plt.grid()
    return h, b2, b


def pearsonr_(x, y, axis=0):
    x_m = np.mean(x, axis=axis, keepdims=True)
    y_m = np.mean(y, axis=axis, keepdims=True)
    x_dm = x - x_m
    y_dm = y - y_m
    r = np.sum(x_dm * y_dm, axis=axis) / np.sqrt(np.sum(x_dm**2, axis=axis) * np.sum(y_dm**2, axis=axis))
    return r


def significance_perc_increase(x, y):
    from scipy.stats import pearsonr, norm
    n = len(x)
    var_x = np.sum((x - np.mean(x))**2) / (n - 1)
    vx = np.sqrt(var_x) / np.mean(x)
    var_y = np.sum((y - np.mean(y))**2) / (n - 1)
    vy = np.sqrt(var_y) / np.mean(y)
    k = vx / vy
    r_obs, pval_init = pearsonr(x, y / x)
    rxy, _ = pearsonr(x, y)
    r0 = -np.sqrt((1 - rxy) / 2)
    zscore = np.sqrt(n - 3)/2 * (np.log((1 + r_obs) / (1 - r_obs)) - np.log((1 + r0) / (1 - r0)))
    p_value = norm.sf(abs(zscore))  # one sided
    return p_value, zscore, r_obs, r0, pval_init, k


def compute_deviation_midline(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    max_value = np.max(np.append(x, y))
    min_value = np.min(np.append(x, y))
    d_sq = (max_value - min_value) * np.sqrt(2)
    dist = (x - y) * np.sqrt(2) / 2
    mdist = np.sqrt(np.mean(dist ** 2)) / d_sq
    return mdist

#  --------------------------------  --------------------------------  --------------------------------
# plotting
#  --------------------------------  --------------------------------  --------------------------------
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_polar_hist(x):
    x2 = np.mod(x, 2 * np.pi)
    height1, bins1 = np.histogram(x2, bins='auto')
    n_bins = len(height1)
    width = 2 * np.pi / n_bins
    theta = bins1[:-1] + width / 2
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.bar(theta, height1, width=width, bottom=0.0)


def plotline_interactive(x, y):
    def onpick(event):
        thisline = event.artist
        n_line = int(str(thisline)[12:-1])
        print('channel ' + str(n_line))

    fig = plt.figure()
    ax = plt.subplot(111)
    line = ax.plot(x, y, picker=1)
    fig.canvas.mpl_connect('pick_event', onpick)


def plot_lines_interactive(x, y, label=None):
    def onpick1(event, label):
        thisline = event.artist
        n_line = int(str(thisline)[12:-1])
        if label is not None:
            print(label[n_line])
        else:
            print('channel ' + str(n_line))

    onpick = lambda event: onpick1(event, label)
    fig = plt.figure()
    ax = plt.subplot(111)
    line = plt.plot(x, y, picker=1)
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.grid(True)
    return fig, ax


def plot_3d(x, y, z):
    from mpl_toolkits import mplot3d
    t1, t2 = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(t1.T, t2.T, z, cmap='viridis', edgecolor='none')
    return ax


def plot_boxplot_paired(ax, data1, data2, labels, paired=True, violin=False, notch=True, datapoints=False):
    # does not open a new figure
    n_points = data1.shape[0]
    data1 = np.reshape(data1, (n_points, 1))
    data2 = np.reshape(data2, (n_points, 1))

    if violin:
        import seaborn as sns
        from pandas import DataFrame
        data_all = np.append(data1, data2)
        type = np.append([labels[0]] * n_points, [labels[1]] * n_points)
        data_all = {'data_all': data_all, 'type': type}
        df_thresh = DataFrame(data=data_all)
        sns.set_theme(style="whitegrid")
        ax = sns.violinplot(x='type', y='data_all', data=df_thresh)
        x1, x2 = 0, 1
    else:
        plt.boxplot(np.concatenate((data1, data2), axis=1), labels=labels, notch=notch)
        x1, x2 = 1, 2

    if datapoints:
        for k in range(n_points):
            plt.plot(np.ones((1, 1)) * x1 + np.random.randn(1, 1) * 0.02, data1[k],
                     marker='.', color='lightskyblue', markersize=3)
            plt.plot(np.ones((1, 1)) * x2 + np.random.randn(1, 1) * 0.02, data2[k],
                     marker='.', color='lightskyblue', markersize=3)

    if paired:
        for k in range(n_points):
            x = np.array([x1, x2])
            y = np.array([data1[k], data2[k]])
            plt.plot(x, y, '-', linewidth=.05)
    ax.yaxis.grid(True)


def plot_colorbar(data, cmap, ori='vertical'):
    import matplotlib as mpl
    data = np.sort(data, axis=0)
    # fig = plt.figure()
    if ori == 'vertical':
        fig, ax = plt.subplots(figsize=(1, 6))
        fig.subplots_adjust(right=0.5)
    elif ori == 'horizontal':
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=data[0], vmax=data[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=ori)
    return fig


def plot_colorbar2(data, cmap, ax, ori='vertical'):
    import matplotlib as mpl
    data = np.sort(data, axis=0)
    norm = mpl.colors.Normalize(vmin=data[0], vmax=data[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=ori)

#  --------------------------------  --------------------------------  --------------------------------
# vectors, matrices and tensors
#  --------------------------------  --------------------------------  --------------------------------


def zscore_matrix(mat):
    return (mat - np.mean(mat)) / np.std(mat)


def tensor_mat_prod(tensor, mat, dim, order='F'):
    if mat.shape[1] != tensor.shape[dim]:
        print('the shapes do not match')
        return
    tensor_ = np.swapaxes(tensor, 0, dim)
    tenmat = tensor_.reshape((tensor_.shape[0], -1), order=order)
    tenmat_prod = mat @ tenmat
    ten_prod = tenmat_prod.reshape((mat.shape[0], tensor_.shape[1], tensor_.shape[2]), order=order)
    return ten_prod


def plot_matrix(mat, cmap='viridis', cmap_level_n=50, title='', vmin=None, vmax=None, axes=None):
    from matplotlib import cm as cm
    from matplotlib import pyplot as plt
    cmp = cm.get_cmap(cmap, cmap_level_n)
    matmin = np.min(mat)
    matabsmax = np.max(np.abs(mat))
    if vmax is None:
        vmax = matabsmax
    else:
        vmax = max([vmax, matabsmax])
    if matmin < 0:
        cmp = 'RdBu_r'
    if vmin is None:
        vmin = matmin

    if axes is None:
        # caxes = plt.matshow(mat, cmap=cmp, vmin=vmin, vmax=vmax
        caxes = plt.matshow(mat, cmap=cmp, norm=MidpointNormalize(midpoint=0., vmin=vmin, vmax=vmax))

        plt.colorbar(caxes)
    else:
        # caxes = axes.matshow(mat, cmap=cmp, vmin=vmin, vmax=vmax)
        caxes = axes.matshow(mat, cmap=cmp, norm=MidpointNormalize(midpoint=0., vmin=vmin, vmax=vmax))
        plt.colorbar(caxes)
    plt.title(title)
    plt.grid(False)


def plot_matrix_(mat, cmap='viridis', cmap_level_n=50, title='', vmin=None, vmax=None, axes=None):
    from matplotlib import cm as cm
    from matplotlib import pyplot as plt
    cmp = cm.get_cmap(cmap, cmap_level_n)
    matmin = np.min(mat)
    matabsmax = np.max(np.abs(mat))
    if vmax is None:
        vmax = matabsmax
    else:
        vmax = max([vmax, matabsmax])
    if matmin < 0:
        cmp = 'RdBu_r'
    if vmin is None:
        vmin = matmin

    if axes is None:
        caxes = plt.matshow(mat, cmap=cmp, vmin=vmin, vmax=vmax)
        plt.colorbar(caxes)
    else:
        caxes = axes.matshow(mat, cmap=cmp, vmin=vmin, vmax=vmax)
        plt.colorbar(caxes)
    plt.title(title)
    plt.grid(False)


def plot_matrix2(mat, cmap='viridis', cmap_level_n=50, power=2, title='', vmin=None, vmax=None, axes=None):
    from matplotlib import cm as cm
    from matplotlib import pyplot as plt
    cmp = cm.get_cmap(cmap, cmap_level_n)
    matmin = np.min(mat)
    matabsmax = np.abs(mat)
    if vmax is None:
        vmax = np.max(matabsmax)
    else:
        vmax = max([vmax, matabsmax])
    if matmin < 0:
        vmin = -vmax
    elif vmin is None:
        vmin = matmin

    if axes is None:
        caxes = plt.matshow(power ** mat, cmap=cmp, vmin=vmin, vmax=vmax)
        tt = list(power ** np.array([1, 10, 20, 40, 60, 80, 100]))
        plt.colorbar(caxes, ticks=tt)
    else:
        caxes = axes.matshow(power ** mat, cmap=cmp, vmin=vmin, vmax=vmax)
        tt = list(power ** np.array([1, 10, 20, 40, 60, 80, 100]))
        plt.colorbar(caxes, ticks=tt)
    plt.title(title)
    plt.grid(False)

def threshold_matrix(mat, perc=95, threshold=None, binary=False):
    mat1 = mat.copy()
    if perc is not None:
        threshold = np.percentile(mat1, perc)
    mat1[mat1 < threshold] = 0
    if binary:
        mat1[mat1 != 0] = 1
    return mat1


def blur_matrix(matrix, n_blur):
    mat = matrix.copy()
    dim1, dim2 = mat.shape
    for i1 in range(mat.shape[0]):
        for j1 in range(mat.shape[1]):
            if i1 != j1:
                i_begin = np.max([i1 - n_blur, 0])
                i_end = np.min([i1 + n_blur, dim1])
                j_begin = np.max([j1 - n_blur, 0])
                j_end = np.min([j1 + n_blur, dim2])
                mat_win = matrix[i_begin:i_end, j_begin:j_end]
                mat[i1, j1] = np.mean(mat_win)
    return mat


def eig_power_iteration(A, epsilon=0.01, max_iter=500):
    """
    power method for calculation of eigvalue decomposition
    (c) from http://mlwiki.org/index.php/Power_Iteration
    :param A:
    :return:
    """
    n, d = A.shape
    v = np.ones(d) / np.sqrt(d)
    ev = np.linalg.multi_dot((v, A, v))
    n_iter = 0
    while n_iter <= max_iter:
        n_iter += 1
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)
        ev_new = np.linalg.multi_dot((v_new, A, v_new))
        v = v_new
        ev = ev_new
        if np.abs(ev - ev_new) < epsilon:
            break
    return ev, v


def khatri_rao_(x, y):
    """

    :param x: [mk x mk]
    :param y: [m x m]
    :return:
    """
    k = y.shape[0]
    m = int(x.shape[0] / k)
    z = np.zeros(x.shape)
    for ki in range(k):
        ind1_s = ki * m
        ind1_e = (ki + 1) * m
        for kj in range(k):
            ind2_s = kj * m
            ind2_e = (kj + 1) * m
            x_ = x[ind1_s:ind1_e, ind2_s:ind2_e]
            z[ind1_s:ind1_e, ind2_s:ind2_e] = x_ * y[ki, kj]
    return z
