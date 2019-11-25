import numpy as np
import tqdm


def _meas_gs(d, inds, swap12):
    if swap12:
        return (
            np.sum(d['g2p'][inds] * d['n2p'][inds]) / np.sum(d['n2p'][inds]),
            np.sum(d['g2m'][inds] * d['n2m'][inds]) / np.sum(d['n2m'][inds]),
            np.sum(d['g2'][inds] * d['n2'][inds]) / np.sum(d['n2'][inds]),
            np.sum(d['g1p'][inds] * d['n1p'][inds]) / np.sum(d['n1p'][inds]),
            np.sum(d['g1m'][inds] * d['n1m'][inds]) / np.sum(d['n1m'][inds]),
            np.sum(d['g1'][inds] * d['n1'][inds]) / np.sum(d['n1'][inds]))
    else:
        return (
            np.sum(d['g1p'][inds] * d['n1p'][inds]) / np.sum(d['n1p'][inds]),
            np.sum(d['g1m'][inds] * d['n1m'][inds]) / np.sum(d['n1m'][inds]),
            np.sum(d['g1'][inds] * d['n1'][inds]) / np.sum(d['n1'][inds]),
            np.sum(d['g2p'][inds] * d['n2p'][inds]) / np.sum(d['n2p'][inds]),
            np.sum(d['g2m'][inds] * d['n2m'][inds]) / np.sum(d['n2m'][inds]),
            np.sum(d['g2'][inds] * d['n2'][inds]) / np.sum(d['n2'][inds]))


def _meas_m_c_cancel(dp, dm, inds, swap12, step, g_true):
    g1p_p, g1m_p, g1_p, g2p_p, g2m_p, g2_p = _meas_gs(dp, inds, swap12)
    g1p_m, g1m_m, g1_m, g2p_m, g2m_m, g2_m = _meas_gs(dm, inds, swap12)

    g1 = (g1_p - g1_m) / 2
    R11 = (g1p_p - g1m_p + g1p_m - g1m_m) / 2 / 2 / step

    g2 = (g2_p + g2_m) / 2
    R22 = (g2p_p - g2m_p + g2p_m - g2m_m) / 2 / 2 / step

    return g1 / R11 / g_true - 1, g2 / R22


def estimate_m_and_c(
        data_p, data_m, g_true, swap12=False, step=0.01,
        rng=None, n_boot=500, verbose=True):
    """Estimate m and c w/ errors via boostrapping.

    Parameters
    ----------
    data_p : array-like
        Array of simulation data estimated from `measure_shear_metadetect`
        with a +g_true true shear.
    data_m : array-like
        Array of simulation data estimated from `measure_shear_metadetect`
        with a -g_true true shear.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    rng : np.random.RandomState, int or None
        An RNG to use for drawing the objects. If an int or None is passed,
        the input is used to initialize a new `np.random.RandomState` object.
    n_boot : int, optional
        The number of iterations used for estimating the bootstrap errors.
    verbose : bool, optional
        If True, show a progress bar. Otherwise show nothing.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """
    n_data = data_p.shape[0]
    ms = []
    cs = []
    rng = (rng if isinstance(rng, np.random.RandomState)
           else np.random.RandomState(seed=rng))
    for _ in tqdm.trange(n_boot, total=n_boot, leave=False, disable=not verbose):
        inds = rng.choice(n_data, replace=True, size=n_data)
        _m, _c = _meas_m_c_cancel(data_p, data_m, inds, swap12, step, g_true)
        ms.append(_m)
        cs.append(_c)

    m, c = _meas_m_c_cancel(
        data_p, data_m, np.arange(n_data), swap12, step, g_true)

    return m, np.std(ms), c, np.std(cs)


def estimate_m_and_c_patch_avg(
        data_p, data_m, g_true, swap12=False, step=0.01,
        rng=None, n_boot=500, verbose=True):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    data_p : array-like
        Array of simulation data estimated from `measure_shear_metadetect`
        with a +g_true true shear.
    data_m : array-like
        Array of simulation data estimated from `measure_shear_metadetect`
        with a -g_true true shear.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    rng : np.random.RandomState, int or None
        An RNG to use for drawing the objects. If an int or None is passed,
        the input is used to initialize a new `np.random.RandomState` object.
    n_boot : int, optional
        The number of iterations used for estimating the bootstrap errors.
    verbose : bool, optional
        If True, show a progress bar. Otherwise show nothing.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """

    if swap12:
        g1p = data_p['g2'].copy()
        R11p = (data_p['g2p'] - data_p['g2m']) / 2 / step * g_true
        g2p = data_p['g1'].copy()
        R22p = (data_p['g1p'] - data_p['g1m']) / 2 / step

        g1m = data_m['g2'].copy()
        R11m = (data_m['g2p'] - data_m['g2m']) / 2 / step * g_true
        g2m = data_m['g1'].copy()
        R22m = (data_m['g1p'] - data_m['g1m']) / 2 / step
    else:
        g1p = data_p['g1'].copy()
        R11p = (data_p['g1p'] - data_p['g1m']) / 2 / step * g_true
        g2p = data_p['g2'].copy()
        R22p = (data_p['g2p'] - data_p['g2m']) / 2 / step

        g1m = data_m['g1'].copy()
        R11m = (data_m['g1p'] - data_m['g1m']) / 2 / step * g_true
        g2m = data_m['g2'].copy()
        R22m = (data_m['g2p'] - data_m['g2m']) / 2 / step

    wgts = data_p['n1'].copy()
    wgts /= np.sum(wgts)

    x1 = (R11p + R11m)/2
    y1 = (g1p - g1m) / 2

    x2 = (R22p + R22m) / 2
    y2 = (g2p + g2m) / 2

    rng = (rng if isinstance(rng, np.random.RandomState)
           else np.random.RandomState(seed=rng))
    mvals = []
    cvals = []
    for _ in tqdm.trange(n_boot, total=n_boot, leave=False, disable=not verbose):
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        _wgts = wgts[ind].copy()
        _wgts /= np.sum(_wgts)
        mvals.append(np.mean(y1[ind] * _wgts) / np.mean(x1[ind] * _wgts) - 1)
        cvals.append(np.mean(y2[ind] * _wgts) / np.mean(x2[ind] * _wgts))

    return (
        np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1, np.std(mvals),
        np.mean(y2 * wgts) / np.mean(x2 * wgts), np.std(cvals))


def measure_shear_metadetect(res, *, s2n_cut, t_ratio_cut, cut_interp):
    """Measure the shear parameters for metadetect.

    NOTE: Returns None if nothing can be measured.

    Parameters
    ----------
    res : dict
        The metadetect results.
    s2n_cut : float
        The cut on `wmom_s2n`. Typically 10.
    t_ratio_cut : float
        The cut on `t_ratio_cut`. Typically 1.2.
    cut_interp : bool
        If True, cut on the `ormask` flags.

    Returns
    -------
    data : array-like or None
        A structured numpy array with the columns

            g1p : float
                The mean 1-component shape for the
                plus metadetect measurement.
            n1p : float
                The number of objects used for the g1p measurement.
            g1m : float
                The mean 1-component shape for the
                minus metadetect measurement.
            n1m : float
                The number of objects used for the g1m measurement.
            g1 : float
                The mean 1-component shape for the
                zero-shear metadetect measurement.
            n1 : float
                The number of objects used for the g1 measurement.
            g2p : float
                The mean 2-component shape for the
                plus metadetect measurement.
            n2p : float
                The number of objects used for the g2p measurement.
            g2m : float
                The mean 2-component shape for the
                minus metadetect measurement.
            n2m : float
                The number of objects used for the g2m measurement.
            g2 : float
                The mean 2-component shape for the
                zero-shear metadetect measurement.
            n2 : float
                The number of objects used for the g2 measurement.

        If any one of the n* columns will be zero, None is returned.
    """
    def _mask(dta):
        if cut_interp:
            return (
                (dta['flags'] == 0) &
                (dta['ormask'] == 0) &
                (dta['wmom_s2n'] > s2n_cut) &
                (dta['wmom_T_ratio'] > t_ratio_cut))
        else:
            return (
                (dta['flags'] == 0) &
                (dta['wmom_s2n'] > s2n_cut) &
                (dta['wmom_T_ratio'] > t_ratio_cut))

    data = np.zeros(1, dtype=[
        ('g1p', 'f8'), ('n1p', 'f8'),
        ('g1m', 'f8'), ('n1m', 'f8'),
        ('g1', 'f8'), ('n1', 'f8'),
        ('g2p', 'f8'), ('n2p', 'f8'),
        ('g2m', 'f8'), ('n2m', 'f8'),
        ('g2', 'f8'), ('n2', 'f8'),
        ])

    op = res['1p']
    q = _mask(op)
    if not np.any(q):
        return None
    data['g1p'] = np.mean(op['wmom_g'][q, 0])
    data['n1p'] = len(op['wmom_g'][q, 0])

    om = res['1m']
    q = _mask(om)
    if not np.any(q):
        return None
    data['g1m'] = np.mean(om['wmom_g'][q, 0])
    data['n1m'] = len(om['wmom_g'][q, 0])

    o = res['noshear']
    q = _mask(o)
    if not np.any(q):
        return None
    data['g1'] = np.mean(o['wmom_g'][q, 0])
    data['n1'] = len(o['wmom_g'][q, 0])
    data['g2'] = np.mean(o['wmom_g'][q, 1])
    data['n2'] = len(o['wmom_g'][q, 1])

    op = res['2p']
    q = _mask(op)
    if not np.any(q):
        return None
    data['g2p'] = np.mean(op['wmom_g'][q, 1])
    data['n2p'] = len(op['wmom_g'][q, 1])

    om = res['2m']
    q = _mask(om)
    if not np.any(q):
        return None
    data['g2m'] = np.mean(om['wmom_g'][q, 1])
    data['n2m'] = len(om['wmom_g'][q, 1])

    return data
