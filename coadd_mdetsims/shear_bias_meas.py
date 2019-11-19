import numpy as np


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
        ('g1p', 'f4'), ('n1p', 'f4'),
        ('g1m', 'f4'), ('n1m', 'f4'),
        ('g1', 'f4'), ('n1', 'f4'),
        ('g2p', 'f4'), ('n2p', 'f4'),
        ('g2m', 'f4'), ('n2m', 'f4'),
        ('g2', 'f4'), ('n2', 'f4'),
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
