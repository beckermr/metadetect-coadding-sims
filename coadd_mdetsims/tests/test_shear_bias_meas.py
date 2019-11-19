import numpy as np
import pytest

from ..shear_bias_meas import measure_shear_metadetect, estimate_m_and_c


def test_measure_shear_metadetect_smoke():
    dtype = [('flags', 'i4'), ('wmom_g', 'f4', (2,)),
             ('wmom_s2n', 'f4'), ('wmom_T_ratio', 'f4'), ('ormask', 'i4')]
    rng = np.random.RandomState(seed=10)
    res = {}
    for key in ['noshear', '1p', '1m', '2p', '2m']:
        res[key] = np.zeros(10, dtype=dtype)
        for col in ['wmom_s2n', 'wmom_T_ratio']:
            res[key][col] = rng.uniform(size=10) * 10
        res[key]['wmom_g'][:, 0] = rng.uniform(size=10) * 10
        res[key]['wmom_g'][:, 1] = rng.uniform(size=10) * 10
        res[key]['flags'] = (rng.uniform(size=10) * 1.5).astype(np.int32)
        res[key]['ormask'] = (rng.uniform(size=10) * 1.5).astype(np.int32)

    val = measure_shear_metadetect(
        res, s2n_cut=2, t_ratio_cut=1, cut_interp=False)
    assert val is not None
    for key in ['1p', '1m', '1', '2p', '2m', '2']:
        gkey = 'g' + key
        nkey = 'n' + key
        reskey = key if ('p' in key or 'm' in key) else 'noshear'
        ind = int(key[0]) - 1
        q = ((res[reskey]['wmom_s2n'] > 2) &
             (res[reskey]['wmom_T_ratio'] > 1) &
             (res[reskey]['flags'] == 0))
        assert np.mean(res[reskey]['wmom_g'][q, ind]) == val[gkey][0]
        assert len(res[reskey]['wmom_g'][q, ind]) == val[nkey][0]

    val = measure_shear_metadetect(
        res, s2n_cut=2, t_ratio_cut=1, cut_interp=True)
    assert val is not None
    for key in ['1p', '1m', '1', '2p', '2m', '2']:
        gkey = 'g' + key
        nkey = 'n' + key
        reskey = key if ('p' in key or 'm' in key) else 'noshear'
        ind = int(key[0]) - 1
        q = ((res[reskey]['wmom_s2n'] > 2) &
             (res[reskey]['wmom_T_ratio'] > 1) &
             (res[reskey]['flags'] == 0) &
             (res[reskey]['ormask'] == 0))
        assert np.mean(res[reskey]['wmom_g'][q, ind]) == val[gkey][0]
        assert len(res[reskey]['wmom_g'][q, ind]) == val[nkey][0]


@pytest.mark.parametrize('kind', ['noshear', '1p', '1m', '2p', '2m'])
def test_measure_shear_metadetect_none(kind):
    dtype = [('flags', 'i4'), ('wmom_g', 'f4', (2,)),
             ('wmom_s2n', 'f4'), ('wmom_T_ratio', 'f4'), ('ormask', 'i4')]
    rng = np.random.RandomState(seed=10)
    res = {}
    for key in ['noshear', '1p', '1m', '2p', '2m']:
        res[key] = np.zeros(10, dtype=dtype)
        for col in ['wmom_s2n', 'wmom_T_ratio']:
            res[key][col] = rng.uniform(size=10) * 10
        res[key]['wmom_g'][:, 0] = rng.uniform(size=10) * 10
        res[key]['wmom_g'][:, 1] = rng.uniform(size=10) * 10
        res[key]['flags'] = (rng.uniform(size=10) * 1.5).astype(np.int32)
        res[key]['ormask'] = (rng.uniform(size=10) * 1.5).astype(np.int32)

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        # we make sure it would return something if not for the bad flags etc
        val = measure_shear_metadetect(
            res, s2n_cut=2, t_ratio_cut=1, cut_interp=False)
        assert val is not None

        val = measure_shear_metadetect(
            res, s2n_cut=2, t_ratio_cut=1, cut_interp=True)
        assert val is not None

        # now test some columns
        for col, col_val in [
                ('flags', 1), ('wmom_s2n', 0), ('wmom_T_ratio', 0)]:
            old_col_val = res[key][col].copy()
            res[key][col] = col_val
            val = measure_shear_metadetect(
                res, s2n_cut=2, t_ratio_cut=1, cut_interp=False)
            assert val is None
            res[key][col] = old_col_val

        old_flags = res[key]['ormask'].copy()
        res[key]['ormask'] = 1
        val = measure_shear_metadetect(
            res, s2n_cut=2, t_ratio_cut=1, cut_interp=True)
        assert val is None
        res[key]['ormask'] = old_flags


@pytest.mark.parametrize('g_true,step,swap12', [
    (0.01, 0.02, True),
    (0.01, 0.02, False),
    (0.005, 0.05, False)])
def test_estimate_m_and_c(g_true, step, swap12):
    rng = np.random.RandomState(seed=10)

    def _shear_meas(g_true, _step, e1, e2):
        if _step == 0:
            _gt = g_true * (1.0 + 0.01)
            cadd = 0.05 * 10
        else:
            _gt = g_true
            cadd = 0.0
        if swap12:
            return np.mean(e1) + cadd + _step*10, np.mean(10*(_gt+_step)+e2)
        else:
            return np.mean(10*(_gt+_step)+e1), np.mean(e2) + cadd + _step*10

    sn = 0.01
    n_gals = 100000
    n_sim = 1000
    pres = np.zeros(n_sim, dtype=[
        ('g1p', 'f4'), ('n1p', 'f4'),
        ('g1m', 'f4'), ('n1m', 'f4'),
        ('g1', 'f4'), ('n1', 'f4'),
        ('g2p', 'f4'), ('n2p', 'f4'),
        ('g2m', 'f4'), ('n2m', 'f4'),
        ('g2', 'f4'), ('n2', 'f4')])
    mres = np.zeros(n_sim, dtype=[
        ('g1p', 'f4'), ('n1p', 'f4'),
        ('g1m', 'f4'), ('n1m', 'f4'),
        ('g1', 'f4'), ('n1', 'f4'),
        ('g2p', 'f4'), ('n2p', 'f4'),
        ('g2m', 'f4'), ('n2m', 'f4'),
        ('g2', 'f4'), ('n2', 'f4')])
    for i in range(n_sim):
        e1 = rng.normal(size=n_gals) * sn
        e2 = rng.normal(size=n_gals) * sn

        g1, g2 = _shear_meas(g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(g_true, step, e1, e2)
        g1m, g2m = _shear_meas(g_true, -step, e1, e2)
        pres['g1p'][i] = g1p
        pres['g1m'][i] = g1m
        pres['g1'][i] = g1
        pres['g2p'][i] = g2p
        pres['g2m'][i] = g2m
        pres['g2'][i] = g2

        pres['n1p'][i] = n_gals
        pres['n1m'][i] = n_gals
        pres['n1'][i] = n_gals
        pres['n2p'][i] = n_gals
        pres['n2m'][i] = n_gals
        pres['n2'][i] = n_gals

        g1, g2 = _shear_meas(-g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(-g_true, step, e1, e2)
        g1m, g2m = _shear_meas(-g_true, -step, e1, e2)
        mres['g1p'][i] = g1p
        mres['g1m'][i] = g1m
        mres['g1'][i] = g1
        mres['g2p'][i] = g2p
        mres['g2m'][i] = g2m
        mres['g2'][i] = g2

        mres['n1p'][i] = n_gals
        mres['n1m'][i] = n_gals
        mres['n1'][i] = n_gals
        mres['n2p'][i] = n_gals
        mres['n2m'][i] = n_gals
        mres['n2'][i] = n_gals

    m, _, c, _ = estimate_m_and_c(
        pres, mres, g_true, swap12=swap12, step=step)

    assert np.allclose(m, 0.01)
    assert np.allclose(c, 0.05)
