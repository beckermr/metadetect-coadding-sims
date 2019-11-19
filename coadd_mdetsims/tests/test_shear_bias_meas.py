import numpy as np
import pytest

from ..shear_bias_meas import measure_shear_metadetect


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
