import numpy as np
import pytest

from ..sim import CoaddingSim


def test_coadding_sim_smoke():
    rng = np.random.RandomState(seed=10)
    sim = CoaddingSim(
        rng=rng,
        gal_type='exp',
        psf_type='gauss',
        scale=0.263
    )
    sim.get_mbobs()


def test_coadding_sim_seeding():
    wcs_kws = {
        'position_angle_range': (0, 360),
        'dither_range': (-5, 5),
        'scale_frac_std': 0.03,
        'shear_std': 0.1
    }

    rng1 = np.random.RandomState(seed=10)
    sim1 = CoaddingSim(
        rng=rng1,
        gal_type='exp',
        psf_type='gauss',
        scale=0.263,
        wcs_kws=wcs_kws,
        n_coadd=5,
        n_bands=3,
    )
    mbobs1, bi1 = sim1.get_mbobs(return_band_images=True)

    rng2 = np.random.RandomState(seed=10)
    sim2 = CoaddingSim(
        rng=rng2,
        gal_type='exp',
        psf_type='gauss',
        scale=0.263,
        wcs_kws=wcs_kws,
        n_coadd=5,
        n_bands=3,
    )
    mbobs2, bi2 = sim2.get_mbobs(return_band_images=True)

    for b1, b2 in zip(bi1, bi2):
        for i1, i2 in zip(b1, b2):
            assert np.allclose(i1, i2)

    for obsl1, obsl2 in zip(mbobs1, mbobs2):
        assert np.allclose(obsl1[0].image, obsl2[0].image)
        assert np.allclose(obsl1[0].noise, obsl2[0].noise)
        assert np.allclose(obsl1[0].weight, obsl2[0].weight)
        assert np.allclose(obsl1[0].bmask, obsl2[0].bmask)
        assert (
            obsl1[0].jacobian.get_galsim_wcs() ==
            obsl2[0].jacobian.get_galsim_wcs())

        assert np.allclose(obsl1[0].psf.image, obsl2[0].psf.image)
        assert np.allclose(obsl1[0].psf.weight, obsl2[0].psf.weight)
        assert (
            obsl1[0].psf.jacobian.get_galsim_wcs() ==
            obsl2[0].psf.jacobian.get_galsim_wcs())


def test_coadding_sim_outputs_basic():
    wcs_kws = {
        'position_angle_range': (0, 360),
        'dither_range': (-5, 5),
        'scale_frac_std': 0.03,
        'shear_std': 0.1
    }
    rng = np.random.RandomState(seed=10)
    sim = CoaddingSim(
        rng=rng,
        gal_type='exp',
        psf_type='gauss',
        scale=0.263,
        n_coadd=5,
        n_bands=3,
        dim=227,
        wcs_kws=wcs_kws
    )
    mbobs, bi = sim.get_mbobs(return_band_images=True)

    assert len(mbobs) == 3
    assert len(mbobs[0]) == 1
    assert len(mbobs[1]) == 1
    assert len(mbobs[2]) == 1

    assert len(bi[0]) == 5
    assert len(bi[1]) == 5
    assert len(bi[2]) == 5

    for obslist in mbobs:
        assert obslist[0].image.shape == (227, 227)
        assert np.allclose(
            obslist[0].jacobian.get_galsim_wcs().pixelArea(), 0.263**2)

        assert obslist[0].psf.image.shape == (53, 53)
        assert np.allclose(
            obslist[0].psf.jacobian.get_galsim_wcs().pixelArea(), 0.263**2)


def test_coadding_sim_raises_twice_called():
    rng = np.random.RandomState(seed=10)
    sim = CoaddingSim(
        rng=rng,
        gal_type='exp',
        psf_type='gauss',
        scale=0.263
    )
    sim.get_mbobs()

    with pytest.raises(AssertionError):
        sim.get_mbobs()
