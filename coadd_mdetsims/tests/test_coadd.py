import numpy as np
import galsim

import pytest

from ..coadd import coadd_psfs
from ..wcs_gen import gen_affine_wcs


@pytest.mark.parametrize('crazy_psf', [False, True])
@pytest.mark.parametrize('crazy_wcs', [False, True])
def test_coadd_psfs(crazy_wcs, crazy_psf):
    n_coadd = 10
    rng = np.random.RandomState(seed=42)

    if crazy_psf:
        psfs = []
        for _ in range(n_coadd):
            _fwhm = 2.9 * (1.0 + rng.normal() * 0.1)
            _g1 = rng.normal() * 0.3
            _g2 = rng.normal() * 0.3
            psfs.append(galsim.Gaussian(fwhm=_fwhm).shear(g1=_g1, g2=_g2))
    else:
        psfs = []
        for _ in range(n_coadd):
            psfs.append(galsim.Gaussian(fwhm=2.9).shear(g1=-0.1, g2=0.3))

    scale = 0.213

    coadd_psf_dim = 103
    coadd_psf_cen = (coadd_psf_dim - 1)/2

    se_psf_dim = 103
    se_psf_cen = (se_psf_dim - 1)/2

    coadd_wgts = rng.uniform(size=n_coadd)
    coadd_wgts /= np.sum(coadd_wgts)

    world_origin = galsim.PositionD(
        x=coadd_psf_cen * scale,
        y=coadd_psf_cen * scale)
    origin = galsim.PositionD(x=se_psf_cen, y=se_psf_cen)

    se_wcs_objs = []
    for _ in range(n_coadd):
        if crazy_wcs:
            wcs = gen_affine_wcs(
                rng=rng,
                position_angle_range=(0, 360),
                dither_range=(5, 5),
                scale=scale,
                scale_frac_std=0.05,
                shear_std=0.1,
                world_origin=world_origin,
                origin=origin)
        else:
            wcs = gen_affine_wcs(
                rng=rng,
                position_angle_range=(0, 0),
                dither_range=(0, 0),
                scale=scale,
                scale_frac_std=0.0,
                shear_std=0.0,
                world_origin=world_origin,
                origin=origin)

        se_wcs_objs.append(wcs)

    se_psfs = []
    for wcs, psf in zip(se_wcs_objs, psfs):
        # we need to offset the PSF images to match the image origin in the
        # stamp
        offset = np.array([
            wcs.x0 - se_psf_cen,
            wcs.y0 - se_psf_cen])
        se_psfs.append(psf.drawImage(
            nx=se_psf_dim,
            ny=se_psf_dim,
            wcs=wcs.local(world_pos=world_origin),
            offset=offset).array)

    coadd_psf = coadd_psfs(
        se_psfs, se_wcs_objs, coadd_wgts,
        scale, coadd_psf_dim)

    true_coadd_psf = galsim.Sum(
        [psf.withFlux(wgt) for psf, wgt in zip(psfs, coadd_wgts)]
    ).drawImage(
        nx=coadd_psf_dim,
        ny=coadd_psf_dim,
        scale=scale).array
    true_coadd_psf /= np.sum(true_coadd_psf)

    if not crazy_wcs:
        rtol = 0
        atol = 5e-8
    else:
        rtol = 0
        atol = 5e-5

    if not np.allclose(coadd_psf, true_coadd_psf, rtol=rtol, atol=atol):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axs = plt.subplots(nrows=2, ncols=2)
        sns.heatmap(true_coadd_psf, ax=axs[0, 0])
        sns.heatmap(coadd_psf, ax=axs[0, 1])
        sns.heatmap(coadd_psf - true_coadd_psf, ax=axs[1, 0])
        sns.heatmap(
            np.abs(coadd_psf - true_coadd_psf) <
            atol + rtol * np.abs(true_coadd_psf),
            ax=axs[1, 1])
        print(np.max(np.abs(coadd_psf - true_coadd_psf)))

    assert np.allclose(coadd_psf, true_coadd_psf, rtol=rtol, atol=atol)
