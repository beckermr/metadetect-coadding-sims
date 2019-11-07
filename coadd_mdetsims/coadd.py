import numpy as np

from .lanczos import lanczos_resample_one


def coadd_psfs(
        se_psfs, se_wcs_objs, coadd_wgts,
        coadd_scale, coadd_dim):
    """Coadd the PSFs.

    Note that this routine assumes that the PSFs in the SE image have their
    centers at the image origin and that they should be interpolated to the
    coadd plane so they are centered at the world origin.

    Parameters
    ----------
    se_psfs : list of np.ndarray
        The list of SE PSF images to coadd.
    se_wcs_objs : list of galsim.BaseWCS or children
        The WCS objects for each of the SE PSFs.
    coadd_wgts : 1d array-like object of floats
        The relative coaddng weights for each of the SE PSFs.
    coadd_scale : float
        The pixel scale of desired coadded PSF image.
    coadd_dim : int
        The number of pixels desired for the final coadd PSF.

    Returns
    -------
    psf : np.ndarray
        The coadded PSF image.
    """

    # coadd pixel coords
    y, x = np.mgrid[0:coadd_dim, 0:coadd_dim]
    u = x.ravel() * coadd_scale
    v = y.ravel() * coadd_scale

    coadd_image = np.zeros((coadd_dim, coadd_dim), dtype=np.float64)

    wgts = coadd_wgts / np.sum(coadd_wgts)

    for se_psf, se_wcs, wgt in zip(se_psfs, se_wcs_objs, wgts):
        se_x, se_y = se_wcs.toImage(u, v)
        im, _ = lanczos_resample_one(se_psf / se_wcs.pixelArea(), se_y, se_x)
        coadd_image += (im.reshape((coadd_dim, coadd_dim)) * wgt)
    coadd_image *= (coadd_scale**2)

    return coadd_image
