import logging
# import os
import functools

import numpy as np

import ngmix
import galsim
import fitsio

from .coadd import coadd_image_noise_interpfrac, coadd_psfs
from .wcs_gen import gen_affine_wcs
from .render_sim import render_objs_with_psf_shear
# from .defaults import WLDEBLEND_DES_FACTOR, WLDEBLEND_LSST_FACTOR

LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _cached_catalog_read(fname):
    return fitsio.read(fname)


class CoaddingSim(object):
    """An end-to-end simulation for metadetect testing.

    Parameters
    ----------
    rng : np.random.RandomState, int or None
        An RNG to use for drawing the objects. If an int or None is passed,
        the input is used to initialize a new `np.random.RandomState` object.
    n_coadd : int, optional
        The number of single epoch images **per band** in a coadd.
    g1 : float, optional
        The simulated shear for the 1-axis.
    g2 : float, optional
        The simulated shear for the 2-axis.
    dim : int, optional
        The total dimension of the coadd image.
    buff : int, optional
        The width of the buffer region in the coadd image.
    noise : float or list of floats, optional
        The noise for a single epoch image. Can be different per band.
    ngal : float, optional
        The number of objects to simulate per arcminute.
    ngal_factor : float, optional
        A factor to change the number density in the sims. It is set to 0.6
        automatically when using the wldeblend galaxy type for DES and 0.45
        when using this type for LSST.
    n_bands : int, optional
        The number of bands to simulate.
    shear_scene : bool, optional
        Whether or not to shear the full scene.

    WCS Parameters
    --------------
    scale : float
        The pixel scale of the coadd image.
    wcs_kws : dict or None, optional
        A dictionary with entries describing how the single epoch image WCS
        transformations should be generated.

        Possible entries are:

            position_angle_range : tuple of floats
                The range of position angles to select from for rotating the
                image WCS coordinares.
            scale_frac_std : float
                The fractional variance in the image pixel scale.
            shear_std : float
                The standard deviation of the Gaussian shear put into the SE
                WCS solutions.
            dither_range : tuple of floats
                The lowest and highest dither in coadd pixels of the center
                of each SE image.

    Galaxy Parameters
    -----------------
    gal_grid : int or None
        If not `None`, galaxies are laid out on a grid of `gal_grid` x
        `gal_grid` dimensions in the central part of the image.
    gal_type : str
        The kind of galaxy to simulate.
    gal_kws : dict or None, optional
        Extra keyword arguments to use when building galaxy objects.

        For gal_type == 'wldeblend', these keywords can be

            'survey_name' : str
                The name of survey in all caps, e.g. 'DES', 'LSST'.
            'catalog' : str
                A path to the catalog to draw from. If this keyword is not
                given, you need to have the one square degree catsim catalog
                in the current working directory or in the directory given by
                the environment variable 'CATSIM_DIR'.
            'bands' : list of str
                A list of strings with the desired bands.

    PSF Parameters
    --------------
    psf_type : str
        The kind of PSF to simulate.
    psf_kws : dict or None, optional
        Extra keyword arguments to pass to the constructors for PSF objects.
        See the doc strings of the PSF object `PowerSpectrumPSF`. You can also
        supply the keywords `fwhm`, `fwhm_frac_std` and `shear_std` to set
        the overall size of the PSF, the fractional variance in the size, and
        the variance in the PSF shape for each epoch.

    Methods
    -------
    get_mbobs(return_band_images=False)
        Make a simulated MultiBandObsList for metadetect.

    Attributes
    ----------
    area_sqr_arcmin : float
        The effective area simulated in square arcmin assuming the pixel
        scale is in arcsec.

    Notes
    -----
    The valid kinds of galaxies are

        'exp' : Sersic objects at very high s/n with n = 1
        # 'wldeblend' : a sample drawn from the WeakLensingDeblending package

    The valid kinds of PSFs are

        'gauss' : a FWHM 0.9 arcsecond Gaussian
        # 'ps' : a PSF from power spectrum model for shape variation and
        #     cubic model for size variation
    """
    def __init__(
            self, *,
            rng,
            scale=0.263,
            n_coadd=1,
            g1=0.02, g2=0.0,
            dim=225, buff=25,
            noise=180,
            ngal=150.0,
            ngal_factor=None,
            n_bands=1,
            shear_scene=True,
            wcs_kws=None,
            gal_grid=None,
            gal_type='exp',
            gal_kws=None,
            psf_type='gauss',
            psf_kws=None):
        self.rng = (
            rng if isinstance(rng, np.random.RandomState)
            else np.random.RandomState(seed=rng))
        self.noise_rng = np.random.RandomState(
            seed=self.rng.randint(1, 2**32-1))
        self.gal_type = gal_type
        self.psf_type = psf_type
        self.n_coadd = n_coadd
        self.g1 = g1
        self.g2 = g2
        self.shear_scene = shear_scene
        self.dim = dim
        self.buff = buff
        self.ngal = ngal
        self.gal_grid = gal_grid
        self.im_cen = (dim - 1) / 2
        self.psf_kws = psf_kws
        self.gal_kws = gal_kws
        self.noise = np.array(noise) * np.ones(n_bands)
        self.ngal_factor = ngal_factor

        self.area_sqr_arcmin = ((self.dim - 2*self.buff) * scale / 60)**2

        # the SE image could be rotated, so we make it big nough to cover the
        # whole coadd region
        dfac = np.sqrt(2)
        self.se_dim = int(np.ceil(self.dim * dfac)) + 10
        if self.se_dim % 2 == 0:
            self.se_dim = self.se_dim + 1

        self._galsim_rng = galsim.BaseDeviate(
            seed=self.rng.randint(low=1, high=2**32-1))

        # wcs info
        self.scale = scale
        self.coadd_wcs = galsim.PixelScale(self.scale)
        self.wcs_kws = wcs_kws

        # frac of a single dimension that is used for drawing objects
        frac = 1.0 - self.buff * 2 / self.dim

        # half of the width of center of the patch that has objects
        self.pos_width = self.dim * frac * 0.5 * self.scale

        # for wldeblend galaxies, we have to adjust some of the input
        # parameters since they are computed self consisently from the
        # input catalog and/or package defaults
        if self.gal_type == 'wldeblend':
            # self._extra_init_for_wldeblend()
            raise ValueError("wldeblend is not supported yet!")

        # given the input number of objects to simulate per square arcminute,
        # compute the number we actually need
        if self.ngal_factor is None:
            self.ngal_factor = 1
        LOGGER.info('ngal adjustment factor: %f', self.ngal_factor)

        self.nobj = int(
            self.ngal * self.ngal_factor * self.area_sqr_arcmin)

        self.shear_mat = galsim.Shear(g1=self.g1, g2=self.g2).getMatrix()

        # reset nobj to the number in a grid if we are using one
        if self.gal_grid is not None:
            self.nobj = self.gal_grid * self.gal_grid

        self.n_bands = len(self.noise)

        # because of the caching of psfs and the wcs below, we only
        # allow the sim class to b used once
        # this attribute gets set to True after it is used
        self.called = False

        LOGGER.info('simulating %d bands', self.n_bands)

        # info about coadd PSF image
        self._psf_dim = 53
        self._psf_cen = (self._psf_dim - 1)/2

    # def _extra_init_for_wldeblend(self):
    #     # guard the import here
    #     import descwl
    #
    #     # make sure to find the proper catalog
    #     gal_kws = self.gal_kws or {}
    #     if 'catalog' not in gal_kws:
    #         fname = os.path.join(
    #             os.environ.get('CATSIM_DIR', '.'),
    #             'OneDegSq.fits')
    #     else:
    #         fname = gal_kws['catalog']
    #
    #     self._wldeblend_cat = _cached_catalog_read(fname)
    #     self._wldeblend_cat['pa_disk'] = self.rng.uniform(
    #         low=0.0, high=360.0, size=self._wldeblend_cat.size)
    #     self._wldeblend_cat['pa_bulge'] = self._wldeblend_cat['pa_disk']
    #
    #     # set the survey name and exposure times
    #     if 'survey_name' not in gal_kws:
    #         survey_name = 'DES'
    #     else:
    #         survey_name = gal_kws['survey_name']
    #
    #     if survey_name == 'DES':
    #         exptime = 90
    #         if self.n_coadd != 10:
    #             LOGGER.warning(
    #                 'simulating DES with descwl - '
    #                 'input n_coadd != 10!')
    #     elif survey_name == 'LSST':
    #         exptime = 15
    #         if self.n_coadd != 360:
    #             LOGGER.warning(
    #                 'simulating LSST with descwl - '
    #                 'input n_coadd != 360!')
    #     else:
    #         raise ValueError("Survey '%s' is not valid!" % survey_name)
    #
    #     bands = gal_kws.get('bands', ['r', 'i', 'z'])
    #     LOGGER.debug('simulating bands: %s', bands)
    #
    #     self._surveys = []
    #     self._builders = []
    #     noises = []
    #     for iband, band in enumerate(bands):
    #         # make the survey and code to build galaxies from it
    #         pars = descwl.survey.Survey.get_defaults(
    #             survey_name=survey_name,
    #             filter_band=band)
    #
    #         pars['survey_name'] = survey_name
    #         pars['filter_band'] = band
    #         pars['pixel_scale'] = self.scale
    #
    #         # note in the way we call the descwl package, the image width
    #         # and height is not actually used
    #         pars['image_width'] = self.dim
    #         pars['image_height'] = self.dim
    #
    #         # reset the exposure times as needed
    #         if survey_name == 'DES':
    #             pars['exposure_time'] = exptime
    #         elif survey_name == 'LSST':
    #             pars['exposure_time'] = pars['exposure_time'] / self.n_coadd
    #
    #         # some versions take in the PSF and will complain if it is not
    #         # given
    #         try:
    #             _svy = descwl.survey.Survey(**pars)
    #         except Exception:
    #             pars['psf_model'] = None
    #             _svy = descwl.survey.Survey(**pars)
    #
    #         self._surveys.append(_svy)
    #         self._surveys.append(descwl.survey.Survey(**pars))
    #         self._builders.append(descwl.model.GalaxyBuilder(
    #             survey=self._surveys[iband],
    #             no_disk=False,
    #             no_bulge=False,
    #             no_agn=False,
    #             verbose_model=False))
    #
    #         noises.append(np.sqrt(self._surveys[iband].mean_sky_level))
    #
    #     self.noise = noises
    #
    #     # when we sample from the catalog, we need to pull the right number
    #     # of objects. Since the default catalog is one square degree
    #     # and we fill a fraction of the image, we need to set the
    #     # base source density `ngal`. This is in units of number per
    #     # square arcminute.
    #     self.ngal = self._wldeblend_cat.size / (60 * 60)
    #
    #     # we use a factor of 0.6 to make sure the depth matches that in
    #     # the real data
    #     if self.ngal_factor is None:
    #         if survey_name == 'DES':
    #             self.ngal_factor = WLDEBLEND_DES_FACTOR
    #         elif survey_name == 'LSST':
    #             self.ngal_factor = WLDEBLEND_LSST_FACTOR
    #         else:
    #             raise ValueError("Survey '%s' is not valid!" % survey_name)
    #
    #     LOGGER.info('catalog density: %f per sqr arcmin', self.ngal)

    def get_mbobs(self, return_band_images=False):
        """Make a simulated MultiBandObsList for metadetect.

        The underlying simulation is done per epoch and then coadded.

        Parameters
        ----------
        return_band_images : bool
            If True, return a list of list of numpy arrays holding the
            SE images in each band.

        Returns
        -------
        mbobs : MultiBandObsList
        """

        assert not self.called, "you can only call a sim object once!"
        self.called = True

        all_band_obj, uv_offsets = self._get_band_objects()

        method = 'auto'
        LOGGER.debug("using draw method '%s'", method)

        uv_cen = galsim.PositionD(
            x=self.im_cen * self.scale,
            y=self.im_cen * self.scale)

        mbobs = ngmix.MultiBandObsList()

        if return_band_images:
            band_images = []

        for band in range(self.n_bands):
            # generate the data I need
            band_objects = [o[band] for o in all_band_obj]
            wcs_objs = self._get_all_epoch_wcs_objs(band)

            # draw the images
            se_images = []
            for epoch, wcs in enumerate(wcs_objs):
                _psf_func = self._get_psf_model_function(
                    band=band, epoch=epoch)
                se_im = render_objs_with_psf_shear(
                    objs=band_objects,
                    psf_function=_psf_func,
                    uv_offsets=uv_offsets,
                    uv_cen=uv_cen,
                    wcs=wcs,
                    img_dim=self.se_dim,
                    method=method,
                    g1=self.g1,
                    g2=self.g2,
                    shear_scene=self.shear_scene)
                se_images.append(se_im.array)

            if return_band_images:
                band_images.append(se_images)

            # add noise, maybe mask them, get noise/wt images and coadd
            (coadd_im, coadd_noise, coadd_intp,
             coadd_bmask, coadd_wgts) = self._add_noise_and_coadd(
                band=band, wcs_objs=wcs_objs, se_images=se_images)

            LOGGER.debug("coadd weights for band %d: %s", band, coadd_wgts)

            # coadd the PSFs
            coadd_psf = self._coadd_psfs(
                band=band, wcs_objs=wcs_objs,
                coadd_wgts=coadd_wgts, method=method)

            # make the final obs
            obs_jac = ngmix.jacobian.Jacobian(
                row=self.im_cen,
                col=self.im_cen,
                wcs=self.coadd_wcs.jacobian())

            psf_jac = ngmix.jacobian.Jacobian(
                row=self._psf_cen,
                col=self._psf_cen,
                wcs=self.coadd_wcs.jacobian())

            psf_obs = ngmix.Observation(
                coadd_psf,
                weight=0.0 * coadd_psf + 1.0 / self.noise[band]**2,
                jacobian=psf_jac)

            obs = ngmix.Observation(
                coadd_im,
                weight=0.0 * coadd_im + 1.0 / np.var(coadd_noise),
                bmask=coadd_bmask,
                ormask=coadd_bmask.copy(),
                jacobian=obs_jac,
                psf=psf_obs,
                noise=coadd_noise)
            obs.meta['fmask'] = coadd_intp

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        if return_band_images:
            return mbobs, band_images
        else:
            return mbobs

    def _add_noise_and_coadd(self, *, band, wcs_objs, se_images):
        LOGGER.info('coadding %d images for band %d',
                    len(se_images), band)
        se_noises = []
        se_interp_fracs = []
        coadd_wgts = []
        final_se_images = []
        for se_im in se_images:
            se_im += self.noise_rng.normal(
                scale=self.noise[band], size=se_im.shape)
            se_nse = self.noise_rng.normal(size=se_im.shape) * self.noise[band]

            # if self.mask_and_interp:
            #     final_se_im, se_nse, bad_msk = self._mask_and_interp(
            #         se_im, se_nse)
            #     se_interp_frac = bad_msk.astype(np.float32)
            # else:
            final_se_im = se_im
            se_interp_frac = np.zeros_like(se_im)

            final_se_images.append(final_se_im)
            se_noises.append(se_nse)
            se_interp_fracs.append(se_interp_frac)

            coadd_wgts.append(1.0 / self.noise[band]**2)

        coadd_wgts = np.array(coadd_wgts)

        # coadd them
        coadd_im, coadd_noise, coadd_intp = coadd_image_noise_interpfrac(
            final_se_images, se_noises, se_interp_fracs, wcs_objs,
            coadd_wgts, self.scale, self.dim)

        coadd_bmask = np.zeros_like(coadd_im, dtype=np.int32)
        coadd_bmask[coadd_intp > 0] = 1

        return coadd_im, coadd_noise, coadd_intp, coadd_bmask, coadd_wgts

    def _coadd_psfs(self, *, band, wcs_objs, coadd_wgts, method):

        se_psf_dim = int(np.ceil(self._psf_dim * np.sqrt(2)))
        if se_psf_dim % 2 == 0:
            se_psf_dim += 1
        se_psf_cen = (se_psf_dim - 1)/2

        world_origin = galsim.PositionD(
            x=self._psf_cen * self.scale,
            y=self._psf_cen * self.scale)

        se_psfs = []
        psf_wcs_objs = []
        for epoch, wcs in enumerate(wcs_objs):
            # use the world origin to get the subpixel offset of the SE
            # PSF center
            pos = wcs.toImage(world_origin)
            dx = pos.x - int(pos.x + 0.5)
            dy = pos.y - int(pos.y + 0.5)

            # set the SE image origin to the SE image PSF location with the
            # subpixel offset
            image_origin = galsim.PositionD(
                x=se_psf_cen + dx,
                y=se_psf_cen + dy)

            # get the proper WCs object for caodding
            # this object needs to have the world origin at the final coadd
            # PSF center and the image origin at the SE image PSF center
            psf_wcs_objs.append(
                wcs.jacobian(
                    image_origin
                ).withOrigin(
                    image_origin, world_origin))

            # now draw the SE image PSF w/ the subpixel offset
            local_wcs = wcs.local(world_pos=world_origin)

            psf = self._get_psf_model_function(
                band=band, epoch=epoch)(x=pos.x, y=pos.y)

            se_psfs.append(psf.drawImage(
                nx=se_psf_dim,
                ny=se_psf_dim,
                wcs=local_wcs,
                method=method,
                offset=galsim.PositionD(x=dx, y=dy)).array.copy())

        LOGGER.info('coadding %d PSF images for band %d',
                    len(se_psfs), band)

        coadd_psf = coadd_psfs(
            se_psfs, psf_wcs_objs, coadd_wgts,
            self.scale, self._psf_dim)

        return coadd_psf / np.sum(coadd_psf)

    def _get_all_epoch_wcs_objs(self, band):
        if not hasattr(self, '_band_wcs_objs'):
            default_wcs_kws = dict(
                position_angle_range=(0, 0),
                scale_frac_std=0,
                shear_std=0,
                dither_range=(0, 0))
            if self.wcs_kws is not None:
                default_wcs_kws.update(self.wcs_kws)

            LOGGER.info('using WCS kwargs %s', default_wcs_kws)

            se_cen = (self.se_dim - 1) / 2
            world_origin = galsim.PositionD(
                x=self.im_cen * self.scale,
                y=self.im_cen * self.scale)
            origin = galsim.PositionD(x=se_cen, y=se_cen)

            self._band_wcs_objs = []
            for _ in range(self.n_bands):

                wcs_objs = []
                for _ in range(self.n_coadd):
                    wcs_objs.append(gen_affine_wcs(
                        rng=self.rng,
                        scale=self.scale,
                        world_origin=world_origin,
                        origin=origin,  # this gets a dither
                        **default_wcs_kws
                    ))

                self._band_wcs_objs.append(wcs_objs)

        LOGGER.info('found %d WCS objects for band %d',
                    len(self._band_wcs_objs[band]), band)

        return self._band_wcs_objs[band]

    def _get_dudv(self):
        if self.gal_grid is not None:
            yind, xind = np.unravel_index(
                self._gal_grid_ind, (self.gal_grid, self.gal_grid))
            dg = self.pos_width * 2 / self.gal_grid
            self._gal_grid_ind += 1
            return (
                yind * dg + dg/2 - self.pos_width,
                xind * dg + dg/2 - self.pos_width)
        else:
            return self.rng.uniform(
                low=-self.pos_width,
                high=self.pos_width,
                size=2)

    def _get_nobj(self):
        if self.gal_grid is not None:
            return self.nobj
        else:
            return self.rng.poisson(self.nobj)

    def _get_gal_exp(self):
        flux = 10**(0.4 * (30 - 18))
        half_light_radius = 0.5

        _gal = []
        for _ in range(self.n_bands):
            obj = galsim.Sersic(
                half_light_radius=half_light_radius,
                n=1,
            ).withFlux(flux)
            _gal.append(obj)

        return _gal

    # def _get_gal_wldeblend(self):
    #     rind = self.rng.choice(self._wldeblend_cat.size)
    #     angle = self.rng.uniform() * 360
    #
    #     gals = [
    #         self._builders[band].from_catalog(
    #             self._wldeblend_cat[rind], 0, 0,
    #             self._surveys[band].filter_band).model.rotate(
    #                 angle * galsim.degrees)
    #         for band in range(len(self._builders))]
    #
    #     return gals

    def _get_band_objects(self):
        """Get a list of effective PSF-convolved galsim images w/ their
        offsets in the image.

        Returns
        -------
        all_band_objs : list of lists
            A list of lists of objects in each band.
        uv_offsets : list of galsim.PositionD
            A list of galsim positions for each object.
        """
        all_band_obj = []
        uv_offsets = []

        nobj = self._get_nobj()

        LOGGER.info('drawing %d objects for a %f square arcmin patch',
                    nobj, self.area_sqr_arcmin)

        if self.gal_grid is not None:
            self._gal_grid_ind = 0

        for i in range(nobj):
            # unsheared offset from center of uv image
            du, dv = self._get_dudv()
            duv = galsim.PositionD(x=du, y=dv)

            # get the galaxy
            if self.gal_type == 'exp':
                gals = self._get_gal_exp()
            # elif self.gal_type == 'wldeblend':
            #     gals = self._get_gal_wldeblend()
            else:
                raise ValueError('gal_type "%s" not valid!' % self.gal_type)

            all_band_obj.append(gals)
            uv_offsets.append(duv)

        return all_band_obj, uv_offsets

    # def _make_ps_psfs(self):
    #     kwargs = self.psf_kws or {}
    #     self._ps_psfs = []
    #     for band in range(self.n_bands):
    #         band_psfs = []
    #         for epoch in range(self.n_coadd):
    #             band_psfs.append(
    #                 PowerSpectrumPSF(
    #                     rng=self.rng,
    #                     im_width=self.dim,
    #                     buff=self.dim/2,
    #                     scale=self.scale,
    #                     **kwargs)
    #             )
    #         self._ps_psfs.append(band_psfs)

    def _get_psf_model_function(self, *, band, epoch):
        if not hasattr(self, '_psf_fwhms'):
            kws = self.psf_kws or {}
            fwhm = kws.get('fwhm', 0.9)
            fwhm_std = kws.get('fwhm_frac_std', 0.0)
            shear_std = kws.get('shear_std', 0.0)
            self._psf_fwhm = fwhm
            self._psf_fwhms = []
            self._psf_shears = []
            for _ in range(self.n_bands):
                self._psf_fwhms.append(
                    (1.0 + self.rng.normal(size=self.n_coadd) * fwhm_std) *
                    fwhm)

                g1s = self.rng.normal(size=self.n_coadd) * shear_std
                g2s = self.rng.normal(size=self.n_coadd) * shear_std
                self._psf_shears.append([
                    galsim.Shear(g1=g1, g2=g2) for g1, g2 in zip(g1s, g2s)])

        LOGGER.debug('PSF fwhms for band %d: %s', band, self._psf_fwhms)
        LOGGER.debug('PSF shears for band %d: %s', band, self._psf_shears)

        def _psf_model_func(*, x, y):
            if self.psf_type == 'gauss':
                psf = galsim.Gaussian(
                    fwhm=self._psf_fwhms[band][epoch]
                ).shear(
                    self._psf_shears[band][epoch]
                ).withFlux(1.0)
                return psf
            # elif self.psf_type == 'wldeblend':
            #     return self._surveys[band].psf_model.dilate(
            #         self._psf_fwhms[band][epoch] / self._psf_fwhm
            #     ).shear(
            #         self._psf_shears[band][epoch])
            # elif self.psf_type == 'ps':
            #     if not hasattr(self, '_ps_psfs'):
            #         self._make_ps_psfs()
            #     return self._ps_psfs[band][epoch].getPSF(
            #         galsim.PositionD(x=x, y=y))
            else:
                raise ValueError('psf_type "%s" not valid!' % self.psf_type)

        return _psf_model_func
