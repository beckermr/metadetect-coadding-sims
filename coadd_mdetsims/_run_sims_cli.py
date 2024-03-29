#!/usr/bin/env python

import sys
import click
import numpy as np
import schwimmbad
import multiprocessing
import logging
from functools import partial
import fitsio

from coadd_mdetsims.sim import CoaddingSim
from coadd_mdetsims.simple_sim import SimpleSim
from coadd_mdetsims.shear_bias_meas import (
    estimate_m_and_c, measure_shear_metadetect)
from coadd_mdetsims.config import load_config
from coadd_mdetsims.defaults import T_RATIO_CUT, S2N_CUTS
from metadetect.metadetect import Metadetect

LOGGER = logging.getLogger(__name__)


def _deal_with_logging(n_sims):
    if n_sims == 1:
        for lib in [__name__, 'ngmix', 'metadetect', 'coadd_mdetsims']:
            lgr = logging.getLogger(lib)
            hdr = logging.StreamHandler(sys.stdout)
            hdr.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            lgr.setLevel(logging.DEBUG)
            lgr.addHandler(hdr)


def _deal_with_mpi(n_sims):
    try:
        if n_sims > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            n_ranks = comm.Get_size()
            HAVE_MPI = True
        else:
            raise Exception()  # punt to the except clause
    except Exception:
        n_ranks = 1
        rank = 0
        comm = None
        HAVE_MPI = False

    if HAVE_MPI and n_ranks > 1:
        n_workers = n_ranks if n_sims > 1 else 1
    else:
        n_workers = multiprocessing.cpu_count() if n_sims > 1 else 1

    USE_MPI = HAVE_MPI and n_ranks > 1

    # short cut if the number of threads is set 1

    return USE_MPI, comm, rank, n_ranks, n_workers


def _add_shears(cfg, swap12, plus=True):
    g1 = 0.02
    g2 = 0.0

    if not plus:
        g1 *= -1

    if swap12:
        g1, g2 = g2, g1

    cfg.update({'g1': g1, 'g2': g2})


def _run_sim(seed, *, sim_config, shear_meas_config,
             swap12, cut_interp, sim_class):
    try:
        # pos shear
        rng = np.random.RandomState(seed=seed)
        _add_shears(sim_config, swap12, plus=True)
        if swap12:
            assert sim_config['g1'] == 0.0
            assert sim_config['g2'] == 0.02
        else:
            assert sim_config['g1'] == 0.02
            assert sim_config['g2'] == 0.0
        sim = sim_class(rng=rng, **sim_config)

        mbobs = sim.get_mbobs()
        md = Metadetect(shear_meas_config, mbobs, rng)
        md.go()

        pres = {}
        for s2n_cut in [10, 15, 20]:
            pres[s2n_cut] = measure_shear_metadetect(
                md.result,
                s2n_cut=s2n_cut,
                t_ratio_cut=T_RATIO_CUT,
                cut_interp=cut_interp)
            if pres[s2n_cut] is None:
                raise Exception("None result for pres s2n %f" % s2n_cut)

        dens = len(md.result['noshear']) / sim.area_sqr_arcmin
        LOGGER.info('found %f objects per square arcminute', dens)

        # neg shear
        rng = np.random.RandomState(seed=seed)
        _add_shears(sim_config, swap12, plus=False)
        if swap12:
            assert sim_config['g1'] == 0.0
            assert sim_config['g2'] == -0.02
        else:
            assert sim_config['g1'] == -0.02
            assert sim_config['g2'] == 0.0
        sim = sim_class(rng=rng, **sim_config)

        mbobs = sim.get_mbobs()
        md = Metadetect(shear_meas_config, mbobs, rng)
        md.go()

        mres = {}
        for s2n_cut in S2N_CUTS:
            mres[s2n_cut] = measure_shear_metadetect(
                md.result,
                s2n_cut=s2n_cut,
                t_ratio_cut=T_RATIO_CUT,
                cut_interp=cut_interp)
            if mres[s2n_cut] is None:
                raise Exception("None result for mres s2n %f" % s2n_cut)

        dens = len(md.result['noshear']) / sim.area_sqr_arcmin
        LOGGER.info('found %f objects per square arcminute', dens)

        retvals = (pres, mres)
    except Exception as e:
        print(repr(e))
        retvals = None

    return retvals


@click.command()
@click.option('--seed', default=42, type=int,
              help='RNG seed to init the sims.')
@click.option('--output-file', default='data.fits', type=str,
              help='File to write outputs to.')
@click.option('--serial', is_flag=True, default=False, type=bool,
              help="Run all sims serially.")
@click.argument('n_sims', type=int)
def main(n_sims, seed, output_file, serial):
    """Run N_SIMS metadetect simulation patches and estimate the shear."""

    # logging and MPI/workers
    _deal_with_logging(n_sims)
    use_mpi, comm, rank, n_ranks, n_workers = _deal_with_mpi(n_sims)

    # and the config
    (sim_config, run_config, shear_meas_config,
     swap12, cut_interp) = load_config('config.yaml')

    use_old_sim = sim_config.pop('straight_to_coadd', False)
    if use_old_sim:
        sim_class = SimpleSim
    else:
        sim_class = CoaddingSim

    __run_sim = partial(
        _run_sim,
        sim_config=sim_config,
        shear_meas_config=shear_meas_config,
        swap12=swap12,
        cut_interp=cut_interp,
        sim_class=sim_class)

    if rank == 0:
        print('running metadetect', flush=True)
        print('config:', sim_config, flush=True)
        print('swap 12:', swap12)
        print('use mpi:', use_mpi, flush=True)
        print("n_ranks:", n_ranks, flush=True)
        print("n_workers:", n_workers, flush=True)

        if use_old_sim:
            print('sim type: straight-to-coadd')
            assert False, (
                "Straight-to-coadd sims are disbaled for safety! Remove this "
                "assert if you actually want to use it!")
        else:
            print('sim type: full coadding')

    seeds = np.random.RandomState(seed=seed).randint(
        low=1,
        high=2**30,
        size=n_sims).tolist()

    if n_workers == 1:
        outputs = [__run_sim(seeds[0])]
    elif serial:
        outputs = [__run_sim(seed) for seed in seeds]
    else:
        if not use_mpi:
            pool = schwimmbad.JoblibPool(
                n_workers, backend='multiprocessing', verbose=40)
        else:
            pool = schwimmbad.choose_pool(mpi=use_mpi, processes=n_workers)

        outputs = pool.map(__run_sim, seeds)
        pool.close()

    outputs = [o for o in outputs if o is not None]
    pres, mres = zip(*outputs)

    all_data = {}

    for s2n in S2N_CUTS:
        data_p = np.concatenate([p[s2n] for p in pres])
        data_m = np.concatenate([m[s2n] for m in mres])

        all_data['p%d' % s2n] = data_p
        all_data['m%d' % s2n] = data_m

        m, msd, c, csd = estimate_m_and_c(
                data_p, data_m, 0.02, swap12=swap12, step=0.01,
                rng=np.random.RandomState(seed=42),
                n_boot=500, verbose=False)

        print("""\
s2n: {s2n}
    # of sims: {n_sims}
    noise cancel m   : {m:f} +/- {msd:f}
    noise cancel c   : {c:f} +/- {csd:f}""".format(
            n_sims=len(data_p),
            m=m,
            msd=msd,
            c=c,
            csd=csd,
            s2n=s2n), flush=True)

    with fitsio.FITS(output_file, 'rw', clobber=True) as fits:
        for s2n in S2N_CUTS:
            for s in ['p', 'm']:
                ext = '%s%d' % (s, s2n)
                fits.write(all_data[ext], extname=ext)
