import click
import glob
import fitsio
import joblib
import numpy as np

from coadd_mdetsims.shear_bias_meas import estimate_m_and_c
from coadd_mdetsims.config import load_config
from coadd_mdetsims.defaults import S2N_CUTS


def _resum_cols(d):
    for col in ['1p', '1m', '1', '2p', '2m', '2']:
        gcol = 'g%s' % col
        ncol = 'n%s' % col
        d[gcol][0] = np.sum(d[gcol] * d[ncol]) / np.sum(d[ncol])

    for col in ['1p', '1m', '1', '2p', '2m', '2']:
        ncol = 'n%s' % col
        d[ncol][0] = np.sum(d[ncol])

    return d[0:1]


def _func(fname, patch_bootstrap):
    try:
        pres = {}
        mres = {}
        with fitsio.FITS(fname, 'r') as fits:
            for s2n in S2N_CUTS:
                ext = '%s%d' % ('p', s2n)
                pres[s2n] = fits[ext].read()

                ext = '%s%d' % ('m', s2n)
                mres[s2n] = fits[ext].read()

                if not patch_bootstrap:
                    pres[s2n] = _resum_cols(pres[s2n])
                    mres[s2n] = _resum_cols(mres[s2n])

        return (pres, mres)
    except Exception:
        return None


@click.command()
@click.option('--patch-bootstrap', is_flag=True, default=False, type=bool,
              help="Bootstrap over patches instead of sets of patches.")
def main(patch_bootstrap):
    tmpdir = 'condor_outputs'
    _, _, _, swap12, _ = load_config('config.yaml')

    files = glob.glob('%s/data*.fits' % tmpdir)
    print('found %d outputs' % len(files))
    if len(files) == 0:
        return

    io = [joblib.delayed(_func)(fname, patch_bootstrap) for fname in files]
    outputs = joblib.Parallel(
        verbose=10,
        n_jobs=-1,
        pre_dispatch='2*n_jobs',
        max_nbytes=None)(io)

    outputs = [o for o in outputs if o is not None]
    print('found %d non-None outputs' % len(outputs))
    if len(outputs) == 0:
        return

    pres, mres = zip(*outputs)

    for s2n in S2N_CUTS:
        data_p = np.concatenate([p[s2n] for p in pres])
        data_m = np.concatenate([m[s2n] for m in mres])

        m, msd, c, csd = estimate_m_and_c(
                data_p, data_m, 0.02, swap12=swap12, step=0.01,
                rng=np.random.RandomState(seed=42),
                n_boot=500, verbose=True)

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
