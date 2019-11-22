import glob
import fitsio
import joblib
import numpy as np

from coadd_mdetsims.shear_bias_meas import estimate_m_and_c
from coadd_mdetsims.config import load_config
from coadd_mdetsims.defaults import S2N_CUTS


def _func(fname):
    try:
        pres = {}
        mres = {}
        with fitsio.FITS(fname, 'r') as fits:
            for s2n in S2N_CUTS:
                ext = '%s%d' % ('p', s2n)
                pres[s2n] = fits[ext].read()

                ext = '%s%d' % ('m', s2n)
                mres[s2n] = fits[ext].read()
        return (pres, mres)
    except Exception:
        return None


def main():
    tmpdir = 'condor_outputs'
    _, _, _, swap12, _ = load_config('config.yaml')

    files = glob.glob('%s/data*.fits' % tmpdir)
    print('found %d outputs' % len(files))
    io = [joblib.delayed(_func)(fname) for fname in files]
    outputs = joblib.Parallel(
        verbose=10,
        n_jobs=-1,
        pre_dispatch='2*n_jobs',
        max_nbytes=None)(io)

    outputs = [o for o in outputs if o is not None]
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
