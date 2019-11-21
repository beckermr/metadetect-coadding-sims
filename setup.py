from setuptools import setup, find_packages

setup(
    name="coadd_mdetsims",
    version="0.1",
    packages=find_packages(),
    scripts=[
        'scripts/coadd-mdetsims-run-sims',
        'scripts/coadd-mdetsims-setup-condor-jobs']
)
