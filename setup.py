from setuptools import setup, find_packages

setup(
    name="coadd_mdetsims",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'coadd-mdetsims-run-sims=coadd_mdetsims._run_sims_cli:main',
            'coadd-mdetsims-setup-condor-jobs=coadd_mdetsims._setup_condor_jobs_cli:main',
            ],
    }
)
