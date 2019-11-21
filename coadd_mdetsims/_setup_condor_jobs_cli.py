import os
from coadd_mdetsims.config import load_config

PREAMBLE = """\
#
# always keep these at the defaults
#
Universe        = vanilla
kill_sig        = SIGINT
+Experiment     = "astro"
# copy env. variables to the job
GetEnv          = True

#
# options below you can change safely
#

# don't send email
Notification    = Never

# Run this executable.  executable bits must be set
Executable      = {script_name}

# A guess at the memory usage, including virtual memory
Image_Size       =  1000000

# this restricts the jobs to use the the shared pool
# Do this if your job will exceed 2 hours
#requirements = (cpu_experiment == "sdcc")

# each time the Queue command is called, it makes a new job
# and sends the last specified arguments. job_name will show
# up if you use the condortop job viewer

"""


def main():
    _, run_config, _, _, _ = load_config('config.yaml')

    N_PATCHES_PER_JOB = run_config['n_patches_per_job']
    N_PATCHES = run_config['n_patches']
    N_JOBS_PER_SCRIPT = run_config['n_jobs_per_script']

    def _append_job(fp, num, output_dir):
        fp.write("""\
    +job_name = "sim-{num:05d}"
    Arguments = {n_patches} {num} {output_dir}
    Queue

    """.format(n_patches=N_PATCHES_PER_JOB, num=num, output_dir=output_dir))

    n_jobs = N_PATCHES // N_PATCHES_PER_JOB
    n_scripts = n_jobs // N_JOBS_PER_SCRIPT

    cwd = os.path.abspath(os.getcwd())
    try:
        os.makedirs(os.path.join(cwd, 'condor_outputs'))
    except Exception:
        pass

    try:
        os.makedirs(os.path.join(cwd, 'condor_outputs', 'logs'))
    except Exception:
        pass

    try:
        os.makedirs(os.path.join(cwd, 'condor_scripts'))
    except Exception:
        pass

    script_name = os.path.join(cwd, "condor_scripts/job_condor.sh")
    output_dir = os.path.join(cwd, "condor_outputs")

    with open(script_name, 'w') as fp:
        fp.write("""\
    #!/bin/bash

    export OMP_NUM_THREADS=1

    if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
        # the condor system creates a scratch directory for us,
        # and cleans up afterward
        tmpdir=$_CONDOR_SCRATCH_DIR
        export TMPDIR=$tmpdir
    else
        # otherwise use the TMPDIR
        tmpdir='.'
        mkdir -p $tmpdir
    fi

    source activate bnl

    echo `which python`

    # change to correct dir just to be sure
    cd $3/..

    # about 1 to 1.6 hours per job
    # args are nsims, seed, odir
    coadd-mdetsims-run-sims \\
      --seed=$2 \\
      --output-file=${tmpdir}/data_${2}.fits \\
      --serial \\
      $1 >& ${tmpdir}/log_${2}.oe

    mv ${tmpdir}/log_${2}.oe $3/logs/.
    mv ${tmpdir}/data_${2}.fits $3/.

    """)

    os.system('chmod u+x %s' % script_name)

    script = PREAMBLE.format(script_name=script_name)

    job_ind = 1
    for snum in range(n_scripts):
        with open('condor_scripts/condor_job_%05d.desc' % snum, 'w') as fp:
            fp.write(script)
            for num in range(job_ind, job_ind + N_JOBS_PER_SCRIPT):
                _append_job(fp, num, output_dir)

        job_ind += N_JOBS_PER_SCRIPT
