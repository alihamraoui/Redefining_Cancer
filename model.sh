#!/bin/bash

################################ Slurm options #################################

### Job name
#SBATCH --job-name=deep

### Output
#SBATCH --output=agc-%j.out  # both STDOUT and STDERR
##SBATCH -o slurm.%N.%j.out  # STDOUT file with the Node name and the Job ID
##SBATCH -e slurm.%N.%j.err  # STDERR file with the Node name and the Job ID

### Limit run time "days-hours:minutes:seconds"
#SBATCH --time=24:00:00

### Requirements
#SBATCH --partition=fast
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=2GB

### Email
##SBATCH --mail-user=email@address
##SBATCH --mail-type=ALL

################################################################################

echo '########################################'
echo 'Date:' $(date --iso-8601=seconds)
echo 'User:' $USER
echo 'Host:' $HOSTNAME
echo 'Job Name:' $SLURM_JOB_NAME
echo 'Job Id:' $SLURM_JOB_ID
echo 'Directory:' $(pwd)
echo '########################################'
echo 'agc: v0.1'
echo '-------------------------'
echo 'Main module versions:'


start0=`date +%s`

# modules loading
conda --version
python --version

echo '-------------------------'
echo 'PATH:'
echo $PATH
echo '-------------------------'

# remove display to make qualimap run:

# What you actually want to launch
python model.py
# move logs
mkdir -p slurm_output
mv *.out slurm_output

echo '########################################'
echo 'Job finished' $(date --iso-8601=seconds)
end=`date +%s`
runtime=$((end-start0))
minute=60
echo "---- Total runtime $runtime s ; $((runtime/minute)) min ----"