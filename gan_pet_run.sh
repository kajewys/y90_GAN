#!/bin/bash -l

# If you need any help, please email farm-hpc@ucdavis.edu

#SBATCH --job-name=gan_pet

# Run this to see what partitions you have access to:
# sacctmgr -s list user $USER format=partition

#SBATCH --partition=qigpu
#SBATCH --mail-user=wysoczynski@ucdavis.edu
#SBATCH --mail-type=END

#SBATCH --output=gan_pet_%J.out
#SBATCH --error=gan_pet_error_%J.out

# Request 4 CPUs and 16 GB of RAM from 1 node:
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00              #HH:MM:SS
#SBATCH --export=ALL

# The useful part of your job goes below
# run one thread for each one the user asks the queue for

hostname
echo "starting at `date` on `hostname`"
set -u
which python3
python3 -V
echo $PATH
conda env list
python3 ~/gan_pet/gan_pet_mini/gan_pet.py

exit 0