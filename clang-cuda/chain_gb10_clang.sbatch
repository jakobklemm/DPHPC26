#!/bin/bash
#SBATCH --job-name=gb10-clang-chain
#SBATCH --account=dphpc
#SBATCH --gpus=gb10:1
#SBATCH --output=/scratch/users/tlemagnan/clang-campaign/jobs/logs/%x-%j.out
# One app's clang campaign in ONE job (same node -> no between-job noise): sweep, then random.
# Usage: sbatch -t HH:MM:SS chain_gb10_clang.sbatch <llama|gromacs> <random-count>
# Resumable: resubmit the same command; finished configurations are skipped.
APP=$1; COUNT=$2
J=/scratch/users/tlemagnan/clang-campaign/jobs/campaign_gb10_clang.sbatch
bash $J $APP gb10-clang-sweep sweep; rc1=$?
bash $J $APP gb10-clang-random$COUNT random --count $COUNT --min-flags 1; rc2=$?
echo "chain done $(date): sweep exit $rc1, random exit $rc2"
exit $(( rc1 || rc2 ))
