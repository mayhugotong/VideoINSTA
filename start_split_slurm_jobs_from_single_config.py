import os
import sys
import ast

from dotenv import load_dotenv


def main(config_file_path, job_name, intervals, num_cpus, num_gpus, nodelist, exclude):
    load_dotenv()

    # define the list of end indices
    splits = ast.literal_eval(intervals)

    # Loop over each end index in the list
    for split in splits:
        start, end = split
        job_name_with_index = f"{job_name}_{start}_{end}"

        # Create a unique SLURM script for each end index
        # TODO adapt the script to your needs (depending on your clusters SLURM configuration)
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name_with_index}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH -o slurm/{job_name_with_index}.out
#SBATCH -e slurm/{job_name_with_index}.err
#SBATCH --time=10-0
{f"#SBATCH --exclude={exclude}" if exclude else ""}
{f"#SBATCH --nodelist={nodelist}" if nodelist else ""}

echo "Executing slurm job {job_name_with_index} for exp {config_file_path}"

# venv
eval "$(conda shell.bash hook)"
conda activate video-reasoning-got

# make sure the CUDA_HOME is set correctly
export CUDA_HOME=/usr
# check CUDA_HOME
which nvcc
echo $CUDA_HOME

# run the experiment
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --conf {config_file_path} --start_video_index {start} --end_video_index {end}
"""
        script_path = f"temp_slurm_script_{start}_{end}.sh"
        with open(script_path, "w") as script_file:
            script_file.write(script_content)

        # Submit the SLURM script
        os.system(f"sbatch {script_path}")

        # Remove the SLURM script after execution
        os.remove(script_path)


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 8:
        print("Usage: python script.py <config_base_path> <job_name> <intervals> <num_cpus=8> <num_gpus=1> <nodelist=''> <exclude=''>")
        sys.exit(1)

    try:
        cpus = int(sys.argv[4])
    except IndexError:
        cpus = 8

    try:
        gpus = int(sys.argv[5])
    except IndexError:
        gpus = 1

    try:
        nodes = sys.argv[6]
    except IndexError:
        nodes = ""

    try:
        excludes = sys.argv[7]
    except IndexError:
        excludes = ""

    main(sys.argv[1], sys.argv[2], sys.argv[3], cpus, gpus, nodes, excludes)
