#!/bin/bash
#SBATCH --job-name=CILRS            
#SBATCH --ntasks=5
#SBATCH –-mem-per-cpu=10000    
#SBATCH --ntasks-per-node=1    
#SBATCH --partition=long       
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=7:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sherz22@ku.edu.tr    
nvidia-smi
module load python/3.9.5
module load cuda/11.4
module load cudnn/8.2.2/cuda-11.4
python cilrs_train.py