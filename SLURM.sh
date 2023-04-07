#!/bin/bash
#SBATCH --job-name=CILRS            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --partition=ai       
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=7:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sherz22@ku.edu.tr    
module load python/3.6.1
moddule load cuda/11.4
module load 8.2.2/cuda-11.4 
activate () {
  . ~/CVADHW1/venv/Scripts/activate
}   
python cilrs_train.py