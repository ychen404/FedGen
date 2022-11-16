import h5py
import glob, os
import numpy as np
import sys
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
import pdb

# print(sys.argv[1])
# dir = 'results/EMnist_alpha0.05_ratio0.1'

dir = 'results/Mnist-alpha0.1-ratio0.5'
# dir = sys.argv[1]
files = []
for file in glob.glob(os.path.join(dir, '*.h5')):
    files.append(file)

print(files)
        
print(40 * '#')
print('mean top acc')
print(40 * '#')

fedavg_res = []
FedEnsemble_res = []
FedGen_res = []
FedDistill_res = []
FedProx_res = []
FedOurs_res = []

for file in files:
    with h5py.File(file, 'r') as f:
        data = f.get('glob_acc')
        # print(file.split('/')[2].split('_')[1], ',', np.max(data))
        name = file.split('/')[2].split('_')[1]
        if 'fedavg' in name.lower():        
            fedavg_res.append(np.max(data))
        if 'fedensemble' in name.lower():        
            FedEnsemble_res.append(np.max(data))
        if 'fedgen' in name.lower():        
            FedGen_res.append(np.max(data))
        if 'feddistill' in name.lower():        
            FedDistill_res.append(np.max(data))
        if 'fedprox' in name.lower():        
            FedProx_res.append(np.max(data))
        if 'fedours' in name.lower():        
            FedOurs_res.append(np.max(data))

print(f"method, mean, std")
print(f"FedAvg, {100 * np.mean(np.array(fedavg_res))}, {100 * np.std(np.array(fedavg_res))}")
print(f"FedProx, {100 * np.mean(np.array(FedProx_res))}, {100 * np.std(np.array(FedProx_res))}")
print(f"FedEnsemble, {100 * np.mean(np.array(FedEnsemble_res))}, {100 * np.std(np.array(FedEnsemble_res))}")
print(f"FedDistill, {100 * np.mean(np.array(FedDistill_res))}, {100 * np.std(np.array(FedDistill_res))}")
print(f"FedGen, {100 * np.mean(np.array(FedGen_res))}, {100 * np.std(np.array(FedGen_res))}")
print(f"FedOurs, {100 * np.mean(np.array(FedOurs_res))}, {100 * np.std(np.array(FedOurs_res))}")

    