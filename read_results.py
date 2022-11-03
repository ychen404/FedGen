import h5py
import glob, os
import numpy as np

from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble

dir = 'results'
files = []
for file in glob.glob(os.path.join(dir, '*.h5')):
    files.append(file)

print(files)


for file in files:
    with h5py.File(file, 'r') as f:
        data = f.get('glob_acc')[-1]
        print(file.split('/')[1], data)
        
print(20 * '#')
print('top acc')
print(20 * '#')

fedavg_res = []
FedEnsemble_res = []
FedGen_res = []
FedDistill_res = []
FedProx_res = []

for file in files:
    with h5py.File(file, 'r') as f:
        data = f.get('glob_acc')
        print(file.split('/')[1].split('_')[1], ',', np.max(data))

        if 'fedavg' in file.lower():        
            fedavg_res.append(np.max(data))

