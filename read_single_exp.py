import h5py
import glob, os
import numpy as np
import sys
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
import pdb


dir = 'results/EMnist-alpha100.0-ratio0.1'
# dir = 'results/ours'

# dir = sys.argv[1]
files = ["results/EMnist-alpha0.5-ratio0.1_FedDFGen_0.01_10u_32b_20_0.h5", 
        "results/EMnist-alpha0.5-ratio0.1_FedDFGen_0.01_10u_32b_20_1.h5",
        "results/EMnist-alpha0.5-ratio0.1_FedDFGen_0.01_10u_32b_20_2.h5" ]

# for file in glob.glob(os.path.join(dir, '*.h5')):
#     files.append(file)

print(files)
        
print(40 * '#')
print('mean top acc')
print(40 * '#')


res = []

for file in files:
    with h5py.File(file, 'r') as f:
        data = f.get('glob_acc')
        # print(file.split('/')[2].split('_')[1], ',', np.max(data))
        # name = file.split('/')[2].split('_')[1]
        res.append(np.max(data))
        
print(f"method, mean, std")
print(f"---, {100 * np.mean(np.array(res))}, {100 * np.std(np.array(res))}")
