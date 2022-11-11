from utils.model_utils import create_model
import pdb
import torch


model = create_model('cnn', 'mnist', 'FedAvg')
input = torch.rand(1, 1, 28, 28)
pdb.set_trace()
output = model[0](input)
print(output)