from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
import pdb
import sys

sys.path.append('/home/users/yitao/Code/FedGen/utils')
sys.path.append('/home/users/yitao/Code/FedGen/FLAlgorithms')
print(sys.path)
# import model_utils
from model_utils import create_model




model = create_model('cnn', 'emnist' 'fedavg')

pdb.set_trace()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)

