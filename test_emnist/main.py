from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
import pdb

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)

pdb.set_trace()