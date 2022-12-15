from utils.model_utils import create_model, read_data, read_user_data
import pdb
import torch
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = nn.Linear(128, 47)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def my_read_data(path):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    # train_data_dir, test_data_dir, proxy_data_dir, public_data_dir = get_data_dir(args.dataset, args.num_users)
    train_data_dir = path + '/train/'
    test_data_dir = path + '/test/'
    public_data_dir = path + '/public/'

    # train_data_dir, test_data_dir, proxy_data_dir, public_data_dir = get_data_dir(dataset)

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}
    public_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                # The cdata contains the following keys: ['users', 'user_data', 'num_samples']
                # 'users' contains all the user names from 'f_00000' to f_00019'
                # 'user_data' contains all the data categorized by user names
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)

        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])

    ######## public data ########
    if public_data_dir and os.path.exists(public_data_dir):
        public_files = os.listdir(public_data_dir)
        public_files = [f for f in public_files if f.endswith('.json') or f.endswith(".pt")]
        for f in public_files:
            file_path = os.path.join(public_data_dir, f)
            if file_path.endswith(".pt"):
                with open(file_path, 'rb') as inf:
                    cdata = torch.load(inf)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
            else:
                raise TypeError("Data format not recognized: {}".format(file_path))
            public_data.update(cdata['data'])

    proxy_data_dir = 'data/proxy_data/emnist-n10/'

    if proxy_data_dir and os.path.exists(proxy_data_dir):
        proxy_files=os.listdir(proxy_data_dir)
        proxy_files=[f for f in proxy_files if f.endswith('.json') or f.endswith(".pt")]
        for f in proxy_files:
            file_path=os.path.join(proxy_data_dir, f)
            if file_path.endswith(".pt"):
                with open(file_path, 'rb') as inf:
                    cdata=torch.load(inf)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as inf:
                    cdata=json.load(inf)
            else:
                raise TypeError("Data format not recognized: {}".format(file_path))
            proxy_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data, proxy_data, public_data


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--net', default='res18')
parser.add_argument('--bs', default='32')
parser.add_argument('--dataset', default='emnist')
args = parser.parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
testset = EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)


# path = '/home/users/yitao/Code/FedGen/data/EMnist/u2-letters-alpha0.5-ratio0.9/train/train.pt'
path = '/home/users/yitao/Code/FedGen/data/EMnist/u2-letters-alpha0.5-ratio0.9'
dataset = 'EMnist-alpha0.5-ratio0.9'

data = my_read_data(path)
total_users = len(data[0])
print("Users in total: {}".format(total_users))

trainloaders = []
testloaders = []
total_train_samples = 0
classes = []

for i in range(total_users):
    id, train_data , test_data = read_user_data(i, data, dataset=dataset)
    total_train_samples += len(train_data)
    # for data in train_data:
    #     classes.append(data[1])
    # pdb.set_trace()
    trainloaders.append(DataLoader(train_data, int(args.bs), shuffle=True, drop_last=True))
    testloaders.append(DataLoader(test_data, int(args.bs), drop_last=False))

print(f"Total train samples length: {total_train_samples}")
# pdb.set_trace()
# data = torch.load(path)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.bs), shuffle=True, num_workers=4)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# l = trainset.classes
# l.sort()
# print(f"number of classes: {len(l)}")
# print(l)

# print(f"train len:{len(trainloader)}, test len: {len(testloader)}")

# the model is having problem??
net, _ = create_model('cnn', 'emnist', 'FedAvg')
# net = Net()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
net = net.to(device)

def get_next_train_batch(trainloader, count_labels=True):

    iter_trainloader = iter(trainloader)

    try:
        # Samples a new batch for personalizing
        (X, y) = next(iter_trainloader)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        iter_trainloader = iter(trainloader)
        (X, y) = next(iter_trainloader)
    result = {'X': X, 'y': y}
    if count_labels:
        unique_y, counts=torch.unique(y, return_counts=True)
        unique_y = unique_y.detach().numpy()
        counts = counts.detach().numpy()
        result['labels'] = unique_y
        result['counts'] = counts
    return result

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloaders[0]):
    
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)['output']
        # pdb.set_trace()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 20 == 0:
            print(f"batch index: {batch_idx}, Loss: {train_loss/(batch_idx+1)}, Acc: {100.*correct/total}")
    
    result = get_next_train_batch(trainloaders[0])
    X, y = result['X'], result['y']
    X, y = X.to(device), y.to(device)
    # pdb.set_trace()
    optimizer.zero_grad()
    outputs = net(X)['output']
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += y.shape[0]
    correct += predicted.eq(y).sum().item()

    print(f"Total: {total}, Loss: {train_loss/(total)}, Acc: {100.*correct/total}")
    # return train_loss
    # return train_loss/(batch_idx+1)
    return train_loss/(total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloaders[0]):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

list_loss = []

for epoch in range(start_epoch, start_epoch+50):
    trainloss = train(epoch)
    # test(epoch)
    
    list_loss.append(trainloss)
    # print(list_loss)
# print(list_loss)