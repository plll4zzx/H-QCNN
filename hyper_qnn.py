#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = {
    'name': 'Zhaoxi Zhang',
    'Email': 'zhaoxi_zhang@163.com',
    'QQ': '809536596',
    'Created': ''
}

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pennylane as qml
from pennylane import numpy as np
import zzxDataset

num_class=[0,1]
qbit=5
dev = qml.device("default.qubit", wires=qbit)
device = torch.device("cpu")

@qml.qnode(dev, interface='torch')
def circuit(inputs, out_dim=2, qnum=5):
    for idx in range(qnum):
        qml.PauliX(wires=idx)

    for idx in range(qnum):
        qml.Hadamard(wires=idx)

    state=0
    for input in inputs:
        qml.RY(input, wires=state % qnum)
        qml.CNOT(wires=[state% qnum, (state+out_dim) % qnum])
        state+=1

    expval=[]
    for idx in range(out_dim):
        expval.append(qml.expval(qml.PauliZ(idx)))
    return tuple(expval)

class HQNN(nn.Module):
    def __init__(self):
        super(HQNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 2**qbit)

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
        x = torch.sigmoid(x)

        x=x*np.pi
        n_qubits=qbit
        q_out = torch.Tensor(0, len(num_class))
        q_out = q_out.to(device)
        for elem in x:
            q_out_elem = circuit(elem, out_dim=len(num_class), qnum=qbit).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        # x=q_out*0.5+0.5
        x=q_out*10
        # x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.long()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.long()
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # if pred.item()==target.item():
            #     correct+=1
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_data(batch_size=16, classes=[0,1]):
    kwargs = {
        'batch_size': batch_size,
        'shuffle':True
    }

    train_x, train_y, test_x, test_y=zzxDataset.get_data(num_class=classes)

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_dataset,**kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    return train_loader, test_loader


def main():

    torch.manual_seed(1)

    train_loader, test_loader=get_data(classes=num_class)

    model = HQNN().to(device)

    pretrained_model = "mnist_hqnn_"+str(len(num_class))+".pt"
    # model.load_state_dict(torch.load(pretrained_model))
    # test(model, device, test_loader)

    optimizer = optim.Adadelta(model.parameters(), lr=0.1)

    for epoch in range(1, 4 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model.state_dict(), "mnist_hqnn_"+str(len(num_class))+".pt")


if __name__ == '__main__':
    main()
