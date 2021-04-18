"""
Script to test the performance of CapsNet on MNIST.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from caps_net import CapsNet
import config
from torchvision import datasets, transforms

sys.path.insert(0, '../src')


def load_mnist(transform):
    data_train = datasets.MNIST(root="./data/mnist/",
                                transform=transform,
                                train=True,
                                download=True)

    data_test = datasets.MNIST(root="./data/mnist/",
                               transform=transform,
                               train=False)
    return data_train, data_test


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        optimizer.zero_grad()
        output = model(data)
        output = model.get_probabilities(output).to(config.DEVICE)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = model(data).to(config.DEVICE)
            output = model.get_probabilities(output).to(config.DEVICE)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    transform = transforms.Compose([transforms.Resize(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_train, data_test = load_mnist(transform)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=64,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=64,
                                                   shuffle=True)

    model = CapsNet(img_size=28,
                    img_channels=1,
                    conv_out_channels=256,
                    out_channels=16,
                    num_classes=10,
                    conv_kernel_size=9).to(config.DEVICE)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()  # sum up batch loss

    train(model, data_loader_train, optimizer, criterion, epoch=20)
    test(model, data_loader_test, criterion)
    return None


if __name__ == "__main__":
    main()
