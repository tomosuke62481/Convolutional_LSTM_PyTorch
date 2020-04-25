import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from convolution_lstm import ConvLSTM
from MovingMNIST import MovingMNIST

def save_image(img, path, clip_by_min_max=True):
    np_img = np.array(img.cpu())
    if clip_by_min_max:
        np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min())
    else:
        np_img = np.clip(np_img, 0, 1)
    np_img = np.uint8(np_img * 255.0)
    pil_img = Image.fromarray(np_img)
    pil_img.save(path)
    return

def main():
    epochs = 10

    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = MovingMNIST(root='.data/mnist', train=True, transform=transform, download=True)
    test_set = MovingMNIST(root='.data/mnist', train=False, transform=transform, download=True)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))


    convlstm = ConvLSTM(input_channels=1, hidden_channels=[32, 16, 16], kernel_size=5, step=11,
                        effective_step=[9, 10]).cuda()
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adadelta(convlstm.parameters())
    # optimizer = optim.RMSprop(convlstm.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        for batch_idx, (input, target) in enumerate(train_loader):
            input = input.float() / 255.0
            input = input.transpose(0, 1).reshape(10, -1, 1, 64, 64)
            input = input.cuda()
            optimizer.zero_grad()
            output = convlstm(input)
            loss = 0
            target = target.float() / 255.0
            for i, out in enumerate(output):
                tar = target[:, i, :, :]
                tar = tar.reshape(-1, 1, 64, 64)
                tar = tar.cuda()
                loss += loss_fn(out, tar)
                if batch_idx % 10 == 0:
                    target_path = 'target{:02d}_{:04d}_{:02d}.png'.format(epoch, batch_idx, i)
                    save_image(tar[0].reshape(64, 64), target_path)
                    pred_path = 'pred{:02d}_{:04d}_{:02d}.png'.format(epoch, batch_idx, i)
                    save_image(out[0].detach().reshape(64, 64), pred_path)
            loss.backward()
            optimizer.step()
            print('Loss: {}'.format(loss.item()))
    return

if __name__ == '__main__':
    main()
