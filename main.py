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

class SequenceTransform:
    def __call__(self, x):
        return x.float() / 255.0

def main():
    epochs = 10

    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    transform = transforms.Compose([
        SequenceTransform()
    ])

    train_set = MovingMNIST(root='.data/mnist', train=True, transform=transform, target_transform=transform, download=True)
    test_set = MovingMNIST(root='.data/mnist', train=False, transform=transform, target_transform=transform, download=True)

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

    hidden_channels = [32, 16, 16]
    kernel_size = 5
    predict_steps = 10
    encoder = ConvLSTM(input_channels=1, hidden_channels=hidden_channels, kernel_size=kernel_size, step=10, effective_step=[]).cuda()
    forecaster = ConvLSTM(input_channels=1, hidden_channels=hidden_channels, kernel_size=kernel_size, step=predict_steps, effective_step=list(range(predict_steps))).cuda()
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    param = list(encoder.parameters()) + list(forecaster.parameters())
    # optimizer = optim.Adadelta(param)
    optimizer = optim.RMSprop(param, lr=1e-3, alpha=0.9)

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        for batch_idx, (input, target) in enumerate(train_loader):
            input = input.transpose(0, 1).reshape(10, -1, 1, 64, 64).cuda()
            optimizer.zero_grad()
            internal_state = [] # without this internal_state is reused for some reason
            _, internal_state = encoder(input, internal_state)
            # zero input for forecaster because it is unconditioned
            _, b, c, h, w = input.size()
            zero_input = Variable(torch.zeros(predict_steps, b, c, h, w), requires_grad=False).cuda()
            output, _ = forecaster(zero_input, internal_state)
            loss = 0
            for i, out in enumerate(output):
                tar = target[:, i, :, :].reshape(-1, 1, 64, 64).cuda()
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
