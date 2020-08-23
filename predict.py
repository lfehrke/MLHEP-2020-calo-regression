import numpy as np

test_file = "data_test.npz"
data_real = np.load(test_file, allow_pickle=True)

# This is the calorimeter response:
energy = data_real['EnergyDeposit']

# These are the quantities we want to predict
# momentum = data_real['ParticleMomentum'][:,:2]
# coordinate = data_real['ParticlePoint'][:,:2]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim as optim

energy = data_real['EnergyDeposit']
X = energy[:,None,...] # adding Channels dimension
# Y = np.concatenate([coordinate, momentum], axis=1)

# valIndices = np.load("valIndices.npy")
# X = X[valIndices]
# Y = Y[valIndices]

mean = [14.53326275, 14.53554257]
std = [4.57695691, 4.69248437]

def getScaledFilm(X, mean, std):
    energy_density = X.reshape(-1, 30, 30) / X.reshape(-1, 30, 30).sum(axis=(1, 2), keepdims=True)
    cell_coords = np.stack([*np.meshgrid(
        np.arange(energy.shape[1]),
        np.arange(energy.shape[2])
    )], axis=-1)[None,...]

    center_of_mass = (energy_density[...,None] * cell_coords).sum(axis=(1, 2))

    return (center_of_mass - mean) / std


def make_torch_dataset(X, F, batch_size, shuffle=True):
    X = torch.tensor(X).float()
    F = torch.tensor(F).float()
    ds = utils.TensorDataset(X, F)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        pin_memory=True, shuffle=shuffle
    )

import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inChannels,
                               out_channels=outChannels,
                               kernel_size=kernelSize,
                               padding=padding,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=outChannels,
                               out_channels=outChannels,
                               kernel_size=kernelSize,
                               padding=padding,
                               stride=1)

        self.bn1 = nn.BatchNorm2d(outChannels)
        self.bn2 = nn.BatchNorm2d(outChannels)

        self.stride = stride
        if inChannels != outChannels:
            self.convSkip1 = nn.Conv2d(in_channels=inChannels,
                                      out_channels=outChannels,
                                      kernel_size=1,
                                      stride=1)
        else:
            self.convSkip1 = None


    def forward(self, x):
        skip = x
        out = self.bn1(swish(self.conv1(x)))
        out = self.bn2(swish(self.conv2(out)))

        if self.convSkip1:
            skip = self.convSkip1(skip)
        if self.stride > 1:
            skip = nn.AvgPool2d((self.stride, self.stride), padding= 1 if x.shape[2] % 2 == 1 else 0)(skip)

        out += skip

        return out


class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas


class FilmRegressor(nn.Module):
    def __init__(self):
        super(FilmRegressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.makeLayer(3, 32, 32, 1)
        self.layer2 = self.makeLayer(3, 32, 64, 2)
        self.layer3 = self.makeLayer(3, 64, 128, 2)
        #self.layer4 = self.makeLayer(7, 128, 256, 2)

        self.pool4 = nn.AvgPool2d((2,2))

        # self.fc1 = nn.Linear(4*4*64, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.fc2 = nn.Linear(256, 256)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.fc3 = nn.Linear(256, 1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, padding=0, stride=1)
        self.pool5 = nn.AvgPool2d((4, 4))


        self.film1 = FiLM()
        self.film2 = FiLM()
        self.film3 = FiLM()
        self.film4 = FiLM()
        self.film5 = FiLM()

        self.filmfc1 = nn.Linear(2, 256)
        self.filmfc2 = nn.Linear(256, 2*32 + 2*32 + 2*64 + 2*128) # + 2*256)
        self.filmbn1 = nn.BatchNorm1d(256)

    def makeLayer(self, nLayers, inChannels, outChannels, stride):
        return nn.Sequential(
            ConvBlock(inChannels, outChannels, stride=stride), *[ConvBlock(outChannels, outChannels, stride=1) for _ in range(nLayers-1)]
        )

    def forward(self, x, f):
#        film = swish(self.filmbn1(self.filmfc1(f)))
#        film = self.filmfc2(film)

        x = swish(self.bn1(self.conv1(x)))
#        x = self.film1(x, film[:, :32], film[:, 32:64])
        x = self.layer1(x)
#        x = self.film2(x, film[:, 64:96], film[:, 96:128])
        x = self.layer2(x)
#        x = self.film3(x, film[:, 128:192], film[:, 192:256])
        x = self.layer3(x)
#        x = self.film4(x, film[:, 256:384], film[:, 384:512])
        #x = self.layer4(x)
        #x = self.film5(x, film[:, 512:768], film[:, 768:1024])

        x = self.pool4(x)
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        return self.pool5(self.conv4(x)).reshape(len(x), -1)
        # x = self.pool4(x).reshape(len(x), -1)
        # x = swish(self.bn2(self.fc1(x)))
        # return self.fc3(x)

class FilmRegressor2(nn.Module):
    def __init__(self):
        super(FilmRegressor2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.makeLayer(5, 32, 32, 1)
        self.layer2 = self.makeLayer(5, 32, 64, 2)
        self.layer3 = self.makeLayer(5, 64, 128, 2)

        self.pool4 = nn.AvgPool2d((2,2))

        self.fc1 = nn.Linear(64 + 4, 256)
        self.bn21 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn31 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding=0, stride=1)
        self.pool5 = nn.AvgPool2d((4, 4))


        self.film1 = FiLM()
        self.film2 = FiLM()
        self.film3 = FiLM()
        self.film4 = FiLM()
        self.film5 = FiLM()

        self.filmfc1 = nn.Linear(4, 256)
        self.filmfc2 = nn.Linear(256, (2*32 + 2*32 + 2*64 + 2*128))
        self.filmbn1 = nn.BatchNorm1d(256)

    def makeLayer(self, nLayers, inChannels, outChannels, stride):
        return nn.Sequential(
            ConvBlock(inChannels, outChannels, stride=stride), *[ConvBlock(outChannels, outChannels, stride=1) for _ in range(nLayers-1)]
        )

    def forward(self, x, f):
        film = swish(self.filmbn1(self.filmfc1(f)))
        film = self.filmfc2(film)

        x = swish(self.bn1(self.conv1(x)))
        x = self.film1(x, film[:, :32], film[:, 32:64])
        x = self.layer1(x)
        x = self.film2(x, film[:, 64:96], film[:, 96:128])
        x = self.layer2(x)
        x = self.film3(x, film[:, 128:192], film[:, 192:256])
        x = self.layer3(x)
        x = self.film4(x, film[:, 256:384], film[:, 384:512])


        x = self.pool4(x)

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = self.pool5(x)

        x = swish(self.bn21(self.fc1(torch.cat([x.reshape(len(x), -1), f], dim=-1))))
        x = swish(self.bn31(self.fc2(x)))
        return self.fc3(x)



device = torch.device('cuda')
regressor = FilmRegressor().to(device)
regressor.load_state_dict(torch.load("firstModel.pt"))
regressor.eval()
regressor2 = FilmRegressor2().to(device)
regressor2.load_state_dict(torch.load("secondModel.pt"))
regressor2.eval()

from PIL import Image, ImageOps
import copy
import math

def rotateInv(labels, angles=180):
    rot_labels = torch.zeros_like(labels)
    rot_labels[:, 0] = math.cos(math.radians(-angles)) * labels[:, 0] + math.sin(math.radians(-angles)) * labels[:, 1]
    rot_labels[:, 1] = math.cos(math.radians(-angles)) * labels[:, 1] - math.sin(math.radians(-angles)) * labels[:, 0]
    rot_labels[:, 2] = math.cos(math.radians(-angles)) * labels[:, 2] + math.sin(math.radians(-angles)) * labels[:, 3]
    rot_labels[:, 3] = math.cos(math.radians(-angles)) * labels[:, 3] - math.sin(math.radians(-angles)) * labels[:, 2]
    return rot_labels

def flipInv(labels):
    rot_labels = copy.copy(labels)
    rot_labels[:, 1] = -rot_labels[:, 1]
    rot_labels[:, 3] = -rot_labels[:, 3]

    return rot_labels

def mirrorInv(labels):
    rot_labels = copy.copy(labels)
    rot_labels[:, 0] = -rot_labels[:, 0]
    rot_labels[:, 2] = -rot_labels[:, 2]

    return rot_labels

def predict(ds, regressor, regressor2):
    preds = []
    for X_batch, F_batch in ds:
        X_batch, F_batch =  X_batch.to(device), F_batch.to(device)
        singlePreds = regressor(X_batch, F_batch).detach()
        singlePreds[:, :2] = regressor2(X_batch, torch.cat((F_batch, singlePreds[:, 2:]), dim=-1)).detach().cpu()
        preds.append(singlePreds)

    preds = torch.cat(preds, dim=0)
    return preds


def rotate(imgs, angles=180):
    rot_imgs = np.stack([
        np.array(
            Image.fromarray(img[0, :, :]).rotate(angles, Image.BILINEAR)
        )
        for img in imgs
    ], axis=0).reshape(-1, 1, 30, 30)

    return rot_imgs


def flip(imgs):
    rot_imgs = np.stack([
        np.array(
            ImageOps.flip(Image.fromarray(img[0, :, :]))
        )
        for img in imgs
    ], axis=0).reshape(-1, 1, 30, 30)

    return rot_imgs


def mirror(imgs):
    rot_imgs = np.stack([
        np.array(
            ImageOps.mirror(Image.fromarray(img[0, :, :]))
        )
        for img in imgs
    ], axis=0).reshape(-1, 1, 30, 30)

    return rot_imgs


BATCH_SIZE = 128
preds = torch.zeros((10, X.shape[0], 4))

ds = make_torch_dataset(np.log1p(X), getScaledFilm(X, mean, std), BATCH_SIZE, shuffle=False)
preds[0] = predict(ds, regressor, regressor2)

X_rot = flip(X)
ds = make_torch_dataset(np.log1p(X_rot), getScaledFilm(X_rot, mean, std), BATCH_SIZE, shuffle=False)
preds[1] = flipInv(predict(ds, regressor, regressor2))

X_rot = mirror(X)
ds = make_torch_dataset(np.log1p(X_rot), getScaledFilm(X_rot, mean, std), BATCH_SIZE, shuffle=False)
preds[2] = mirrorInv(predict(ds, regressor, regressor2))

for i, angle in enumerate(np.linspace(45, 315, 7)):
    X_rot = rotate(X, angle)
    ds = make_torch_dataset(np.log1p(X_rot), getScaledFilm(X_rot, mean, std), BATCH_SIZE, shuffle=False)
    preds[3 + i] = rotateInv(predict(ds, regressor, regressor2), angle)

preds = torch.mean(preds, dim=0)

coordinate_test, momentum_test = (
    preds.detach().numpy()[:, :2],
    preds.detach().numpy()[:, 2:],
)

prediction_file = "data_test_prediction.npz"
np.savez_compressed(prediction_file,
                    ParticlePoint=coordinate_test,
                    ParticleMomentum=momentum_test)
