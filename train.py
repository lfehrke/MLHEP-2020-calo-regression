import numpy as np

train_file = "data_train.npz"
data_real = np.load(train_file, allow_pickle=True)

# This is the calorimeter response:
energy = data_real['EnergyDeposit']

# These are the quantities we want to predict
momentum = data_real['ParticleMomentum'][:,:2]
coordinate = data_real['ParticlePoint'][:,:2]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim as optim

from tqdm import tqdm

trainIndices = np.load("trainIndices.npy")
valIndices = np.load("valIndices.npy")


energy = data_real['EnergyDeposit']
X = energy[:,None,...] # adding Channels dimension
Y = np.concatenate([coordinate, momentum], axis=1)

X_train = X[trainIndices]
X_val = X[valIndices]
Y_train = Y[trainIndices]
Y_val = Y[valIndices]

energy_density = X_train.reshape(-1, 30, 30) / X_train.reshape(-1, 30, 30).sum(axis=(1, 2), keepdims=True)
cell_coords = np.stack([*np.meshgrid(
    np.arange(energy.shape[1]),
    np.arange(energy.shape[2])
)], axis=-1)[None,...]

center_of_mass = (energy_density[...,None] * cell_coords).sum(axis=(1, 2))
mean = np.mean(center_of_mass, axis=0)
std = np.std(center_of_mass, axis=0)

# print(mean, std)
# exit()

def getScaledFilm(X, mean, std):
    energy_density = X.reshape(-1, 30, 30) / X.reshape(-1, 30, 30).sum(axis=(1, 2), keepdims=True)
    cell_coords = np.stack([*np.meshgrid(
        np.arange(energy.shape[1]),
        np.arange(energy.shape[2])
    )], axis=-1)[None,...]

    center_of_mass = (energy_density[...,None] * cell_coords).sum(axis=(1, 2))

    return (center_of_mass - mean) / std


def make_torch_dataset(X, Y, F, batch_size, shuffle=True):
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()
    F = torch.tensor(F).float()
    ds = utils.TensorDataset(X, Y, F)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        pin_memory=True, shuffle=shuffle
    )


def rotate(imgs, labels, angles=180):
    from PIL import Image, ImageOps
    import copy, math

    rot_imgs = np.stack([
        np.array(
            Image.fromarray(img[0, :, :]).rotate(angles, Image.BILINEAR)
        )
        for img in imgs
    ], axis=0).reshape(-1, 1, 30, 30)
    rot_labels = np.zeros(labels.shape)
    rot_labels[:, 0] = math.cos(math.radians(angles)) * labels[:, 0] + math.sin(math.radians(angles)) * labels[:, 1]
    rot_labels[:, 1] = math.cos(math.radians(angles)) * labels[:, 1] - math.sin(math.radians(angles)) * labels[:, 0]
    rot_labels[:, 2] = math.cos(math.radians(angles)) * labels[:, 2] + math.sin(math.radians(angles)) * labels[:, 3]
    rot_labels[:, 3] = math.cos(math.radians(angles)) * labels[:, 3] - math.sin(math.radians(angles)) * labels[:, 2]

    return rot_imgs, rot_labels

def flip(imgs, labels):
    from PIL import Image, ImageOps
    import copy, math

    rot_imgs = np.stack([
        np.array(
            ImageOps.flip(Image.fromarray(img[0, :, :]))
        )
        for img in imgs
    ], axis=0).reshape(-1, 1, 30, 30)
    rot_labels = copy.copy(labels)
    rot_labels[:, 1] = -rot_labels[:, 1]
    rot_labels[:, 3] = -rot_labels[:, 3]

    return rot_imgs, rot_labels

def mirror(imgs, labels):
    from PIL import Image, ImageOps
    import copy, math

    rot_imgs = np.stack([
        np.array(
            ImageOps.mirror(Image.fromarray(img[0, :, :]))
        )
        for img in imgs
    ], axis=0).reshape(-1, 1, 30, 30)
    rot_labels = copy.copy(labels)
    rot_labels[:, 0] = -rot_labels[:, 0]
    rot_labels[:, 2] = -rot_labels[:, 2]

    return rot_imgs, rot_labels

BATCH_SIZE = 256

x_conc = [X_train]
y_conc = [Y_train]
for angle in np.linspace(45, 315, 7): #np.linspace(10, 350, 35):
    print(angle)
    x_trans, y_trans = rotate(X_train, Y_train, angles=angle)
    x_conc.append(x_trans)
    y_conc.append(y_trans)

x_90, y_90 = flip(X_train, Y_train)
x_conc.append(x_90)
y_conc.append(y_90)
x_270, y_270 = mirror(X_train, Y_train)
x_conc.append(x_270)
y_conc.append(y_270)


X_train90, Y_train90 = rotate(X_train, Y_train, angles=90)
X_train180, Y_train180 = rotate(X_train, Y_train, angles=180)
X_train270, Y_train270 = rotate(X_train, Y_train, angles=270)

X_train = np.concatenate(x_conc, axis=0)
del x_conc
Y_train = np.concatenate(y_conc, axis=0)
del y_conc

ds_train = make_torch_dataset(np.log1p(X_train) , Y_train[:, :], getScaledFilm(X_train, mean, std), BATCH_SIZE)
ds_val = make_torch_dataset(np.log1p(X_val) , Y_val[:, :], getScaledFilm(X_val, mean, std), BATCH_SIZE, shuffle=False)

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
        #self.layer4 = self.makeLayer(3, 128, 256, 2)

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
        self.filmfc2 = nn.Linear(256, (2*32 + 2*32 + 2*64 + 2*128)) # + 2*256)
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
        #x = self.layer4(x)
        #x = self.film5(x, film[:, 512:768], film[:, 768:1024])

        x = self.pool4(x)

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = self.pool5(x)

        x = swish(self.bn21(self.fc1(torch.cat([x.reshape(len(x), -1), f], dim=-1))))
        x = swish(self.bn31(self.fc2(x)))
        return self.fc3(x)

        #x = swish(self.bn2(self.conv2(x)))
        #x = swish(self.bn3(self.conv3(x)))
        #return self.pool5(self.conv4(x)).reshape(len(x), -1)



class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.makeLayer(5, 32, 32, 1)
        self.layer2 = self.makeLayer(3, 32, 64, 2)
        self.layer3 = self.makeLayer(3, 64, 128, 2)

        self.pool4 = nn.AvgPool2d((2,2))

        self.fc1 = nn.Linear(4*4*128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)

    def makeLayer(self, nLayers, inChannels, outChannels, stride):
        return nn.Sequential(
            ConvBlock(inChannels, outChannels, stride=stride), *[ConvBlock(outChannels, outChannels, stride=1) for _ in range(nLayers-1)]
        )

    def forward(self, x):
        out = x
        out = self.bn1(swish(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool4(out).reshape(len(x), -1)
        out = self.bn2(swish(self.fc1(out)))
        return self.fc3(out)


device = torch.device('cuda')
regressor = FilmRegressor().to(device)
# from torchsummary import summary
# summary(regressor, X.shape[1:])

stepsPerEpoch = int(X_train.shape[0] / BATCH_SIZE)
learning_rate = 1e-3
opt = optim.Adam(regressor.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=5e-5, max_lr=1e-3, step_size_up=stepsPerEpoch * 10, cycle_momentum=False)

def metric_relative_mse(y_true, y_pred):
    return (
        (y_true - y_pred).pow(2).mean(dim=0) / y_true.pow(2).mean(dim=0)
    )

def metric_relative_mse_total(y_true, y_pred):
    return metric_relative_mse(y_true, y_pred).sum()

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self,yhat,y):
        return ((y - yhat).pow(2).mean(dim=0) / y.pow(2).mean(dim=0)).sum()


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.log(torch.cosh(ey_t + 1e-12)).mean(dim=0).sum()


# loss_fn = RMSELoss().to(device)
# loss_fn = torch.nn.L1Loss().to(device)
loss_fn = LogCoshLoss().to(device)


def run_training(epochs=5):
    losses_train = []
    losses_val = []
    metrics_train = []
    metrics_val = []
    per_component_metrics_train = []
    per_component_metrics_val = []
    best_score = 10

    for epoch in range(epochs):
        print(epoch, "/", epochs)
        avg_loss, avg_metrics, avg_per_component_metrics = [], [], []
        for batch_X, batch_Y, batch_F in ds_train:
            batch_X, batch_Y, batch_F = batch_X.to(device), batch_Y.to(device), batch_F.to(device)

            pred = regressor(batch_X, batch_F)

            l2_reg = None
            for W in regressor.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)

            loss = loss_fn(pred, batch_Y) + 1e-3 * l2_reg

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            avg_loss.append(loss.item())
            avg_metrics.append(
                metric_relative_mse_total(batch_Y, pred).item()
            )
            avg_per_component_metrics.append(
                metric_relative_mse(batch_Y, pred).detach().cpu().numpy()
            )

        losses_train.append(np.mean(avg_loss))
        metrics_train.append(
            np.mean(avg_metrics)
        )

        per_component_metrics_train.append(
            np.mean(avg_per_component_metrics, axis=0)
        )

        avg_loss, avg_metrics, avg_per_component_metrics = [], [], []
        for batch_X, batch_Y, batch_F in ds_val:
            batch_X, batch_F = batch_X.to(device),  batch_F.to(device)

            pred = regressor(batch_X, batch_F).detach().cpu()
            loss = loss_fn(pred, batch_Y)

            avg_loss.append(loss.item())
            avg_metrics.append(
                metric_relative_mse_total(batch_Y, pred).item()
            )
            avg_per_component_metrics.append(
                metric_relative_mse(batch_Y, pred).detach().cpu().numpy()
            )
        losses_val.append(np.mean(avg_loss))
        metrics_val.append(np.mean(avg_metrics))
        per_component_metrics_val.append(
            np.mean(avg_per_component_metrics, axis=0)
        )

        if metrics_val[-1] < best_score:
            torch.save(regressor.state_dict(), "firstModel.pt")
            best_score = metrics_val[-1]
        print(losses_train[-1], metrics_train[-1], per_component_metrics_train[-1], losses_val[-1] , metrics_val[-1], per_component_metrics_val[-1])

        ms_train = np.array(per_component_metrics_train).T
        ms_val = np.array(per_component_metrics_val).T

    return metrics_train, ms_train, metrics_val, ms_val

_, _, metrics_val, _ = run_training(100)
print(np.min(metrics_val), np.argmin(metrics_val))
regressor = FilmRegressor().to(device)
regressor.load_state_dict(torch.load("firstModel.pt"))

def labelRotateInv(labels, angles=180):
    import copy, math
    # rotAngle = 360 - angles
    rot_labels = torch.zeros_like(labels)
    rot_labels[:, 0] = math.cos(math.radians(-angles)) * labels[:, 0] + math.sin(math.radians(-angles)) * labels[:, 1]
    rot_labels[:, 1] = math.cos(math.radians(-angles)) * labels[:, 1] - math.sin(math.radians(-angles)) * labels[:, 0]
    rot_labels[:, 2] = math.cos(math.radians(-angles)) * labels[:, 2] + math.sin(math.radians(-angles)) * labels[:, 3]
    rot_labels[:, 3] = math.cos(math.radians(-angles)) * labels[:, 3] - math.sin(math.radians(-angles)) * labels[:, 2]
    return rot_labels

def flipInv(labels):
    import copy, math

    rot_labels = copy.copy(labels)
    rot_labels[:, 1] = -rot_labels[:, 1]
    rot_labels[:, 3] = -rot_labels[:, 3]

    return rot_labels

def mirrorInv(labels):
    import copy, math
    rot_labels = copy.copy(labels)
    rot_labels[:, 0] = -rot_labels[:, 0]
    rot_labels[:, 2] = -rot_labels[:, 2]

    return rot_labels


preds = torch.zeros((10, X_val.shape[0], 4))


a = torch.tensor(np.log1p(X_val), device=device).float()
b = torch.tensor(getScaledFilm(X_val, mean, std), device=device).float()
preds[0] = regressor(a, b).detach().cpu()
print(metric_relative_mse(torch.tensor(Y_val), preds[0]))
print(preds[0])
exit()
del a, b
torch.cuda.empty_cache()
# preds.append(singlePreds)
rotImage, _ = flip(X_val, Y_val)
a = torch.tensor(np.log1p(rotImage), device=device).float()
b =  torch.tensor(getScaledFilm(rotImage, mean, std), device=device).float()
singlePreds = regressor(a, b).detach().cpu()
preds[1] = flipInv(singlePreds)
del a, b
torch.cuda.empty_cache()

# preds.append(singlePreds)
rotImage, _ = mirror(X_val, Y_val)
preds[2] = mirrorInv(regressor(torch.tensor(np.log1p(rotImage), device=device).float(), torch.tensor(getScaledFilm(rotImage, mean, std), device=device).detach().float()).cpu())
# preds.append(singlePreds)

print(metric_relative_mse(torch.tensor(Y_val), torch.mean(preds[0:3], dim=0)))


for i, angle in enumerate(np.linspace(45, 315, 7)):
    print(angle)
    rotImage, _ = rotate(X_val, Y_val, angles=angle)
    a = torch.tensor(np.log1p(rotImage), device=device).float()
    b = torch.tensor(getScaledFilm(rotImage, mean, std), device=device).float()
    singlePreds = regressor(a, b).detach().cpu()
    preds[i + 3] = labelRotateInv(singlePreds, angle)

#print(metric_relative_mse(torch.tensor(Y_val), torch.mean(preds[0:10], dim=0)))

print(metric_relative_mse(torch.tensor(Y_val), torch.mean(preds, dim=0)))



#exit()
regressor2 = FilmRegressor2().to(device)
# from torchsummary import summary
# summary(regressor, X.shape[1:])

stepsPerEpoch = int(X_train.shape[0] / BATCH_SIZE)
learning_rate = 1e-3
opt2 = optim.Adam(regressor2.parameters(), lr=learning_rate)
scheduler2 = torch.optim.lr_scheduler.CyclicLR(opt2, base_lr=5e-5, max_lr=1e-3, step_size_up=stepsPerEpoch * 10, cycle_momentum=False)
loss_fn2 = RMSELoss().to(device)


def run_training2(epochs=5):
    losses_train = []
    losses_val = []
    metrics_train = []
    metrics_val = []
    per_component_metrics_train = []
    per_component_metrics_val = []
    best_score = 10

    for epoch in range(epochs):
        print(epoch, "/", epochs)
        avg_loss, avg_metrics, avg_per_component_metrics = [], [], []
        for batch_X, batch_Y, batch_F in ds_train:
            batch_X, batch_Y, batch_F = batch_X.to(device), batch_Y.to(device), batch_F.to(device)

            coord = regressor(batch_X, batch_F)
            pred = regressor2(batch_X, torch.cat([batch_F, coord[:, 2:]], dim=-1))

            l2_reg = None
            for W in regressor.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)

            loss = loss_fn2(pred, batch_Y[:, :2]) + 1e-3 * l2_reg

            opt2.zero_grad()
            loss.backward()
            opt2.step()
            scheduler2.step()

            avg_loss.append(loss.item())
            avg_metrics.append(
                metric_relative_mse_total(batch_Y[:, :2], pred).item()
            )
            avg_per_component_metrics.append(
                metric_relative_mse(batch_Y[:, :2], pred).detach().cpu().numpy()
            )

        losses_train.append(np.mean(avg_loss))
        metrics_train.append(
            np.mean(avg_metrics)
        )

        per_component_metrics_train.append(
            np.mean(avg_per_component_metrics, axis=0)
        )

        avg_loss, avg_metrics, avg_per_component_metrics = [], [], []
        for batch_X, batch_Y, batch_F in ds_val:
            batch_X, batch_Y, batch_F = batch_X.to(device), batch_Y.to(device), batch_F.to(device)

            coord = regressor(batch_X, batch_F).detach()
            pred = regressor2(batch_X, torch.cat([batch_F, coord[:, 2:]], dim=-1))
            loss = loss_fn2(pred, batch_Y[:, :2])

            avg_loss.append(loss.item())
            avg_metrics.append(
                metric_relative_mse_total(batch_Y[:, :2], pred).item()
            )
            avg_per_component_metrics.append(
                metric_relative_mse(batch_Y[:, :2], pred).detach().cpu().numpy()
            )
        losses_val.append(np.mean(avg_loss))
        metrics_val.append(np.mean(avg_metrics))
        per_component_metrics_val.append(
            np.mean(avg_per_component_metrics, axis=0)
        )

        if metrics_val[-1] < best_score:
            torch.save(regressor2.state_dict(), "secondModel.pt")
            best_score = metrics_val[-1]
        print(losses_train[-1], metrics_train[-1], per_component_metrics_train[-1], losses_val[-1] , metrics_val[-1], per_component_metrics_val[-1])

        ms_train = np.array(per_component_metrics_train).T
        ms_val = np.array(per_component_metrics_val).T

    return metrics_train, ms_train, metrics_val, ms_val


_, _, metrics_val, _ = run_training2(40)
print(np.min(metrics_val), np.argmin(metrics_val))

regressor2 = FilmRegressor2().to(device)
regressor2.load_state_dict(torch.load("secondModel.pt"))


def labelRotateInvSmall(labels, angles=180):
    import copy, math
    # rotAngle = 360 - angles
    rot_labels = torch.zeros_like(labels)
    rot_labels[:, 0] = math.cos(math.radians(-angles)) * labels[:, 0] + math.sin(math.radians(-angles)) * labels[:, 1]
    rot_labels[:, 1] = math.cos(math.radians(-angles)) * labels[:, 1] - math.sin(math.radians(-angles)) * labels[:, 0]
    return rot_labels

def flipInvSmall(labels):
    import copy, math

    rot_labels = copy.copy(labels)
    rot_labels[:, 1] = -rot_labels[:, 1]

    return rot_labels

def mirrorInvSmall(labels):
    import copy, math
    rot_labels = copy.copy(labels)
    rot_labels[:, 0] = -rot_labels[:, 0]

    return rot_labels


preds = torch.zeros((10, X_val.shape[0], 2))


def predictRegressor2(model, batch_size, inp, film):
    preds = []
    nLoops = int(inp.shape[0] / batch_size) + 1
    for i in range(nLoops):
        if i+1 == nLoops:
            preds.append(model(inp[i*batch_size:], film[i*batch_size:]).detach().cpu())
        else:
            preds.append(model(inp[i*batch_size:(i+1)*batch_size], film[i*batch_size:(i+1)*batch_size]).detach().cpu())

    return torch.cat(preds, dim=0)

a = torch.tensor(np.log1p(X_val), device=device).float()
b = torch.tensor(getScaledFilm(X_val, mean, std), device=device).float()
coord = regressor(a, b).detach()
preds[0] = regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()   #predictRegressor2(regressor2, 128, a, torch.cat([b, coord[:, 2:]], dim=-1))  #regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()
print(metric_relative_mse(torch.tensor(Y_val[:, :2]), preds[0]))
del a, b
torch.cuda.empty_cache()
# preds.append(singlePreds)
rotImage, _ = flip(X_val, Y_val)
a = torch.tensor(np.log1p(rotImage), device=device).float()
b = torch.tensor(getScaledFilm(rotImage, mean, std), device=device).float()
coord = regressor(a, b).detach()
singlePreds = regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()  #predictRegressor2(regressor2, 128, a, torch.cat([b, coord[:, 2:]], dim=-1))  #regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()
preds[1] = flipInvSmall(singlePreds)
del a, b
torch.cuda.empty_cache()

# preds.append(singlePreds)
rotImage, _ = mirror(X_val, Y_val)
a = torch.tensor(np.log1p(rotImage), device=device).float()
b = torch.tensor(getScaledFilm(rotImage, mean, std), device=device).float()
coord = regressor(a, b).detach()
singlePreds = regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()  #predictRegressor2(regressor2, 128, a, torch.cat([b, coord[:, 2:]], dim=-1))  #regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()
preds[2] = mirrorInvSmall(singlePreds)
del a, b
torch.cuda.empty_cache()
# preds.append(singlePreds)

print(metric_relative_mse(torch.tensor(Y_val[:, :2]), torch.mean(preds[0:3], dim=0)))


for i, angle in enumerate(np.linspace(45, 315, 7)):
    print(angle)
    rotImage, _ = rotate(X_val, Y_val, angles=angle)
    a = torch.tensor(np.log1p(rotImage), device=device).float()
    b = torch.tensor(getScaledFilm(rotImage, mean, std), device=device).float()
    coord = regressor(a, b).detach()
    singlePreds = regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()  #predictRegressor2(regressor2, 128, a, torch.cat([b, coord[:, 2:]], dim=-1))  #regressor2(a, torch.cat([b, coord[:, 2:]], dim=-1)).detach().cpu()
    preds[i + 3] = labelRotateInvSmall(singlePreds, angle)


print(metric_relative_mse(torch.tensor(Y_val[:, :2]), torch.mean(preds, dim=0)))
