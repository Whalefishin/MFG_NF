import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vae_mnist_model import VAE
import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt




batch_size = 100
input_dim = 784
hidden_dim_1 = 512
hidden_dim_2 = 256
latent_dim = 16

vae = VAE(x_dim=input_dim, h_dim1= hidden_dim_1, h_dim2=hidden_dim_2, z_dim=latent_dim)

# MNIST Dataset
train_dataset = datasets.MNIST(root='./experiments/dataset/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./experiments/dataset/data', train=False, transform=transforms.ToTensor(), download=True)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# directory
log_dir   = './results/vae_mnist'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

save_path = os.path.join(log_dir, 'vae_mnist.t')
vae.load_state_dict(torch.load(save_path))


# get latent vectors and their labels
latent_all   = []
latent_01234_train = []
latent_56789_train = []
latent_01234_test = []
latent_56789_test = []
label_all    = []
label_01234_train = []
label_56789_train = []
label_01234_test  = []
label_56789_test  = []

img_01234_train = []
img_56789_train = []
img_01234_test  = []
img_56789_test  = []

for (x,y) in train_loader:
    mask = y <=4
    img_01234_train.append(x[mask].detach().clone())
    img_56789_train.append(x[~mask].detach().clone())
    x = vae.encoder(x.view(-1, 784))[0]
    latent_all.append(x)
    label_all.append(y)

    latent_01234_train.append(x[mask])
    latent_56789_train.append(x[~mask])
    label_01234_train.append(y[mask])
    label_56789_train.append(y[~mask])

for (x,y) in test_loader:
    mask = y <=4
    img_01234_test.append(x[mask].detach().clone())
    img_56789_test.append(x[~mask].detach().clone())
    x = vae.encoder(x.view(-1, 784))[0]
    latent_all.append(x)
    label_all.append(y)

    latent_01234_test.append(x[mask])
    latent_56789_test.append(x[~mask])
    label_01234_test.append(y[mask])
    label_56789_test.append(y[~mask])

latent_all         = torch.cat(latent_all).detach().cpu().numpy()
latent_01234_train = torch.cat(latent_01234_train).detach().cpu().numpy()
latent_56789_train = torch.cat(latent_56789_train).detach().cpu().numpy()
latent_01234_test  = torch.cat(latent_01234_test).detach().cpu().numpy()
latent_56789_test  = torch.cat(latent_56789_test).detach().cpu().numpy()
label_all          = torch.cat(label_all).detach().cpu().numpy()
# label_01234  = torch.cat(label_01234_test).detach().cpu().numpy()
# label_56789  = torch.cat(label_56789_test).detach().cpu().numpy()
img_01234_train = torch.cat(img_01234_train).detach().cpu().numpy()
img_56789_train = torch.cat(img_56789_train).detach().cpu().numpy()
img_01234_test  = torch.cat(img_01234_test).detach().cpu().numpy()
img_56789_test  = torch.cat(img_56789_test).detach().cpu().numpy()

# # t-SNE on all latent vectors
# n_subset = 20000
# X_embedded = TSNE(n_components=2, init='random').fit_transform(latent_all[:n_subset])

# # plotting
# # plt.set_cmap('jet')
# t = label_all[:n_subset]
# plt.scatter(X_embedded[:,0], X_embedded[:,1], c=t, s=10)
# plt.colorbar()
# plt.show()

# save stuff
np.save(os.path.join(log_dir, 'latent_01234_train.npy'), latent_01234_train)
np.save(os.path.join(log_dir, 'latent_56789_train.npy'), latent_56789_train)
# np.save(os.path.join(log_dir, 'label_01234_train.npy'), label_01234_train)
# np.save(os.path.join(log_dir, 'label_56789_train.npy'), label_56789_train)
np.save(os.path.join(log_dir, 'latent_01234_test.npy'), latent_01234_test)
np.save(os.path.join(log_dir, 'latent_56789_test.npy'), latent_56789_test)
# np.save(os.path.join(log_dir, 'label_01234_test.npy'), label_01234_test)
# np.save(os.path.join(log_dir, 'label_56789_test.npy'), label_56789_test)

np.save(os.path.join(log_dir, 'img_01234_train.npy'), img_01234_train)
np.save(os.path.join(log_dir, 'img_56789_train.npy'), img_56789_train)
np.save(os.path.join(log_dir, 'img_01234_test.npy'), img_01234_test)
np.save(os.path.join(log_dir, 'img_56789_test.npy'), img_56789_test)