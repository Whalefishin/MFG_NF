import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from vae_mnist_model import VAE
import os


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD, BCE, KLD


def train(epoch):
    vae.train()
    train_loss = 0
    train_CE   = 0 
    train_D_KL = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss, CE, D_KL = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        optimizer.step()

        train_loss += float(loss)
        train_CE   += float(CE)
        train_D_KL += float(D_KL)
        
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, CE: {:.6f}, D_KL: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data), 
                CE.item() / len(data), D_KL.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}, Average CE: {:.4f}, Average D_KL: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset), train_CE / len(train_loader.dataset), 
        train_D_KL / len(train_loader.dataset)))


def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            loss, CE, D_KL = loss_function(recon, data, mu, log_var)
            test_loss += loss.item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# Training Parameters
batch_size = 100
num_epochs = 50
# num_epochs = 50

# Model Parameters
input_dim = 784
hidden_dim_1 = 512
hidden_dim_2 = 256
# latent_dim = 10
latent_dim = 16


# MNIST Dataset
train_dataset = datasets.MNIST(root='./experiments/dataset/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./experiments/dataset/data', train=False, transform=transforms.ToTensor(), download=True)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Define model
vae = VAE(x_dim=input_dim, h_dim1= hidden_dim_1, h_dim2=hidden_dim_2, z_dim=latent_dim)

if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters())

for epoch in range(1, num_epochs+1):
    train(epoch)
    test()

# #Save sample image
# with torch.no_grad():
#     z = torch.randn(64, latent_dim).cuda()
#     sample = vae.decoder(z).cuda()
    
#     save_image(sample.view(64, 1, 28, 28), './results/MNIST/sample' + '.png')


# directory
log_dir   = 'results/vae_mnist'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

save_path = os.path.join(log_dir, 'vae_mnist.t')
torch.save(vae.state_dict(), save_path)