import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pdb

class Encoder(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        '''
        Args:
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, stride=2)
        self.linear = nn.Linear(3 * 13 * 13, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden1 = self.conv1(x)                     # shape = [batch_size, 3, 13, 13]
        hidden1 = hidden1.view(-1, 3 * 13 * 13)     # shape = [batch_size, 507]
        hidden2 = F.relu(self.linear(hidden1))      # shape = [batch_size, hidden_dim]
        z_mu = self.mu(hidden2)                     # shape = [batch_size, 2]
        z_var = self.var(hidden2)                   # shape = [batch_size, 2]
        return z_mu, z_var

class Decoder(nn.Module):
    def __init__(self, z_dim):
        '''
            z_dim: A integer indicating the latent size.
        '''
        super().__init__()
        self.linear = nn.Linear(z_dim, 26 * 26)
        self.deconv1 = nn.ConvTranspose2d(1, 3, 3)
        self.out = nn.Linear(3*28*28, 784)

    def forward(self, x):
        hidden1 = F.relu(self.linear(x))        # shape = [batch_size, 676]
        hidden1 = hidden1.view(-1, 1, 26, 26)   # shape = [batch_size, 1, 26, 26]
        hidden2 = self.deconv1(hidden1)         # shape = [batch_size, 3, 28, 28]
        hidden2 = hidden2.view(-1, 3*28*28)     # shape = [batch_size, 2352]
        hidden3 = self.out(hidden2)             # shape = [batch_size, 784]
        predicted = torch.sigmoid(hidden3)
        return predicted

class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z_mu, z_var = self.enc(x)   # encoder, sample from the distribution having latent parameters z_mu, z_var

        std = torch.sqrt(torch.exp(z_var))              #
        eps = torch.randn_like(std)             # reparameterize
        z_sample = eps.mul(std).add_(z_mu)      #

        predicted = self.dec(z_sample)   # decode
        return predicted, z_mu, z_var, z_sample

def train():
    model.train()       # set the train mode
    train_loss = 0      # loss of the epoch

    for i, (x, _) in enumerate(train_iterator):
        x = x.to(device)            # tensor to device
        optimizer.zero_grad()       # set the gradients to zero
        x_sample, z_mu, z_var, z_sample = model(x)        # forward pass

        recon_loss = F.binary_cross_entropy(x_sample, x.view(-1, 28 * 28), reduction='sum')        # reconstruction loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)       # kl divergence loss
        loss = recon_loss + kl_loss     # total loss
        if(i%100 == 1): print('iteration: {}/{}   Loss: {}'.format(i, len(train_iterator), loss))
        loss.backward()         # backward pass
        train_loss += loss.item()
        optimizer.step()        # update the weights
    return train_loss

def test():
    model.eval()        # set the evaluation mode
    test_loss = 0       # test loss for the data
    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, _) in enumerate(test_iterator):
            x = x.to(device)
            x_sample, z_mu, z_var, z_sample = model(x)        # forward pass

            recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)        # reconstruction loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)       # kl divergence loss
            loss = recon_loss + kl_loss         # total loss
            test_loss += loss.item()
    return test_loss

# encodes and decodes images from the test_set and plots them
def generate_and_save_images(model, epoch, test_iterator, path_folder=r'C:\Users\Markus Miller\Desktop\Uni\Machine Learning\ex06\Bilder\\'):
    batch = next(iter(test_iterator))[0]
    x = batch.to(device)
    x_sample, z_mu, z_var, z_sample = model(x)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(x_sample[i].view(28, 28).detach().numpy(), cmap='gray')
        plt.axis('off')

    plt.savefig(path_folder + r'image_at_epoch_{:04d}.png'.format(epoch))

# encodes and decodes images with different labels from the test_set and plots them
def generate_and_save_images2(test_dataset, model, epoch, loss, path_folder=r'C:\Users\Markus Miller\Desktop\Uni\Machine Learning\ex06\Bilder\\'):
    images = test_dataset.data/255.0
    labels = test_dataset.targets
    different_clothes = [images[labels == i][0] for i in range(labels.max() + 1)]
    predictions = []
    for x_sample in different_clothes:
        x_sample = x_sample.to(device)
        x_sample = x_sample.view(1, 1, 28, 28)

        x_recon, z_mu, z_var, z_sample = model(x_sample.float())
        predictions.append(x_recon)

    fig, ax = plt.subplots(10, 2)
    for i in range(labels.max() + 1):

        ax[i][0].imshow(different_clothes[i], cmap='gray')
        ax[i][1].imshow(predictions[i].view(28, 28).detach().numpy(), cmap='gray')
        ax[i][0].axis('off')
        ax[i][1].axis('off')
    plt.suptitle('Loss: {}'.format(loss))
    plt.savefig(path_folder + 'image_classes_at_epoch_{:04d}.pdf'.format(epoch))

def plot_grid(decoder, path_folder=r'C:\Users\Markus Miller\Desktop\Uni\Machine Learning\ex06\Bilder\\'):
    xx, yy = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5))
    grid = np.zeros((xx.shape[0], xx.shape[0], 2))
    grid[:, :, 0] = xx
    grid[:, :, 1] = yy
    grid = grid.reshape(81, 2)
    latent_img = decoder(torch.from_numpy(grid).float().to(device)).view(-1, 28, 28)

    for i in range(81):
        plt.subplot(9, 9, i+1)
        plt.imshow(latent_img[i].detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.savefig(path_folder + r'grid.png')

def plot_latent_space(train_dataset, model, path_folder=r'C:\Users\Markus Miller\Desktop\Uni\Machine Learning\ex06\Bilder\\'):
    images = train_dataset.data[:3000] / 255.0
    labels = train_dataset.targets[:3000]
    images = images.view(3000, 1, 28, 28)
    images = images.to(device)
    x_sample, z_mu, z_var, z_sample = model(images)
    z_sample = z_sample.detach().numpy()
    colors = ['black', 'darkgrey', 'red', 'darkorange', 'yellow', 'green', 'teal', 'blue', 'darkviolet', 'saddlebrown']
    groups = train_dataset.classes
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(10):
        z_class = z_sample[labels == i]
        x, y = z_class[:, 0], z_class[:, 1]
        ax.scatter(x, y, alpha=0.8, c=colors[i], label=groups[i])
    plt.legend()
    plt.savefig(path_folder + r'latent_space.pdf')

if __name__=='__main__':
    transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms)

    BATCH_SIZE = 64  # number of data points in each batch
    N_EPOCHS = 20  # times to run the model on complete data
    HIDDEN_DIM = 256  # hidden dimension for encoder
    LATENT_DIM = 2  # latent vector dimension
    lr = 1e-4  # learning rate

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    encoder = Encoder(HIDDEN_DIM, LATENT_DIM)
    decoder = Decoder(LATENT_DIM)
    model = VAE(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for e in range(N_EPOCHS):
        train_loss = train()
        test_loss = test()

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        generate_and_save_images(model, e+1, test_iterator)
        generate_and_save_images2(test_dataset, model, e+1, test_loss)
        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
    plot_latent_space(train_dataset, model)
    plot_grid(decoder)

