"""
===========================================
VAE
Auto-Encoding Variational Bayes
Diederik P. Kingma， Max Welling
Publihsed in 2014
===========================================
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pydpm._model._VAE import VAE
import argparse
import os

os.makedirs("images", exist_ok=True)
parser = argparse.ArgumentParser()
# GPU
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--h_dim1", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--z_dim", type=int, default=2, help="dimensionality of the z latent space")
parser.add_argument("--h_dim2", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument('--load_epoch', type=int, default=50,help = 'load this epoch')
parser.add_argument('--checkpoints_interval', type=int, default=10)
parser.add_argument('--checkpoints_path', type=str, default='save_models/debug_vae')
parser.add_argument('--checkpoints_file', type=str, default='', help='save file' )
parser.add_argument('--load_checkpoints_file', type=str, default='',help='load file')
opt = parser.parse_args()
# choose gpu
if torch.cuda.is_available() and opt.gpu_id >= 0:
    device = torch.device('cuda:%d' % opt.gpu_id)
else:
    device = torch.device('cpu')

# build model
vae = VAE(x_dim=opt.img_size**2, h_dim1=opt.h_dim1, h_dim2=opt.h_dim2, z_dim=opt.z_dim,device=device)

# MNIST Dataset
train_dataset = datasets.MNIST(root='../data/mnist/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../data/mnist/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)

optimizer = optim.Adam(vae.parameters(),lr=opt.lr)

# models path
if not os.path.exists(opt.checkpoints_path):
    os.makedirs(opt.checkpoints_path)
# Loss function
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE, KLD

def train(epoch):
    vae.train()
    train_loss = 0
    Kl_loss = 0
    Llh = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        batch_size = data.size(0)
        data = data.view(batch_size, vae.x_dim).to(device)
        optimizer.zero_grad()
        recon_x, mu, log_var = vae(data)
        llh, kl_loss = loss_function(recon_x, data, mu, log_var)
        loss = llh + kl_loss
        loss.backward()
        train_loss += loss.item()
        Kl_loss += kl_loss.item()
        Llh += llh.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f} KL_Loss:{:.4f}  llh_Loss:{:.4f}'.format(epoch, train_loss / len(train_loader.dataset), Kl_loss/ len(train_loader.dataset), Llh/ len(train_loader.dataset) ))
    return train_loss / len(train_loader.dataset), llh


def test():
    vae.eval()
    test_loss = 0
    kl_Losst = 0
    Llht = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784).to(device)
            recon_x, mu, log_var = vae(data)
            llh, kl_loss = loss_function(recon_x, data, mu, log_var)
            test_loss += (llh.item() + kl_loss.item())
            kl_Losst += kl_loss.item()
            Llht += llh.item()
    test_loss /= len(test_loader.dataset)
    kl_Losst /= len(test_loader.dataset)
    Llht /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} KL_Loss:{:.4f}  llh_Loss:{:.4f}'.format(test_loss, kl_Losst, Llht))
    return test_loss, Llht


# ----------
#  Training
# ----------
save_path = os.path.join(opt.checkpoints_path, opt.checkpoints_file)
for epoch in range(1, opt.n_epochs):
    if (opt.checkpoints_interval > 0
            and (epoch + 1) % opt.checkpoints_interval == 0):
        vae.save(epoch, save_path)
    trainloss, trainLlh = train(epoch)
    testloss, testLlh = test()

# Load model
vae.load(opt.load_epoch, os.path.join(opt.checkpoints_path, opt.load_checkpoints_file))
# sample image
print('sample image,please wait!')
with torch.no_grad():
    sample = vae.sample(64)
    save_image(sample.view(64,1,28, 28), './images/l_sample_' + '.png')
    show_image = vae.show()
    save_image(show_image.view(144, 1, 28, 28), './images/l_show_' + '.png',nrow = 12)

print('complete!!!')