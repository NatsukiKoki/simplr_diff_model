from numpy import true_divide
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets
from torch.utils.data.dataset import Subset
from torchvision.utils import save_image
import math
from IPython.display import Image
import matplotlib.pyplot as plt
from torch.optim import Adam
import random
from tqdm import tqdm

import os
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn


if not os.path.exists('./checkpoint/'):
    os.makedirs('./checkpoint/')
if not os.path.exists('./diffusion_result/'):
    os.makedirs('./diffusion_result/')
IMG_SIZE = 32
BATCH_SIZE = 128

class Simple_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=36,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(stride=2,kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=36,out_channels=72,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=72,out_channels=144,kernel_size=3,padding=1)
        self.fc1=nn.Linear(in_features=144*4*4,out_features=1024)
        self.fc2=nn.Linear(in_features=1024,out_features=512)
        self.fc3=nn.Linear(in_features=512,out_features=256)
        self.out=nn.Linear(in_features=256,out_features=300)
    def forward(self,t):
        #print(t.shape)
        t=F.relu(self.conv1(t))
        t=self.pool(t)
        #print(t.shape)
        t=F.relu(self.conv2(t))
        t=self.pool(t)
        t=F.relu(self.conv3(t))
        t=self.pool(t)
        #print(t.shape)
        t=t.flatten(start_dim=1)
        #print(t.shape)
        t=F.relu(self.fc1(t))
        #print(t.shape)
        t=F.relu(self.fc2(t))
        ##print(t.shape)
        t=F.relu(self.fc3(t))
        #print(t.shape)
        t=self.out(t)
        #print(t.shape)
        return t

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.MNIST(root="./data", download=True,train=True, 
                                         transform=data_transform)

    test = torchvision.datasets.MNIST(root="./data", download=True,train=False,
                                         transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0,:, :, :]
    image=reverse_transforms(image)
    plt.imshow(image)

# Simulate forward diffusion
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (32, 64, 128 , 256, 512)
        up_channels = (512, 256, 128, 64, 32)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            #print(x.shape)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
            #print(x.shape)
        return self.output(x)


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)
@torch.no_grad()
def sample_timestep(x, t,model):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(model,cnn,kk):
    # Sample noise
    # Simulate forward diffusion
    data = load_transformed_dataset()
    dataloader = DataLoader(data, 128, shuffle=True, drop_last=True)
    image = next(iter(dataloader))[0]
    noise = torch.randn_like(image)
    img=noise
    img=img.cuda()
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = T
    if not num_images%10 == 0:
        num_images=num_images//10
        num_images*=10
    num_images_=10
    stepsize=num_images//10

    for i in range(0,num_images)[::-1]:
        tp = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, tp,model)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images_, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.tight_layout()
    plt.savefig(f"./diffusion_result/{kk}.png")
    plt.show()
    plt.close()

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(args):
    comment=f'diffusion batch size {args.batch_size} learning rate {args.lr}'
    tb=SummaryWriter(comment=comment)
    model = SimpleUnet()
    model.to(device)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benckmark = True
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    images,labels=next(iter(dataloader))
    grid=torchvision.utils.make_grid(images)
    tb.add_image("images",grid)
    t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
    x_noisy, noise = forward_diffusion_sample(images.cuda(), t, device)
    tb.add_graph(model,(x_noisy,t))
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        step=0
        for batch in tqdm(dataloader,total=len(dataloader),leave=False):
            optimizer.zero_grad()
            step+=1
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()
        for name, weight in model.named_parameters():
            tb.add_histogram(f'{name}.weight',weight,epoch)
            tb.add_histogram(f'{name}.weight.grad',weight.grad,epoch)

        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        if (epoch + 1) % 1 == 0:
            torch.save(model, "./checkpoint/DIFF_{}.tar".format(epoch+1))
    tb.close()
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args([])
if not os.path.exists("./checkpoint/DIFF_300.tar"):
    train_loop(args)
#Cnn=Simple_CNN()
Cnn=torch.load("./checkpoint/cnn_100.tar")
Cnn.cuda()


model=torch.load("./checkpoint/DIFF_300.tar")
model.cuda()
for i in range(10):
    sample_plot_image(model,Cnn,i)