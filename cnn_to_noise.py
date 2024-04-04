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
from itertools import product
if not os.path.exists('./checkpoint/'):
    os.makedirs('./checkpoint/')
if not os.path.exists('./diffusion_result/'):
    os.makedirs('./diffusion_result/')
if not os.path.exists("./with_noise_detect/"):
    os.makedirs("./with_noise_detect/")
    
IMG_SIZE = 32
BATCH_SIZE = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

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
def train_loop(args):
    comment=f'batch size cnn {args.batch_size} learning rate {args.lr}'
    tb=SummaryWriter(comment=comment)
    model = Simple_CNN()
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
    tb.add_graph(model,x_noisy)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        step=0
        total_loss=0
        total_correct=0
        for batch in tqdm(dataloader,total=len(dataloader),leave=False):
            optimizer.zero_grad()
            step+=1
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            image=batch[0].to(device)
            image_noisy,noise=forward_diffusion_sample(image,t,device)
            preds=model(image)
            loss=F.cross_entropy(preds,t)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*args.batch_size
            total_correct+=get_num_correct(preds,t)
        tb.add_scalar("loss",total_loss,epoch)
        tb.add_scalar("Number correct",total_correct,epoch)
        tb.add_scalar("Accuracy",total_correct/step,epoch)
        for name, weight in model.named_parameters():
            tb.add_histogram(f'{name}.weight',weight,epoch)
            tb.add_histogram(f'{name}.weight.grad',weight.grad,epoch)
        print("epoch: ",epoch," total correct: ",total_correct," total loss: ",total_loss)
        if (epoch + 1) % 10 == 0:
                torch.save(model, "./checkpoint/cnn_{}.tar".format(epoch+1))
    tb.close()
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args([])
if not os.path.exists("./checkpoint/cnn_100.tar"):
    train_loop(args)
model=torch.load("./checkpoint/cnn_100.tar")
model.cuda()