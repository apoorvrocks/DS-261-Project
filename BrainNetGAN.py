import pandas as pd 
import numpy as np
import torch_geometric
import torch.nn as nn
import torch
from torch_geometric.data import InMemoryDataset
import pickle
from nilearn.connectome import ConnectivityMeasure
import networkx as nx
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from torch_geometric import transforms

from torch.utils.tensorboard import SummaryWriter
from Dataset import AD_Real_Images,CN_Real_Images
from models import LargerGATModel,LargerGCNModel

import warnings
warnings.filterwarnings("ignore")


torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class MyDataset(Dataset):
    def __init__(self, data,labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        
        return self.data[idx],self.labels[idx]

def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_obj(dictionary, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
    f.close()


AAL_data_path = '/home/hiren/Apoorv Pandey/Dataset/'

AD_dict = load_obj(AAL_data_path + 'AD')
CN_dict = load_obj(AAL_data_path + 'CN')

AD_train = load_obj(AAL_data_path + 'AD_train_full')
AD_val = load_obj(AAL_data_path + 'AD_val_full')
AD_test = load_obj(AAL_data_path + 'AD_test_full')

CN_train = load_obj(AAL_data_path +'CN_train_full')
CN_val = load_obj(AAL_data_path +'CN_val_full')
CN_test = load_obj(AAL_data_path + 'CN_test_full')

CN_sub_data = pd.read_csv(AAL_data_path + 'CN_sub_data.csv')
CN_sub_data = CN_sub_data.set_index('Subject ID')

AD_sub_data = pd.read_csv(AAL_data_path + 'AD_sub_data.csv')
AD_sub_data = AD_sub_data.set_index('Subject ID')

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes,bias=False):
        super(E2EBlock, self).__init__()
        self.d = 116 #example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes,planes,(1,self.d),bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes,planes,(self.d,1),bias=bias)

        
    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d,3) + torch.cat([b]*self.d,2)

class Discriminator(torch.nn.Module):
    
    def __init__(self, img_size = 116,num_classes=2):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.in_planes = 1 #example.size(1)
        self.d = 116 #example.size(3)
        
        self.e2econv1 = E2EBlock(1,32,bias=True)
        self.e2econv2 = E2EBlock(32,64,bias=True)
        self.E2N = torch.nn.Conv2d(64,1,(1,self.d))
        self.N2G = torch.nn.Conv2d(1,256,(self.d,1))
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,3)
        self.dense3 = torch.nn.Linear(3,1)
        self.label_embed  = nn.Embedding(1,img_size*img_size)
        
    def forward(self, x):
        #self.labels = self.label_embed(labels).view(labels.size(0),1,self.img_size,self.img_size)
        #x = torch.cat([x,self.labels],dim=1)
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out),negative_slope=0.33) 
        out = F.leaky_relu(self.E2N(out),negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out),negative_slope=0.33),p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out),negative_slope=0.33),p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out),negative_slope=0.33),p=0.5)
        out = F.leaky_relu(self.dense3(out),negative_slope=0.33)
        
        return out
class Generator(nn.Module):
    
    def __init__(self, noise_d,num_classes=2):
        super(Generator, self).__init__()
        #self.label_embed = nn.Embedding(1,label_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_d, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 6670),
            nn.Tanh(),
        )
        
    def forward(self, x):
        
        #label_emb = self.label_embed(labels).squeeze(1)
        #x = torch.cat([x, label_emb], dim=1)
        x = self.net(x)
        batch_size = x.size(0)
        y = torch.ones((batch_size,116,116)).to(device)   ## 
        for i in range(batch_size):
            
            r,c = torch.triu_indices(116,116,1)
            y[i,r,c] = x[i]
            y[i,c,r] = x[i]
        
        #print("before ",y)
        '''
        for i in range(batch_size):
            y[i] = Tensor(get_adj_mat(y[i].detach().cpu().numpy(), th_p,th_n))
        print("after ",y)
        '''
        return y

class Classifier(torch.nn.Module):
    
    def __init__(self, img_size = 116,num_classes=2):
        super(Classifier, self).__init__()

        self.img_size = img_size
        self.in_planes = 1 #example.size(1)
        self.d = 116 #example.size(3)
        
        self.e2econv1 = E2EBlock(1,32,bias=True)
        self.e2econv2 = E2EBlock(32,64,bias=True)
        self.E2N = torch.nn.Conv2d(64,1,(1,self.d))
        self.N2G = torch.nn.Conv2d(1,256,(self.d,1))
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,3)
        self.dense3 = torch.nn.Linear(3,1)
        self.label_embed  = nn.Embedding(1,img_size*img_size)
    def forward(self, x):
        #self.labels = self.label_embed(labels).view(labels.size(0),1,self.img_size,self.img_size)
        #x = torch.cat([x,self.labels],dim=1)
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out),negative_slope=0.33) 
        out = F.leaky_relu(self.E2N(out),negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out),negative_slope=0.33),p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out),negative_slope=0.33),p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out),negative_slope=0.33),p=0.5)
        out = F.leaky_relu(self.dense3(out),negative_slope=0.33)
        
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm - 1)**2)
    #### END CODE HERE ####
    return penalty


if __name__==__'main__':
    lambda_gp = 10
    lr = 1e-4
    features_dim = 6670     # 116*116 connectivity matrices
    noise_dim = 100
    label_dim = 20
    batch_size = 32
    beta1 = 0.5
    n_epochs = 100
    real_label = 1.
    fake_label = 0.
    n_critic = 5

    # Initialize generator and discriminator
    num_classes = 2

    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    path = '/home/hiren/Apoorv Pandey/Code/GAN/BrainNet_GAN/'


    import matplotlib.pyplot as plt


    criterion = nn.BCEWithLogitsLoss()
    batches_done = 0
    fake_images = []

    Dloss = []
    Gloss = []
    Closs = []
    step = 0
    for i in tqdm(range(0,10)):
        
        CN_Real_Dataset = CN_Real_Images(f'/home/hiren/Apoorv Pandey/Dataset/CN_Training_Data/Seed={i}',CN_train[i])
        dataloader = DataLoader(CN_Real_Dataset, batch_size=32, shuffle=True)
        
        generator = Generator(noise_dim).to(device)
        discriminator = Discriminator().to(device)
        classifier = Classifier().to(device)
        pruning_layer = Attention_Pruning(116).to(device)
        
        optimizer_G = torch.optim.AdamW(generator.parameters(), lr, betas=(beta1, 0.999))
        optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr, betas=(beta1, 0.999))
        optimizer_C = torch.optim.AdamW(classifier.parameters(), lr, betas=(beta1, 0.999))
        
        generator.train()
        discriminator.train()
        
        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in enumerate(dataloader):
                
                data = data.squeeze(1).to(device)
                torch.cuda.empty_cache()
                batch_size = data.shape[0]
                noise = torch.randn(batch_size, noise_dim).to(device)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                optimizer_C.zero_grad()
                # Sample noise as generator input

                # Generate a batch of images
                fake = generator(noise)
                fake_pruned = pruning_layer(fake)
                real_pruned = pruning_layer(data)
                real_imgs = real_pruned.view(-1,1, 116,116).to(device)
                #print(fake)
                fake_imgs = fake_pruned.view(-1,1, 116,116).to(device)
                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity = discriminator(fake_imgs.detach())
                # Gradient penalty
                epsilon = torch.rand(len(real_imgs), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(discriminator, real_imgs, fake_imgs.detach(), epsilon)
                gp = gradient_penalty(gradient)
                #gradient_penalty = gradient_penalty(discriminator, real_imgs, fake_imgs)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
                label_pred_real = classifier(real_imgs)
                label_pred_fake = classifier(fake_imgs.detach())
                real_label = torch.ones(batch_size,1,dtype = torch.float).to(device)
                fake_label = torch.zeros(batch_size,1,dtype = torch.float).to(device)
                c_loss = 0.5*(criterion(label_pred_real,real_label) + criterion(label_pred_fake,fake_label))
                c_loss.backward(retain_graph = True)
                optimizer_C.step()
                d_loss.backward()
                optimizer_D.step()
                

                # Train the generator every n_critic steps
                if batch_idx % n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    optimizer_G.zero_grad()
                    fake = generator(noise)
                    fake_imgs = fake.view(-1,1, 116,116).to(device)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                    optimizer_G.step()
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f]"
                        % (epoch, n_epochs, batch_idx, len(dataloader), d_loss.item(), g_loss.item(),c_loss.item())
                    )
                    
                    batches_done += n_critic
                Dloss.append(d_loss.item())
                Gloss.append(g_loss.item())
                Closs.append(c_loss.item())
                
        
        print('Discriminator/Classifier/Generator Loss')
        plt.figure(0)
        plt.xlabel('Epochs')
        plt.ylabel(f'Discrimator/Generator/Classifier Loss')
        plt.plot(np.arange(0,len(Dloss)),Dloss,color = 'yellow',label = 'Discriminator')
        plt.plot(np.arange(0,len(Gloss)),Gloss,color = 'blue',label = 'Generator')
        plt.plot(np.arange(0,len(Closs)),Closs,color = 'red',label = 'Classifier')
        plt.legend()
        plt.savefig(f'/home/hiren/Apoorv Pandey/Code/GAN/BrainNet_GAN/Pruned_CN_Loss_Curve_seed{i}.jpg')
        plt.close()
        
        state = {'epoch': epoch,'Model_state_dict': generator.state_dict(),'discriminator': discriminator.state_dict(),
                'classifier': classifier.state_dict(),'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D':optimizer_D.state_dict(),'optimizer_C': optimizer_C.state_dict(),}

        torch.save(state, path + f'CN_Fake_Pruned_seed{i}.pt')

    generator_CN = Generator(noise_dim).to(device)
    generator_AD = Generator(noise_dim).to(device)

    model_CN = torch.load('/home/apoorv/dataset/BrainGAN_Fake/Brain_GAN_CN.pt')
    model_AD = torch.load('/home/apoorv/dataset/BrainGAN_Fake/Brain_GAN_AD.pt')
    #generator_CN.load_state_dict(model_CN['Model_state_dict'])
    generator_AD.load_state_dict(model_AD['Model_state_dict'])
    torch.manual_seed(42)

    noise_AD = torch.randn(165,noise_dim).to(device)
    AD_fake_images = generator_AD(noise_AD).detach().cpu().numpy()
    noise_CN = torch.randn(375,noise_dim).to(device)
    CN_fake_images = generator_CN(noise_CN).detach().cpu().numpy()
    save_obj(CN_fake_images,'/home/apoorv/dataset/BrainGAN_Fake/CN_fake') ## fake generated CN images
    save_obj(AD_fake_images,'/home/apoorv/dataset/BrainGAN_Fake/AD_fake')  ## fake generated AD images