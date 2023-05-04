#!/usr/bin/env python
# coding: utf-8

# # Buckler et al. 2009
# 
# > This file aims to reproduce the findings of [Buckler et al 2009](https://www.science.org/doi/10.1126/science.1174276?url_ver=Z39.88-2003), "The Genetic Architecture of Maize Flowering Time".

# It used data from panzea
# - Phenotypic data panzea\phenotypes\Buckler_etal_2009_Science_flowering_time_data-090807\
# - Genotypic Data panzea\genotypes\GBS\v27\ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023.vcf.gz
# - Genomic Data ... 

# In[ ]:





# In[1]:


use_gpu_num = 1

import os
import pandas as pd
import numpy as np
import re

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

# TODO fixme

device = "cuda" if torch.cuda.is_available() else "cpu"
if use_gpu_num in [0, 1]: 
    torch.cuda.set_device(use_gpu_num)
print(f"Using {device} device")

import tqdm

import plotly.graph_objects as go
import plotly.express as px

# [e for e in os.listdir() if re.match(".+\\.txt", e)]


# In[2]:


nam_overview = pd.read_table('../ext_data/zma/panzea/phenotypes/Buckler_etal_2009_Science_flowering_time_data-090807/NAMSum0607FloweringTraitBLUPsAcross8Envs.txt')
nam_overview


# In[3]:


data = pd.read_table('../ext_data/zma/panzea/phenotypes/Buckler_etal_2009_Science_flowering_time_data-090807/markergenotypes062508.txt', skiprows=1
                    ).reset_index().rename(columns = {'index': 'Geno_Code'})
data


# In[4]:


px.scatter_matrix(data.loc[:, ['days2anthesis', 'days2silk', 'asi']])


# In[ ]:





# In[ ]:





# In[5]:


d2a = np.array(data['days2anthesis'])
d2s = np.array(data['days2silk'])
asi = np.array(data['asi'])

xs = np.array(data.drop(columns = ['days2anthesis', 'days2silk', 'asi', 'pop', 'Geno_Code']))

n_obs = xs.shape[0]

np_seed = 9070707
rng = np.random.default_rng(np_seed)  # can be called without a seed

test_pr = 0.2

test_n = round(n_obs*test_pr)
idxs = np.linspace(0, n_obs-1, num = n_obs).astype(int)
rng.shuffle(idxs)

test_idxs = idxs[0:test_n]
train_idxs = idxs[test_n:-1]


# In[6]:


# make up tensors
def calc_cs(x): return [np.mean(x, axis = 0), np.std(x, axis = 0)]


# In[7]:


def apply_cs(xs, cs_dict_entry): return ((xs - cs_dict_entry[0]) / cs_dict_entry[0])


# In[ ]:





# In[8]:


scale_dict = {
    'd2a':calc_cs(d2a[train_idxs]),
    'd2s':calc_cs(d2s[train_idxs]),
    'asi':calc_cs(asi[train_idxs]),
    'xs' :calc_cs(xs[train_idxs])
}


# In[9]:


y1 = apply_cs(d2a, scale_dict['d2a'])
y2 = apply_cs(d2s, scale_dict['d2s'])
y3 = apply_cs(asi, scale_dict['asi'])

# No need to cs xs -- 0-2 scale
# apply_cs(xs, scale_dict['xs'])

y1_train = torch.from_numpy(y1[train_idxs]).to(device).float()[:, None]
y2_train = torch.from_numpy(y2[train_idxs]).to(device).float()[:, None]
y3_train = torch.from_numpy(y3[train_idxs]).to(device).float()[:, None]
xs_train = torch.from_numpy(xs[train_idxs]).to(device).float()

y1_test = torch.from_numpy(y1[test_idxs]).to(device).float()[:, None]
y2_test = torch.from_numpy(y2[test_idxs]).to(device).float()[:, None]
y3_test = torch.from_numpy(y3[test_idxs]).to(device).float()[:, None]
xs_test = torch.from_numpy(xs[test_idxs]).to(device).float()


# In[10]:


class CustomDataset(Dataset):
    def __init__(self, y1, y2, y3, xs, transform = None, target_transform = None):
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.xs = xs
        self.transform = transform
        self.target_transform = target_transform    
    
    def __len__(self):
        return len(self.y1)
    
    def __getitem__(self, idx):
        y1_idx = self.y1[idx]
        y2_idx = self.y2[idx]
        y3_idx = self.y3[idx]
        xs_idx = self.xs[idx]
        
        if self.transform:
            xs_idx = self.transform(xs_idx)
            
        if self.target_transform:
            y1_idx = self.transform(y1_idx)
            y2_idx = self.transform(y2_idx)
            y3_idx = self.transform(y3_idx)
        return xs_idx, y1_idx, y2_idx, y3_idx


# In[11]:


training_dataloader = DataLoader(
    CustomDataset(
        y1 = y1_train,
        y2 = y2_train,
        y3 = y3_train,
        xs = xs_train
    ), 
    batch_size = 64, 
    shuffle = True)

testing_dataloader = DataLoader(
    CustomDataset(
        y1 = y1_test,
        y2 = y2_test,
        y3 = y3_test,
        xs = xs_test
    ), 
    batch_size = 64, 
    shuffle = True)

xs.shape


# ## Version 1, Predict `y1` (Anthesis)

# In[12]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()    
        self.x_network = nn.Sequential(
            nn.Linear(1106, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1))
        
    def forward(self, x):
        x_out = self.x_network(x)
        return x_out

model = NeuralNetwork().to(device)
# print(model)


# In[13]:


xs_i, y1_i, y2_i, y3_i = next(iter(training_dataloader))
model(xs_i).shape # try prediction on one batch


# In[14]:


def train_loop(dataloader, model, loss_fn, optimizer, silent = False):
    size = len(dataloader.dataset)
    for batch, (xs_i, y1_i, y2_i, y3_i) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(xs_i)
        loss = loss_fn(pred, y1_i) # <----------------------------------------

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(y1_i) # <----------------
            if not silent:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                
def train_error(dataloader, model, loss_fn, silent = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0

    with torch.no_grad():
        for xs_i, y1_i, y2_i, y3_i in dataloader:
            pred = model(xs_i)
            train_loss += loss_fn(pred, y1_i).item() # <----------------------
            
    train_loss /= num_batches
    return(train_loss) 

            
def test_loop(dataloader, model, loss_fn, silent = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for xs_i, y1_i, y2_i, y3_i in dataloader:
            pred = model(xs_i)
            test_loss += loss_fn(pred, y1_i).item() # <-----------------------

    test_loss /= num_batches
    if not silent:
        print(f"Test Error: Avg loss: {test_loss:>8f}")
    return(test_loss) 


def train_nn(
    training_dataloader,
    testing_dataloader,
    model,
    learning_rate = 1e-3,
    batch_size = 64,
    epochs = 500
):
    # Initialize the loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_df = pd.DataFrame([i for i in range(epochs)], columns = ['Epoch'])
    loss_df['TrainMSE'] = np.nan
    loss_df['TestMSE']  = np.nan

    for t in tqdm.tqdm(range(epochs)):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(training_dataloader, model, loss_fn, optimizer, silent = True)

        loss_df.loc[loss_df.index == t, 'TrainMSE'
                   ] = train_error(training_dataloader, model, loss_fn, silent = True)
        
        loss_df.loc[loss_df.index == t, 'TestMSE'
                   ] = test_loop(testing_dataloader, model, loss_fn, silent = True)
        
    return([model, loss_df])


# In[15]:


model, loss_df = train_nn(
    training_dataloader,
    testing_dataloader,
    model,
    learning_rate = 1e-3,
    batch_size = 64,
    epochs = 500
)


# In[16]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TrainMSE,
                    mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TestMSE,
                    mode='lines', name='Test'))
fig.show()


# In[34]:


# ! conda install captum -c pytorch -y


# In[19]:


# imports from captum library
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation


# In[ ]:





# In[20]:


ig = IntegratedGradients(model)
ig_nt = NoiseTunnel(ig)
dl = DeepLift(model)
gs = GradientShap(model)
fa = FeatureAblation(model)

ig_attr_test = ig.attribute(xs_test, n_steps=50)
ig_nt_attr_test = ig_nt.attribute(xs_test)
dl_attr_test = dl.attribute(xs_test)
gs_attr_test = gs.attribute(xs_test, xs_train)
fa_attr_test = fa.attribute(xs_test)


# In[21]:


[e.shape for e in [ig_attr_test,
ig_nt_attr_test,
dl_attr_test,
gs_attr_test,
fa_attr_test]]


# In[22]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
                         y = ig_nt_attr_test.cpu().detach().numpy().mean(axis=0),
                         mode='lines', name='Test'))
fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
                         y = dl_attr_test.cpu().detach().numpy().mean(axis=0),
                         mode='lines', name='Test'))
fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
                         y = gs_attr_test.cpu().detach().numpy().mean(axis=0),
                         mode='lines', name='Test'))
fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
                         y = fa_attr_test.cpu().detach().numpy().mean(axis=0),
                         mode='lines', name='Test'))
fig.show()


# In[23]:


len(dl_attr_test.cpu().detach().numpy().mean(axis = 0))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Version 2, Predict `y1` (Anthesis), `y2` (Silking), and `y3` (ASI)
# 
# Here each model will predict 3 values. The loss function is still mse, but the y tensors are concatenated

# In[37]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()    
        self.x_network = nn.Sequential(
            nn.Linear(1106, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3))
        
    def forward(self, x):
        x_out = self.x_network(x)
        return x_out

model = NeuralNetwork().to(device)
# print(model)


# In[38]:


xs_i, y1_i, y2_i, y3_i = next(iter(training_dataloader))
model(xs_i).shape # try prediction on one batch


# In[ ]:





# In[ ]:





# In[39]:


def train_loop(dataloader, model, loss_fn, optimizer, silent = False):
    size = len(dataloader.dataset)
    for batch, (xs_i, y1_i, y2_i, y3_i) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(xs_i)
        loss = loss_fn(pred, torch.concat([y1_i, y2_i, y3_i], axis = 1)) # <----------------------------------------

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(y1_i) # <----------------
            if not silent:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                
def train_error(dataloader, model, loss_fn, silent = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0

    with torch.no_grad():
        for xs_i, y1_i, y2_i, y3_i in dataloader:
            pred = model(xs_i)
            train_loss += loss_fn(pred, torch.concat([y1_i, y2_i, y3_i], axis = 1)).item() # <----------------------
            
    train_loss /= num_batches
    return(train_loss) 

            
def test_loop(dataloader, model, loss_fn, silent = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for xs_i, y1_i, y2_i, y3_i in dataloader:
            pred = model(xs_i)
            test_loss += loss_fn(pred, torch.concat([y1_i, y2_i, y3_i], axis = 1)).item() # <-----------------------

    test_loss /= num_batches
    if not silent:
        print(f"Test Error: Avg loss: {test_loss:>8f}")
    return(test_loss) 


def train_nn(
    training_dataloader,
    testing_dataloader,
    model,
    learning_rate = 1e-3,
    batch_size = 64,
    epochs = 500
):
    # Initialize the loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_df = pd.DataFrame([i for i in range(epochs)], columns = ['Epoch'])
    loss_df['TrainMSE'] = np.nan
    loss_df['TestMSE']  = np.nan

    for t in tqdm.tqdm(range(epochs)):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(training_dataloader, model, loss_fn, optimizer, silent = True)

        loss_df.loc[loss_df.index == t, 'TrainMSE'
                   ] = train_error(training_dataloader, model, loss_fn, silent = True)
        
        loss_df.loc[loss_df.index == t, 'TestMSE'
                   ] = test_loop(testing_dataloader, model, loss_fn, silent = True)
        
    return([model, loss_df])


# In[40]:


model, loss_df = train_nn(
    training_dataloader,
    testing_dataloader,
    model,
    learning_rate = 1e-3,
    batch_size = 64,
    epochs = 500
)


# In[54]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TrainMSE,
                    mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TestMSE,
                    mode='lines', name='Test'))
fig.show()


# In[55]:


model, loss_df = train_nn(
    training_dataloader,
    testing_dataloader,
    model,
    learning_rate = 1e-3,
    batch_size = 64,
    epochs = 5000
)


# In[56]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TrainMSE,
                    mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TestMSE,
                    mode='lines', name='Test'))
fig.show()


# In[ ]:





# In[ ]:





# In[25]:


'../ext_data/zma/panzea/phenotypes/'


# In[27]:


# pd.read_table('../ext_data/zma/panzea/phenotypes/traitMatrix_maize282NAM_v15-130212.txt', low_memory = False)


# In[33]:


# pd.read_excel('../ext_data/zma/panzea/phenotypes/traitMatrix_maize282NAM_v15-130212_TraitDescritptions.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




