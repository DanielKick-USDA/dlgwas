#!/usr/bin/env python
# coding: utf-8

# # Tian et al. 2011 Model 1
# 
# > This file aims to reproduce the findings of *Tian et al. 2011*, "Genome-wide association study of leaf architecture in the
# maize nested association mapping population".

# <!-- It used data from panzea
# - Phenotypic data panzea\phenotypes\Buckler_etal_2009_Science_flowering_time_data-090807\
# - Genotypic Data panzea\genotypes\GBS\v27\ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023.vcf.gz
# - Genomic Data ...  -->

# In[ ]:


get_ipython().system('conda install -c conda-forge pytorch-gpu -y')


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
from tqdm import tqdm

import plotly.graph_objects as go
import plotly.express as px

# [e for e in os.listdir() if re.match(".+\\.txt", e)]


# In[ ]:


import dlgwas
# from dlgwas.dna import *

from dlgwas.kegg import ensure_dir_path_exists
from dlgwas.kegg import get_cached_result
from dlgwas.kegg import put_cached_result


# In[ ]:


# set up directory for notebook artifacts
# nb_name = '11_TianEtAl2011'
# ensure_dir_path_exists(dir_path = '../models/'+nb_name)
# ensure_dir_path_exists(dir_path = '../reports/'+nb_name)


# ##  Load Data

# In[ ]:


with open('../ext_data/zma/panzea/phenotypes/Tian_etal_2011_NatGen_leaf_pheno_data-110221/Tian_etal_2011_NatGen_readme.txt', 
          'r') as f:
    dat = f.read()
print(dat)


# In[ ]:


data = pd.read_excel('../ext_data/zma/panzea/phenotypes/Tian_etal_2011_NatGen_leaf_pheno_data-110221/Tian_etal2011NatGenet.leaf_trait_phenotype.xlsx')
data


# ## Find Marker data to use along with the phenotypic data here
# 

# In[ ]:


samples = list(set(data['sample']))


# In[ ]:


# this can take a while to calculate so it's worth cacheing
save_path = '../models/10_TianEtAl2011/samples_and_matches.pkl'

samples_and_matches = get_cached_result(save_path=save_path)


# In[ ]:


samples_one_match = [e for e in samples_and_matches if len(e['matches']) == 1]

print("Warning: "+str(len(samples_and_matches)-len(samples_one_match)
    )+" samples ("+str(round(100*((len(samples_and_matches)-len(samples_one_match))/len(samples_and_matches))) 
    )+"%) have zero matches or more than one match in AGPv4. The first is being used.")


# ## Filter data to only those with unambiguous genotypes

# In[ ]:


original_rows = data.shape[0]

# mask to restrict to only those samples with one or more GBS marker set in AGPv4
mask = [True if e in [e1['sample'] for e1 in 
                      [e for e in samples_and_matches if len(e['matches']) >= 1]
              ] else False for e in data['sample'] ]
data = data.loc[mask,].reset_index().drop(columns = 'index')
print(str(original_rows - data.shape[0])+' rows dropped.')


# ### Misc

# In[ ]:


# Useful for converting between the physical location and site
AGPv4_site = pd.read_table('../data/zma/panzea/genotypes/GBS/v27/'+'ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_PositionList.txt')
AGPv4_site.head()


# ## Taxa Groupings

# In[ ]:


taxa_groupings = pd.read_table('../data/zma/panzea/genotypes/GBS/v27/ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_TaxaList.txt')
taxa_groupings = taxa_groupings.loc[:, ['Taxa', 'Tassel4SampleName', 'Population']]

taxa_groupings[['sample', 'sample2']] = taxa_groupings['Taxa'].str.split(':', expand = True)

taxa_groupings = taxa_groupings.loc[:, ['sample', 'Population']].drop_duplicates()

# Restrict to those in data
taxa_groupings = data[['sample']].merge(taxa_groupings, how = 'left')
taxa_groupings


# In[ ]:


temp = [e for e in list(set(taxa_groupings.Population))]
temp.sort()
temp


# In[ ]:





# In[ ]:


temp = taxa_groupings.copy()
temp['sample'] = 1

fig = px.treemap(temp, 
                 path=[px.Constant("All Populations:"), 'Population'], values='sample')
# fig.update_traces(root_color="lightgrey")
# fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()


# ## Finalize Data

# In[ ]:


# Define holdout sets (Populations)

uniq_pop = list(set(taxa_groupings['Population']))
print(str(len(uniq_pop))+" Unique Holdout Groups.")
taxa_groupings['Holdout'] = None
for i in range(len(uniq_pop)):
    mask = (taxa_groupings['Population'] == uniq_pop[i])
    taxa_groupings.loc[mask, 'Holdout'] = i

taxa_groupings


# In[ ]:


Holdout_Int = 0
print("Holding out: "+uniq_pop[Holdout_Int])

mask = (taxa_groupings['Holdout'] == Holdout_Int)
train_idxs = list(taxa_groupings.loc[~mask, ].index)
test_idxs = list(taxa_groupings.loc[mask, ].index)


# In[ ]:


y1 = data['leaf_length']
y2 = data['leaf_width']
y3 = data['upper_leaf_angle']
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)


# ### Retrieve xs
# Can we hold all the xs in memory? A ballpark estimate has the full marker dataset as 4.5 Gb. so let's try it!
# 

# In[ ]:


# Non-Hilbert Version
save_path = '../models/10_TianEtAl2011/markers/'
xs = np.zeros(shape = (len(y1), 943455, 4))

failed_idxs = []

for i in tqdm(range(len(y1))):
    save_file_path = save_path+'m'+str(i)+'.npz'
    if os.path.exists(save_file_path):
        xs[i, :, :] = np.load(save_file_path)['arr_0']
    else:
        failed_idxs += [i]
if failed_idxs != []:
    print(str(len(failed_idxs))+' indexes could not be retrieved. Examine `failed_idxs` for more information.')


# In[ ]:


# # Hilbert version
# save_path = '../models/'+nb_name+'/hilbert/'
# xs = np.zeros(shape = (len(y1), 1024, 1024, 4))

# failed_idxs = []

# for i in tqdm(range(len(y1))):
#     save_file_path = save_path+'h'+str(i)+'.npy'
#     if os.path.exists(save_file_path):
#         xs[i, :, :, :] = np.load(save_file_path)
#     else:
#         failed_idxs += [i]
# if failed_idxs != []:
#     print(str(len(failed_idxs))+' indexes could not be retrieved. Examine `failed_idxs` for more information.')


# ### Scale data

# In[ ]:


def calc_cs(x): return [np.mean(x, axis = 0), np.std(x, axis = 0)]

def apply_cs(xs, cs_dict_entry): return ((xs - cs_dict_entry[0]) / cs_dict_entry[0])

scale_dict = {
    'y1':calc_cs(y1[train_idxs]),
    'y2':calc_cs(y2[train_idxs]),
    'y3':calc_cs(y3[train_idxs])
}


# In[ ]:


y1 = apply_cs(y1, scale_dict['y1'])
y2 = apply_cs(y2, scale_dict['y2'])
y3 = apply_cs(y3, scale_dict['y3'])


# In[ ]:


# Running the below seems to crash the session.

# Need to process the below without crashing the session.
# - Cycle data on and off gpu
#     - http://localhost:8895/notebooks/GenomeExplore/notebooks/snps_modeling.ipynb
# - Premake matricies and only load in np arrays
# - Possibly *read* in data from disk. Look at image processing for ideas.


# In[ ]:


# y1_train = torch.from_numpy(y1[train_idxs])[:, None]#.to(device).float()
# y2_train = torch.from_numpy(y2[train_idxs])[:, None]#.to(device).float()
# y3_train = torch.from_numpy(y3[train_idxs])[:, None]#.to(device).float()
# xs_train = torch.from_numpy(xs[train_idxs])#.to(device).float()

# y1_test = torch.from_numpy(y1[test_idxs])[:, None]#.to(device).float()
# y2_test = torch.from_numpy(y2[test_idxs])[:, None]#.to(device).float()
# y3_test = torch.from_numpy(y3[test_idxs])[:, None]#.to(device).float()
# xs_test = torch.from_numpy(xs[test_idxs])#.to(device).float()


# In[ ]:





# In[ ]:


# class CustomDataset(Dataset):
#     def __init__(self, y1, y2, y3, xs, transform = None, target_transform = None):
#         self.y1 = y1
#         self.y2 = y2
#         self.y3 = y3
#         self.xs = xs
#         self.transform = transform
#         self.target_transform = target_transform    
    
#     def __len__(self):
#         return len(self.y1)
    
#     def __getitem__(self, idx):
#         y1_idx = self.y1[idx]
#         y2_idx = self.y2[idx]
#         y3_idx = self.y3[idx]
#         xs_idx = self.xs[idx]
        
#         if self.transform:
#             xs_idx = self.transform(xs_idx)
            
#         if self.target_transform:
#             y1_idx = self.transform(y1_idx)
#             y2_idx = self.transform(y2_idx)
#             y3_idx = self.transform(y3_idx)
#         return xs_idx, y1_idx, y2_idx, y3_idx


# In[ ]:


# training_dataloader = DataLoader(
#     CustomDataset(
#         y1 = y1_train,
#         y2 = y2_train,
#         y3 = y3_train,
#         xs = xs_train
#     ), 
#     batch_size = 64, 
#     shuffle = True)

# testing_dataloader = DataLoader(
#     CustomDataset(
#         y1 = y1_test,
#         y2 = y2_test,
#         y3 = y3_test,
#         xs = xs_test
#     ), 
#     batch_size = 64, 
#     shuffle = True)

# xs.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# data = pd.read_table('../ext_data/zma/panzea/phenotypes/Buckler_etal_2009_Science_flowering_time_data-090807/markergenotypes062508.txt', skiprows=1
#                     ).reset_index().rename(columns = {'index': 'Geno_Code'})
# data


# In[ ]:


# px.scatter_matrix(data.loc[:, ['days2anthesis', 'days2silk', 'asi']])


# In[ ]:


# d2a = np.array(data['days2anthesis'])
# d2s = np.array(data['days2silk'])
# asi = np.array(data['asi'])


# In[ ]:


# xs = np.array(data.drop(columns = ['days2anthesis', 'days2silk', 'asi', 'pop', 'Geno_Code']))

# n_obs = xs.shape[0]

# np_seed = 9070707
# rng = np.random.default_rng(np_seed)  # can be called without a seed

# test_pr = 0.2

# test_n = round(n_obs*test_pr)
# idxs = np.linspace(0, n_obs-1, num = n_obs).astype(int)
# rng.shuffle(idxs)

# test_idxs = idxs[0:test_n]
# train_idxs = idxs[test_n:-1]


# ## Make tensors

# In[ ]:


# y1_train = torch.from_numpy(y1[train_idxs]).to(device).float()[:, None]
# y2_train = torch.from_numpy(y2[train_idxs]).to(device).float()[:, None]
# y3_train = torch.from_numpy(y3[train_idxs]).to(device).float()[:, None]
# xs_train = torch.from_numpy(xs[train_idxs]).to(device).float()

# y1_test = torch.from_numpy(y1[test_idxs]).to(device).float()[:, None]
# y2_test = torch.from_numpy(y2[test_idxs]).to(device).float()[:, None]
# y3_test = torch.from_numpy(y3[test_idxs]).to(device).float()[:, None]
# xs_test = torch.from_numpy(xs[test_idxs]).to(device).float()


# In[ ]:


# class CustomDataset(Dataset):
#     def __init__(self, y1, y2, y3, xs, transform = None, target_transform = None):
#         self.y1 = y1
#         self.y2 = y2
#         self.y3 = y3
#         self.xs = xs
#         self.transform = transform
#         self.target_transform = target_transform    
    
#     def __len__(self):
#         return len(self.y1)
    
#     def __getitem__(self, idx):
#         y1_idx = self.y1[idx]
#         y2_idx = self.y2[idx]
#         y3_idx = self.y3[idx]
#         xs_idx = self.xs[idx]
        
#         if self.transform:
#             xs_idx = self.transform(xs_idx)
            
#         if self.target_transform:
#             y1_idx = self.transform(y1_idx)
#             y2_idx = self.transform(y2_idx)
#             y3_idx = self.transform(y3_idx)
#         return xs_idx, y1_idx, y2_idx, y3_idx


# In[ ]:


# training_dataloader = DataLoader(
#     CustomDataset(
#         y1 = y1_train,
#         y2 = y2_train,
#         y3 = y3_train,
#         xs = xs_train
#     ), 
#     batch_size = 64, 
#     shuffle = True)

# testing_dataloader = DataLoader(
#     CustomDataset(
#         y1 = y1_test,
#         y2 = y2_test,
#         y3 = y3_test,
#         xs = xs_test
#     ), 
#     batch_size = 64, 
#     shuffle = True)

# xs.shape


# ## Version 1, Predict `y1` (Anthesis)

# In[ ]:


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()    
#         self.x_network = nn.Sequential(
#             nn.Linear(1106, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 1))
        
#     def forward(self, x):
#         x_out = self.x_network(x)
#         return x_out

# model = NeuralNetwork().to(device)
# # print(model)

# xs_i, y1_i, y2_i, y3_i = next(iter(training_dataloader))
# model(xs_i).shape # try prediction on one batch

# def train_loop(dataloader, model, loss_fn, optimizer, silent = False):
#     size = len(dataloader.dataset)
#     for batch, (xs_i, y1_i, y2_i, y3_i) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(xs_i)
#         loss = loss_fn(pred, y1_i) # <----------------------------------------

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(y1_i) # <----------------
#             if not silent:
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                
# def train_error(dataloader, model, loss_fn, silent = False):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     train_loss = 0

#     with torch.no_grad():
#         for xs_i, y1_i, y2_i, y3_i in dataloader:
#             pred = model(xs_i)
#             train_loss += loss_fn(pred, y1_i).item() # <----------------------
            
#     train_loss /= num_batches
#     return(train_loss) 

            
# def test_loop(dataloader, model, loss_fn, silent = False):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss = 0

#     with torch.no_grad():
#         for xs_i, y1_i, y2_i, y3_i in dataloader:
#             pred = model(xs_i)
#             test_loss += loss_fn(pred, y1_i).item() # <-----------------------

#     test_loss /= num_batches
#     if not silent:
#         print(f"Test Error: Avg loss: {test_loss:>8f}")
#     return(test_loss) 


# def train_nn(
#     training_dataloader,
#     testing_dataloader,
#     model,
#     learning_rate = 1e-3,
#     batch_size = 64,
#     epochs = 500
# ):
#     # Initialize the loss function
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#     loss_df = pd.DataFrame([i for i in range(epochs)], columns = ['Epoch'])
#     loss_df['TrainMSE'] = np.nan
#     loss_df['TestMSE']  = np.nan

#     for t in tqdm.tqdm(range(epochs)):
#         # print(f"Epoch {t+1}\n-------------------------------")
#         train_loop(training_dataloader, model, loss_fn, optimizer, silent = True)

#         loss_df.loc[loss_df.index == t, 'TrainMSE'
#                    ] = train_error(training_dataloader, model, loss_fn, silent = True)
        
#         loss_df.loc[loss_df.index == t, 'TestMSE'
#                    ] = test_loop(testing_dataloader, model, loss_fn, silent = True)
        
#     return([model, loss_df])

# model, loss_df = train_nn(
#     training_dataloader,
#     testing_dataloader,
#     model,
#     learning_rate = 1e-3,
#     batch_size = 64,
#     epochs = 500
# )

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TrainMSE,
#                     mode='lines', name='Train'))
# fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TestMSE,
#                     mode='lines', name='Test'))
# fig.show()

# # ! conda install captum -c pytorch -y


# # imports from captum library
# from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
# from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation



# ig = IntegratedGradients(model)
# ig_nt = NoiseTunnel(ig)
# dl = DeepLift(model)
# gs = GradientShap(model)
# fa = FeatureAblation(model)

# ig_attr_test = ig.attribute(xs_test, n_steps=50)
# ig_nt_attr_test = ig_nt.attribute(xs_test)
# dl_attr_test = dl.attribute(xs_test)
# gs_attr_test = gs.attribute(xs_test, xs_train)
# fa_attr_test = fa.attribute(xs_test)

# [e.shape for e in [ig_attr_test,
# ig_nt_attr_test,
# dl_attr_test,
# gs_attr_test,
# fa_attr_test]]

# fig = go.Figure()
# fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
#                          y = ig_nt_attr_test.cpu().detach().numpy().mean(axis=0),
#                          mode='lines', name='Test'))
# fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
#                          y = dl_attr_test.cpu().detach().numpy().mean(axis=0),
#                          mode='lines', name='Test'))
# fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
#                          y = gs_attr_test.cpu().detach().numpy().mean(axis=0),
#                          mode='lines', name='Test'))
# fig.add_trace(go.Scatter(x = np.linspace(0, 1106-1, 1106),
#                          y = fa_attr_test.cpu().detach().numpy().mean(axis=0),
#                          mode='lines', name='Test'))
# fig.show()

# len(dl_attr_test.cpu().detach().numpy().mean(axis = 0))















# ## Version 2, Predict `y1` (Anthesis), `y2` (Silking), and `y3` (ASI)

# Here each model will predict 3 values. The loss function is still mse, but the y tensors are concatenated

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()    
#         self.x_network = nn.Sequential(
#             nn.Linear(1106, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 3))
        
#     def forward(self, x):
#         x_out = self.x_network(x)
#         return x_out

# model = NeuralNetwork().to(device)
# # print(model)

# xs_i, y1_i, y2_i, y3_i = next(iter(training_dataloader))
# model(xs_i).shape # try prediction on one batch





# def train_loop(dataloader, model, loss_fn, optimizer, silent = False):
#     size = len(dataloader.dataset)
#     for batch, (xs_i, y1_i, y2_i, y3_i) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(xs_i)
#         loss = loss_fn(pred, torch.concat([y1_i, y2_i, y3_i], axis = 1)) # <----------------------------------------

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(y1_i) # <----------------
#             if not silent:
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                
# def train_error(dataloader, model, loss_fn, silent = False):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     train_loss = 0

#     with torch.no_grad():
#         for xs_i, y1_i, y2_i, y3_i in dataloader:
#             pred = model(xs_i)
#             train_loss += loss_fn(pred, torch.concat([y1_i, y2_i, y3_i], axis = 1)).item() # <----------------------
            
#     train_loss /= num_batches
#     return(train_loss) 

            
# def test_loop(dataloader, model, loss_fn, silent = False):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss = 0

#     with torch.no_grad():
#         for xs_i, y1_i, y2_i, y3_i in dataloader:
#             pred = model(xs_i)
#             test_loss += loss_fn(pred, torch.concat([y1_i, y2_i, y3_i], axis = 1)).item() # <-----------------------

#     test_loss /= num_batches
#     if not silent:
#         print(f"Test Error: Avg loss: {test_loss:>8f}")
#     return(test_loss) 


# def train_nn(
#     training_dataloader,
#     testing_dataloader,
#     model,
#     learning_rate = 1e-3,
#     batch_size = 64,
#     epochs = 500
# ):
#     # Initialize the loss function
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#     loss_df = pd.DataFrame([i for i in range(epochs)], columns = ['Epoch'])
#     loss_df['TrainMSE'] = np.nan
#     loss_df['TestMSE']  = np.nan

#     for t in tqdm.tqdm(range(epochs)):
#         # print(f"Epoch {t+1}\n-------------------------------")
#         train_loop(training_dataloader, model, loss_fn, optimizer, silent = True)

#         loss_df.loc[loss_df.index == t, 'TrainMSE'
#                    ] = train_error(training_dataloader, model, loss_fn, silent = True)
        
#         loss_df.loc[loss_df.index == t, 'TestMSE'
#                    ] = test_loop(testing_dataloader, model, loss_fn, silent = True)
        
#     return([model, loss_df])

# model, loss_df = train_nn(
#     training_dataloader,
#     testing_dataloader,
#     model,
#     learning_rate = 1e-3,
#     batch_size = 64,
#     epochs = 500
# )

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TrainMSE,
#                     mode='lines', name='Train'))
# fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TestMSE,
#                     mode='lines', name='Test'))
# fig.show()

# model, loss_df = train_nn(
#     training_dataloader,
#     testing_dataloader,
#     model,
#     learning_rate = 1e-3,
#     batch_size = 64,
#     epochs = 5000
# )

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TrainMSE,
#                     mode='lines', name='Train'))
# fig.add_trace(go.Scatter(x=loss_df.Epoch, y=loss_df.TestMSE,
#                     mode='lines', name='Test'))
# fig.show()





# '../ext_data/zma/panzea/phenotypes/'

# # pd.read_table('../ext_data/zma/panzea/phenotypes/traitMatrix_maize282NAM_v15-130212.txt', low_memory = False)

# # pd.read_excel('../ext_data/zma/panzea/phenotypes/traitMatrix_maize282NAM_v15-130212_TraitDescritptions.xlsx')

