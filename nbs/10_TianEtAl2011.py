#!/usr/bin/env python
# coding: utf-8

# # Tian et al. 2011
# 
# > This file aims to reproduce the findings of *Tian et al. 2011*, "Genome-wide association study of leaf architecture in the
# maize nested association mapping population".

# <!-- It used data from panzea
# - Phenotypic data panzea\phenotypes\Buckler_etal_2009_Science_flowering_time_data-090807\
# - Genotypic Data panzea\genotypes\GBS\v27\ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023.vcf.gz
# - Genomic Data ...  -->

# In[6]:


# !cd ../ && pip install -e '.[dev]'


# In[1]:


use_gpu_num = 1

import os
import pandas as pd
import numpy as np
import re

# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch import nn

# TODO fixme

# device = "cuda" if torch.cuda.is_available() else "cpu"
# if use_gpu_num in [0, 1]: 
#     torch.cuda.set_device(use_gpu_num)
# print(f"Using {device} device")

import tqdm
from tqdm import tqdm

import plotly.graph_objects as go
import plotly.express as px

# [e for e in os.listdir() if re.match(".+\\.txt", e)]


# In[2]:


import dlgwas
from dlgwas.dna import *

from dlgwas.kegg import ensure_dir_path_exists
from dlgwas.kegg import get_cached_result
from dlgwas.kegg import put_cached_result


# In[ ]:


# set up directory for notebook artifacts
nb_name = '10_TianEtAl2011'
ensure_dir_path_exists(dir_path = '../models/'+nb_name)
ensure_dir_path_exists(dir_path = '../reports/'+nb_name)


# ##  Load Data

# In[ ]:


with open('../ext_data/zma/panzea/phenotypes/Tian_etal_2011_NatGen_leaf_pheno_data-110221/Tian_etal_2011_NatGen_readme.txt', 
          'r') as f:
    dat = f.read()
print(dat)


# In[ ]:


data = pd.read_excel('../ext_data/zma/panzea/phenotypes/Tian_etal_2011_NatGen_leaf_pheno_data-110221/Tian_etal2011NatGenet.leaf_trait_phenotype.xlsx')
data


# In[ ]:





# ## Find Marker data to use along with the phenotypic data here
# 

# In[ ]:


samples = list(set(data['sample']))


# In[ ]:





# In[ ]:


# this can take a while to calculate so it's worth cacheing
save_path = '../models/'+nb_name+'/samples_and_matches.pkl'

samples_and_matches = get_cached_result(save_path=save_path)

if None == samples_and_matches:
    samples_and_matches = [{
        'sample': sample,
        'matches': find_AGPv4(taxa = sample) } for sample in tqdm.tqdm(samples)]

    put_cached_result(
        save_path = save_path,
        save_obj = samples_and_matches
    )


# In[ ]:





# In[ ]:





# In[ ]:


#TODO: Some of these samples have multiple possible matches. 
# For the time being I'm usign the first one.
[e for e in samples_and_matches if len(e['matches']) > 1][0:10]


# In[ ]:


samples_one_match = [e for e in samples_and_matches if len(e['matches']) == 1]

print("Warning: "+str(len(samples_and_matches)-len(samples_one_match)
    )+" samples ("+str(round(100*((len(samples_and_matches)-len(samples_one_match))/len(samples_and_matches))) 
    )+"%) have zero matches or more than one match in AGPv4. The first is being used.")


# In[ ]:





# ## Filter data to only those with unambiguous genotypes

# In[ ]:


original_rows = data.shape[0]

# mask to restrict to only those samples with one or more GBS marker set in AGPv4
mask = [True if e in [e1['sample'] for e1 in 
                      [e for e in samples_and_matches if len(e['matches']) >= 1]
              ] else False for e in data['sample'] ]
data = data.loc[mask,].reset_index().drop(columns = 'index')
print(str(original_rows - data.shape[0])+' rows dropped.')


# In[ ]:


# ys = data.loc[:, ['leaf_length', 'leaf_width', 'upper_leaf_angle']]

# geno_sample = data.loc[:, 'sample']
# pop_sample  = data.loc[:, 'pop']


# In[ ]:


# Sample withing Group
data.assign(n = 1).groupby(['pop', 'sample']).agg(nsum = ('n', np.mean)).reset_index().sort_values('nsum')


# In[ ]:


# But sample is also usable as a uid
data.assign(n = 1).groupby(['sample']).agg(nsum = ('n', np.mean)).reset_index().sort_values('nsum')


# ## Retrieve Marker data

# ### Initial approach: Convert lists to arrays

# The original approach to converting marker lists to np arrays was straighforward but required looping over the marker lists such that it takes about a second per sample. This puts the conversion at ~1h. 
# ```
# n_samples = len(list(data['sample']))
# 
# markers = np.zeros(shape = (
#     n_samples, 
#     len(get_AGPv4(taxa = 'taxa'))-1, # don't include taxa
#     4
# ))
# 
# for i in tqdm(range(n_samples)):
#     search_taxa = data['sample'][i]
#     markers[i, :, :] = list_to_ACGT(
#             in_seq = get_AGPv4(
#                 taxa = find_AGPv4(
#                     taxa = search_taxa)[0] 
#             )[1:]
#         )
# ```
# 

# In[ ]:


n_samples = len(list(data['sample']))

markers = np.zeros(shape = (
    n_samples, 
    len(get_AGPv4(taxa = 'taxa'))-1, # don't include taxa
    4
))



import time
times_list = []

for i in tqdm(range(2)):
    times = []
    times += [time.time()]
    search_taxa = data['sample'][i]
    times += [time.time()] #---- 0
    aa = find_AGPv4(taxa = search_taxa)[0]
    times += [time.time()] #---- 1
    bb = get_AGPv4(taxa =  aa)[1:]
    times += [time.time()] #---- 2
    cc = list_to_ACGT(in_seq = bb) # <-- This is where almost all of the time is coming from
    times += [time.time()] #---- 3
    markers[i, :, :] = cc
    times += [time.time()] #---- 4
    times_list += [times]


# In[ ]:


diff_times = []
for times in times_list:
    diff_times += [ [times[i+1]-times[i] for i in range(len(times)-1)]]


# In[ ]:


px.imshow(np.asarray(diff_times))


# ### Improved approach: convert dataframes to arrays
# In addition to caching the data, alternate approaches are available. Here I tested a function that would work off of a dataframe rather than a list. 

# In[ ]:


# What about a dataframe based version of list_to_ACGT?

def df_to_ACGT(
    in_df, # This should be a dataframe containing samples and SNPs
    sample_axis, # this is an int with the axis of the samples. If samples are not in the 0th axis they will be swapped and returned there.
    progress = False,
    silent = False
):
    # Note! in_df may have samples second! if so then.
    irows, jcols = in_df.shape

    # Convert IUPAC codes into pr ACGT -------------------------------------------
    encode_dict = {
        #     https://www.bioinformatics.org/sms/iupac.html
        #     A     C     G     T
        'A': [1,    0,    0,    0   ],
        'C': [0,    1,    0,    0   ],
        'G': [0,    0,    1,    0   ],
        'T': [0,    0,    0,    1   ],
        'K': [0,    0,    0.5,  0.5 ],
        'M': [0.5,  0.5,  0,    0   ],
        'N': [0.25, 0.25, 0.25, 0.25],
        'R': [0.5,  0,    0.5,  0   ],
        'S': [0,    0.5,  0.5,  0   ],
        'W': [0.5,  0,    0,    0.5 ],
        'Y': [0,    0.5,  0,    0.5 ],
        #     Other values (assumed empty)
        #     A     C     G     T
    #      '': [0,    0,    0,    0   ],
    #     '-': [0,    0,    0,    0   ],
    #     '0': [0,    0,    0,    0   ],
    }

    # fix newline in last row
    for j in range(jcols):
        in_df[j] = in_df[j].str.replace('\n', '')

    not_in_dict = [e for e in set(in_df[j]) if e not in list(encode_dict.keys())]

    if not_in_dict != []:
        if silent != True:
            print("Waring: The following are not in the encoding dictionary and will be set as missing.\n"+str(not_in_dict))

    # output matrix
    GMat = np.zeros(shape = [irows,
                             jcols, 
                             4])
    # convert all nucleotides to probabilities
    if progress == True:
        for nucleotide in tqdm(encode_dict.keys()):
            mask = (in_df == nucleotide)
            GMat[mask, :] = encode_dict[nucleotide] 
    else:
        for nucleotide in encode_dict.keys():
            mask = (in_df == nucleotide)
            GMat[mask, :] = encode_dict[nucleotide] 

    # if needed rotate to have desired shape    
    if sample_axis != 0:
        GMat = np.swapaxes(GMat, 0, sample_axis)

    return(GMat)


# Confirm that these are equivalent:

# In[ ]:


search_taxa = data['sample'][i]
aa = find_AGPv4(taxa = search_taxa)[0]
bb = get_AGPv4(taxa =  aa)[1:]
cc = list_to_ACGT(in_seq = bb) 

dd = df_to_ACGT(
    in_df = pd.DataFrame(bb),# This should be a dataframe containing samples and SNPs
    sample_axis = 0, # this is an int with the axis of the samples. If samples are not in the 0th axis they will be swapped and returned there.
    progress = False
)

cc.shape ==  dd.squeeze().shape


# Check if this is faster. Previous version takes ~1sec/iter.

# In[ ]:


j = 2

vals = pd.concat([pd.DataFrame(get_AGPv4(taxa = find_AGPv4(taxa = data['sample'][i])[0])[1:]
                       ) for i in tqdm(range(j))], axis = 1)
vals.columns = [i for i in range(j)]

vals.shape

vals = df_to_ACGT(
    in_df = vals,# This should be a dataframe containing samples and SNPs
    sample_axis = 1# this is an int with the axis of the samples. If samples are not in the 0th axis they will be swapped and returned there.
)


# In[ ]:





# This is slow as well. Still an improvement over but not an impressive one.

# In[ ]:


# save_path = '../models/'+nb_name+'/markers/'

# for e in os.listdir(save_path):
#     np.savez_compressed(save_path+e[0:-1]+'z',
#                         np.load(save_path+e))

# # np.savez_compressed('../models/'+nb_name+'/markers/test_zip.npz', vals[0, :, :])


# In[ ]:


# numpy.savez_compressed


# In[ ]:


# Loading and processing is the time consuming part. 
save_path = '../models/'+nb_name+'/markers/'
ensure_dir_path_exists(save_path)

n_samples = len(list(data['sample']))

for i in  tqdm(range(n_samples)):
    save_file_path = save_path+'m'+str(i)+'.npz'
    if os.path.exists(save_file_path) == False:
        markers = pd.DataFrame(get_AGPv4(taxa = find_AGPv4(taxa = data['sample'][i])[0])[1:])    

        markers = df_to_ACGT(
            in_df = markers,# This should be a dataframe containing samples and SNPs
            sample_axis = 1,# this is an int with the axis of the samples. If samples are not in the 0th axis they will be swapped and returned there.
            silent = True
        )

        np.savez_compressed(save_file_path, markers)
# 4683/4683 [59:54<00:00,  1.30it/s]


# ### Apply Hilbert Transform

# In[ ]:


# This should be done on as many samples as possible at one time. 
# Transforming 1 sample and transforming 11 each take 7 and 7.74 seconds.
# markers_hilbert = np_3d_to_hilbert(
#     in_seq = np.load('../models/10_TianEtAl2011/markers/m0.npy')
# )


# In[ ]:


# load_path = '../models/'+nb_name+'/markers/'
# save_path = '../models/'+nb_name+'/hilbert/'
# ensure_dir_path_exists(save_path)

# cached_markers = os.listdir(load_path)
# cached_hilbert_markers = os.listdir(save_path)
# # only process markers that haven't been transformed already
# cached_markers = [ee for ee in cached_markers if ee not in ['m'+e[1:] for e in cached_hilbert_markers]]

# # if need be, this can always be chunked
# # process these as a large array for speed.
# print('Applying Hilbert Transformation.')
# markers_hilbert = np_3d_to_hilbert(
#     in_seq = np.concatenate([np.load(load_path+e) for e in cached_markers])
# )

# # then save each sample separately so they can be loaded as needed.
# print('Saving.')
# for i in tqdm(range(len(cached_markers))):
#     np.save(save_path+'h'+cached_markers[i][1:], markers_hilbert[i])


# In[ ]:


chunk_size = 500
import math

load_path = '../models/'+nb_name+'/markers/'
save_path = '../models/'+nb_name+'/hilbert/'
ensure_dir_path_exists(save_path)

cached_markers = os.listdir(load_path)
cached_hilbert_markers = os.listdir(save_path)
# only process markers that haven't been transformed already
cached_markers = [ee for ee in cached_markers if ee not in ['m'+e[1:] for e in cached_hilbert_markers]]

n_chunks = math.ceil(len(cached_markers)/chunk_size)
for ith_chunk in range(n_chunks):

    print(str(ith_chunk)+'/'+str(n_chunks))
    chunk_start = ith_chunk*chunk_size
    chunk_stop = min((ith_chunk+1)*chunk_size, 
                     len(cached_markers)+1 ) # Last chunk may not be evenly divisible


    cached_marker_slice = cached_markers[chunk_start:chunk_stop]
    # if need be, this can always be chunked. 
    # process these as a large array for speed.
    print('Applying Hilbert Transformation.')

    markers_hilbert = np_3d_to_hilbert(
        #                                                v--- zipped np arrays use position (arr_0, arr_1, ...) if no name is passed
        in_seq = np.concatenate([np.load(load_path+e)['arr_0'] for e in cached_marker_slice ]) 
    )

    #then save each sample separately so they can be loaded as needed.
    print('Saving.')
    for i in tqdm(range(len(cached_marker_slice))):
        np.savez_compressed(save_path+'h'+cached_marker_slice[i][1:-1]+'z', markers_hilbert[i])    


# In[ ]:





# In[ ]:





# In[ ]:





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


# In[ ]:


# fig = px.treemap(taxa_groupings.loc[:, ['Population', 'sample']], 
#                  path=[px.Constant("All"), 'Population', 'sample'])
# # fig.update_traces(root_color="lightgrey")
# fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
# fig.show()


# In[ ]:





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





# In[ ]:


Holdout_Int = 0
print("Holding out: "+uniq_pop[Holdout_Int])

mask = (taxa_groupings['Holdout'] == Holdout_Int)
train_idxs = list(taxa_groupings.loc[~mask, ].index)
test_idxs = list(taxa_groupings.loc[mask, ].index)


# In[ ]:





# In[ ]:


y1 = data['leaf_length']
y2 = data['leaf_width']
y3 = data['upper_leaf_angle']


# ### Retrieve xs
# Can we hold all the xs in memory? A ballpark estimate has the full marker dataset as 4.5 Gb. so let's try it!
# 

# In[ ]:


# Non-Hilbert Version
save_path = '../models/'+nb_name+'/markers/'
xs = np.zeros(shape = (len(y1), 943455, 4))

failed_idxs = []

for i in tqdm(range(len(y1))):
    save_file_path = save_path+'m'+str(i)+'.npy'
    if os.path.exists(save_file_path):
        xs[i, :, :] = np.load(save_file_path)
    else:
        failed_idxs += [i]
if failed_idxs != []:
    print(str(len(failed_idxs))+' indexes could not be retrieved. Examine `failed_idxs` for more information.')


# In[ ]:


# Hilbert version
save_path = '../models/'+nb_name+'/hilbert/'
xs = np.zeros(shape = (len(y1), 1024, 1024, 4))

failed_idxs = []

for i in tqdm(range(len(y1))):
    save_file_path = save_path+'h'+str(i)+'.npy'
    if os.path.exists(save_file_path):
        xs[i, :, :, :] = np.load(save_file_path)
    else:
        failed_idxs += [i]
if failed_idxs != []:
    print(str(len(failed_idxs))+' indexes could not be retrieved. Examine `failed_idxs` for more information.')


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


# y1_train = torch.from_numpy(y1[train_idxs]).to(device).float()[:, None]
# y2_train = torch.from_numpy(y2[train_idxs]).to(device).float()[:, None]
# y3_train = torch.from_numpy(y3[train_idxs]).to(device).float()[:, None]
# xs_train = torch.from_numpy(xs[train_idxs]).to(device).float()

# y1_test = torch.from_numpy(y1[test_idxs]).to(device).float()[:, None]
# y2_test = torch.from_numpy(y2[test_idxs]).to(device).float()[:, None]
# y3_test = torch.from_numpy(y3[test_idxs]).to(device).float()[:, None]
# xs_test = torch.from_numpy(xs[test_idxs]).to(device).float()


# # No need to cs xs 


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

