#!/usr/bin/env python
# coding: utf-8

# # Retrieve Nucleotide Data 
# 
# > This notebook will likely be broken into at least two

# In[ ]:


#| default_exp dna


# <!-- #| hide
# from nbdev.showdoc import * -->

# In[2]:


import os

import numpy as np
import pandas as pd

import plotly.express as px

# import hilbertcurve
# from hilbertcurve.hilbertcurve import HilbertCurve

    # !conda install openpyxl -y
    # ! conda install h5py -y
# ! conda install hilbertcurve -y
# ! pip install hilbertcurve


# ## Access genome records

# Working with these records proved tricky. Ultimately I need the nucleotide data in a tensor, but after using tassel to save  the data (`ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023`) as a table (along with position list and taxa list) it's too big to easily load (>30Gb). As a work around to easily access specific genomes, I split the table into a separate file for the header and each genome so that these files can be read piecemeal. See the Readme below for more details.

# In[3]:


#| export

def read_txt(path, 
             **kwargs # Intended to allow for explicit 'encoding' to be passed into open the file
            ):
    if 'encoding' in kwargs.keys():
        print(kwargs)
        with open(path, 'r', encoding  = kwargs['encoding']) as f:
            data = f.read()        
    else:    
        with open(path, 'r') as f:
            data = f.read()
            
    return(data)


# In[4]:


#| export

def print_txt(path):
    print(read_txt(path = path))


# In[5]:


AGPv4_path = '../data/zma/panzea/genotypes/GBS/v27/'


# In[6]:


print_txt(path = AGPv4_path+'Readme')


# This last point was completed with the following shell script.

# In[7]:


print_txt(path = AGPv4_path+'rename_all.sh')


# With that done, and with the summary files from tassel (position and taxa), the genomes can be individually loaded as needed.

# In[8]:


# Other than listing the taxa this isn't expected to be of much use for our purposes.
AGPv4_taxa=pd.read_table(AGPv4_path+'ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_TaxaList.txt')
AGPv4_taxa.head()


# In[9]:


# Useful for converting between the physical location and site
AGPv4_site = pd.read_table(AGPv4_path+'ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_PositionList.txt')
AGPv4_site.head()


# Retrieving a genome by taxa name:

# In[10]:


# The genomes are in a folder with an identical name as their source table
table_directory = AGPv4_path+'ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/'
# Note however that the naming is altered to not use ':'
os.listdir(table_directory)[0:3]


# In[11]:


#| export

def taxa_to_filename(taxa = '05-397:250007467'): return(taxa.replace(':', '__'))


# In[12]:


taxa_to_filename(taxa = '05-397:250007467')


# In[ ]:


#| export
def find_AGPv4_genome(
    taxa, # should be the desired taxa or a regex fragment (stopping before the __). E.g. 'B73' or 'B\d+'
    **kwargs # optionally pass in a genome list (this allows for a different path or precomputing if we're finding a lot of genomes)
    ):
    "Search for existing marker sets __"
    if 'genome_files' not in kwargs.keys():
        import os
        genome_files = os.listdir(
    '../data/zma/panzea/genotypes/GBS/v27/ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/')
    else:
        genome_files = kwargs['genome_files']
    import re
    return( [e for e in genome_files if re.match(taxa+'__.+', e)] )


# In[ ]:


#| export
def get_AGPv4( 
    taxa,
    **kwargs 
    ):
    "Retrieve an existing marker set"
    if 'table_directory' not in kwargs.keys():
        table_directory = '../data/zma/panzea/genotypes/GBS/v27/ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/'
    else:
        table_directory = kwargs['table_directory']
        
    with open(table_directory+taxa, 'r') as f:
        data = f.read()    
    data = data.split('\t')
    return(data)


# In[13]:


# def get_AGPv4( 
#     taxa,
#     table_directory = '../data/zma/panzea/genotypes/GBS/v27/ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/'
# ):
#     with open(table_directory+taxa, 'r') as f:
#         data = f.read()    
#     data = data.split('\t')
#     return(data)


# In[14]:


get_AGPv4('05-397__250007467')[0:4]


# In addition to returning a specific taxa, the table's headers can be retieved with "taxa".

# In[15]:


get_AGPv4(taxa = 'taxa')[0:4]


# Converting between site and chromosome/position requires the `AGPv4_site` dataframe. A given record contains the taxa as well as the nucleotides, so with that entry excluded the chromosome / position can be paired up.

# In[16]:


len(get_AGPv4(taxa = 'taxa')), AGPv4_site.shape


# In[17]:


ith_taxa = '05-397:250007467'
res = get_AGPv4(taxa_to_filename(taxa = ith_taxa))   # Retrieve record
temp = AGPv4_site.loc[:, ['Chromosome', 'Position']]  
temp[res[0]] = res[1:]                               # Add Col. with Nucleotides
temp.head()


# ## Look at SNP coverage

# In[18]:


mask = (temp.Chromosome == 1)

temp_pos = temp.loc[mask, ['Position']]


# In[19]:


temp_pos['Shift'] = 0
temp_pos.loc[1: , ['Shift']] = np.array(temp_pos.Position)[:-1]
temp_pos['Diff'] = temp_pos['Position'] - temp_pos['Shift']

temp_pos.loc[0, 'Diff'] = None


# In[20]:


temp_pos


# In[ ]:


# px.histogram(temp_pos, x = 'Diff')


# ## Demonstrate Hilbert Curve

# In[ ]:


# # demonstrating the hilbert curve
# temp = np.linspace(1, 100, num= 50)
# # px.scatter(x = temp, y = [0 for e in range(len(temp))], color = temp)
# px.imshow(temp.reshape((1, temp.shape[0])))


# In[ ]:


# hilbert_curve = HilbertCurve(p = 10, # iterations i.e. hold 4^p positions
#                              n = 2    # dimensions
#                             )
# distances = list(range(len(temp)))

# points = hilbert_curve.points_from_distances(distances)
# # px.line(pd.DataFrame(points, columns = ['i', 'j']), x = 'i', y = 'j')


# In[ ]:


# dim_0 = np.max(np.array(points)[:, 0])+1 # add 1 to account for 0 indexing
# dim_1 = np.max(np.array(points)[:, 1])+1
# temp_mat = np.zeros(shape = [dim_0, dim_1])
# temp_mat[temp_mat == 0] = np.nan         #  empty values being used for visualization

# for i in range(len(temp)):
# #     print(i)
#     temp_mat[points[i][0], points[i][1]] = temp[i]
    
# # temp2 = pd.DataFrame(points, columns = ['i', 'j'])
# # temp2['value'] = temp
# # px.scatter(temp2, x = 'i', y = 'j', color = 'value')

# px.imshow(temp_mat)


# In[ ]:


# # Data represented need not be continuous -- it need only have int positions
# # a sequence or a sequence with gaps can be encoded
# hilbert_curve = HilbertCurve(p = 10, # iterations i.e. hold 4^p positions
#                              n = 2    # dimensions
#                             )


# fake_dists = list(range(len(temp)))
# # Introdude a gap in the sequence
# fake_dists = [e if e>10 else e+5 for e in fake_dists]
# distances = fake_dists

# points = hilbert_curve.points_from_distances(distances)
# dim_0 = np.max(np.array(points)[:, 0])+1 # add 1 to account for 0 indexing
# dim_1 = np.max(np.array(points)[:, 1])+1
# temp_mat = np.zeros(shape = [dim_0, dim_1])
# temp_mat[temp_mat == 0] = np.nan         #  empty values being used for visualization

# for i in range(len(temp)):
# #     print(i)
#     temp_mat[points[i][0], points[i][1]] = temp[i]
# px.imshow(temp_mat)


# ### Hilbert curve for one sequence

# In[ ]:


# temp_pos['Present'] = 1


# In[ ]:


# temp_pos.shape[0]


# In[ ]:





# In[ ]:


# def calc_needed_hilbert_p(n_needed = 1048576,
#                           max_p = 20):
#     out = None
#     for i in range(1, max_p):
#         if 4**i > n_needed:
#             out = i
#             break
#     return(out)

# calc_needed_hilbert_p(n_needed=147150)


# In[ ]:


# temp_pos['Position']


# In[ ]:


# # Data represented need not be continuous -- it need only have int positions
# # a sequence or a sequence with gaps can be encoded
# hilbert_curve = HilbertCurve(p = 10, # iterations i.e. hold 4^p positions
#                              n = 2    # dimensions
#                             )


# fake_dists = list(range(len(temp)))
# # Introdude a gap in the sequence
# fake_dists = [e if e>10 else e+5 for e in fake_dists]
# distances = fake_dists

# points = hilbert_curve.points_from_distances(distances)
# dim_0 = np.max(np.array(points)[:, 0])+1 # add 1 to account for 0 indexing
# dim_1 = np.max(np.array(points)[:, 1])+1
# temp_mat = np.zeros(shape = [dim_0, dim_1])
# temp_mat[temp_mat == 0] = np.nan         #  empty values being used for visualization

# for i in range(len(temp)):
# #     print(i)
#     temp_mat[points[i][0], points[i][1]] = temp[i]
# px.imshow(temp_mat)


# In[ ]:


#| hide
import nbdev; nbdev.nbdev_export()

