#!/usr/bin/env python
# coding: utf-8

# # Retrieve genomic data for phenotypes
# 
# > This notebook will likely be broken into at least two

# In[ ]:


#| default_exp core


# <!-- #| hide
# from nbdev.showdoc import * -->

# In[ ]:


import os

import numpy as np
import pandas as pd

import plotly.express as px

# import hilbertcurve
# from hilbertcurve.hilbertcurve import HilbertCurve


import dlgwas
from dlgwas.dna import *

# ! conda install openpyxl -y
# ! conda install hilbertcurve -y
# ! pip install hilbertcurve


# ## Load phenotypic data to explore

# For this I'm using data referenced in *Wallace et al 2014* which is available through panzea. This study refers to data from 9 studies (including itself) as a source of phenotypes for the NAM data. This combination of a large set of published GWAS hits, phenotypes, and many rils makes it ideal for use here. 

# This file contains results I can use to check if my approaches are producing similar hits.

# In[ ]:


Wallace_etal_2014_PLoSGenet_GWAS_hits = pd.read_table('../ext_data/zma/panzea/GWASResults/Wallace_etal_2014_PLoSGenet_GWAS_hits-150112.txt')
Wallace_etal_2014_PLoSGenet_GWAS_hits.head()


# This file on I think matches *Buckler et al 2009*.

# In[ ]:


temp = pd.read_excel('../ext_data/zma/panzea/GWASResults/JointGLMModels090324QTLLocations.xlsx', 
                     skiprows=1
                    ).rename(columns = {
    'Unnamed: 1': 'ASI', 
    'Unnamed: 2': 'Days to Anthesis', 
    'Unnamed: 3': 'Days to Silk'
})
temp = temp.loc[temp.index == 0]
temp.head()


# In[ ]:


temp = pd.read_excel('../ext_data/zma/panzea/GWASResults/JointGLMModels090324QTLLocations.xlsx', 
                     skiprows=4
                    ).rename(columns = {
    'Unnamed: 1': 'ASI', 
    'Unnamed: 2': 'Days to Anthesis', 
    'Unnamed: 3': 'Days to Silk'
})
temp.head()


# In[ ]:





# In[ ]:


# pull in some of the data that Wallace et al 2014 point to:

buckler_2009_path = '../ext_data/zma/panzea/phenotypes/Buckler_etal_2009_Science_flowering_time_data-090807/'
os.listdir(buckler_2009_path)


# In[ ]:


nam_dts = pd.read_table(buckler_2009_path+'NAM_DaysToSilk.txt', encoding="ISO-8859-1")
nam_dts_taxa = list(nam_dts.loc[:, 'accename'].drop_duplicates())


# In[ ]:


# Look for the right taxa


# In[ ]:


genome_files.sort()
genome_files[0:10]


# In[ ]:


# I think we need to match everything before the __
def find_AGPv4_genome(
    taxa, # should be the desired taxa or a regex fragment (stopping before the __). E.g. 'B73' or 'B\d+'
    **kwargs # optionally pass in a genome list (this allows for a different path or precomputing if we're finding a lot of genomes)
    ):
    if 'genome_files' not in kwargs.keys():
        import os
        genome_files = os.listdir(
    '../data/zma/panzea/genotypes/GBS/v27/ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/')
    else:
        genome_files = kwargs['genome_files']
    import re
    return( [e for e in genome_files if re.match(taxa+'__.+', e)] )


# In[ ]:


genome_files = os.listdir(
    '../data/zma/panzea/genotypes/GBS/v27/ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/')


# In[ ]:


possible_matches = [{'taxa': e,
                     'matches': find_AGPv4_genome(
                         taxa = e,
                         genome_files = genome_files
)} for e in nam_dts_taxa]


# In[ ]:


# how many have more than one match?
len(
[[len(e['matches']), e] for e in possible_matches if len(e['matches']) != 1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'Z018E0021'




# ith_taxa = '05-397:250007467'
# res = get_AGPv4(taxa_to_filename(taxa = ith_taxa))   # Retrieve record
# temp = AGPv4_site.loc[:, ['Chromosome', 'Position']]  
# temp[res[0]] = res[1:]                               # Add Col. with Nucleotides
# temp.head()


# In[ ]:


pd.read_table(buckler_2009_path+'NAMSum0607FloweringTraitBLUPsAcross8Envs.txt', encoding="ISO-8859-1")


# In[ ]:


pd.read_table(buckler_2009_path+'NAM_TasselingDate.txt', encoding="ISO-8859-1")


# In[ ]:


# pd.read_table(buckler_2009_path+'markergenotypes062508.txt', encoding="ISO-8859-1")


# In[ ]:


pd.read_table(buckler_2009_path+'NAMSum0607FloweringTraitBLUPsAcross8Envs.txt', encoding="ISO-8859-1")


# In[ ]:


# os.listdir('../data/zma/panzea/genotypes/GBS/v27/ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


AGPv4_path = '../data/zma/panzea/genotypes/GBS/v27/'


# In[ ]:


# Other than listing the taxa this isn't expected to be of much use for our purposes.
AGPv4_taxa=pd.read_table(AGPv4_path+'ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_TaxaList.txt')
AGPv4_taxa.head()


# In[ ]:


# Useful for converting between the physical location and site
AGPv4_site = pd.read_table(AGPv4_path+'ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_PositionList.txt')
AGPv4_site.head()


# Retrieving a genome by taxa name:

# In[ ]:


# The genomes are in a folder with an identical name as their source table
table_directory = AGPv4_path+'ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023_Table/'
# Note however that the naming is altered to not use ':'
os.listdir(table_directory)[0:3]


# In[ ]:


taxa_to_filename(taxa = '05-397:250007467')


# In[ ]:


def get_AGPv4(taxa):
    with open(table_directory+taxa, 'r') as f:
        data = f.read()    
    data = data.split('\t')
    return(data)


# In[ ]:


get_AGPv4('05-397__250007467')[0:4]


# In addition to returning a specific taxa, the table's headers can be retieved with "taxa".

# In[ ]:


get_AGPv4(taxa = 'taxa')[0:4]


# Converting between site and chromosome/position requires the `AGPv4_site` dataframe. A given record contains the taxa as well as the nucleotides, so with that entry excluded the chromosome / position can be paired up.

# In[ ]:


len(get_AGPv4(taxa = 'taxa')), AGPv4_site.shape


# In[ ]:


ith_taxa = '05-397:250007467'
res = get_AGPv4(taxa_to_filename(taxa = ith_taxa))   # Retrieve record
temp = AGPv4_site.loc[:, ['Chromosome', 'Position']]  
temp[res[0]] = res[1:]                               # Add Col. with Nucleotides
temp.head()


# ## Look at SNP coverage

# In[ ]:


mask = (temp.Chromosome == 1)

temp_pos = temp.loc[mask, ['Position']]


# In[ ]:





# In[ ]:


temp_pos['Shift'] = 0
temp_pos.loc[1: , ['Shift']] = np.array(temp_pos.Position)[:-1]
temp_pos['Diff'] = temp_pos['Position'] - temp_pos['Shift']

temp_pos.loc[0, 'Diff'] = None


# In[ ]:


temp_pos


# In[ ]:


# px.histogram(temp_pos, x = 'Diff')


# In[ ]:





# In[ ]:





# In[ ]:





# ## Demonstrate Hilbert Curve

# In[ ]:


# demonstrating the hilbert curve
temp = np.linspace(1, 100, num= 50)
# px.scatter(x = temp, y = [0 for e in range(len(temp))], color = temp)
px.imshow(temp.reshape((1, temp.shape[0])))


# In[ ]:


hilbert_curve = HilbertCurve(p = 10, # iterations i.e. hold 4^p positions
                             n = 2    # dimensions
                            )
distances = list(range(len(temp)))

points = hilbert_curve.points_from_distances(distances)
# px.line(pd.DataFrame(points, columns = ['i', 'j']), x = 'i', y = 'j')


# In[ ]:


dim_0 = np.max(np.array(points)[:, 0])+1 # add 1 to account for 0 indexing
dim_1 = np.max(np.array(points)[:, 1])+1
temp_mat = np.zeros(shape = [dim_0, dim_1])
temp_mat[temp_mat == 0] = np.nan         #  empty values being used for visualization

for i in range(len(temp)):
#     print(i)
    temp_mat[points[i][0], points[i][1]] = temp[i]
    
# temp2 = pd.DataFrame(points, columns = ['i', 'j'])
# temp2['value'] = temp
# px.scatter(temp2, x = 'i', y = 'j', color = 'value')

px.imshow(temp_mat)


# In[ ]:


# Data represented need not be continuous -- it need only have int positions
# a sequence or a sequence with gaps can be encoded
hilbert_curve = HilbertCurve(p = 10, # iterations i.e. hold 4^p positions
                             n = 2    # dimensions
                            )


fake_dists = list(range(len(temp)))
# Introdude a gap in the sequence
fake_dists = [e if e>10 else e+5 for e in fake_dists]
distances = fake_dists

points = hilbert_curve.points_from_distances(distances)
dim_0 = np.max(np.array(points)[:, 0])+1 # add 1 to account for 0 indexing
dim_1 = np.max(np.array(points)[:, 1])+1
temp_mat = np.zeros(shape = [dim_0, dim_1])
temp_mat[temp_mat == 0] = np.nan         #  empty values being used for visualization

for i in range(len(temp)):
#     print(i)
    temp_mat[points[i][0], points[i][1]] = temp[i]
px.imshow(temp_mat)


# ### Hilbert curve for one sequence

# In[ ]:


temp_pos['Present'] = 1


# In[ ]:


temp_pos.shape[0]


# In[ ]:





# In[ ]:


def calc_needed_hilbert_p(n_needed = 1048576,
                          max_p = 20):
    out = None
    for i in range(1, max_p):
        if 4**i > n_needed:
            out = i
            break
    return(out)

calc_needed_hilbert_p(n_needed=147150)


# In[ ]:


temp_pos['Position']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Data represented need not be continuous -- it need only have int positions
# a sequence or a sequence with gaps can be encoded
hilbert_curve = HilbertCurve(p = 10, # iterations i.e. hold 4^p positions
                             n = 2    # dimensions
                            )


fake_dists = list(range(len(temp)))
# Introdude a gap in the sequence
fake_dists = [e if e>10 else e+5 for e in fake_dists]
distances = fake_dists

points = hilbert_curve.points_from_distances(distances)
dim_0 = np.max(np.array(points)[:, 0])+1 # add 1 to account for 0 indexing
dim_1 = np.max(np.array(points)[:, 1])+1
temp_mat = np.zeros(shape = [dim_0, dim_1])
temp_mat[temp_mat == 0] = np.nan         #  empty values being used for visualization

for i in range(len(temp)):
#     print(i)
    temp_mat[points[i][0], points[i][1]] = temp[i]
px.imshow(temp_mat)


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





# In[ ]:





# In[ ]:




