from torch_geometric.data import InMemoryDataset

import numpy as np
import networkx as nx
from nilearn.connectome import ConnectivityMeasure
import torch
import pickle 
import torch_geometric
import pandas as pd
from tqdm import tqdm


AAL_data_path = '/home/hiren/Apoorv Pandey/Dataset/'
AD_sub_path = '/home/hiren/Apoorv Pandey/Dataset/AD_sub_data.csv'
CN_sub_path = '/home/hiren/Apoorv Pandey/Dataset/CN_sub_data.csv'
AD_sub_data = pd.read_csv(AD_sub_path)
CN_sub_data = pd.read_csv(CN_sub_path)

def get_age(sub_id,sub_data):

    return sub_data[sub_data['Subject ID']==sub_id[:-8]]['Age'].mean()

def get_gender(sub_id,sub_data):
    return list(sub_data[sub_data['Subject ID']==sub_id[:-8]]['Sex'])[0]

def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

AD_dict = load_obj(AAL_data_path + 'AD')
CN_dict = load_obj(AAL_data_path + 'CN')


def get_correlation_matrix(timeseries,msr):
    
    correlation_measure = ConnectivityMeasure(kind = msr)
    correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
    return correlation_matrix

def get_upper_triangular_matrix(matrix):
    
    upp_mat = []
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            upp_mat.append(matrix[i][j])
    return upp_mat

def get_adj_mat(correlation_matrix, th_value_p, th_value_n):
    
    adj_mat = []
    k = 0
    for i in correlation_matrix:
        row = []
        for j in i:
            if j>0:
                if j > th_value_p:
                    row.append(j)
                else:
                    row.append(0)
            else:
                if abs(j) > th_value_n:
                    row.append(abs(j))
                else:
                    row.append(0)
        adj_mat.append(row)
    return adj_mat

def absolutise(matrix):
    adj_matrix = np.array(matrix)
    return np.abs(adj_matrix)

def get_threshold_value(ad_timeseries, cn_timeseries, measure, threshold_percent):
    
    ad_corr_mats = [get_correlation_matrix(ts, measure) for ts in ad_timeseries]
    cn_corr_mats = [get_correlation_matrix(ts, measure) for ts in cn_timeseries]
    ad_upper = [get_upper_triangular_matrix(matrix) for matrix in ad_corr_mats]
    cn_upper = [get_upper_triangular_matrix(matrix) for matrix in cn_corr_mats]

    all_correlation_values = ad_upper + cn_upper
    all_correlation_values = np.array(all_correlation_values).flatten()
    
    all_correlation_values_pos=[]
    all_correlation_values_neg=[]
    for i in all_correlation_values:
        if i==1:
            continue
        elif i>0:
            all_correlation_values_pos.append(i)
        else:  
            all_correlation_values_neg.append(abs(i))

    all_correlation_values_pos = np.array(all_correlation_values_pos)
    all_correlation_values_pos = np.sort(all_correlation_values_pos)[::-1]
    
    all_correlation_values_neg = np.array(all_correlation_values_neg)
    all_correlation_values_neg = np.sort(all_correlation_values_neg)[::-1]

    th_val_index = (len(all_correlation_values)*threshold_percent)//100
    
    return all_correlation_values_pos[int(th_val_index)], all_correlation_values_neg[int(th_val_index)]


import random

class AD_CN_Dataset(InMemoryDataset):
    def __init__(self, root, AD_list,CN_list,thr, transform=None, pre_transform=None):
        self.thr = thr
        self.AD_list = AD_list
        self.CN_list = CN_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graphs = []    
        th_p,th_n = get_threshold_value(list(AD_dict.values()), list(CN_dict.values()), 'correlation', self.thr)
        for sub_id in (self.AD_list):
            
            correlation_matrix = get_correlation_matrix(AD_dict[sub_id], 'correlation')
            adj_mat = get_adj_mat(correlation_matrix, th_p,th_n)
            G = nx.from_numpy_matrix(np.array(adj_mat), create_using=nx.DiGraph)
            data=torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            data.sub_id=sub_id
            
            data.y = torch.tensor([0])
            graphs.append(data)
            
        for sub_id in (self.CN_list):
            correlation_matrix = get_correlation_matrix(CN_dict[sub_id], 'correlation')
            adj_mat = get_adj_mat(correlation_matrix, th_p,th_n)
            G = nx.from_numpy_matrix(np.array(adj_mat), create_using=nx.DiGraph)
            data=torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            data.sub_id=sub_id
            data.y = torch.tensor([1])
            graphs.append(data)
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
    


class AD_CN_Dataset_Oversampled(InMemoryDataset):
    def __init__(self, root, AD_list,CN_list,thr, transform=None, pre_transform=None):
        self.thr = thr
        self.AD_list = AD_list
        self.CN_list = CN_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graphs = []    
        extra = len(self.CN_list)%len(self.AD_list)
        th_p,th_n = get_threshold_value(list(AD_dict.values()), list(CN_dict.values()), 'correlation', self.thr)
        

        for sub_id in (self.AD_list):
            
            correlation_matrix = get_correlation_matrix(AD_dict[sub_id], 'correlation')
            adj_mat = get_adj_mat(correlation_matrix, th_p,th_n)
            G = nx.from_numpy_matrix(np.array(adj_mat), create_using=nx.DiGraph)
            data=torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            data.sub_id=sub_id
            data.age = torch.tensor(get_age(sub_id=sub_id,sub_data=AD_sub_data),dtype=torch.float)
            if get_gender(sub_id=sub_id,sub_data=AD_sub_data) == 'M':
                data.gender = torch.tensor([1],dtype=torch.int) 
            else:
                data.gender = torch.tensor([0],dtype=torch.int) 
            
            data.y = torch.tensor([0])
            graphs.append(data)
        
        graphs = graphs + graphs
        graphs = graphs + random.sample(graphs[0:len(self.AD_list)], extra) ## Oversampling
        #print(f'AD len = {len(graphs)}')
        for sub_id in (self.CN_list):
            correlation_matrix = get_correlation_matrix(CN_dict[sub_id], 'correlation')
            adj_mat = get_adj_mat(correlation_matrix, th_p,th_n)
            G = nx.from_numpy_matrix(np.array(adj_mat), create_using=nx.DiGraph)
            data=torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            data.sub_id=sub_id
            data.age = torch.tensor(get_age(sub_id=sub_id,sub_data=CN_sub_data),dtype=torch.float)
            if get_gender(sub_id=sub_id,sub_data=CN_sub_data) == 'M':
                data.gender = torch.tensor([1],dtype=torch.int) 
            else:
                data.gender = torch.tensor([0],dtype=torch.int) 
            data.y = torch.tensor([1])
            graphs.append(data)
        #print(f'AD len = {len(graphs)}')
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

  
class AD_CN_Images():
    def __init__(self, AD_list,CN_list,thr):
        self.thr = thr
        self.AD_list = AD_list
        self.CN_list = CN_list
        
    def process(self):
        
        graphs = [] 
        labels = []
        th_p,th_n = get_threshold_value(list(AD_dict.values()), list(CN_dict.values()), 'correlation', self.thr)
        for sub_id in (self.AD_list):
            
            correlation_matrix = get_correlation_matrix(AD_dict[sub_id], 'correlation')
            
            adj_mat = get_adj_mat(correlation_matrix, th_p,th_n)
            data = torch.tensor(adj_mat,dtype=torch.float).unsqueeze(0)
            graphs.append(data)
            
        for sub_id in (self.CN_list):
                          
            correlation_matrix = get_correlation_matrix(CN_dict[sub_id], 'correlation')
            adj_mat = get_adj_mat(correlation_matrix, th_p,th_n)
            

            data = torch.tensor(adj_mat,dtype=torch.float).unsqueeze(0)
            graphs.append(data)
        graphs = torch.cat(graphs,dim=0)
        
        return graphs
        
class AD_Real_Images(torch.utils.data.Dataset):
    def __init__(self, root,AD_list,transform=None, target_transform=None):
        self.AD_list = AD_list
        graphs = [] 
        for sub_id in (self.AD_list):

            correlation_matrix = get_correlation_matrix(AD_dict[sub_id], 'correlation')
            data = torch.tensor(correlation_matrix,dtype=torch.float).unsqueeze(0)
            graphs.append(data)
        
        self.graphs = graphs
        
    def __len__(self):
        return len(self.graphs)  
    
    def __getitem__(self, idx):
        
        return self.graphs[idx]
        
class CN_Real_Images(torch.utils.data.Dataset):
    def __init__(self, root,CN_list,transform=None, target_transform=None):
        self.CN_list = CN_list
        graphs = [] 
        for sub_id in (self.CN_list):
            
            correlation_matrix = get_correlation_matrix(CN_dict[sub_id], 'correlation')
            data = torch.tensor(correlation_matrix,dtype=torch.float).unsqueeze(0)
            graphs.append(data)
        
        self.graphs = graphs
        
    def __len__(self):
        return len(self.graphs)  
    
    def __getitem__(self, idx):
        
        return self.graphs[idx]        
    

        
class AD_Dataset(InMemoryDataset):
    def __init__(self, root, AD_list, transform=None, pre_transform=None):
        self.AD_list = AD_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graphs = []
        for sub_id in (self.AD_list):
            
            correlation_matrix = get_correlation_matrix(AD_dict[sub_id], 'correlation')
            G = nx.from_numpy_matrix(np.array(correlation_matrix), create_using=nx.DiGraph)
            data=torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            data.sub_id=sub_id
            
            data.y = torch.tensor([0])
            graphs.append(data)
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        

class CN_Dataset(InMemoryDataset):
    def __init__(self, root, CN_list, transform=None, pre_transform=None):
        self.CN_list = CN_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graphs = []
        for sub_id in (self.CN_list):
            
            correlation_matrix = get_correlation_matrix(CN_dict[sub_id], 'correlation')
            G = nx.from_numpy_matrix(np.array(correlation_matrix), create_using=nx.DiGraph)
            data = torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            data.sub_id = sub_id
            
            data.y = torch.tensor([1])
            graphs.append(data)
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        
        
class AD_fake_Dataset(InMemoryDataset):
    def __init__(self, root, samples, transform=None, pre_transform=None):
        self.samples = samples
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graphs = []
        for sample in (self.samples):
            
            G = nx.from_numpy_matrix(np.array(sample), create_using=nx.DiGraph)
            data = torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(sample, dtype=torch.float)
            
            data.y = torch.tensor([0])
            graphs.append(data)
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        

class CN_fake_Dataset(InMemoryDataset):
    def __init__(self, root, samples, transform=None, pre_transform=None):
        self.samples = samples
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graphs = []
        for sample in (self.samples):
            
            G = nx.from_numpy_matrix(np.array(sample), create_using=nx.DiGraph)
            data = torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(sample, dtype=torch.float)
            
            data.y = torch.tensor([1])
            graphs.append(data)
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        

class AD_CN_Dataset_Augmented(InMemoryDataset):
    def __init__(self, root, AD_real,CN_real,AD_fake,CN_fake, transform=None, pre_transform=None):
        self.AD_real = AD_real
        self.CN_real = CN_real
        self.AD_fake = AD_fake
        self.CN_fake = CN_fake
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graphs = []
        
        extra = len(self.CN_real)%len(self.AD_real)
        for sub_id in (self.AD_real):
            
            correlation_matrix = get_correlation_matrix(AD_dict[sub_id], 'correlation')
            G = nx.from_numpy_matrix(np.array(correlation_matrix), create_using=nx.DiGraph)
            data=torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            
            data.y = torch.tensor([0])
            graphs.append(data)
        graphs = graphs + graphs
        graphs = graphs + random.sample(graphs[0:len(self.AD_real)], extra) ## Oversampling
        for sub_id in (self.CN_real):
            
            correlation_matrix = get_correlation_matrix(CN_dict[sub_id], 'correlation')
            G = nx.from_numpy_matrix(np.array(correlation_matrix), create_using=nx.DiGraph)
            data=torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(correlation_matrix, dtype=torch.float)
            
            data.y = torch.tensor([1])
            graphs.append(data)
            
        for sample in (self.AD_fake):
            sample = sample.cpu().numpy()
            G = nx.from_numpy_matrix(sample, create_using=nx.DiGraph)
            data = torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(sample, dtype=torch.float)

            data.y = torch.tensor([0])
            graphs.append(data)
        for sample in (self.CN_fake):
            sample = sample.cpu().numpy()
            G = nx.from_numpy_matrix(sample, create_using=nx.DiGraph)
            data = torch_geometric.utils.from_networkx(G)
            data.x = torch.tensor(sample, dtype=torch.float)
            
            data.y = torch.tensor([1])
            graphs.append(data)
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])        


if __name__=='__main__':

    data_path = '/home/hiren/Apoorv Pandey/Dataset/'
    AD_train = load_obj(data_path + 'AD_train_full')
    AD_val = load_obj(data_path + 'AD_val_full')
    AD_test = load_obj(data_path + 'AD_test_full')

    CN_train = load_obj(data_path + 'CN_train_full')
    CN_val = load_obj(data_path + 'CN_val_full')
    CN_test = load_obj(data_path + 'CN_test_full')

    print(AD_sub_data.head())

    for threshold_percentage in [0.5,2,4,6,8,10]:
        for i in tqdm(range(10)):
        
            train_dataset = AD_CN_Dataset_Oversampled(f'./Oversampled_Training_Data/Seed={i}Thr={threshold_percentage}',AD_train[i],CN_train[i],threshold_percentage)  
            val_dataset = AD_CN_Dataset_Oversampled(f'./Oversampled_Validation_Data/Seed={i}Thr={threshold_percentage}',AD_val[i],CN_val[i],threshold_percentage)    
            test_dataset = AD_CN_Dataset_Oversampled(f'./Oversampled_Test_Data/Seed={i}Thr={threshold_percentage}',AD_test[i],CN_test[i],threshold_percentage)  


            
