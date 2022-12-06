import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv,GINConv,GraphConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F



class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, dropout, batch_norm=True, residual=False
    ):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
        self.conv = GraphConv(in_features, out_features)

    def forward(self, x, edge_index,edge_weight):
        if edge_weight is not None:
            h = self.conv(x, edge_index,edge_weight)
        else:
            h = self.conv(x,edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)

class GINLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        dropout,
        batch_norm,
        mlp_hidden_dim=None,
        residual=True,
        train_eps=False,
    ):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = in_features

        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
        gin_net = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        self.conv = GINConv(gin_net, train_eps=train_eps)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        if self.residual:
            h = h + x
        return self.dropout(h)



class GATLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        dropout,edge_dropout,
        num_heads=1,
        batch_norm = False,
        residual=False
    ):
        super().__init__()


        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features * num_heads) if batch_norm else nn.Identity()
        
        self.conv = GATConv(in_features, out_features, heads = num_heads, dropout = edge_dropout,edge_dim = 1)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            h = self.conv(x, edge_index,edge_attr = edge_attr)
        else:
             h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels_dims,heads):
        super(GAT, self).__init__()
        torch.manual_seed(0)
    
        self.hidden_channels_dims = hidden_channels_dims
        self.layers = nn.ModuleList() 
        for i in range(len(hidden_channels_dims)-1):
            if i==0:
                self.layers.append(GATLayer(hidden_channels_dims[i],hidden_channels_dims[i+1],0.5,heads))
            else:
                self.layers.append(GATLayer(hidden_channels_dims[i]*heads,hidden_channels_dims[i+1],0.5,heads))
        self.lin = nn.Linear(hidden_channels_dims[-1]*heads, 2)

    def forward(self, x, edge_index, edge_attr, batch):

        for layer in self.layers:
            x = layer(x,edge_index,edge_attr)

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return x





class PointWiseMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self,x):
        return self.mlp(x)


class LargerGCNModel(nn.Module):
    def __init__(self, hidden_dim, depth, num_node_features, num_classes,
                 batch_norm=False,
                 residual=False,
                 add_mlp=False,
                 ):
        super(LargerGCNModel, self).__init__()
        
        def build_gnn_layer(is_first = False, is_last = False):
            return GCNLayer(
                    in_features = hidden_dim,
                    out_features = hidden_dim,
                    activation = nn.Identity() if is_last else F.relu,
                    dropout = 0. if is_last else 0.5,
                    edge_dropout = 0,
                    batch_norm = batch_norm, residual = residual)
        


        graph_pooling_operation = global_mean_pool
       
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        layers = [
            build_gnn_layer(is_first=i == 0, is_last=i == (depth-1))
            for i in range(depth)
        ]

        if add_mlp:
            mlp_layer = PointWiseMLP(hidden_dim)
            layers.insert(1, mlp_layer)

        self.layers = nn.ModuleList(layers)

       
        self.pooling_fun = graph_pooling_operation
     
        dim_before_class = hidden_dim

        self.classif = nn.Identity()
        self.classif =  torch.nn.Sequential(
            nn.Linear(dim_before_class, hidden_dim // 2),
             nn.ReLU(),
             nn.Linear(hidden_dim // 2, hidden_dim // 4),
             nn.ReLU(),
             nn.Linear(hidden_dim // 4, num_classes)
         )
        
    
    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index,data.weight
        
        x = self.embedding(x)
        
        edge_weights = dict()
        for layer in self.layers:
            x = layer(x, edge_index=edge_index, edge_weight = edge_weight)
            
        x = self.pooling_fun(x, data.batch)
        x = self.classif(x)

        return x,edge_weights


class LargerGATModel(nn.Module):
    def __init__(self, hidden_dim, depth, num_node_features, num_classes,mode,
                 batch_norm=False,
                 residual=False):
        super(LargerGATModel, self).__init__()
        self.mode = mode
        num_heads=2
        def build_gnn_layer(is_first = False, is_last = False):
            return GATLayer(
                    in_features = (hidden_dim * num_heads),
                    out_features = (hidden_dim * num_heads) if is_last else hidden_dim,
                    activation = nn.Identity() if is_last else F.relu,
                    dropout = 0. if is_last else 0.5,
                    edge_dropout = 0,
                    batch_norm = batch_norm, residual = residual,
                    num_heads = 1 if is_last else num_heads)
        


        output_dim = hidden_dim * 2
        self.age_embedding = nn.Linear(1,output_dim)
        self.gender_embedding = nn.Embedding(2,output_dim)
        graph_pooling_operation = global_mean_pool
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim * num_heads)

        layers = [
            build_gnn_layer(is_first=i == 0, is_last=i == (depth-1))
            for i in range(depth)
        ]


        self.layers = nn.ModuleList(layers)
        
        self.pooling_fun = graph_pooling_operation
        
        if self.mode=='Without_fMRI':
            dim_before_class = output_dim * 2
        elif self.mode=='Fusion':
            dim_before_class = output_dim * 3
        else:
            dim_before_class = output_dim

        self.classif = nn.Identity()
        self.classif =  torch.nn.Sequential(
            nn.Linear(dim_before_class, hidden_dim // 2),
             nn.ReLU(),
             nn.Linear(hidden_dim // 2, hidden_dim // 4),
             nn.ReLU(),
             nn.Linear(hidden_dim // 4, num_classes)
         )
        

    def forward(self, data):
        
        x, edge_index,edge_attr,age,gender = data.x, data.edge_index,data.edge_attr,data.age.unsqueeze(1),data.gender.unsqueeze(1)
        
        if self.mode=='Without_fMRI':
            age_embedding = self.age_embedding(age).squeeze(1)
            gender_embedding = self.gender_embedding(gender).squeeze(1)
            x = torch.cat([age_embedding,gender_embedding],dim=1)
            x = self.classif(x)
        
        elif self.mode=='Only_fMRI':
          
            x = self.embedding(x)

            for layer in self.layers:
                x = layer(x, edge_index = edge_index,edge_attr = edge_attr)    
                
            x = self.pooling_fun(x, data.batch)
            x = self.classif(x)
        
        else:
            x = self.embedding(x)

            for layer in self.layers:
                x = layer(x, edge_index = edge_index,edge_attr = edge_attr)    
            #print(f'before pooling x size:{x.size()}') 
            x = self.pooling_fun(x, data.batch)
            age_embedding = self.age_embedding(age).squeeze(1)
            gender_embedding = self.gender_embedding(gender).squeeze(1)
            #print(f'After pooling {x.size(),age_embedding.size(),gender_embedding.size()}')
            x = torch.cat([x,age_embedding,gender_embedding],dim=1)
            x = self.classif(x)
        return x

    



    