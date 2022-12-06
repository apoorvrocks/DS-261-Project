import torch
import time
from tqdm import tqdm
import statistics as stat
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from Dataset import AD_CN_Dataset_Oversampled
from models import LargerGATModel,LargerGCNModel
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
import pickle
import  warnings
warnings.filterwarnings("ignore")

def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def update_metrics(_dict,acc,f1,rocauc):
    _dict['acc'] = _dict['acc'] + acc
    _dict['f1'] = _dict['f1'] + f1
    _dict['rocauc'] = _dict['rocauc'] + rocauc
    return _dict

data_path = '/home/hiren/Apoorv Pandey/Dataset/'
base_path = '/home/hiren/Apoorv Pandey/AIMI/Project'
seeds=10
device = 'cuda:0'


AD_dict = load_obj(data_path + 'AD')
CN_dict = load_obj(data_path + 'CN')

AD_train = load_obj(data_path + 'AD_train_full')
AD_val = load_obj(data_path + 'AD_val_full')
AD_test = load_obj(data_path + 'AD_test_full')

CN_train = load_obj(data_path + 'CN_train_full')
CN_val = load_obj(data_path + 'CN_val_full')
CN_test = load_obj(data_path + 'CN_test_full')

class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation acc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation acc improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = 0
        self.val_acc_max = 0
        self.val_roc_auc_max = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_f1, model):

        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        '''Saves model when validation acc increase.'''
        
        torch.save(model.state_dict(), self.path)
        self.val_f1_max = val_f1


def metrics(loader,model):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            out = model(data)
            pred = out.argmax(dim=1)
            y_pred += pred.cpu().detach().tolist()
            y_true += data.y.cpu().detach().tolist()
    
    return f1_score(y_true,y_pred),roc_auc_score(y_true,y_pred),accuracy_score(y_true,y_pred)


def plots(train_scores,val_scores,test_scores,path):
    avg_train_scores,avg_val_scores,avg_test_scores = [],[],[]
    early_stopping_index_train,early_stopping_index_val,early_stopping_index_test = 70,70,70
    for i in range(len(train_scores)):
        if -1 in train_scores[i]:
            early_stopping_index_train = min(early_stopping_index_train,i)
            break
        avg_train_scores.append(sum(train_scores[i])/len(train_scores[i]))
        
    for i in range(len(val_scores)):
        if -1 in val_scores[i]:
            early_stopping_index_val = min(early_stopping_index_val,i)
            break
        avg_val_scores.append(sum(val_scores[i])/len(val_scores[i]))
    for i in range(len(test_scores)):
        if -1 in test_scores[i]:
            early_stopping_index_test = min(early_stopping_index_test,i)
            break
        avg_test_scores.append(sum(test_scores[i])/len(test_scores[i]))   
    print(len(avg_train_scores))
    plt.figure(0)
    plt.plot(np.arange(len(avg_train_scores)),avg_train_scores,label='train')
    plt.plot(np.arange(len(avg_val_scores)),avg_val_scores,label='val')
    plt.plot(np.arange(len(avg_test_scores)),avg_test_scores,label='test')

    plt.legend()
    plt.savefig(path)
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval(mode):
    config = {'depth': [2,3,4],'hidden_dim':[32,64,128],'threshold_percentage': [0.5,2,4,6,8,10]}
    best_hyperparameters = {}
    best_val_acc = 0.0
    logfile_path = f'./best_config_{mode}.txt'

    for depth in tqdm(config['depth']):
        for hidden_dim in (config['hidden_dim']):
            for threshold_percentage in config['threshold_percentage']:

                lr = 1e-3
                avg_train_metrics = {'acc':0,'f1':0,'rocauc':0}
                avg_val_metrics =  {'acc':0,'f1':0,'rocauc':0}
                avg_test_metrics =  {'acc':0,'f1':0,'rocauc':0}
                start = time.time()
                with open(logfile_path,'a') as f:
                    print(f'Depth:{depth} Hidden dim {hidden_dim} threshold_percentage :{threshold_percentage} staretd',file =f)
                for i in (range(seeds)):
                    
                    
                    train_dataset = AD_CN_Dataset_Oversampled(f'/home/hiren/Apoorv Pandey/AIMI/Project/OverSampled_Training_Data/Seed={i}Thr={threshold_percentage}',AD_train[i],CN_train[i],threshold_percentage)
                    val_dataset = AD_CN_Dataset_Oversampled(f'/home/hiren/Apoorv Pandey/AIMI/Project/OverSampled_Val_Data/Seed={i}Thr={threshold_percentage}',AD_val[i],CN_val[i],threshold_percentage)
                    test_dataset = AD_CN_Dataset_Oversampled(f'/home/hiren/Apoorv Pandey/AIMI/Project/OverSampled_Test_Data/Seed={i}Thr={threshold_percentage}',AD_test[i],CN_test[i],threshold_percentage)
                    
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    val_loader = DataLoader(val_dataset,batch_size=32, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
                    model = LargerGATModel(hidden_dim = hidden_dim,depth=depth,num_node_features=116,num_classes=2,mode=mode,
                                        batch_norm=True, residual=True).to(device)
                    
                    checkpt_path = base_path + f"/Model:GAT/{mode}/checkpoints_th:{threshold_percentage}/depth:{depth}/hidden_dim:{hidden_dim}/lr:{lr}/checkpoint_seed_{i}.pt"
                    model.load_state_dict(torch.load(checkpt_path))
                    train_f1,train_rocauc,train_acc = metrics(train_loader,model)
                    val_f1,val_rocauc,val_acc = metrics(val_loader,model)
                    test_f1,test_rocauc,test_acc = metrics(test_loader,model)

                
                    avg_train_metrics = update_metrics(avg_train_metrics,train_acc,train_f1,train_rocauc)
                    avg_val_metrics = update_metrics(avg_val_metrics,val_acc,val_f1,val_rocauc)
                    avg_test_metrics = update_metrics(avg_test_metrics,test_acc,test_f1,test_rocauc)
                if best_val_acc < avg_val_metrics['acc']/seeds:
                    best_val_acc = avg_val_metrics['acc']/seeds
                
                    best_hyperparameters = {'depth':depth,'hidden_dim':hidden_dim,'threshold_percentage':threshold_percentage}
                end = time.time()
                with open(logfile_path,'a') as f:
                    print(f'Time taken for Depth:{depth} Hidden dim {hidden_dim} threshold_percentage :{threshold_percentage} = {end-start}',file =f)
                    num_params = count_parameters(model)
                    print(f'Model Parameters:{num_params}',file = f)
                    
    with open(logfile_path,'a') as f:
        print(f'Mode:{mode}, best_hyperparameters={best_hyperparameters}',file = f)
    return best_hyperparameters


def train(mode):

    
    epochs = 70
    Patience = 20
    

    criterion = torch.nn.CrossEntropyLoss()
    train_f1s_epochwise,val_f1s_epochwise,test_f1s_epochwise = [[-1]*seeds for i in range(epochs)],[[-1]*seeds for i in range(epochs)],[[-1]*seeds for i in range(epochs)]
    train_rocaucs_epochwise,val_rocaucs_epochwise,test_rocaucs_epochwise = [[-1]*seeds for i in range(epochs)],[[-1]*seeds for i in range(epochs)],[[-1]*seeds for i in range(epochs)]
    train_accs_epochwise,val_accs_epochwise,test_accs_epochwise = [[-1]*seeds for i in range(epochs)],[[-1]*seeds for i in range(epochs)],[[-1]*seeds for i in range(epochs)]
    final_train_accs,final_val_accs,final_test_accs  = [],[],[]
    final_train_rocaucs,final_val_rocaucs,final_test_rocaucs = [],[],[]
    final_train_f1s,final_val_f1s,final_test_f1s = [],[],[]


    
    config = {'depth': [4],'hidden_dim':[128],'threshold_percentage': [0.5,2,4,6,8,10]}
    best_hyperparameters = {}
    best_val_acc = 0.0
    scores = [] 
    logfile_path = f'./logfile_{mode}.txt'
    for depth in tqdm(config['depth']):
        for hidden_dim in tqdm(config['hidden_dim']):
            for threshold_percentage in config['threshold_percentage']:
                lr = 1e-3
                
                start = time.time()
                with open(logfile_path,'a') as f:
                    print(f'Depth:{depth} Hidden dim {hidden_dim} threshold_percentage :{threshold_percentage} staretd',file =f)
                
                avg_train_metrics = {'acc':0,'f1':0,'rocauc':0}
                avg_val_metrics =  {'acc':0,'f1':0,'rocauc':0}
                avg_test_metrics =  {'acc':0,'f1':0,'rocauc':0}
                for i in tqdm(range(seeds)):
                    
                
                    train_dataset = AD_CN_Dataset_Oversampled(f'/home/hiren/Apoorv Pandey/AIMI/Project/OverSampled_Training_Data/Seed={i}Thr={threshold_percentage}',AD_train[i],CN_train[i],threshold_percentage)
                    val_dataset = AD_CN_Dataset_Oversampled(f'/home/hiren/Apoorv Pandey/AIMI/Project/OverSampled_Val_Data/Seed={i}Thr={threshold_percentage}',AD_val[i],CN_val[i],threshold_percentage)
                    test_dataset = AD_CN_Dataset_Oversampled(f'/home/hiren/Apoorv Pandey/AIMI/Project/OverSampled_Test_Data/Seed={i}Thr={threshold_percentage}',AD_test[i],CN_test[i],threshold_percentage)
                    
                    losses = []
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    val_loader = DataLoader(val_dataset,batch_size=32, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

                    model = LargerGATModel(hidden_dim = hidden_dim,depth=depth,num_node_features=116,num_classes=2,mode=mode,
                                        batch_norm=True, residual=True).to(device)
                
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
                    os.makedirs(base_path + f"/Model:GAT/{mode}/checkpoints_th:{threshold_percentage}/depth:{depth}/hidden_dim:{hidden_dim}/lr:{lr}", exist_ok=True)
                    checkpt_path = base_path + f"/Model:GAT/{mode}/checkpoints_th:{threshold_percentage}/depth:{depth}/hidden_dim:{hidden_dim}/lr:{lr}/checkpoint_seed_{i}.pt"
                    
                    early_stopping = EarlyStopping(patience=Patience, verbose=True, path=checkpt_path)
                    
                    for epoch in tqdm(range(0,epochs)):
                        
                        mean_loss = 0
                        itr = 0
                        for data in train_loader:  # Iterate in batches over the training dataset.
                            model.train()
                            data = data.to(device)
                            out= model(data)  # Perform a single forward pass.
                            loss = criterion(out, data.y)  # Compute the loss.
                            loss.backward()  # Derive gradients.
                            mean_loss += loss.item()
                            itr += 1
                            optimizer.step()  # Update parameters based on gradients.
                            optimizer.zero_grad()  # Clear gradients.
                            torch.cuda.empty_cache()
                        mean_loss = mean_loss/itr
                        losses.append(mean_loss)
                        scheduler.step(mean_loss)
                        train_f1,train_rocauc,train_acc = metrics(train_loader,model)
                        val_f1,val_rocauc,val_acc = metrics(val_loader,model)
                        test_f1,test_rocauc,test_acc = metrics(test_loader,model)
                        '''
                        train_metrics = {'train_acc':train_acc,'train_f1':train_f1,'train_rocauc':train_rocauc}
                        val_metrics = {'val_acc':val_acc,'val_f1':val_f1,'val_rocauc':val_rocauc}
                        test_metrics = {'test_acc':test_acc,'test_f1':test_f1,'test_rocauc':test_rocauc}
                        wandb.log({'train_acc':train_acc,'train_f1':train_f1,'train_rocauc':train_rocauc,\
                                'val_acc':val_acc,'val_f1':val_f1,'val_rocauc':val_rocauc,\
                                'test_acc':test_acc,'test_f1':test_f1,'test_rocauc':test_rocauc})
                        '''
                        train_f1s_epochwise[epoch][i] = train_f1
                        val_f1s_epochwise[epoch][i] = val_f1
                        test_f1s_epochwise[epoch][i] = test_f1
                        train_accs_epochwise[epoch][i] = train_acc
                        val_accs_epochwise[epoch][i] = val_acc
                        test_accs_epochwise[epoch][i] = test_acc
                        train_rocaucs_epochwise[epoch][i] = train_rocauc
                        val_rocaucs_epochwise[epoch][i] = val_rocauc
                        test_rocaucs_epochwise[epoch][i] = test_rocauc
                        
                        
                        
                        early_stopping(val_f1, model)
                        if early_stopping.early_stop:
                            with open(logfile_path,'a') as f:
                    
                                print(f"threshold percentage = {threshold_percentage}, Seed = {i},Early stopping at epoch:{epoch}",file =f)
                                print(f" Seed = {i},Early stopping at epoch:{epoch}",file =f)
                                print(f'F1 scores Epoch: {epoch:03d}, Train f1: {train_f1:.4f},Test f1:{test_f1:.4f}, Val f1: {val_f1:.4f}',file =f)
                                print(f'Acc scores Epoch: {epoch:03d}, Train Acc: {train_acc:.4f},Test acc:{test_acc:.4f}, Val acc: {val_acc:.4f}',file =f)
                                print(f'ROC-AUC scores Epoch: {epoch:03d}, Train ROC-AUC: {train_rocauc:.4f},Test ROC-AUC:{test_rocauc:.4f}, Val ROC-AUC: {val_rocauc:.4f}',file =f)
                            break
                        
                        if epoch==epochs-1:
                            with open(logfile_path,'a') as f:
                    
                                print(f"threshold percentage = {threshold_percentage}, Seed = {i},Early stopping at epoch:{epoch}",file =f)
                                print(f" Seed = {i},Early stopping at epoch:{epoch}",file =f)
                                print(f'F1 scores Epoch: {epoch:03d}, Train f1: {train_f1:.4f},Test f1:{test_f1:.4f}, Val f1: {val_f1:.4f}',file =f)
                                print(f'Acc scores Epoch: {epoch:03d}, Train Acc: {train_acc:.4f},Test acc:{test_acc:.4f}, Val acc: {val_acc:.4f}',file =f)
                                print(f'ROC-AUC scores Epoch: {epoch:03d}, Train ROC-AUC: {train_rocauc:.4f},Test ROC-AUC:{test_rocauc:.4f}, Val ROC-AUC: {val_rocauc:.4f}',file =f)
                    plt.figure(0)
                    plt.plot(np.arange(len(losses)),losses,label='train losses')
                    plt.legend()
                    plt.show()
                    plt.close()
                    model = LargerGATModel(hidden_dim = hidden_dim,depth=depth,num_node_features=116,num_classes=2,mode=mode,
                                        batch_norm=True, residual=True).to(device)
                    model.load_state_dict(torch.load(checkpt_path))
                    train_f1,train_rocauc,train_acc = metrics(train_loader,model)
                    val_f1,val_rocauc,val_acc = metrics(val_loader,model)
                    test_f1,test_rocauc,test_acc = metrics(test_loader,model)

                    final_train_accs.append(train_acc),final_test_accs.append(test_acc),final_val_accs.append(val_acc)
                    final_train_f1s.append(train_f1),final_test_f1s.append(test_f1),final_val_f1s.append(val_f1)
                    final_train_rocaucs.append(train_rocauc), final_val_rocaucs.append(val_rocauc),final_test_rocaucs.append(test_rocauc)

                    avg_train_metrics = update_metrics(avg_train_metrics,train_acc,train_f1,train_rocauc)
                    avg_val_metrics = update_metrics(avg_val_metrics,val_acc,val_f1,val_rocauc)
                    avg_test_metrics = update_metrics(avg_test_metrics,test_acc,test_f1,test_rocauc)


                plots(train_f1s_epochwise,val_f1s_epochwise,test_f1s_epochwise,base_path + f"/Model:GAT/{mode}/checkpoints_th:{threshold_percentage}/depth:{depth}/hidden_dim:{hidden_dim}/lr:{lr}/F1.jpg")
                plots(train_accs_epochwise,val_accs_epochwise,test_accs_epochwise,base_path + f"/Model:GAT/{mode}/checkpoints_th:{threshold_percentage}/depth:{depth}/hidden_dim:{hidden_dim}/lr:{lr}/Acc.jpg")
                plots(train_rocaucs_epochwise,val_rocaucs_epochwise,test_rocaucs_epochwise,base_path + f"/Model:GAT/{mode}/checkpoints_th:{threshold_percentage}/depth:{depth}/hidden_dim:{hidden_dim}/lr:{lr}/RocAuc.jpg")
                    

                scores.append({'train_acc':avg_train_metrics['acc']/seeds,'train_f1':avg_train_metrics['f1']/seeds,'train_rocauc':avg_train_metrics['rocauc']/seeds,\
                                    'val_acc':avg_val_metrics['acc']/seeds,'val_f1':avg_val_metrics['f1']/seeds,'val_rocauc':avg_val_metrics['rocauc']/seeds,\
                                    'test_acc':avg_test_metrics['acc']/seeds,'test_f1':avg_test_metrics['rocauc']/seeds,'test_rocauc':avg_test_metrics['rocauc']/seeds})
                if best_val_acc < avg_val_metrics['acc']/seeds:
                    best_val_acc = avg_val_metrics['acc']/seeds
                
                    best_hyperparameters = {'depth':depth,'hidden_dim':hidden_dim,'threshold_percentage':threshold_percentage}

                with open(base_path + f"/Model:GAT/{mode}/checkpoints_th:{threshold_percentage}/depth:{depth}/hidden_dim:{hidden_dim}/lr:{lr}/report_latest", "w") as file:
                    file.write( f"train_avg_accuracy={stat.mean(final_train_accs)*100:0.2f} +- {stat.stdev(final_train_accs)*100:0.2f}%\n"
                                f"train_avg_f1={stat.mean(final_train_f1s):0.2f}+-{stat.stdev(final_train_f1s):0.2f}%\n"
                                f"Train AUC-ROC = {stat.mean(final_train_rocaucs)*100:0.2f}+-{stat.stdev(final_train_rocaucs)*100:0.2f}%\n"
                                f"val_avg_accuracy={stat.mean(final_val_accs)*100:0.2f}+-{stat.stdev(final_val_accs)*100:0.2f}%\n"
                                f"val_avg_f1={stat.mean(final_val_f1s):0.2f}+-{stat.stdev(final_val_f1s):0.2f}%\n"
                                f"Val AUC-ROC = {stat.mean(final_val_rocaucs)*100:0.2f}+-{stat.stdev(final_val_rocaucs)*100:0.2f}%\n"    
                                f"test_avg_accuracy={stat.mean(final_test_accs)*100:0.2f}+-{stat.stdev(final_test_accs)*100:0.2f}%\n"
                                f"test_avg_f1={stat.mean(final_test_f1s):0.2f}+-{stat.stdev(final_test_f1s):0.2f}%\n"
                                f"Test AUC-ROC = {stat.mean(final_test_rocaucs)*100:0.2f}+-{stat.stdev(final_test_rocaucs)*100:0.2f}%\n")
                end = time.time()
                with open(logfile_path,'a') as f:
                    print(f'Time taken for Depth:{depth} Hidden dim {hidden_dim} threshold_percentage :{threshold_percentage} = {end-start}',file =f)

    return best_hyperparameters

best_hyperparameters = train('Fusion')

print(f'Mode:FusionI, best_hyperparameters={best_hyperparameters}')

best_hyperparameters = train('Without_fMRI')

print(f'Mode:Without_fMRI, best_hyperparameters={best_hyperparameters}')

best_hyperparameters = train('Only_fMRI')
print(f'Mode:Only_fMRI, best_hyperparameters={best_hyperparameters}')
    

                


