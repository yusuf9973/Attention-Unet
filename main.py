import importlib
import matplotlib.pyplot as plt 
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_optimizer as optim
from torch.optim import RAdam   
from codecarbon.emissions_tracker import EmissionsTracker
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time_list = []
torch.manual_seed(42) 
def train(x_train,y_train,n_classes,name,names,d,batch_size,num_epochs,n_channels):
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    train_dataset = CustomDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  
    class_counts = torch.zeros(n_classes) 
    for i in range(n_classes):
        class_counts[i] = (y_train == i).sum() 
    total_count = class_counts.sum() 
    class_weights = total_count / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    module = importlib.import_module(f"unet_variants.{name}")
    UNet = getattr(module, "UNet") 
    f, axarr = plt.subplots(len(names), 1, figsize=(10, 4*len(names)))
    for q in range(len(names)):
        tracker = EmissionsTracker(save_to_file=True, output_file='data/'+names[q]+d+'_my_emissions.csv', log_level="ERROR")
        tracker.start()
        t1 = time.time()
        model = UNet(n_channels, n_classes, bilinear=False,index=q)
        model.to(device)
        print(names[q])
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        learning_rate = 0.0001
        text= names[q]+ '\n'
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        l_t = len(train_dataloader)
        l_v = len(val_dataloader) 
        best_acc = 0.0
        best_epoch = 0                                               
        early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.01, path='model/es/' + names[q] + d+'.pth')
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_acc = 0.0
            t = 0
            v = 0
            model.train()
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.permute(0, 3, 1, 2)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = torch.exp(outputs)
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_acc += acc.item()
                t+=1
                print(f'\rEpoch {epoch+1}: Train Progress: {int((t/l_t)*100)}%',end="")
            train_losses.append(running_loss / len(train_dataloader))
            train_accs.append(running_acc / len(train_dataloader))
            val_loss = 0.0
            val_acc = 0.0
            model.eval()
            with torch.no_grad():
                for i ,(inputs, labels) in enumerate(val_dataloader):
                    inputs = inputs.permute(0, 3, 1, 2)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    outputs = torch.exp(outputs)
                    loss = criterion(outputs, labels)
                    acc = (outputs.argmax(dim=1) == labels).float().mean()
                    val_loss += loss.item()
                    val_acc += acc.item()
                    v+=1
                    print(f'\rEpoch {epoch+1}: Train Progress: {int((t/l_t)*100)}% Validation Progress: {int((v/l_v)*100)}%  ',end="")
            val_losses.append(val_loss / len(val_dataloader))
            val_accs.append(val_acc / len(val_dataloader))        
            scheduler.step(val_loss)
            print(f'\rEpoch {epoch+1}: Training Loss: {running_loss / len(train_dataloader):.4f}, Training Accuracy: {running_acc / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {val_acc / len(val_dataloader):.4f}')
            text += f'Epoch {epoch+1}: Training Loss: {running_loss / len(train_dataloader):.4f}, Training Accuracy: {running_acc / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {val_acc / len(val_dataloader):.4f} \n'
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(),'model/'+names[q]+d+'.pth')
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch+1}')
                break
        with open("output/train_output_"+names[q]+d+".txt", "w") as file:
            file.write(text)
        t2 = time.time()
        tracker.stop()
        time_list.append((t2-t1)/ 60.0)
    with open('data/time'+name+d+'.pkl', 'wb') as file:
        pickle.dump(time_list, file)
    axarr[i].plot(train_losses, label="Training Loss", color="blue")
    axarr[i].plot(val_losses, label="Validation Loss", color="red") 
    axarr[i].plot(train_accs, label="Training Accuracy", color="green") 
    axarr[i].plot(val_accs, label="Validation Accuracy", color="orange") 
    axarr[i].set_xlabel("Epochs")
    axarr[i].set_ylabel("Metrics")
    axarr[i].set_title(f"{names[i]} Metrics Curve")
    axarr[i].legend()
    plt.savefig("output/imgs/train_models"+name+d+".png",bbox_inches='tight')
    plt.close()  
