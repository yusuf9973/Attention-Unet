models = [UNet,AttenUNet,UNetplusplus,SegCaps]  
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
def train(x_train,y_train,n_classes,names,d,batch_size,num_epochs,n_channels,n,learning_rate):
    global time_list
    global emissions
    metrics = []
    y_train = torch.from_numpy(y_train).long()
    x_train = torch.from_numpy(x_train).float()
    device = 'cuda'
    torch.manual_seed(42)
    class_counts = torch.zeros(n_classes)
    for i in range(n_classes):
        class_counts[i] = (y_train == i).sum()
    total_count = class_counts.sum()
    class_weights = total_count / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)
    b = batch_size
    temp = -1
    batch = []
    for q in range(len(names)):                      
        try:
            if temp !=q:
                print(names[q]+d) 
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            tracker = EmissionsTracker(save_to_file=True, output_file='my_emissions.csv', log_level="ERROR")
            tracker.start()
            t1 = time.time()
            model = models[n](n_channels, n_classes, bilinear=False,index=q)
            model.to(device)
            best_acc = 0.0
            train_dataloader = DataLoader(train_dataset, b, shuffle=True)
            val_dataloader = DataLoader(val_dataset, b, shuffle=True)
            l_t = len(train_dataloader)
            l_v = len(val_dataloader)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
                print(f'Epoch {epoch+1}: Training Loss: {running_loss / len(train_dataloader):.4f}, Training Accuracy: {running_acc / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {val_acc / len(val_dataloader):.4f}, batch_size: {b} ')
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(),f'model/{names[q]}{d}.pth')                          
            t2 = time.time()
            tracker.stop()
            df = pd.read_csv('my_emissions.csv')
            df = df['emissions']
            total_emissions = df.sum()
            emissions.append(total_emissions)
            os.remove('my_emissions.csv')
            metrics.append({'train_losses': train_losses, 'train_accs': train_accs, 'val_losses': val_losses, 'val_accs': val_accs}) 
            time_list.append((t2 - t1) / 60.0)
            batch.append(b)
            b = batch_size
            q+=1
        except Exception as e:
            if temp != q:
                print(e,"for batch_size:",batch_size)
                temp = q
            if b > 1:
                b -=1
            else:
                print("batch_size can't be reduced further")
                return
    f, axarr = plt.subplots(len(names), 1, figsize=(10, 4*len(names))) 
    for q in range(len(names)):
      epochs = range(1, len(train_losses) + 1)
      try:
        axarr[q].plot(epochs, metrics[q]['train_losses'], marker='o', linestyle='-', color='blue', label="Training Loss")
        axarr[q].plot(epochs, metrics[q]['val_losses'], marker='o', linestyle='-', color='red', label="Validation Loss")
        axarr[q].plot(epochs, metrics[q]['train_accs'], marker='o', linestyle='-', color='green', label="Training Accuracy")
        axarr[q].plot(epochs, metrics[q]['val_accs'], marker='o', linestyle='-', color='orange', label="Validation Accuracy")
        axarr[q].set_ylim(0, 3)
        num_ticks = 25 
        axarr[q].set_yticks(np.linspace(0, 3, num_ticks))
        axarr[q].set_xlabel("Epochs")
        axarr[q].set_ylabel("Metrics")
        axarr[q].set_title(f"{names[q]+d} (batch_size:{batch[q]},num_epochs:{num_epochs},learning_rate:{learning_rate})")
        axarr[q].legend()
      except:
        ax = axarr 
        ax.plot(epochs, metrics[q]['train_losses'], marker='o', linestyle='-', color='blue', label="Training Loss")
        ax.plot(epochs, metrics[q]['val_losses'], marker='o', linestyle='-', color='red', label="Validation Loss")
        ax.plot(epochs, metrics[q]['train_accs'], marker='o', linestyle='-', color='green', label="Training Accuracy")
        ax.plot(epochs, metrics[q]['val_accs'], marker='o', linestyle='-', color='orange', label="Validation Accuracy")
        ax.set_ylim(0, 3)
        num_ticks = 25 
        ax.set_yticks(np.linspace(0, 3, num_ticks))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metrics")
        ax.set_title(f"{names[q]+d} (batch_size:{batch[q]}, num_epochs:{num_epochs}, learning_rate:{learning_rate})")
        ax.legend()
    plt.tight_layout()
    plt.savefig("output/train_models_"+names[0]+d+".png",bbox_inches='tight')
    plt.close()           
