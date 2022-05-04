import random

import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders import *
import tqdm
import matplotlib.pyplot as plt

# ______________________________________General Parameters_______________________________________
num_epochs = 20
learning_rate = 0.001
batch_size = 5
# ______________________________________Cuda connection___________________________________________
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# ______________________________________Data___________________________________________________
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'

DATA_DIR = '/home/ubuntu/ASL'

# Data Loader in utilsConv3D
loader, dataset = get_loader(
                             keyword='train',
                             batch_size=batch_size,
                             transform=frame_transform
                            )

val_loader = get_loader(
                            keyword='test',
                            batch_size=batch_size,
                            transform=frame_transform
                            )

class CNN3d(nn.Module):
    def __init__(self):
        super(CNN3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.drop1 = nn.Dropout(0.5)

        self.conv1b = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm1b = nn.BatchNorm3d(128)
        self.pool1b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.drop1b = nn.Dropout(0.5)

        self.conv2 = nn.Conv3d(128,256, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.norm2 = nn.BatchNorm3d(256)
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.norm3 = nn.BatchNorm3d(512)
        self.drop3 = nn.Dropout(0.5)

        self.global_avg_pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(2,2,2), padding=(0,1,1)) #nn.AdaptiveAvgPool3d((1,6,6))
        #self.global_avg_pool2 = nn.AdaptiveAvgPool3d((None,6,6))
        self.linear1 = nn.Linear(24576,1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 32)
        self.linear4 = nn.Linear(32, 1)
        self.act = torch.relu
        self.dropout = nn.Dropout(0.5)




    def forward(self, x):
        x = self.drop1(self.pool1(self.act(self.norm1(self.conv1(x))))) #1
        x = self.drop1b(self.pool1b(self.norm1b(self.act(self.conv1b(x)))))  # 1b
        x = self.drop2(self.norm2(self.pool2(self.act(self.conv2b(self.act(self.conv2(x)))))))
        x = self.drop3(self.norm3(self.pool3(self.act(self.conv3b(self.conv3(x))))))
        x = self.global_avg_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.linear4(self.act(self.linear3(self.dropout(self.act(self.linear2(self.dropout(self.act(self.linear1(x)))))))))

        return x

# _______________________________________Loss Function__________________________________________
class video2class(nn.Module):
    def __init__(self, convolution):
        super(video2class, self).__init__()
        self.encoder = convolution

    def forward(self, source, target):
        batch_size = source.shape[0]
        target_len = target.shape[0]
        target_size = 1 #binary
        #outputs = torch.zeros(target_len, batch_size, target_size)
        output = self.encoder(source)
        #for t in range(target_len):
        #    output = self.encoder(source)
        #    outputs[t] = output
        return output

# ________________________________________Training______________________________________________
CONV = CNN3d()
MODEL = video2class(CONV).to(device)
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=learning_rate)
CRITERION = nn.BCEWithLogitsLoss()

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(iterator)):
        #Adjust input shape

        inputs = inputs.view(inputs.shape[0], inputs.shape[2], inputs.shape[1], inputs.shape[-2], inputs.shape[-1])
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(inputs, labels)
        loss = criterion(output.squeeze(), labels)
        acc = binary_acc(output, labels.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(iterator)):

            inputs = inputs.view(inputs.shape[0], inputs.shape[2], inputs.shape[1], inputs.shape[-2], inputs.shape[-1])
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, labels)
            loss = criterion(output.squeeze(), labels)
            acc = binary_acc(output, labels.unsqueeze(1))
            epoch_loss += loss.item()
            epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def print_loss(train_loss, eval_loss):
    plt.figure(figsize=(10,10))
    plt.plot(np.array(train_loss), label = 'Train Loss')
    plt.plot(np.array(eval_loss), label = 'Validation Loss')
    plt.title('Training Loss Plot')
    plt.show()


#
CLIP = 1
best_acc = 0

training_loss = []
validation_loss = []
for epoch in range(1, num_epochs+1):
    print(f'Epoch {epoch}  training')
    train_loss, train_accuracy = train(MODEL, loader, OPTIMIZER, CRITERION, CLIP)
    training_loss.append(train_loss)
    print(f'Epoch {epoch}: |  Training Loss = {train_loss:.5f} | Acc: {train_accuracy:.3f}')
    print(f'Epoch {epoch}  evaluating')
    val_loss, val_acc = evaluate(MODEL, val_loader, CRITERION)
    validation_loss.append(val_loss)
    print(f'Epoch {epoch}: |  Test loss = {val_loss:.5f} | Acc: {val_acc:.3f}')
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(MODEL.state_dict(), 'model_{}.pt'.format('CLASSIFICATION'))
        print("The model has been saved!")
        print(f'\tBest Train Accuracy: {train_accuracy:.3f} | Best Test Accuracy {val_acc:.3f}')


print_loss(train_loss, validation_loss)













