import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# configuration
img_dir = './dataset/images/'
metadata_file ='./dataset/HAM10000_metadata'

# preprocessing the data - 
#   splitting into training and testing sets
#   One hot encoding on labels
df = pd.read_csv(metadata_file)
y = df.pop('dx')
y = y.to_frame()
encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
# print('Training set shape X-', X_train.shape, 'y-', y_train.shape) 
# print('Training set shape X-', X_test.shape, 'y-', y_test.shape)

# custom dataset for returning tuples of imgs and labels
class SkinLesionsDataset(Dataset):
    def __init__(self, X, y, transform = None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_name = self.X.iloc[idx]['image_id']
        label = self.y[idx]
        img = Image.open(f'{img_dir}{img_name}.jpg')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
# CNN Model Architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # feature extraction with Convolutions, Relu, and max_pooling layers
        self.conv1 = nn.Conv2d(3, 24, 3)
        self.conv2 = nn.Conv2d(24, 18, 3)
        self.conv3 = nn.Conv2d(18, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # classification with linear layers
        self.linear1 = nn.Linear(6*59*59, 4096)
        self.linear2 = nn.Linear(4096, 512)
        self.linear3 = nn.Linear(512, 128)
        # no softamx activation needed bacause we are using Cross Entropy loss
        # In pytorch, it includes softmax
        self.linear4 = nn.Linear(128, 7) # 7 classes
        

    def forward(self, x):
        # input size = (3,128,128)
        x = nn.functional.relu(self.conv1(x))  # Output size = (24,126,126)
        x = self.pool(x)  # Output size = (24,63,63)
        x = nn.functional.relu(self.conv2(x))  # Output size = (18,61,61)
        x = nn.functional.relu(self.conv3(x))  # Output size = (6,59,59)
        x = torch.flatten(x,1)  # Output size = (1,6*59*59)
        x = nn.functional.relu(self.linear1(x))  # Output size = (1,4096)
        x = nn.functional.relu(self.linear2(x))  # Output size = (1,512)
        x = nn.functional.relu(self.linear3(x))  # Output size = (1,128)
        x = self.linear4(x)  # Output size = (1,7)
        return x
    
# Training loop
def train_model(model, optimizer, loss_func, num_epochs, device, train_dataloader, test_dataloader):
    print('Training started...')
    print('==============================================================')
    for current_epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_train_predictions = 0
        total_train_predictions = 0
        for i, (batch_images, batch_labels) in enumerate(train_dataloader):
            imgs =  batch_images.to(device)
            labels = batch_labels.clone().detach().to(device)
            # Forward pass, get the output from the model and
            # calculate the loss by comparing the model output and true labels
            model_output = model(imgs)
            loss = loss_func(model_output, labels)
            
            # Backpropagate the calculated loss
            optimizer.zero_grad() #zeroing the gradients
            loss.backward()
            optimizer.step()

            # update running loss and accuracy
            running_loss += loss.item()
            _, predictions = torch.max(model_output, 1)  # Get the higest probability predictions
            _,labels = torch.max(labels,1)
            total_train_predictions += len(predictions)
            correct_train_predictions += (predictions ==labels).sum().item()

        # evaluate the model after every epoch
        model.eval()
        correct_test_predictions = 0
        total_test_predictions = 0

        with torch.no_grad():
            for i, (batch_images, batch_labels) in enumerate(test_dataloader):
                imgs =  batch_images.to(device)
                labels = batch_labels.clone().detach().to(device)
                # Forward pass, get the output from the model and
                model_output = model(imgs)

                # calculating the accuracies
                _, predictions = torch.max(model_output, 1)  # Get the higest probability predictions
                _,labels = torch.max(labels,1)
                total_test_predictions += len(predictions)
                correct_test_predictions += (predictions ==labels).sum().item() 

        running_loss = running_loss/len(train_dataloader)
        train_accuracy = correct_train_predictions/total_train_predictions
        test_accuracy = correct_test_predictions/total_test_predictions
        print(f'Epoch [{current_epoch+1}/{num_epochs}] - ')
        print(f'   Training Accuracy: {round(train_accuracy*100,4)}% Validation Accuracy: {round(test_accuracy*100,4)}% Loss: {running_loss}')
    
    print('==============================================================')
    print('Training Completed!')
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
batch_size = 4
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

training_dataset = SkinLesionsDataset(X_train, y_train, transform=transform)
testing_dataset = SkinLesionsDataset(X_test, y_test, transform=transform)

train_dataloader = DataLoader(training_dataset, batch_size)
test_dataloader = DataLoader(testing_dataset, batch_size)

model = CNNModel().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_model(model, optimizer, loss_function, num_epochs, device, train_dataloader, test_dataloader)