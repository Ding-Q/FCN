import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
import cv2

print("PyTorch Version",torch.__version__)

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.block1_conv1 = nn.Conv2d(3, 8, 3, stride=2)
        self.block1_conv2 = nn.Conv2d(8, 8, 3, stride=2)
        self.block2_conv1 = nn.Conv2d(8, 16, 3, stride=2)
        #self.block2_conv2 = nn.Conv2d(16, 16, 3, stride=2)
        self.block3_conv1 = nn.Conv2d(16, 32, 3, stride=2)
        #self.block3_conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.block3_conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.block4_conv1 = nn.Conv2d(32, 32, 3, stride=2)
        #self.block4_conv2 = nn.Conv2d(64, 64, 3, stride=2)
        #self.block4_conv3 = nn.Conv2d(64, 64, 3, stride=2)
        #self.block5_conv1 = nn.Conv2d(64, 64, 3, stride=2)
        #self.block5_conv2= nn.Conv2d(64, 64, 3, stride=2)
        self.block5_conv3= nn.Conv2d(32, 32, 3, stride=2)
        self.fc6_conv=nn.Conv2d(32, 32, 1, stride=2)
        #self.fc7_conv=nn.Conv2d(32, 32, 1, stride=2)
        #self.fr_conv=nn.Conv2d(32, 32, 1, stride=2)
        self.deconv1=nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0)
        self.deconv2=nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0)
        self.deconv3=nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0)
        self.deconv4=nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0)
        self.deconv5=nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0)
        self.deconv6=nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0)
        self.deconv7=nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0)
        self.deconv8=nn.ConvTranspose2d(8, 1, 3, stride=2, padding=0)
        #self.deconv9=nn.ConvTranspose2d(8, 1, 3, stride=2, padding=0)
    def forward(self, x):
        x = F.relu(self.block1_conv1(x))
        x = F.relu(self.block1_conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.block2_conv1(x))
        #x = F.relu(self.block2_conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.block3_conv1(x))
        #x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.block4_conv1(x))
        #x = F.relu(self.block4_conv2(x))
        #x = F.relu(self.block4_conv3(x))
        #x = F.max_pool2d(x, 2, 2)
        #x = F.relu(self.block5_conv1(x))
        #x = F.relu(self.block5_conv2(x))
        x = F.relu(self.block5_conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.fc6_conv(x))
        x = F.dropout(x, 0.5)
        #x = F.relu(self.fc7_conv(x))
        #import pdb; pdb.set_trace()
        x = F.dropout(x, 0.5)
        #x = F.relu(self.fr_conv(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = F.relu(self.deconv6(x))
        x = F.relu(self.deconv7(x))
        x = F.relu(self.deconv8(x))
        #x = F.relu(self.deconv9(x))
        
        return x
model = FCN()

#summary(model, input_size=(3, 960, 960), batch_size=4)

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FCN_Dataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir('FCN/road'))
    
    def __getitem__(self, idx):
        img_name = os.listdir('FCN/road')[idx]
        imgA = cv2.imread('FCN/road/'+img_name)
        imgA = cv2.resize(imgA, (960,960))
        imgB = cv2.imread('FCN/label_road/'+img_name, 0)
        imgB = cv2.resize(imgB, (511,511))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 1)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)
            
        return imgA, imgB

img_road = FCN_Dataset(transform)

train_size = int(0.9 * len(img_road))
test_size = len(img_road) - train_size
train_dataset, test_dataset = random_split(img_road, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
print(len(train_dataloader.dataset),
len(test_dataloader.dataset), train_dataloader.dataset[7][1].shape) 

for batch_idx, (road, label_road)in enumerate(train_dataloader):
    print (road.shape) #label_road.shape)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
unloader = transforms.ToPILImage()
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    #image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    plt.pause(1)
    
plt.figure()
imshow(train_dataloader.dataset[20][1])
'''
lr = 0.01
momentum = 0.5
model = FCN().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(model, device, train_dataloader, optimizer, epoch, log_interval=10):
        model.train()
        for batch_idx, (road, label_road)in enumerate(train_dataloader):
            road, label_road = road.to(device), label_road.to(device)
            optimizer.zero_grad()
            #import pdb; pdb.set_trace()
            output = model(road)
            output = torch.sigmoid(output)
            #print(output)
            loss = F.binary_cross_entropy(output, label_road)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print("Train Epoch: {} [{}/{}] ({:0f}%)\tloss: {:.6f}"
                     .format(epoch, batch_idx * len(road), len(train_dataloader.dataset),
                                100. * batch_idx / len(train_dataloader), loss.item()))

epochs = 10
for epoch in range(1, epochs + 1):
    train(model, device, train_dataloader, optimizer, epoch
         
         )

save_model = True
if save_model:
    torch.save(model.state_dict(),"road_FCN.pt")
