import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from network import DigitNet

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 1e-4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root='/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
    
model = DigitNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch {} | Step {} | Loss {:.4f}'.format(epoch+1, i+1, loss.item()))
            
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        _, predictions = torch.max(output.data, 1)
        
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        
    print('Accuracy: {}%'.format(correct/total*100))
    
torch.save(model.state_dict(), 'model.ckpt')