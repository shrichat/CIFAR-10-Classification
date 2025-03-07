import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from torchvision.models import resnet18
from google.colab import drive





drive.mount('/content/drive')




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model=resnet18(pretrained=True)
        self.model.fc=nn.Linear(self.model.fc.in_features, 10)


        #resnet18 resnet50

    def forward(self, x):
        return self.model(x)



def savecheckpoint(model,optimizer,epoch,accuracy_list,base_dir='/content/drive/My Drive/checkpoints/'):


  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    print("yes")

  filename=os.path.join(base_dir, f"checkpoint_{epoch+1}.pth")
  checkpoint={'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epoch': epoch,
              'accuracy_list': accuracy_list}
  torch.save(checkpoint, filename)
  print(f"Checkpoint was saved at epoch {epoch+1} as {filename}")





def load_checkpoint(model,optimizer,filename="checkpoint.pth"):
  if os.path.isfile(filename):
    checkpoint=torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy_list = checkpoint['accuracy_list']
    print(f"Checkpoint loaded, resuming from epoch {epoch+1}")
    return model, optimizer, epoch, accuracy_list
  else:
    print("No checkpoint found, starting from scratch")
    return model, optimizer, -1, []





#transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.RandomRotation(10),transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



class_indices = [[] for i in range(10)]
for idx, (img, label) in enumerate(trainset):
    class_indices[label].append(idx)


non_iid_splits = []
for i in range(5):
    split_indices = []
    labels = random.sample(range(10), 2)
    for label in labels:
        split_indices.extend(random.sample(class_indices[label], 5000))
    non_iid_splits.append(split_indices)



datasets = [torch.utils.data.Subset(trainset, indices) for indices in non_iid_splits]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def train_and_evaluate(model, optimizer, batch_size, epochs, start_epoch=0, accuracy_list=[]):
    criterion = nn.CrossEntropyLoss()
    batch_bag=[]


    for epoch in range(start_epoch, epochs):
        selected_dataset = random.choice(datasets)
        trainloader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size, shuffle=True)
        running_loss = 0.0



        for i, data in enumerate(trainloader):
          if i==0: #using only 1 mini batch to get the updates
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            #for name, param in model.named_parameters():
              #if param.grad is not None:
                #print(f'{name} gradient : {param.grad.mean()}')



            gradients=[param.grad.clone() for param in model.parameters()]#storing copies of the gradients in batch bag
            batch_bag.append(gradients)

            running_loss+=loss.item()
            break

        if (epoch+1)%100==0: #printing loss every 1000 epochs
            avg_loss=running_loss
            print(f'Epoch[{epoch + 1},{epochs}] loss: {avg_loss:.3f}')


        #calculate accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct/total
        accuracy_list.append(accuracy)
        if (epoch+1)%100==0:
          print(f'Validation accuracy {epoch + 1}: {accuracy:.2f}%')



        if (epoch+1)%32==0: #updating model every 32 epochs
          avg_gradients=[torch.mean(torch.stack([batch_bag[i][j] for i in range(len(batch_bag))]), dim=0) for j in range(len(batch_bag[0]))]

          with torch.no_grad():
            for param, avg_grad in zip(model.parameters(), avg_gradients):
                param-=optimizer.param_groups[0]['lr']*avg_grad

          batch_bag=[]


        if (epoch+1)%50==0:
          savecheckpoint(model, optimizer, epoch, accuracy_list)



    return accuracy_list







mini_batch_size=2
epochs=20000

model= SimpleCNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0002)
model, optimizer, start_epoch, accuracy_list = load_checkpoint(model, optimizer)





batch_size_2_20000_epochs = train_and_evaluate(model, optimizer, mini_batch_size, epochs, start_epoch=start_epoch, accuracy_list=accuracy_list)

#plotting the graph
plt.figure(figsize=(12, 6))

plt.plot(range(start_epoch+1, start_epoch+epochs+1), batch_size_2_20000_epochs, label=f'Mini batch size 2, {epochs} epochs')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.show()