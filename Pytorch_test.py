#the code is heavily influenced by the pytorch tutorial, so I've added comments explaining what I've learnt.


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#downloading training data
training_data = datasets.FashionMNIST(#sets training dataset to the dataset received from FashionMNIST, which contains images of Zalando's article images
    root = "data",
    train = True, #defines whether the dataset is for training or testing
    download = True,
    transform = ToTensor(), )

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(), )

batch_size = 64 #sets batch size for dataloaders
train_dataloader = DataLoader(training_data, batch_size = batch_size) 
test_dataloader = DataLoader(test_data, batch_size = batch_size) #sets data aside for testing and training

for X, y in test_dataloader: #for each feature and row, 
    print(f"Shape of X [N, C, H, W]: {X.shape}") #print the dimensions of the array, 
    print(f"Shape of y: {y.shape} {y.dtype}") 
    break

device = "cuda" if torch.cuda.is_available() else "cpu" #uses cpu is cuda doesnt exist
print(f"Using {device} device")

class NeuralNetwork(nn.Module): #defining a class for the neural network
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() #flattens a range of dims into tensors
        self.linear_relu_stack = nn.Sequential( #creates a sequential container: modules are added in the same order they are passed 
            nn.Linear(28*28, 512), #applies a linear relationship in the form y = mX + c
            nn.ReLU(),
            nn.Linear(512, 512), #this part also defines the number of input and output features
            nn.ReLU(),
            nn.Linear(512, 10) )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

def train(dataloader, model, loss_fn, optimizer): #trains model for accuracy
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader): 
        X, y - X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) #calculates the error (cost) 

        #back propagation that adjusts the model's parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0: #if remainder of batch/100 is 0,
            loss, current = loss.iten(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

#ascertain performance against the test data
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.n_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100 * correct):>.1f}%, Avg loss: {test_loss:>8f} \n")