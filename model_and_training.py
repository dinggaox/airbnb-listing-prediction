# Importing relevant Libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class AirbnbPredictor(nn.Module):
    '''
    Class that holds regression model for Airbnb listing price prediction

    input : feature tensor of N x _______
    output : float value of price estimate (normalized)
    '''

    def __init__(self, N):
        super(AirbnbPredictor, self).__init__()
        self.name = 'AirbnbPredictor'
        self.features = nn.Sequential(
            nn.Linear(N, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return x


def evaluate(model, loader, criterion, threshold=0.15):
    """ Evaluate the network on the validation set.

    Args:
        model: PyTorch neural network object
        loader: PyTorch data loader for the dataset
        criterion: The loss function

    Returns:
        acc: A scalar for the avg classification acc over the validation set
        loss: A scalar for the average loss function over the validation set
    """
    total_correct = 0
    total_loss = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, price = data
        if GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        # TODO GET PREDICTIONS HERE
        total_correct += 0 # TODO GET TOTAL CORRECT HERE WITH THRESHOLD
        total_epoch += len(labels)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

    acc = (float(total_epoch - total_correct) / total_epoch)
    loss = float(total_loss) / (i + 1)

    return acc, loss


def train_net(model, batch_size, learning_rate, threshold, num_epochs, name='default'):
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    if GPU:
        torch.cuda.manual_seed_all(1000)

    ########################################################################
    # PyTorch data loader objects
    train_loader, val_loader, test_loader = get_data_loader('../input/')
    print("Data loaded. Starting training:")

    ########################################################################
    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    ########################################################################
    # Train the network
    for epoch in range(num_epochs):
        total_correct = 0
        total_train_loss = 0.
        total_epoch = 0

        for i, data in enumerate(train_loader): # ,0)?
            inputs, price = data
            if GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate the statistics
            # TODO GET PREDICTIONS HERE
            total_correct += 0 # TODO GET TOTAL CORRECT HERE WITH THRESHOLD
            total_train_loss += loss.item()
            total_epoch += len(labels)
            
        train_acc[epoch] = float(total_correct) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_acc[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)

        print(("Epoch {}: Train acc: {}, Train loss: {} | "+
               "Validation acc: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_acc[epoch],
                   train_loss[epoch],
                   val_acc[epoch],
                   val_loss[epoch]))

        # Save the current model (checkpoint) to a file
        model_path = "MODEL{}_NAME{}_EPOCH{}".format(net.name, name, epoch)
        torch.save(net.state_dict(), model_path)

    print('Finished Training')

    return train_acc, train_loss, val_acc, val_loss