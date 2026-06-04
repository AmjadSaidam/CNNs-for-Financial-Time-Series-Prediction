import torch 
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader as dataloader
import numpy as np
import comet_ml

class CNN_model(nn.Module): 
    """convolutional neural network"""
    def __init__(self, fcn_hidden_lyr: int = 3, n_per_lyr: int = 30, f_sz = 3, p_sz = 4):
        super().__init__()
        # low level features (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 24, kernel_size=f_sz) # in_channel = dim of input, out_channel = 
        self.pool1 = nn.MaxPool2d(kernel_size=p_sz)
        # mid level featues
        self.conv2 = nn.Conv2d(24, 36, kernel_size=f_sz) 
        self.pool2 = nn.MaxPool2d(kernel_size=p_sz)

        # FCN 
        layers = []
        self.flatten = nn.Flatten()
        layers.append(nn.LazyLinear(out_features = n_per_lyr)) # infer input dim
        layers.append(nn.ReLU())
        for _ in range(fcn_hidden_lyr-1): # hidden layers 
            layers.append(nn.Linear(n_per_lyr, n_per_lyr))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_per_lyr, 2)) # output two logits (must for cross entropy loss to work)
        self.fc = nn.Sequential(*layers) # feed forward 

    def forward(self, x):
        # convolv and pool 
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # apply FCN 
        x = self.flatten(x)
        return self.fc(x) # output logit log(P(y_i | x_i) / 1-P(y_i | x_i)) = log(p_i / 1-p_i)
    
class CustomImageDataset():
    def __init__(self, dataset: list[np.ndarray], labels: np.ndarray):
        self.image = dataset
        self.label = labels
        pass

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = torch.tensor(self.image[idx], dtype = torch.float32).unsqueeze(0) # add extra dimension, ouput is (1, w, h)
        label = torch.tensor(self.label[idx])
        return image, label

class CNN():
    def __init__(self, fcn_hidden_lyr: int = 3, n_per_lyr: int = 30, f_sz = 3, p_sz = 4):
        self.model = CNN_model(fcn_hidden_lyr, n_per_lyr, f_sz, p_sz) # updated on training steps 

    # loss function and optimiser
    def CNN_train(self, training_loader: CustomImageDataset, epochs: int = 1, learning_rate: float = 0.001, comet: bool = False):
        loss_f = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params = self.model.parameters(), lr = learning_rate)
        exp = None

        # initialise trainer 
        self.model.train()
        for epoch in range(epochs):
            if comet:
                exp = comet_ml.Experiment(project_name = 'gaf-cnn')
            # initialise/re-initialise statistics
            total_loss = 0
            correct_pred = 0
            total_pred = 0
            for _, (image, label) in enumerate(tqdm(training_loader)):
                # output 
                logits = self.model(image) # compute output 
                loss = loss_f(logits, label) # compute logit loss
                # backpropagation 
                optim.zero_grad()
                loss.backward()
                optim.step()
                # predictions and metrics 
                pred = torch.argmax(logits, dim = 1) # returns index of max output
                correct_pred += 1 if (pred == label) else 0
                total_loss += loss.item()          
                total_pred += 1    
                if exp:
                    exp.log_metric('loss', loss.item(), epoch = epoch)
            # get statistics
            total_epoch_loss = total_loss / total_pred
            epoch_accuracy = correct_pred / total_pred
            print("Epoch{}: Epoch-Loss{}, Epoch Accuracy{}".format(epoch, total_epoch_loss, epoch_accuracy))
        assert exp == None, exp.end() 

    def CNN_evaluate(self, test_data: dataloader):
        sigs = []
        # loss function and model eval mode 
        loss_f = nn.CrossEntropyLoss()
        self.model.eval()
        # statistics 
        correct_pred = 0
        test_loss = 0
        total_pred = 0
        with torch.no_grad(): # disable gradiant tracking 
            for image, label in test_data:
                logit_opt = self.model(image) # predict (logits)
                prob = nn.Softmax(dim=1)(logit_opt) # prediction
                pred = torch.argmax(prob, dim = 1)
                loss_v = loss_f(logit_opt, label).item() # loss
                test_loss += loss_v * image.size(0) # number of pixels in output 
                # metrics 
                correct_pred += 1 if (pred == label) else 0
                total_pred += 1
                sigs.append(pred*2 - 1) # map back to [-1, 1] space 
            # model evaluation metrics 
            test_loss /= total_pred
            test_accuracy = correct_pred / total_pred
        return test_loss, test_accuracy, sigs