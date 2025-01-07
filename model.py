# Contains the siamese network model
import torch

class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Dropout(p=0.3),
        )

        # Defining the fully connected layers
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(160000, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2