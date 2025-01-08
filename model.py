# Contains the siamese network model
import torch
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)
from torchinfo import summary


class SiameseResNet(torch.nn.Module):
    """
    Wrapper class for the ResNet model to be used in the Siamese Network
    """

    def __init__(self, arch, weights=None, num_classes=1024):
        super(SiameseResNet, self).__init__()
        self.arch = arch
        self.weights = weights
        self.model = self._get_model(arch)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward_once(self, x):
        # Forward pass
        output = self.model(x)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

    def _get_model(self, arch):
        if arch == "resnet18":
            return resnet18(weights=ResNet18_Weights.DEFAULT)
        elif arch == "resnet34":
            return resnet34(weights=ResNet34_Weights.DEFAULT)
        elif arch == "resnet50":
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        elif arch == "resnet101":
            return resnet101(weights=ResNet101_Weights.DEFAULT)
        elif arch == "resnet152":
            return resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f"Invalid architecture: {arch}")
        

def test_ResNet():
    model = SiameseResNet("resnet50")
    summary(model.model, input_size=(1, 3, 224, 224))


if __name__ == "__main__":
    test_ResNet()
