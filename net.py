import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvNet(nn.Module):
    ''' expects 150x75 images '''

    image_channels = 3
    image_width = 150  # width of one eye
    image_height = 75  # height of one eye
    extra_features = 3

    def __init__(self):
        super(ConvNet, self).__init__()
        # convolutional kernels for image
        self.conv1 = nn.Conv2d(image_channels, 6, 3)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, 3)
        # expected sizes of tensors
        conv1_output = ConvNet.output_size(self.conv1, self.image_height * 2,
                                           self.image_width)
        conv2_output = ConvNet.output_size(self.conv2, *conv1_output)
        # an affine operation: y = Wx + b
        # input from conv network and add in the extra information
        self.fc1 = nn.Linear(
            conv2_output[0] * conv2_output[1] * self.conv2.out_channels +
            self.extra_features, 120)
        self.fc2 = nn.Linear(self.fc1.out_features, 84)

        # x and y coordinates as output
        self.fc3 = nn.Linear(self.fc2.out_features, 2)

    @staticmethod
    def output_size(convlayer, input_y, input_x):
        # as per documentation https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        output_x = (input_x + 2 * convlayer.padding[1] -
                    convlayer.dilation[1] *
                    (convlayer.kernel_size[1] - 1) - 1) // convlayer.stride[1]
        output_y = (input_x + 2 * convlayer.padding[0] -
                    convlayer.dilation[0] *
                    (convlayer.kernel_size[0] - 1) - 1) // convlayer.stride[0]

        return output_y, output_x

    def forward(self, eye_left, eye_right, features):
        x = torch.cat([eye_left, eye_right], dim=1)
        print(x.dim())
        # convolutional layers
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2)
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        # add non-convolutional inputs
        x = torch.cat([x, features], dim=0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
