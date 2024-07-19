import torch
import torchvision.models as models

# Load the pre-trained ConvNeXt-Tiny model
convnext_tiny = models.convnext_tiny(pretrained=True)
# print(convnext_tiny.features)
# print(convnext_tiny.features[0])
# print(convnext_tiny)
# for x in convnext_tiny.features:
#     print(x)
#     print("-" * 50)

import torch.nn as nn


# Define a custom model using a part of ConvNeXt-Tiny
class CustomConvNeXt(nn.Module):
    def __init__(self, original_model):
        super(CustomConvNeXt, self).__init__()
        # Extract the desired layers/blocks
        self.part1 = nn.Sequential(
            original_model.features[0],  # First block
            original_model.features[1],  # Second block
            original_model.features[2],
            original_model.features[3],
            original_model.features[4],
            original_model.features[5],
            original_model.features[6],
            original_model.features[7]

        )
        print(self.part1)
        # You can add more layers or blocks as needed
        # self.part2 = nn.Sequential(...)

    def forward(self, x):
        x = self.part1(x)
        # x = self.part2(x)  # If additional parts are added
        return x


# Instantiate the custom model
custom_model = CustomConvNeXt(convnext_tiny)

# Test the custom model with dummy input
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image
output = custom_model(dummy_input)

print(output.shape)  # Output shape will depend on the layers used
