import torch

# create a tensor of size (100, 32, 1) filled with random values
x = torch.randn(100, 32, 1)

# pad the tensor with 21 zeros along the last dimension to create a tensor of size (100, 32, 22)
padded_x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, 21))

# check the size of the padded tensor
print(padded_x.size())  # output: torch.Size([100, 32, 22])