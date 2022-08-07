import torch
import numpy as np #including relevant libraries

#ways to init a tensor

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

ones = torch.ones_like(x_data) #retains properties of x_data
print(f"Ones Tensor: \n {ones} \n")

rand = torch.rand_like(x_data, dtype=torch.float) #overrides/changes the datatype in x_data
print(f"Random Tensor: \n {rand} \n")

#shape refers to the dimensions of a tensor

shape = (2,3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

#attributes

#attributes can refer to shape, datatype, and the device where it is stored

tensor = torch.rand(3, 4)

print(f"Shape: {tensor.shape} ")
print(f"Datatype: {tensor.dtype} ")
print(f"Device: {tensor.device} ")