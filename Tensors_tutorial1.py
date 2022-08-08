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

tensor1 = torch.rand(3, 4)

print(f"Shape: {tensor1.shape} ")
print(f"Datatype: {tensor1.dtype} ")
print(f"Device: {tensor1.device} ")

#operations:

#if torch.cuda.is_available():
#   tensor = tensor.to("cuda")

tensor2 = torch.ones(4, 4)
print(f"Row 1: {tensor2[0]}")
print(f"Column 1: {tensor2[:, 0]}")
print(f"Final Row: {tensor2[..., -1]}")

#editing an entire column (col 1)
tensor2[:, 1] = 0
print(tensor2)

#combining tensors - tensors are combined by addig new columns to the right

t1 = torch.cat([tensor2, tensor2, tensor2], dim = 1)
print(t1)

#arithmetic operations: 

#multiplication - multiplies these matrices. y1, y2, and y3 are equal.
#multiplies the ENTIRE matrix

y1 = tensor2 @ tensor2.T
y2 = tensor2.matmul(tensor2.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor2, tensor2.T, out = y3)

print(y1)

#element-wise product - again, z1, z2, and z3 are equal: these are different ways of
#multiplying the elements in matrices

z1 = tensor2 * tensor2

z2 = tensor2.mul(tensor2)

z3 = torch.rand_like(tensor2)
torch.mul(tensor2, tensor2, out = z3)

print(z1)

#forming single-element tensors by aggregation
agg = tensor2.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


#In-place operations: store the result in the operand
print(f"{tensor2} \n")
tensor2.add_(5) #adds 5 to every number in the tensor, stores it in tensor2
print(tensor2)

#converting from Tensor to Numpy Array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() #changes made to the tensor will affect the array too 
print(f"n: {n}")

#convert from np array to tensor
n2 = np.ones(5)
t = torch.from_numpy(n2) #changes made to the array reflect in the tensor