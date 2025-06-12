"""
    
    Main File for the PyTorch library and running of project


"""


## 00. pyTorch Fundamentals I.E. Start

import torch
import numpy as np
import pandas as pn
import matplotlib as mpl


def createBreak(topic):
    i = 0
    while True:
        print("\n")
        i += 1
        if i >= 3:
            print("Next Topic: ",topic,"\n")
            break
    pass
#random = torch.random(size=(3,4,10))
#zeros = torch.zeros()
#ones = torch.ones()
# size = (number of arrays, length, dimensions)
#arranged = torch.arange(start=0,end=10,step=3)
#print(arranged)
# can get an out put in the shape like something using 
# .zeros_like(input=Tensor Shape you want)

#Tensor types = torch.float32/torch.float
#basiclly all floats with bits using binary steps

some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")

some_tensor - 10
print(some_tensor, "Wont update untill re-asigned")

some_tensor = some_tensor - 10
print(some_tensor, "like this")

createBreak("Matrix Multiplication")

#Matrix Multiplication
# Multiplying tensors via * normal operators or torch.mul()
# Will result in Tensor(1,2,3) * Tensor(1,2,3) = Tensor(1,4,9)

#While Matrix Multiplication via torch.matmul() or @
#Has to be against two differnt sized tensors
#It will get a complete number

tensor = torch.tensor([1,2,3])
print(tensor*tensor,"Muiltiplied with *")

print(torch.matmul(tensor,tensor),"Matmul() function")

createBreak("MatMul Errors")

# Matmul has the TOTAL value of all them added together

#Becareful about shape errors when multiplying tensors/matrices

#You cannot multiply matrix of the same size 3,2 @ 3,2 
#The inner dimensions which is the columns in the first and rows in the second
#            This guy --> 2,3  @ 3,2 <-- And this one
#The other two can be what ever      
#So! We tranpose one to multiply

tensor_A = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                        [8, 11], 
                        [9, 12]], dtype=torch.float32)

# tensor_A @ tensor_B Will error

print(tensor_A,"\n")

print(tensor_B.T, "Transposed","\n")

#Try
print(torch.matmul(tensor_A,tensor_B.T))

#Also called dot products when multipying them ifykyk ifyd Google it 

#Get Good at above section

createBreak("Linear Layer")

torch.manual_seed(42) # This get explained later ln: ??? A Wizard has stolen the LINE!

# Feed-Forward layer/ Fully Connected Layer

# y = x *A^T +B
# x = input to layer 
# a = weights the values that gets better as the network learns
# b = Bias twords the output i.e. 

linear = torch.nn.Linear(in_features=2 # in_features = matches the inner dimension of input
                        ,out_features=6# out_features = describes the outer value
                        )

x = tensor_A #Defined earlier

output = linear(x) #input shape = 3,2

# in_features Follows the same rules as matmul() Inners have to match
# in_features and out_features being the first matrix

print(output)

createBreak("Aggregation")

x = torch.arange(0, 100, 10)
#0,10,20,30,40,...,...,90

print(x.min(),": Minimum")
print(x.max(),": Maximum")
print(x.type(torch.float32).mean(),": Mean of all") #Have to cast to float first
print(x.sum(),": Sum of All")
# or use torch.min() torch.max() .mean(tensor.type(torch.float32)) torch.sum()

#Find the index of min and max

print("minimum index: ",x.argmin())
print("maximum index: ",x.argmax())

#Sub Topic!!!
# Tensor casting
tensorI8 =  x.type(torch.int8)

tensorf16 =  x.type(torch.float16)

# Lowering the size of the tensor from float32 -> 16 only reduces the amount of information
# Float32 = 0.0000000000000000
# Float16 = 0.00000000 Half the info
# Float8 = 0.0000 #Half Again
# This isnt a completely accurate display just a visual one
