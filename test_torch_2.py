import torch

tensor = torch.FloatTensor([[0,0,0,0,1,1,2,3,0,3,1],[0,1,0,0,1,1,0,0,0,3,1]])

print(tensor)


# print(tensor.sum())
# print(torch.nonzero(tensor.data))
# # print(torch.nonzero(tensor.data.size(0)))
print((tensor == 3).sum())
print((tensor == 3))
print((tensor == 2).sum())
print((tensor == 0).sum())

list = [0,1,2]

rst = list[1] * int((tensor==0).sum())
print(rst)