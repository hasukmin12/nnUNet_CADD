import numpy as np
a = np.zeros((2,2,3))
print(a)
print(a.shape)

# a = np.delete(a, 1,1)

a[:][1] = 1
print()
print(a[:].shape)
print(a[:][1].shape)