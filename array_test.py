import numpy as np

A = np.eye(4)
D = np.array([[0,1],[0,1]])
E = np.array([0,1])
Ex,Ey = np.mgrid[0:2, 0:2]
B = A[0:2,0:2]
C = A[Ex,Ey]
print(B.shape)
print(C.shape)
print(B)
print(C)