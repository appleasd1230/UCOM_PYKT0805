import numpy as np

print(np.c_[1, 2, 3, 4], np.c_[1, 2, 3, 4].shape)
print(np.r_[1, 2, 3, 4], np.r_[1, 2, 3, 4].shape)
a1 = [1, 2]
a2 = [3, 4]
a3 = [5, 6]
print(np.c_[a1, a2, a3])
print(np.r_[a1, a2, a3])
print(np.hstack((a1, a2, a3)))
print(np.vstack((a1, a2, a3)))