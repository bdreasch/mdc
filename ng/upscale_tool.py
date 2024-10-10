from scipy import ndimage
import numpy as np

small = np.array([[1, 2],
                  [3, 4]])

# result = ndimage.zoom(small, 2,  mode="grid-constant", grid_mode=True)
# print(result)
#
# n = 2

result = np.kron(small, np.ones((4, 5)))
result = result / (4*5)
print(result)
