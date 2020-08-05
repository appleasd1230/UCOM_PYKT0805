import matplotlib.pyplot as plt
import numpy as np

range1 = [-1, 3]
p = np.array([3])
print(f"range1 type={type(range1)}, p type={type(p)}")
print(f"range1={range1}, p={p}")
y1_y2 = p * range1 + 5
print(f"y1y2={y1_y2}, y1y2 type={type(y1_y2)}")
plt.plot(range1, y1_y2, c='green')
plt.show()