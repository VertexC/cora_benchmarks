# importing package
import matplotlib.pyplot as plt
import numpy as np
  
# create data
arrays = []
with open("log.txt", "r") as f:
    lines = f.readlines()
    array = []
    for line in lines:
        c = line.strip()
        if c == "":
            arrays.append(array)
            array = []
        else:
            array.append(float(c))
# import pdb; pdb.set_trace()


f_arrays = []
for array in arrays:
    mean_val = np.mean(array)
    if mean_val < 0.015:
        continue
    
    f_arrays.append(array)

plt.figure(1, figsize=(20, 6))
for i, array in enumerate(f_arrays):
    # plot lines
    if i != 2:
        continue
    
    plt.plot(np.arange(len(array)), array, label = str(i))
    plt.legend()
    print(i, np.mean(array))
# import pdb; pdb.set_trace()
min_array = np.min(f_arrays, axis=0)
plt.plot(np.arange(len(min_array)), min_array, label = "min")
plt.legend()
print("min", np.mean(min_array))


plt.show()
plt.savefig("result.png")