import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

with open('output.csv', 'r') as file:
    data = file.read().split(',')

# convert data from strings to numbers
for i in range(len(data)):
    data[i] = float(data[i])

inpSize = int(data[0])
outSize = int(data[1])
pointSize = inpSize + outSize


numPoints = (len(data)-2) / (pointSize)

for n in range(int(numPoints)):
    inputData = data[2 + n * pointSize:2 + n * pointSize + inpSize]
    outputData = data[2 + n * pointSize + inpSize:2 + n * pointSize + inpSize + outSize]
    ax.scatter(inputData[0],inputData[1],outputData[1],marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()