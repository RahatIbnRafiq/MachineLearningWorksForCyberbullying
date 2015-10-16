import numpy as np
import matplotlib.pyplot as plt

with open("test2.txt") as f:
    data = f.read()

data = data.split('\n')


x = [row.strip().split('    ')[0] for row in data]
y = [row.strip().split('    ')[1] for row in data]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Plot title...")    
ax1.set_xlabel('your x label..')
ax1.set_ylabel('your y label...')

ax1.plot(x,y, c='r', label='the data')

leg = ax1.legend()

plt.show()