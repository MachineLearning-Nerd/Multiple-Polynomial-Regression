import matplotlib.pyplot as plt
import numpy as np 
import pandas
from mpl_toolkits.mplot3d import Axes3D

# ax = plt.axes(projection = '3d')

df = pandas.read_csv('Ass1.csv')

voltage = np.asarray(df['Voltage'])
externalforce = np.asarray(df['External Force'])
electron_vel = np.asarray(df[' Electron Velocity'])

print(voltage)
print(externalforce)
print(electron_vel)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(voltage, externalforce, electron_vel, 'gray')
plt.show()