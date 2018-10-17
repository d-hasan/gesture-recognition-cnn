import numpy as np
import matplotlib.pyplot as plt

def loadData():
    instances = np.load('data/instances.npy')
    labels = np.load('data/labels.npy')

    return instances, labels

def calcAverage(instances, labels):
    avgs = []

    for letter in range(26):
        inst = instances[np.where(labels == letter)[0]].reshape((21500, 6))
        avg_values = np.mean(inst, axis = 0)
        avgs.append(avg_values)

    return avgs

def calcStd(instances, labels):
    stds = []

    for letter in range(26):
        inst = instances[np.where(labels==letter)[0]].reshape((21500, 6))
        std_values = np.std(inst, axis=0)
        stds.append(std_values)

    return stds

def plotBar(avgs, stds, letter):
    sensors = ['$a_x$', '$a_y$', '$a_z$', '$\omega_x$', '$\omega_y$', '$\omega_z$']

    title = 'Letter: ' + chr(letter + 97) + ' Average Sensor Values'
    plt.title(title)

    x = np.arange(6)

    plt.bar(x, avgs.reshape((6,)), color='g', width=0.5, yerr=stds.reshape((6,)))
    plt.ylabel('Acceleration')
    plt.xlabel('Sensors')
    plt.xticks(x, sensors)
    plt.ylim(-2, 12)

    for x, y, std in zip(x, avgs.reshape((6,)), stds.reshape((6,))):
        plt.text(x - 0.35, std + y + 0.1, '%0.3f \n +/- %0.3f' % (y, std))

    plt.show()
    plt.clf()

def generatePlots():
    instances, labels = loadData()
    avgs = calcAverage(instances, labels)
    stds = calcStd(instances, labels)

    L = ord('l') - 97
    I = ord('i') - 97
    T = ord('t') - 97

    for letter in [L, I, T]:
        plotBar(avgs[letter], stds[letter], letter)