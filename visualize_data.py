import numpy as np
import matplotlib.pyplot as plt

def loadData():
    instances = np.load('data/instances.npy')
    labels = np.load('data/labels.npy')

    return instances, labels

def plot(instance, label, instance_num, file):
    colors = ['b', 'g', 'r', 'c', 'm', 'k']

    sensors = ['$a_x$', '$a_y$', '$a_z$', '$\omega_x$', '$\omega_y$', '$\omega_z$']

    title = 'Letter: ' + chr(label+97) + ' Instance: ' + str(instance_num)
    plt.title(title)

    for i in range(6):
        values = instance[:,i]
        x = np.linspace(0, 2, 100)
        plt.plot(x, values, label=r'%s' % sensors[i], color=colors[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.xlim(0, 2)
    plt.legend(loc='best')
    plt.savefig(file)
    print('saved files')
    plt.show()
    plt.clf()

def generatePlots():
    instances, labels = loadData()

    L = ord('l') - 97
    M = ord('m') - 97
    letters = [L, M]

    L_index = np.where(labels == L)[0][3:6]
    M_index= np.where(labels == M)[0][3:6]

    L_inst = instances[L_index]
    M_inst = instances[M_index]

    inst = [L_inst, M_inst]

    for i, letter in enumerate(letters):
        for j, instance in enumerate(inst[i]):
            file = 'figures/question2_2/' + chr(letter+97) + '_' + str(j) + '.png'
            plot(instance, letter, j, file)
