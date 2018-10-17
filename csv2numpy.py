from os import listdir
import pandas as pd
import numpy as np

def totalInstances(students=43, cols=6, rows=100, gest_instances=130):
    return (students*gest_instances, rows, cols)

def loadData():
    shape = totalInstances()
    instances = np.empty(shape)
    labels = np.empty((shape[0], 1), dtype=int)

    data_path = 'data/unnamed_train_data'
    students = listdir(data_path)
    row = 0
    for i in range(len(students)):
        student_path = data_path + '/student' + str(i)
        files = listdir(student_path)
        for letter in range(26):
            for iteration in range(5):
                file_path = student_path + '/' + chr(letter+97) + '_' + str(iteration+1) + '.csv'
                instance = np.genfromtxt(file_path, delimiter=",")
                instances[row,] = instance[:,1:]
                labels[row,] = letter
                row += 1
    return instances, labels

def saveData():
    instances, labels = loadData()
    np.save('data/instances.npy', instances)
    np.save('data/labels.npy', labels)
