import numpy as np
from sklearn.model_selection import train_test_split

random_seed = 9

def loadData():
    instances = np.load('data/instances.npy')
    labels = np.load('data/labels.npy')

    return instances, labels

def normalizeData():
    instances, labels = loadData()
    instances_transpose = np.empty((5590, 6, 100))

    for i in range(instances.shape[0]):
        instances[i] = instances[i] - np.mean(instances[i], axis=0)
        instances[i] = instances[i]/np.std(instances[i], axis=0)
        instances_transpose[i] = instances[i].transpose()

    np.save('data/normalized_data.npy', instances_transpose)

    return instances_transpose, labels

def trainValSplit():
    instances, labels = normalizeData()
    X_train, X_val, y_train, y_val = train_test_split(instances, labels, test_size=0.2, random_state=random_seed)
    np.save('data/train_data.npy', X_train)
    np.save('data/train_labels.npy', y_train)
    np.save('data/val_data.npy', X_val)
    np.save('data/val_labels.npy', y_val)

    return X_train, X_val, y_train, y_val
