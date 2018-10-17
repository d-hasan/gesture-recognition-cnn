import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from os import listdir
from shutil import copy
import torch
import pickle
import json

def plot(x, valid_acc, train_acc, args, path):
    train = 1.0 - savgol_filter(train_acc, 3, 2)
    val = 1.0 - savgol_filter(valid_acc, 3, 2)
    title = 'Batch Size: ' + str(args['batch_size']) + ' Learn. Rate: ' + str(args['lr'])
    plt.title(title)
    plt.plot(x, train, label='Training')
    plt.plot(x, val, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='best')
    plt.savefig(path + '/' + 'bs_' + str(args['batch_size']) + '_lr_' + str(args['lr']) + '.png')
    # plt.show()

def getPath(params):
    file = 'models/model_num.json'
    with open(file, 'r') as fp:
        data = json.load(fp)
    model_num = data['model_num']
    path = 'models/model_%d' % model_num
    os.makedirs(path)
    print('model num%d'% model_num)

    model_num += 1
    data = {'model_num': model_num}
    with open(file, 'w') as fp:
        json.dump(data, fp)

    return path

def loadConfig(path):
    with open(path) as file:
        config = json.load(file)

    return config

def saveModel(model, epoch, val_err, path):
    model_name = '/%d_%0.4f.pt' % (epoch, val_err)
    torch.save(model, path+model_name)

def saveData(path, steps, train_err, train_loss, val_err, val_loss, params):
    data = {'steps': steps,
            'train_err': train_err,
            'train_loss': train_loss,
            'val_err': val_err,
            'val_loss': val_loss,
            'params': params
            }
    batch_size = str(params['batch_size'])
    lr = str(params['lr'])
    epochs = str(params['epochs'])
    data_file = path + '/bs_%s_lr_%s_epochs_%s' % (batch_size, lr, epochs) + '.pkl'
    f = open(data_file, 'wb')
    pickle.dump(data, f)
    f.close()

def copyFiles(path):
    copy('main.py', path+'/')
    copy('model.py', path+'/')