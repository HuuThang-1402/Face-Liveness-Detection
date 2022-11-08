import argparse
import os
import numpy as np
import shutil

# # Creating Train / Test folders (One time use)
def split_data(args):
    posCls = '/real'
    negCls = '/fake'

    os.makedirs(args.root +'/train' + posCls)
    os.makedirs(args.root +'/train' + negCls)
    os.makedirs(args.root +'/test' + posCls)
    os.makedirs(args.root +'/test' + negCls)

    for i in [posCls, negCls]:
        # Creating partitions of the data after shuffeling
        currentCls = i
        src = args.root+currentCls # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8)])


        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Testing: ', len(test_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, args.root+"/train"+currentCls)

        for name in test_FileNames:
            shutil.copy(name, args.root+"/test"+currentCls)

    print("That's Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default='datasets')
    
    args = parser.parse_args()
    split_data(args)