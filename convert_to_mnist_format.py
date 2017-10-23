#
# This python script converts a sample of the notMNIST dataset into
# the same file format used by the MNIST dataset. If you have a program
# that uses the MNIST files, you can run this script over notMNIST to
# produce a new set of data files that should be compatible with
# your program.
#
# Instructions:
#
# 1) if you already have a MNIST data/ directory, rename it and create
#    a new one
#
# $ mv data data.original_mnist
# $ mkdir convert_MNIST
#
# 2) Download and unpack the notMNIST data. This can take a long time
#    because the notMNIST data set consists of ~500,000 files
#
# $ curl -o notMNIST_small.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
# $ curl -o notMNIST_large.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
# $ tar xzf notMNIST_small.tar.gz
# $ tar xzf notMNIST_large.tar.gz
#
# 3) Run this script to convert the data to MNIST files, then compress them.
#    These commands will produce files of the same size as MNIST
#    notMNIST is larger than MNIST, and you can increase the sizes if you want.
#
# $ python convert_to_mnist_format.py notMNIST_small test 1000
# $ python convert_to_mnist_format.py notMNIST_large train 6000
# $ gzip convert_MNIST/*ubyte
#
# 4) After update, we cancel output path and replace with 'train', 'test' or test ratio number,
#    it not only work on 10 labels but more,
#    it depends on your subdir number under target folder, you can input or not input more command
#
# Now we define input variable like following:
# $ python convert_to_mnist_format.py target_folder test_train_or_ratio data_number
#
# target_folder: must give minimal folder path to convert data
# test_train_or_ratio: must define 'test' or 'train' about this data,
#                      if you want seperate total data to test and train automatically,
#                      you can input one integer for test ratio,
#                      e.q. if you input 2, it mean 2% data will become test data
# data_number: if you input 0 or nothing, it convert total images under each label folder,
#        e.q.
#          a. python convert_to_mnist_format.py notMNIST_small test 0
#          b. python convert_to_mnist_format.py notMNIST_small test
#          c. python convert_to_mnist_format.py notMNIST_small train 0
#          d. python convert_to_mnist_format.py notMNIST_small train
#

import numpy
import imageio
import glob
import sys
import os
import random

height = 0
width = 0

dstPath = "convert_MNIST"
testLabelPath = dstPath+"/t10k-labels-idx1-ubyte"
testImagePath = dstPath+"/t10k-images-idx3-ubyte"
trainLabelPath = dstPath+"/train-labels-idx1-ubyte"
trainImagePath = dstPath+"/train-images-idx3-ubyte"


def get_subdir(folder):
    listDir = None
    for root, dirs, files in os.walk(folder):
        if not dirs == []:
            listDir = dirs
            break
    listDir.sort()
    return listDir


def get_labels_and_files(folder, number=0):
    # Make a list of lists of files for each label
    filelists = []
    subdir = get_subdir(folder)
    for label in range(0, len(subdir)):
        filelist = []
        filelists.append(filelist)
        dirname = os.path.join(folder, subdir[label])
        for file in os.listdir(dirname):
            if (file.endswith('.png')):
                fullname = os.path.join(dirname, file)
                if (os.path.getsize(fullname) > 0):
                    filelist.append(fullname)
                else:
                    print('file ' + fullname + ' is empty')
        # sort each list of files so they start off in the same order
        # regardless of how the order the OS returns them in
        filelist.sort()

    # Take the specified number of items for each label and
    # build them into an array of (label, filename) pairs
    # Since we seeded the RNG, we should get the same sample each run
    labelsAndFiles = []
    for label in range(0, len(subdir)):
        count = number if number > 0 else len(filelists[label])
        filelist = random.sample(filelists[label], count)
        for filename in filelist:
            labelsAndFiles.append((label, filename))

    return labelsAndFiles


def make_arrays(labelsAndFiles, ratio):
    global height, width
    images = []
    labels = []
    imShape = imageio.imread(labelsAndFiles[0][1]).shape
    if len(imShape) > 2:
        height, width, channels = imShape
    else:
        height, width = imShape
        channels = 1
    for i in range(0, len(labelsAndFiles)):
        # display progress, since this can take a while
        if (i % 100 == 0):
            sys.stdout.write("\r%d%% complete" %
                             ((i * 100) / len(labelsAndFiles)))
            sys.stdout.flush()

        filename = labelsAndFiles[i][1]
        try:
            image = imageio.imread(filename)
            images.append(image)
            labels.append(labelsAndFiles[i][0])
        except:
            # If this happens we won't have the requested number
            print("\nCan't read image file " + filename)

    if ratio == 'train':
        ratio = 0
    elif ratio == 'test':
        ratio = 1
    else:
        ratio = float(ratio) / 100
    count = len(images)
    trainNum = int(count * (1 - ratio))
    testNum = count - trainNum
    if channels > 1:
        trainImagedata = numpy.zeros(
            (trainNum, height, width, channels), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width, channels), dtype=numpy.uint8)
    else:
        trainImagedata = numpy.zeros(
            (trainNum, height, width), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width), dtype=numpy.uint8)
    trainLabeldata = numpy.zeros(trainNum, dtype=numpy.uint8)
    testLabeldata = numpy.zeros(testNum, dtype=numpy.uint8)

    for i in range(trainNum):
        trainImagedata[i] = images[i]
        trainLabeldata[i] = labels[i]

    for i in range(0, testNum):
        testImagedata[i] = images[trainNum + i]
        testLabeldata[i] = labels[trainNum + i]
    print("\n")
    return trainImagedata, trainLabeldata, testImagedata, testLabeldata


def write_labeldata(labeldata, outputfile):
    header = numpy.array([0x0801, len(labeldata)], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(labeldata.tobytes())

def write_imagedata(imagedata, outputfile):
    global height, width
    header = numpy.array([0x0803, len(imagedata), height, width], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(imagedata.tobytes())


def main(argv):
    global idxLabelPath, idxImagePath
    # Uncomment the line below if you want to seed the random
    # number generator in the same way I did to produce the
    # specific data files in this repo.
    # random.seed(int("notMNIST", 36))
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    if len(argv) is 3:
        labelsAndFiles = get_labels_and_files(argv[1])
    elif len(argv) is 4:
        labelsAndFiles = get_labels_and_files(argv[1], int(argv[3]))
    random.shuffle(labelsAndFiles)

    trainImagedata, trainLabeldata, testImagedata, testLabeldata = make_arrays(
        labelsAndFiles, argv[2])

    if argv[2] == 'train':
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
    elif argv[2] == 'test':
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)
    else:
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)


if __name__ == '__main__':
    main(sys.argv)
