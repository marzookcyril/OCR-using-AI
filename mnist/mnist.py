#!/usr/bin/env python3

import numpy as np
import math
import os
import functools
import operator
import gzip
import struct
import array
import tempfile
import matplotlib.pyplot as plt

# class permetant de travailler avec la BDD MNIST, peut parser le fichier et prendre les images avec les labels et la data.
class MNIST():
    # all the data type of the IDX format
    DATA_TYPES = {0x08: 'B',
                  0x09: 'b',
                  0x0b: 'h',
                  0x0c: 'i',
                  0x0d: 'f',
                  0x0e: 'd'}
    
    labelOffset = 4 + 4 # magic number (32 int) + number of items (32 int)
    imageOffset = 4 + 4 + 4 + 4 # magic number (32 int) + number of items (32 int) + rows (32 int) + cols (32 int) 
    
    currentTrainImagePointer = 0
    currentTrainLabelPointer = 0
    currentTestImagePointer  = 0
    currentTestLabelPointer  = 0

    trainImageIndex = -1
    trainLabelIndex = -1
    testImageIndex  = -1
    testLabelIndex  = -1

    trainImageData = []
    trainLabelData = []
    testImageData  = []
    testLabelData  = []
    
    imageSize = 28
    
    # dans l'ordre testLabel, testImage, trainLabel, trainImage
    def __init__(self, dataPath = "data/", fileLabel = ["t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "train-labels-idx1-ubyte", "train-images-idx3-ubyte"]) :
        try:
                self.testLabel  = open(dataPath + fileLabel[0], "rb")
                self.testImage  = open(dataPath + fileLabel[1], "rb")
                self.trainLabel = open(dataPath + fileLabel[2], "rb")
                self.trainImage = open(dataPath + fileLabel[3], "rb")
        except IOError:
            print("Error: a file in the MNIST database is not here, or not name the right way")
            
        self.currentTrainImagePointer = self.imageOffset
        self.currentTestImagePointer  = self.imageOffset
        self.currentTrainLabelPointer = self.labelOffset
        self.currentTestLabelPointer  = self.labelOffset

        self.trainImageData = self.parse(self.trainImage)
        self.trainLabelData = self.parse(self.trainLabel)
        self.testImageData = self.parse(self.testImage)
        self.testLabelData = self.parse(self.testLabel)
        
    def parse(self, data):
        # on read le header et check que tout est ok

        header = data.read(4)
        
        if len(header) != 4:
            print("Invalid IDX file")
            return 0

        zeros, dtype, dim = struct.unpack('>HBB', header)
        
        if (zeros != 0):
            print("Invalid IDX file")
            return 0

        try:
            dtype = self.DATA_TYPES[dtype]
        except KeyError:
            print("Unknown data type")

            
        dim = struct.unpack('>' + 'i' * dim, data.read(4 * dim))
        print(dim)

        rawData = array.array(dtype, data.read())
        rawData.byteswap() # little endian

        return np.array(rawData).reshape(dim)
        
    def getTrainImage(self):
        return self.trainImageData

    def getTrainLabel(self):
        return self.trainLabelData

    def getTestImage(self):
        return self.testImageData

    def getTestLabel(self):
        return self.testLabelData

    def getNextTrainImage(self):
        self.trainImageIndex += 1
        return self.trainImageData[self.trainImageIndex]

    def getNextTrainLabel(self):
        self.trainLabelIndex += 1
        return self.trainLabelData[self.trainLabelIndex]

    def getNextTestImage(self):
        self.testImageIndex += 1
        return self.testImageData[self.testImageIndex]

    def getNextTestLabel(self):
        self.testLabelIndex += 1
        return self.testLabelData[self.testLabelIndex]

def normalize(image):
    maxx = np.amax(image)
    return image/maxx

def plotImage(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
