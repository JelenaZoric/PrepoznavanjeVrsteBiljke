import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_adaptive, threshold_otsu
from skimage.morphology import closing, disk, dilation
from skimage.measure import label, regionprops
import csv
import random
import math
import operator
import cv2

file = open('podaciUkupno.csv', 'w')
fileRS = open('podaciRS.csv', 'w')
fileES = open('podaciES.csv', 'w')
fileRE = open('podaciRE.csv', 'w')
# 'hwratio,eccentricity,solidity,species'
txt_to_write_ukupno = []
txt_to_write_ES = []
txt_to_write_RS = []
txt_to_write_RE = []
brojVl = 13
brojKr = 13


def draw_regions(regs, img_size):
    img_res = np.ndarray((img_size[0], img_size[1]), dtype='float32')
    for reg in regs:
        coords = reg.coords
        for coord in coords:
            img_res[coord[0], coord[1]] = 1.
    return img_res


def skup1():
    for i in range(0, brojKr + brojVl):
        if i <= brojVl - 1:
            img = imread('slike/vl' + i.__str__() + '.jpg')
            vrsta = 0
        else:
            img = imread('slike/kr' + i.__str__() + '.jpg')
            vrsta = 1
        hsl, wsl = img.shape[:2]
        imgGray = rgb2gray(img)
        thresh = threshold_otsu(imgGray)
        imgThreshold = imgGray <= thresh
    #    ret, imgThreshold = cv2.threshold(imgGray, thresh, 255, cv2.THRESH_BINARY)
    # imgTreshold = 1 - threshold_adaptive(imgGray, block_size=75, offset=0.04)
        str_elem = disk(9)
        imgClosed = closing(imgThreshold, str_elem)
        imgClosed2 = closing(imgClosed, str_elem)
    #    if i == 2:
    #        plt.imshow(imgClosed2, 'gray')
    #        plt.show()
        labeledImg = label(imgClosed2)
        regions = regionprops(labeledImg)
        # print('Regioni:{}'.format(len(regions)))
        regions_vineleaf = []
        regions_krompir = []

        for region in regions:
            bbox = region.bbox
            h = bbox[2] - bbox[0]
            w = bbox[3] - bbox[1]
            ratio = float(h) / w
            if ratio < 1.1:
                if h > hsl / 2 and w > wsl / 2:
                    regions_vineleaf.append(region)
                    newlineUkupno = str(ratio) + ',' + str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
                    newlineES = str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
                    newlineRS = str(ratio) + ',' + str(region.solidity) + ',' + str(vrsta)
                    newlineRE = str(region.eccentricity) + ',' + str(ratio) + ',' + str(vrsta)
                    if i == 0:
                        txt_to_write_ukupno.append(newlineUkupno)
                        txt_to_write_ES.append(newlineES)
                        txt_to_write_RS.append(newlineRS)
                        txt_to_write_RE.append(newlineRE)
                    else:
                        txt_to_write_ukupno.append('\n' + newlineUkupno)
                        txt_to_write_ES.append('\n' + newlineES)
                        txt_to_write_RS.append('\n' + newlineRS)
                        txt_to_write_RE.append('\n' + newlineRE)
            else:
                if h > hsl / 2:
                    regions_krompir.append(region)
                    newlineUkupno = str(ratio) + ',' + str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(
                        vrsta)
                    newlineES = str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
                    newlineRS = str(ratio) + ',' + str(region.solidity) + ',' + str(vrsta)
                    newlineRE = str(region.eccentricity) + ',' + str(ratio) + ',' + str(vrsta)
                    txt_to_write_ukupno.append('\n' + newlineUkupno)
                    txt_to_write_ES.append('\n' + newlineES)
                    txt_to_write_RS.append('\n' + newlineRS)
                    txt_to_write_RE.append('\n' + newlineRE)

    file.writelines(txt_to_write_ukupno)
    fileES.writelines(txt_to_write_ES)
    fileRE.writelines(txt_to_write_RE)
    fileRS.writelines(txt_to_write_RS)
    file.close()
    fileRS.close()
    fileRE.close()
    fileES.close()


def flavia():
    for i in range(0, 100):
        if i < 54:
            img = imread('flavia/' + i.__str__() + '.jpg')
            vrsta = 0
        else:
            img = imread('flavia/' + i.__str__() + '.jpg')
            vrsta = 1
        hsl, wsl = img.shape[:2]
        imgGray = rgb2gray(img)
        thresh = threshold_otsu(imgGray)
        imgThreshold = imgGray <= thresh
        #    ret, imgThreshold = cv2.threshold(imgGray, thresh, 255, cv2.THRESH_BINARY)
        # imgTreshold = 1 - threshold_adaptive(imgGray, block_size=75, offset=0.04)
        str_elem = disk(9)
        imgClosed = closing(imgThreshold, str_elem)
        imgClosed2 = closing(imgClosed, str_elem)
        #    if i == 2:
        #        plt.imshow(imgClosed2, 'gray')
        #        plt.show()
        labeledImg = label(imgClosed2)
        regions = regionprops(labeledImg)

        for region in regions:
            bbox = region.bbox
            h = bbox[2] - bbox[0]
            w = bbox[3] - bbox[1]
            ratio = float(h) / w
            # if ratio < 1.1:
            if h > hsl / 2 or w > wsl / 2:
                    newlineUkupno = str(ratio) + ',' + str(region.eccentricity) + ',' + str(
                        region.solidity) + ',' + str(vrsta)
                    newlineES = str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
                    newlineRS = str(ratio) + ',' + str(region.solidity) + ',' + str(vrsta)
                    newlineRE = str(region.eccentricity) + ',' + str(ratio) + ',' + str(vrsta)
                    if i == 0:
                        txt_to_write_ukupno.append(newlineUkupno)
                        txt_to_write_ES.append(newlineES)
                        txt_to_write_RS.append(newlineRS)
                        txt_to_write_RE.append(newlineRE)
                    else:
                        txt_to_write_ukupno.append('\n' + newlineUkupno)
                        txt_to_write_ES.append('\n' + newlineES)
                        txt_to_write_RS.append('\n' + newlineRS)
                        txt_to_write_RE.append('\n' + newlineRE)
            # else:
            #    if h > hsl / 2:
            #        regions_krompir.append(region)
            #        newlineUkupno = str(ratio) + ',' + str(region.eccentricity) + ',' + str(
            #            region.solidity) + ',' + str(
            #            vrsta)
            #        newlineES = str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
            #        newlineRS = str(ratio) + ',' + str(region.solidity) + ',' + str(vrsta)
            #        newlineRE = str(region.eccentricity) + ',' + str(ratio) + ',' + str(vrsta)
            #        txt_to_write_ukupno.append('\n' + newlineUkupno)
            #        txt_to_write_ES.append('\n' + newlineES)
            #        txt_to_write_RS.append('\n' + newlineRS)
            #        txt_to_write_RE.append('\n' + newlineRE)

    file.writelines(txt_to_write_ukupno)
    fileES.writelines(txt_to_write_ES)
    fileRE.writelines(txt_to_write_RE)
    fileRS.writelines(txt_to_write_RS)
    file.close()
    fileRS.close()
    fileRE.close()
    fileES.close()


def load_dataset(filename, skup, training_set=[], test_set=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            if skup == 0:
                if x not in (1, 2, 3, 4, 7, 8, 14, 16, 17, 18, 19, 23):
                    training_set.append(dataset[x])
                else:
                    test_set.append(dataset[x])
            else:
                if 0 <= x < 40 or 50 <= x < 90:
                    training_set.append(dataset[x])
                else:
                    test_set.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1  # da ne uzima u obzir iza poslednje zapete, tj. naziv klase
    for x in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):  # vraca labele klasa najblizih suseda
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        print(response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        # for y in range(len(predictions[x])):
            if testSet[x][-1] == predictions[x]:
                correct += 1
                # break
    return (correct / float(len(testSet))) * 100.0


def main():
    skup1()
    skup = 0

    trainingSet = []
    testSet = []
    # split = 0.67
    load_dataset('podaciUkupno.csv', skup, trainingSet, testSet)
    print 'Trening set: ' + repr(len(trainingSet))  # broj training podataka
    print 'Test set: ' + repr(len(testSet))  # broj test podataka
    predictionsForK = []
    k = 1
    for x in range(len(testSet)):
        neighbors = get_neighbors(trainingSet, testSet[x], k)
    #    result = []
        result = get_response(neighbors)
    #   niz predvidjanja
        predictionsForK.append(result)
        print('> dobijena vrednost')
    #    for y in range(len(result)):
        print '>>'+repr(result)
        print('>>>>> stvarna vrednost=' + repr(testSet[x][-1]))
    accuracy = get_accuracy(testSet, predictionsForK)
    print('Procenat tacnosti prepoznavanja vrste biljke za sve osobine je: ' + repr(accuracy) + '%')

    trainingSet = []
    testSet = []
    load_dataset('podaciES.csv', skup, trainingSet, testSet)
    print 'Trening set: ' + repr(len(trainingSet))  # broj training podataka
    print 'Test set: ' + repr(len(testSet))  # broj test podataka
    predictionsForKES = []
    k = 1
    for x in range(len(testSet)):
        neighbors = get_neighbors(trainingSet, testSet[x], k)
        #    result = []
        result = get_response(neighbors)
        #   niz predvidjanja
        predictionsForKES.append(result)
        print('> dobijena vrednost')
        #    for y in range(len(result)):
        print '>>' + repr(result)
        print('>>>>> stvarna vrednost=' + repr(testSet[x][-1]))
    accuracy = get_accuracy(testSet, predictionsForKES)
    print('Procenat tacnosti prepoznavanja vrste biljke za ekscentricitet i solidnost je: ' + repr(accuracy) + '%')

    trainingSet = []
    testSet = []
    load_dataset('podaciRE.csv', skup, trainingSet, testSet)
    print 'Trening set: ' + repr(len(trainingSet))  # broj training podataka
    print 'Test set: ' + repr(len(testSet))  # broj test podataka
    predictionsForKRE = []
    k = 1
    for x in range(len(testSet)):
        neighbors = get_neighbors(trainingSet, testSet[x], k)
        #    result = []
        result = get_response(neighbors)
        #   niz predvidjanja
        predictionsForKRE.append(result)
        print('> dobijena vrednost')
        #    for y in range(len(result)):
        print '>>' + repr(result)
        print('>>>>> stvarna vrednost=' + repr(testSet[x][-1]))
    accuracy = get_accuracy(testSet, predictionsForKRE)
    print('Procenat tacnosti prepoznavanja vrste biljke za ratio i ekscentricitet je: ' + repr(accuracy) + '%')

    trainingSet = []
    testSet = []
    load_dataset('podaciRS.csv', skup, trainingSet, testSet)
    print 'Trening set: ' + repr(len(trainingSet))  # broj training podataka
    print 'Test set: ' + repr(len(testSet))  # broj test podataka
    predictionsForKRS = []
    k = 1
    for x in range(len(testSet)):
        neighbors = get_neighbors(trainingSet, testSet[x], k)
        #    result = []
        result = get_response(neighbors)
        #   niz predvidjanja
        predictionsForKRS.append(result)
        print('> dobijena vrednost')
        #    for y in range(len(result)):
        print '>>' + repr(result)
        print('>>>>> stvarna vrednost=' + repr(testSet[x][-1]))
    accuracy = get_accuracy(testSet, predictionsForKRS)
    print('Procenat tacnosti prepoznavanja vrste biljke za ratio i solidnost je: ' + repr(accuracy) + '%')


main()
