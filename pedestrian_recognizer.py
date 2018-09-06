#!/usr/bin/python
# -*- coding: latin-1 -*-

# Import the required modules
import cv2, os
import numpy as np
import scipy as sp
import math
import imutils
import random
from PIL import Image
from numpy import linalg as LA
from skimage import data
from sklearn import svm
from numba import double, jit

class Dataset:
    def __init__(self):
        self.paths = []
        self.images = []
        self.hog = []
        self.pyramid_images = []
        self.images_test = []

    #funções comuns entre os datasets
    def get_images_and_labels_pos(self, path_pos):
        c = 0
        for image_path in os.listdir(path_pos):
            if image_path.endswith('.png'):
                c +=1
                path = os.path.join(path_pos, image_path)
                self.paths.append(path)
                # Read the image and convert to grayscale
                image_pil = Image.open(path)
                # Convert the image format into numpy array
                image = np.array(image_pil, 'uint8')

                fullImage = image.copy()
                image = fullImage[16:144,16:80] #select block 128x64

                self.images.append(image)
            if c == 1000:
                break

    def get_images_and_labels_neg(self, path_neg):
        c = 0
        for image_path in os.listdir(path_neg):
            if image_path.endswith('.png'):
                c +=1
                path = os.path.join(path_neg, image_path)
                self.paths.append(path)

                # Read the image and convert to grayscale
                image_pil = Image.open(path)
                # Convert the image format into numpy array
                image = np.array(image_pil, 'uint8')

                fullImage = image.copy()
                #pega 10 janelas da imagem
                for k in range(0,10):
                    i = int(random.uniform(0, (fullImage.shape[0]-128))) #aleatorio entre 0 e image.shape[0]-128
                    j = int(random.uniform(0, (fullImage.shape[1]-64))) #aleatorio entre 0 e image.shape[1]-64

                    image = fullImage[i:i+128,j:j+64] #select block 128x64
                    self.images.append(image)


            if c == 1200: #pega apenas 1200 imagens e 10 janelas cada
                break

    def get_images_and_labels_pos_test(self, path_pos_test):

        for image_path in os.listdir(path_pos_test):
            if image_path.endswith('.png'):
                path = os.path.join(path_pos_test, image_path)
                self.paths.append(path)

                # Read the image and convert to grayscale
                image_pil = Image.open(path)
                # Convert the image format into numpy array
                image = np.array(image_pil, 'uint8')

                self.images_test.append(image)
                # img = image.copy()
                # # print image[0]
                # # print image[1]
                # print "Piramide de imagens positivas de teste"
                # cv2.pyrUp( image, img, ( int(image.shape[0]*2), int(image.shape[1]*2) ))
                # self.pyramid_images.append(img)
                # cv2.imshow( 'sdddas', img )
                # cv2.waitKey()
                # cv2.pyrDown( image, img, ( int(image.shape[0]/2), int(image.shape[1]/2) ))
                # self.pyramid_images.append(img)
                # cv2.imshow( 'sdahhs', img )
                # cv2.waitKey()

    def hardmining(self, path_neg):
        hard_images = []
        hard_paths = []
        hogs = []
        c = 0
        for image_path in os.listdir(path_neg):
            if image_path.endswith('.png'):
                path = os.path.join(path_neg, image_path)
                # Read the image and convert to grayscale
                image_pil = Image.open(path)
                # Convert the image format into numpy array
                image = np.array(image_pil, 'uint8')

                fullImage = image.copy()
                #pega 150 janelas da imagem
                for k in range(0,150):
                    i = int(random.uniform(0, (fullImage.shape[0]-128))) #aleatorio entre 0 e image.shape[0]-128
                    j = int(random.uniform(0, (fullImage.shape[1]-64))) #aleatorio entre 0 e image.shape[1]-64

                    image = fullImage[i:i+128,j:j+64] #select block 128x64
                    hard_hog = hog_calculation(image)
                    hard_hog = np.reshape(hard_hog, (1,1152))
                    result = clf.predict(hard_hog)
                    if result[0] > 0: #falso positivo
                        print ("Falso positivo "+str(c+1))
                        hard_images.append(image)
                        hard_paths.append(path)
                        hard_hog = np.reshape(hard_hog, (128,9))
                        hogs.append(hard_hog)
                        c+=1

            if c == 10000: #para todas as imagens negativas, continuar até dar 10 mil imagens
                break

        return hard_images, hard_paths, hogs

def get_end_points(point, angle, length):
    y, x = point

    y_final = int(y + length * math.sin(angle))
    x_final = int(x + length * math.cos(angle))

    return y_final, x_final

def hog_calculation(image):
    hog_vector = []
    # Calculate gradient
    gx = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True) #retorna matriz de magnitude e angulo

    wlinha = 8 - (image.shape[0]%8)
    wcoluna = 8 - (image.shape[1]%8)
    for k in xrange(0,image.shape[0],wlinha):
        for j in xrange(0,image.shape[1],wcoluna):
            mag_block = mag[k:k+wlinha,j:j+wcoluna] #blocos de 8 em 8
            angle_block = angle[k:k+wlinha,j:j+wcoluna]
            bin_vector =  np.zeros(9, dtype='uint8')

            for l in xrange(mag_block.shape[0]):
                for c in xrange(mag_block.shape[1]):
                    #histograma para 0 20 40 60 80 100 120 140 160 graus
                    value = np.mean(angle_block[l,c])
                    if (value >= 180):
                        value = value - 180
                    if (value < 0):
                        value = value + 180
                    value = value / 20
                    frac, whole = math.modf(value)
                    min_bin = whole * 20
                    max_bin = min_bin + 20
                    if (max_bin > 160):
                        max_bin = 0

                    min_percentual = value*frac
                    max_percentual = value*(1-frac)

                    bin_vector[int(min_bin/20)] += int(np.mean(mag_block[l,c])*min_percentual)
                    bin_vector[int(max_bin/20)] += int(np.mean(mag_block[l,c])*max_percentual)

            hog_vector.append(bin_vector)

            # for l in xrange(len(bin_vector)):
            #
            #     y_final, x_final  = get_end_points((k+1,j+1), math.radians(bin_vector[l]), 4)
            #
            #     cv2.line(img, (j,k), (x_final,y_final), (0,0,255), 1)
            #     cv2.imwrite("image.jpg", img)

    return hog_vector

np.set_printoptions(threshold=np.nan)
# path_pos = '../../../../../nobackup/bcc/lsl16/96X160H96/Train/pos'
path_pos = 'INRIAPerson/96X160H96/Train/pos'
# path_pos = './train_pos'
path_pos_test = './INRIAPerson/Test/pos'
# path_pos_test = './test_pos'
# path_neg = '../../../../../nobackup/bcc/lsl16/Train/neg'
path_neg = 'INRIAPerson/Train/neg'
# path_neg = './train_neg'

print "Lendo imagens positivas"
pos = Dataset()
pos.get_images_and_labels_pos(path_pos)
fast_hog = jit(double[:,:](double[:,:]))(hog_calculation)

print "Lendo imagens negativas"
neg = Dataset()
neg.get_images_and_labels_neg(path_neg)

hogs = []
yp = np.ones(len(pos.images), 'uint8')
k=0
for image in pos.images:

    hogs.append(fast_hog(image))
    # hogs.append(hog_calculation(image))
    print "Calculo de hog na imagem "+str(k+1)+" positiva"
    # file.write('Imagem '+str(k)+'\n')
    k+=1
np.save('posoutput', hogs)

hogs = []
yn = np.zeros(len(neg.images), 'uint8')
k=0
for image in neg.images:
    hogs.append(fast_hog(image))
    # hogs.append(hog_calculation(image))
    print "Calculo de hog na imagem "+str(k+1)+" negativa"
    k=k+1
np.save('negoutput', hogs)
hogs = []

xp = np.load('posoutput.npy')
xn = np.load('negoutput.npy')
number = len(pos.images) + len(neg.images)
hogs = np.concatenate((xp,xn))
hogs = np.reshape(hogs, (number, 1152))
yhogs = np.concatenate((yp,yn))

print "Treino SVM"
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(hogs, yhogs) #X sendo estrutura de m(imagens)*3780 [n_samples, n_features]; y o vetor de falso ou verdadeiro

hogs = []
print "Hardmining"
hard_images, hard_paths, hogs = neg.hardmining(path_neg)
np.save('hardimages', hard_images)
np.save('hardminingoutput', hogs)
hogs = []

xh = np.load('hardminingoutput.npy')
hard_images = np.load('hardimages.npy')
yh = np.zeros(len(hard_images), 'uint8')

number = len(pos.images) + len(neg.images) + len(hard_images)
if (len(xh) != 0):
    xh = np.reshape(xh, (len(xh),128,9))
    hogs = np.concatenate((xp,xn,xh), axis=0)
    yhogs = np.concatenate((yp,yn,yh))
else:
    hogs = np.concatenate((xp,xn), axis=0)
hogs = np.reshape(hogs, (number, 1152))

print "Retreino SVM"
clf.fit(hogs, yhogs) #X sendo estrutura de m(imagens)*3780 [n_samples, n_features]; y o vetor de falso ou verdadeiro

print "Lendo imagens positivas de teste"
pos.get_images_and_labels_pos_test(path_pos_test)
people = []
k = 0
for image in pos.images_test: #para cada imagem pegar 2 escalas na pirâmide, uma maior outra menor
    print ("Sliding window na imagem "+str(k+1)),
    # cv2.imshow('image', image)
    # cv2.waitKey(10)
    c = 0
    shape = 0
    for count in xrange(0, 5):
        print ("com imagem aumentada em "+str(1.10+shape)+" vezes")
        shape += 0.1
        image_maior = cv2.resize(image, (int(image.shape[1]*(1.10+shape)),int(image.shape[0]*(1.10+shape))), interpolation=cv2.INTER_CUBIC) #aumenta em 10%
        cv2.imshow('image_maior', image_maior)
        cv2.waitKey(50)
        for i in xrange(0,image_maior.shape[0],16): #anda 16 a 16 pixels
            for j in xrange(0,image_maior.shape[1],16):
                block = image_maior[i:i+128,j:j+64] #128 linhas, 64 colunas
                # cv2.imshow('block', block)
                # print block.shape[0], block.shape[1]
                block_hog = fast_hog(block)
                #imprimir hog
                # wlinha = 8 - (block.shape[0]%8)
                # wcoluna = 8 - (block.shape[1]%8)
                # img = block.copy()
                # w = 0
                # for lin in xrange(0,block.shape[0],wlinha):
                #     for col in xrange(0,block.shape[1],wcoluna):
                #             for l in xrange(len(block_hog[w])):
                #                 y_final, x_final  = get_end_points((lin+1,col+1), math.radians(block_hog[w][l]), 4)
                #
                #                 cv2.line(img, (col,lin), (x_final,y_final), (0,0,255), 1)
                #             w +=1
                # cv2.imshow('img', img)
                # cv2.waitKey(10)

                block_hog = np.reshape(block_hog, (1,len(block_hog)*9))
                if (len(block_hog) != 1152):
                    block_hog = np.resize(block_hog,(1,1152))
                result = clf.predict(block_hog)
                if result > 0: #achou uma pessoa
                    print ("Possível pessoa encontrada "+str(c+1))
                    c+=1
                    people.append([(i,j), 1.5*shape]) #salva o ponto inicial do bloco e a escala

    # cv2.imshow('image', image)
    # cv2.waitKey(10)
    c = 0
    shape = 0
    for count in xrange(0, 5):
        print ("com imagem reduzida em "+str(0.10+shape)+" vezes")
        shape += 0.1
        image_menor = cv2.resize(image, (int(image.shape[1]*(0.1+shape)),int(image.shape[0]*(0.1+shape))), interpolation=cv2.INTER_CUBIC) #diminui em 10%
        cv2.imshow('image_menor', image_menor)
        cv2.waitKey(50)
        for i in xrange(0,image_menor.shape[0],16): #anda 16 a 16 pixels
            for j in xrange(0,image_menor.shape[1],16):
                block = image_menor[i:i+128,j:j+64] #128 linhas, 64 colunas
                # cv2.imshow('block', block)
                # print block.shape[0], block.shape[1]
                block_hog = fast_hog(block)
                #imprimir hog
                # wlinha = 8 - (block.shape[0]%8)
                # wcoluna = 8 - (block.shape[1]%8)
                # img = block.copy()
                # w = 0
                # for lin in xrange(0,block.shape[0],wlinha):
                #     for col in xrange(0,block.shape[1],wcoluna):
                #             for l in xrange(len(block_hog[w])):
                #                 y_final, x_final  = get_end_points((lin+1,col+1), math.radians(block_hog[w][l]), 4)
                #
                #                 cv2.line(img, (col,lin), (x_final,y_final), (0,0,255), 1)
                #             w +=1
                # cv2.imshow('img', img)
                # cv2.waitKey(10)

                block_hog = np.reshape(block_hog, (1,len(block_hog)*9))
                if (len(block_hog) != 1152):
                    block_hog = np.resize(block_hog,(1,1152))
                result = clf.predict(block_hog)
                if result > 0: #achou uma pessoa
                    print("Possível pessoa encontrada "+str(c+1))
                    c+=1
                    people.append([(i,j), 0.5*shape]) #salva o ponto inicial do bloco e a escala

    #printar imagem original com retangulos nas pessoas
    for person in people:
        x, y = person[0]
        scale = person[1]
        #top-left corner and bottom-right
        cv2.rectangle(image, (x*scale, y*scale), ((x*scale+128), (y*scale+64)), (0,255,0), 2)
        cv2.imshow('people', image)
        cv2.waitKey(10)
    k +=1

## Non-maximum suppression
## Curve miss rate vs FPPW
