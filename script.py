#Algorithm for sorting handwritten digits through SVD factoring.
#The USPS database was used, which has images of 256 pixels.
#Each set of images representing the same number was considered as a vector space in which the SVD factorization was applied. 
# A tolerance factor was applied to determine the amount of principal components to be used in a reduced subspace.
#The classification was made by analyzing the distance between the vectors that represent each image and the subspaces generated.

import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm

#Convert 1D array to an image
def view_image(arr):
    new_arr = -np.reshape(arr,(16,16))
    reshape_img = cv.resize(new_arr, (128,128))
    cv.imshow('test',reshape_img)
    cv.waitKey(0)

#Determines the dimension of the subspace based on the tolerance 
#and returns the generated subspace that represents the set of images of a number.
def space_reduction(xdata,ydata,tolerance,number):
    x = xdata[:,ydata == number]
    U,S,V = np.linalg.svd(x)
    for i in range(np.size(S)):
        sigma_sum = np.sum(S[:i])
        if sigma_sum/np.sum(S)>=tolerance:
            break
    U_red = U[:,:i]
    return U_red

#Calculates the distance between the vector representing the image and the generated subspace.
def distance(x,U):
    V1 = np.linalg.inv(np.dot(U.T,U))
    V2 = np.dot(U,np.dot(V1,U.T))
    d = np.linalg.norm(x-np.dot(V2,x))
    return d

#Calculates the distance between an image and subspaces and returns the prediction
def prediction(test_data,U0,U1,U2,U3,U4,U5,U6,U7,U8,U9):
    dist = np.zeros((10,1))
    predictions = np.zeros(np.size(test_data,1))
    for i in tqdm(range(np.size(test_data,1))):
        dist[0] = distance(test_data[:,i],U0)
        dist[1] = distance(test_data[:,i],U1)
        dist[2] = distance(test_data[:,i],U2)
        dist[3] = distance(test_data[:,i],U3)
        dist[4] = distance(test_data[:,i],U4)
        dist[5] = distance(test_data[:,i],U5)
        dist[6] = distance(test_data[:,i],U6)
        dist[7] = distance(test_data[:,i],U7)
        dist[8] = distance(test_data[:,i],U8) 
        dist[9] = distance(test_data[:,i],U9)
        predictions[i] = np.argmin(dist)
    return predictions

# Reading data base
df = pd.read_csv(r'Data.csv',header=None,delimiter=' ',skip_blank_lines= True)
df = df.dropna(1)

# Separating test and validation set
train, validation, test = np.split(df.sample(frac=1), [int(.6*len(df)),int(.8*len(df))])

xtrain = train.values[:,1:].T
ytrain = train.values[:,0]
xtest = test.values[:,1:].T
ytest = test.values[:,0]
xval = validation.values[:,1:].T
yval = validation.values[:,0]


# Determining the optimal tolerance
print('Tolerance definition')
tolerance_test = np.linspace(0.1,0.9,9)
accuracy = np.zeros(9)
for i in range(9):
    print('Tolerance = %.2f' %tolerance_test[i],end ='')
    U0 = space_reduction(xtrain,ytrain,tolerance_test[i],0)
    U1 = space_reduction(xtrain,ytrain,tolerance_test[i],1)
    U2 = space_reduction(xtrain,ytrain,tolerance_test[i],2)
    U3 = space_reduction(xtrain,ytrain,tolerance_test[i],3)
    U4 = space_reduction(xtrain,ytrain,tolerance_test[i],4)
    U5 = space_reduction(xtrain,ytrain,tolerance_test[i],5)
    U6 = space_reduction(xtrain,ytrain,tolerance_test[i],6)
    U7 = space_reduction(xtrain,ytrain,tolerance_test[i],7)
    U8 = space_reduction(xtrain,ytrain,tolerance_test[i],8)
    U9 = space_reduction(xtrain,ytrain,tolerance_test[i],9) 
    val_pred = prediction(xval,U0,U1,U2,U3,U4,U5,U6,U7,U8,U9)
    accuracy[i] = sum(val_pred==yval)/np.size(yval,0)
    print(' Acurracy : %.2f' %accuracy[i])

plt.figure()
plt.plot(tolerance_test,accuracy)
plt.title('Tolerance determination')
plt.ylabel('Accuracy')
plt.xlabel('Tolerance')
plt.show()



#Testing for accuracy on the test set
print('Test data')
tolerance = tolerance_test[np.argmax(accuracy)]

U0 = space_reduction(xtrain,ytrain,tolerance,0)
U1 = space_reduction(xtrain,ytrain,tolerance,1)
U2 = space_reduction(xtrain,ytrain,tolerance,2)
U3 = space_reduction(xtrain,ytrain,tolerance,3)
U4 = space_reduction(xtrain,ytrain,tolerance,4)
U5 = space_reduction(xtrain,ytrain,tolerance,5)
U6 = space_reduction(xtrain,ytrain,tolerance,6)
U7 = space_reduction(xtrain,ytrain,tolerance,7)
U8 = space_reduction(xtrain,ytrain,tolerance,8)
U9 = space_reduction(xtrain,ytrain,tolerance,9)
test_pred = prediction(xtest,U0,U1,U2,U3,U4,U5,U6,U7,U8,U9)

print(f'Acurracy in test data :%.2f' %(sum(test_pred==ytest)/np.size(ytest,0)))








