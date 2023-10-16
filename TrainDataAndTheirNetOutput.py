



#------------------------------------------------------------------------------  
import numpy as np
from keras.layers import Activation as activation
from keras.layers.core import Dense
from keras import layers
from keras import Input
from keras.models import Model
from keras import optimizers
from keras import callbacks
import random
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# network structure
NumLayer=6
NumNeorons=[5,5,5,5,5,5]#[5,3]
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# random input samples:  
Dim=2
NumData=5
data=4*np.random.rand(NumData,Dim)-2
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
NumData=data.shape[0]
DataDim=data.shape[1]
NetInput = Input(shape=(DataDim,))
x=NetInput
for i in range(NumLayer):
    #x = Dense(NumNeorons[i], activation="sigmoid")(x)
    x = Dense(NumNeorons[i], activation="tanh")(x)
x = Dense(DataDim, activation="linear")(x)
output = x
model=Model(NetInput, output)
model.summary()
#------------------
opt1 = optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt1 = optimizers.SGD(lr=0.3, momentum=0.7, nesterov=False)
model.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
#history = model.fit(data, data, epochs=100000,shuffle=True,verbose=1,callbacks= [reduce_LR ])
history = model.fit(data, data, epochs=1000,shuffle=True,verbose=1)
model.save('model.h5')
#-----------------
#Find attractors
NumRandom=100
RandomData=4*np.random.rand(NumRandom,2)-2
data2=RandomData
N=10000
for i in range(N):
    OutPredict = model.predict(data2)
    data2=OutPredict
#keep uniqe ones
data2=np.round(data2,2)
Attractors=np.unique(data2,axis=0)    
print("NumAttractor",len(Attractors))   
#---------------
#---------------
#Draw attractors, attraction path and phase space for 2 and 3 dimentional data
import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape

#attraction path and phase space
data2=np.random.rand(1,2)
for i in range(NI):
    for j in range(NJ):
        data2[0][0]=X[i, j]
        data2[0][1]=Y[i, j]
        OutPredict = model.predict(data2)
        u[i,j]=OutPredict[0][0]-X[i, j]
        v[i,j]=OutPredict[0][1]-Y[i, j]
X1=X 
Y1=Y
#------------------
from operator import itemgetter
#Basin of attraction
N=100
data2=np.random.rand(1,2)
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
NI, NJ = X.shape
Out = np.zeros((30,30))
for i in range(NI):
    print(i)
    for j in range(NJ):
        data2[0][0]=X[i, j]
        data2[0][1]=Y[i, j]
        for k in range(N):
            OutPredict = model.predict(data2)
            data2=OutPredict
          
        mse = ((data2 - Attractors)**2).mean(axis=1)   
        Out[NI-i-1,j]=min(enumerate(mse), key=itemgetter(1))[0] 

#------------------                
plt.subplot(1,2,1)
for i in range( np.shape(Attractors)[0]):
    plt.plot(Attractors[i][0],Attractors[i][1],'ro')
for i in range( np.shape(data)[0]):
    plt.plot(data[i][0],data[i][1],'b*')
plt.streamplot(X, Y, u, v)
plt.quiver(X, Y, u, v, units='width')
plt.axis('square')
plt.axis([-3, 3, -3, 3])
plt.title('Phase Plane')
#---
plt.subplot(1,2,2)
for i in range( np.shape(Attractors)[0]):
    plt.plot(Attractors[i][0],Attractors[i][1],'ro')
for i in range( np.shape(data)[0]):
    plt.plot(data[i][0],data[i][1],'b*')
plt.imshow(Out, cmap='jet', extent=[-3, 3, -3, 3])  
plt.title('Basin Of Attraction')
#---
PlotName="JUst Train data"
plt.savefig(PlotName, dpi=300)
plt.clf()
#-----------------------------





#----------------------------------------------------------------------------
for i in range (3000):
    dataOut = model.predict(data)
    data2=np.concatenate((data, dataOut), axis=0) 
    data2goal=np.concatenate((data, data), axis=0) 
    history = model.fit(data2, data2goal, epochs=1,shuffle=True,verbose=1)
#-----------------
#Find attractors
NumRandom=100
RandomData=4*np.random.rand(NumRandom,2)-2
data2=RandomData
N=10000
for i in range(N):
    OutPredict = model.predict(data2)
    data2=OutPredict
#keep uniqe ones
data2=np.round(data2,1)
Attractors=np.unique(data2,axis=0)    
print("NumAttractor",len(Attractors))   
#---------------
#---------------
#Draw attractors, attraction path and phase space for 2 and 3 dimentional data
import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape

#attraction path and phase space
data2=np.random.rand(1,2)
for i in range(NI):
    for j in range(NJ):
        data2[0][0]=X[i, j]
        data2[0][1]=Y[i, j]
        OutPredict = model.predict(data2)
        u[i,j]=OutPredict[0][0]-X[i, j]
        v[i,j]=OutPredict[0][1]-Y[i, j]
X1=X 
Y1=Y
#------------------
from operator import itemgetter
#Basin of attraction
N=100
data2=np.random.rand(1,2)
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
NI, NJ = X.shape
Out = np.zeros((30,30))
for i in range(NI):
    print(i)
    for j in range(NJ):
        data2[0][0]=X[i, j]
        data2[0][1]=Y[i, j]
        for k in range(N):
            OutPredict = model.predict(data2)
            data2=OutPredict
          
        mse = ((data2 - Attractors)**2).mean(axis=1)   
        Out[NI-i-1,j]=min(enumerate(mse), key=itemgetter(1))[0] 

#------------------                
plt.subplot(1,2,1)
for i in range( np.shape(Attractors)[0]):
    plt.plot(Attractors[i][0],Attractors[i][1],'ro')
for i in range( np.shape(data)[0]):
    plt.plot(data[i][0],data[i][1],'b*')
plt.streamplot(X, Y, u, v)
plt.quiver(X, Y, u, v, units='width')
plt.axis('square')
plt.axis([-3, 3, -3, 3])
plt.title('Phase Plane')
#---
plt.subplot(1,2,2)
for i in range( np.shape(Attractors)[0]):
    plt.plot(Attractors[i][0],Attractors[i][1],'go')
for i in range( np.shape(data)[0]):
    plt.plot(data[i][0],data[i][1],'b*')
plt.imshow(Out, cmap='jet', extent=[-3, 3, -3, 3])  
plt.title('Basin Of Attraction')
#---
PlotName="WithTrainOutputs"
plt.savefig(PlotName, dpi=300)
plt.clf()
#-----------------------------  


















  
#------------------------------------------------------------------------------
NumData=data.shape[0]
DataDim=data.shape[1]
NetInput = Input(shape=(DataDim,))
x=NetInput
for i in range(NumLayer):
    x = Dense(NumNeorons[i], activation="tanh")(x)
x = Dense(DataDim, activation="linear")(x)
output = x
model=Model(NetInput, output)
model.summary()
#-----
opt1 = optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt1 = optimizers.SGD(lr=0.3, momentum=0.7, nesterov=False)
model.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
history = model.fit(data, data, epochs=1,shuffle=True,verbose=1)
#-----
for i in range (5000):
    dataOut = model.predict(data)
    data2=np.concatenate((data, dataOut), axis=0) 
    data2goal=np.concatenate((data, data), axis=0) 
    history = model.fit(data2, data2goal, epochs=1,shuffle=True,verbose=1)
model.save('model.h5')
#-----
#Find attractors
NumRandom=100
RandomData=4*np.random.rand(NumRandom,2)-2
data2=RandomData
N=10000
for i in range(N):
    OutPredict = model.predict(data2)
    data2=OutPredict
#keep uniqe ones
data2=np.round(data2,1)
Attractors=np.unique(data2,axis=0)    
print("NumAttractor",len(Attractors))   
#----
#Draw attractors, attraction path and phase space for 2 and 3 dimentional data
import numpy as np
import matplotlib.pyplot as plt
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape
#attraction path and phase space
data2=np.random.rand(1,2)
for i in range(NI):
    for j in range(NJ):
        data2[0][0]=X[i, j]
        data2[0][1]=Y[i, j]
        OutPredict = model.predict(data2)
        u[i,j]=OutPredict[0][0]-X[i, j]
        v[i,j]=OutPredict[0][1]-Y[i, j]
X1=X 
Y1=Y
#------
from operator import itemgetter
#Basin of attraction
N=100
data2=np.random.rand(1,2)
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
NI, NJ = X.shape
Out = np.zeros((30,30))
for i in range(NI):
    print(i)
    for j in range(NJ):
        data2[0][0]=X[i, j]
        data2[0][1]=Y[i, j]
        for k in range(N):
            OutPredict = model.predict(data2)
            data2=OutPredict
          
        mse = ((data2 - Attractors)**2).mean(axis=1)   
        Out[NI-i-1,j]=min(enumerate(mse), key=itemgetter(1))[0] 
#---               
plt.subplot(1,2,1)
for i in range( np.shape(Attractors)[0]):
    plt.plot(Attractors[i][0],Attractors[i][1],'ro')
for i in range( np.shape(data)[0]):
    plt.plot(data[i][0],data[i][1],'b*')
plt.streamplot(X, Y, u, v)
plt.quiver(X, Y, u, v, units='width')
plt.axis('square')
plt.axis([-3, 3, -3, 3])
plt.title('Phase Plane')
#---
plt.subplot(1,2,2)
for i in range( np.shape(Attractors)[0]):
    plt.plot(Attractors[i][0],Attractors[i][1],'ro')
for i in range( np.shape(data)[0]):
    plt.plot(data[i][0],data[i][1],'b*')
plt.imshow(Out, cmap='jet', extent=[-3, 3, -3, 3])  
plt.title('Basin Of Attraction')
#---
PlotName="TrainAndItsOutputFromScratch"
plt.savefig(PlotName, dpi=300)
plt.clf()
#-----------------------------------------------------------