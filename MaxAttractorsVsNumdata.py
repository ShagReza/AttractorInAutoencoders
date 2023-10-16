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
from keras import backend as back
back.clear_session()
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
D=20
NumRandom=D*D
RandomData=4*np.random.rand(NumRandom,Dim+1)-2
data0=RandomData
X, Y = np.meshgrid(np.linspace(-2, 2, D), np.linspace(-2, 2,D ))
m=0
for i in range(D):
    for j in range(D):
        data0[m][0]=X[i, j]
        data0[m][1]=Y[i, j]
        m=m+1
#----------------------
        
        
        


#for Dim in range(MaxDim):
Dim=1
#Dim=4
#Dim=9
#Dim=14
#Dim=19
NumLayer=2
#NumLayer=3
#NumLayer=4
NumNor=9
#NumNor=19
#NumNor=29
MaxNumData=50
Repetition=5
Results=[1]*MaxNumData
for NumData in range(MaxNumData):
    MaxA=0
    for RR in range(Repetition):    

        #------------------------------------------------------------------------------
        # random input samples:  
        data=4*np.random.rand(NumData+1,Dim+1)-2
        #------------------------------------------------------------------------------
        # network structure
        NumNeorons=[NumNor+1]*(NumLayer+1)
        #------------------------------------------------------------------------------
        back.clear_session() #to clear net
        DataDim=data.shape[1]
        NetInput = Input(shape=(DataDim,))
        x=NetInput
        for i in range(NumLayer+1):
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
        OutPredict = model.predict(data)
        hist=history.history
        B = hist["mean_squared_error"][-1]
        #------------------------------------------------------------------------------
        if B < 1e-4:
            #Find attractors
            #NumRandom=100
            #RandomData=4*np.random.rand(NumRandom,Dim+1)-2
            data2=data0
            N=1000
            for i in range(N):
                OutPredict = model.predict(data2)
                data2=OutPredict
            #keep uniqe ones
            data2=np.round(data2,1)
            Attractors=np.unique(data2,axis=0)  
            #delete nan ones
            nan_array = np.isnan(Attractors)
            not_nan_array = ~ nan_array
            Attractors = Attractors[not_nan_array]
            MaxA=max(MaxA,int(len(Attractors)/(Dim+1)))
        #------------------------------------------------------------------------------
    Results[NumData]=MaxA
    
    
#-------------------------------   
import matplotlib.pyplot as plt
plt.plot(Results)
plt.xlabel("Number of data")
plt.ylabel("Number of arttractors")
PlotName="Dim=2 model=[10,10,10]"
#PlotName="Dim=2 model=[20,20,20,20]"
#PlotName="Dim=2 model=[30,30,30,30,30]"
#PlotName="Dim=5 model=[30,30,30,30,30]"
#PlotName="Dim=10 model=[30,30,30,30,30]"
#PlotName="Dim=15 model=[30,30,30,30,30]"
#PlotName="Dim=20 model=[30,30,30,30,30]"
plt.savefig(PlotName, dpi=300)
#------------------------------- 

