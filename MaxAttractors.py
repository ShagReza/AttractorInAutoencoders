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
Repetition=3
MaxDim=10
MaxNumData=20
NumDataList=[0,1,4,9,14,19]
MaxNumLayer=5
MaxNumNeorons=20
NumNeoronsList=[0,1,4,9,14,19]
text_file = open("MaxAttractors.txt", "w")
#------------------------------------------------------------------------------



for Dim in range(MaxDim):
    #for NumData in range(MaxNumData):
    for NumData in NumDataList:
        for NumLayer in range(MaxNumLayer):
            #for NumNor in range (MaxNumNeorons):
            for NumNor in NumNeoronsList:
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
                    #------------------------------------------------------------------------------
                    #Find attractors
                    NumRandom=100
                    RandomData=4*np.random.rand(NumRandom,Dim+1)-2
                    data2=RandomData
                    N=10000
                    for i in range(N):
                        OutPredict = model.predict(data2)
                        data2=OutPredict
                    #keep uniqe ones
                    data2=np.round(data2,2)
                    Attractors=np.unique(data2,axis=0)  
                    #delete nan ones
                    nan_array = np.isnan(Attractors)
                    not_nan_array = ~ nan_array
                    Attractors = Attractors[not_nan_array]
                    #
                    MaxA=max(MaxA,int(len(Attractors)/(Dim+1)))
                    #------------------------------------------------------------------------------
       
                #------------------------------------------------------------------------------
                #write results in a file
                text_file = open("MaxAttractors.txt", "a")
                str1=""
                str1='Dim:'+str(Dim+1)+'    NumData:'+str(NumData+1)+'    NumNor:'+str(NumNeorons)+'    MaximumNumOfAttractors:'+str( MaxA)+'\n'
                text_file.write(str1)
                text_file.close()
                #------------------------------------------------------------------------------





