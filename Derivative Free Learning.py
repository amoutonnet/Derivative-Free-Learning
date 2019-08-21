# -*- coding: utf-8 -*-


########################################################################################################
#This code is made to explore the possibilities of derivative-free learning for simple neural networks.#
#It may not be the most proper code, but it works. Have fun.                                           #
########################################################################################################

from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten, Dropout, MaxPooling2D
from keras.layers import BatchNormalization, Activation
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.optimizers import Adam
from tqdm import tqdm
import os

from sklearn.datasets.samples_generator import make_moons,make_circles,make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import time
import warnings
import sys
import lhsmdu
from PIL import Image
plt.close('all')

K.set_image_dim_ordering('tf')
warnings.filterwarnings("ignore")

def reinit_training(file):
    try:
        os.remove(file)
        print("The training of",file,"has been reinitialized.")
    except OSError:
        print("The model",file,"is not trained yet.")

"""
This function takes gross array of weights as argument, and return it in
a single dimension
"""

def to_1D(params):
    test=0
    for i in params:
        if(test==0):
            test=1
            to_return=np.reshape(i,(-1,1))
        else:
            to_return=np.concatenate([to_return,np.reshape(i,(-1,1))],axis=0)
    return to_return

"""
This function take as argument an array of weights in 1D, and return it
in the same dimension as the neural network need it to update its weights
"""

def to_weights_shape(to_modif,params):
    new_params=[]
    update=to_modif
    for i in params:
        size=get_size(i)
        new_params+=[np.reshape(update[:size],np.shape(i))]
        update=update[size:]
    return new_params

"""
This function returns the length of a weights array that we
would have put in 1D.
"""

def get_size(param):
    size=1
    for k in np.shape(param):
        size*=k
    return size

"""
This functions add to array of weights together and returns a new one that is
the sum of the two passed as argument.
"""

def add_two_weights(param1,param2):
    to_return=[]
    for i,j in zip(param1,param2):
        to_return+=[i+j]
    return to_return

def print_sep(mess=None):
    print("\n")
    if(mess):
        toprint="-"*int((65-len(mess))/2)+str(mess)+"-"*int((65-len(mess))/2)
        if(len(toprint)!=65):
            toprint+="-"
        print(toprint)
    else:
        print("-"*65)
    print("\n")
    
def plot_comp(d_tr_l,d_tr_a,d_te_l,d_te_a,g_tr_l,g_tr_a,g_te_l,g_te_a,test,title1,title2,database):
    print_sep("Simple comparaison")
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,figsize=(6,4))
    ax[0].plot(range(1,len(d_tr_l)+1),d_tr_l,'-x', alpha=1,ms=8,zorder=-1)
    ax[0].plot(range(1,len(g_tr_l)+1),g_tr_l,'-x', alpha=1,ms=8,zorder=0)
    if(test):
        ax[0].plot(range(1,len(d_te_l)+1),d_te_l,':v', alpha=0.7,ms=5,linewidth=1.2,zorder=1)
        ax[0].plot(range(1,len(g_te_l)+1),g_te_l,':v', alpha=0.7,ms=5,linewidth=1.2,zorder=2)
    ax[0].set_ylabel("Loss")
    ax[0].set_yscale('log')
    ax[0].grid()
    ax[1].plot(range(1,len(d_tr_a)+1),d_tr_a,'-x',label=title1, alpha=1,ms=8,zorder=-1)
    ax[1].plot(range(1,len(g_tr_a)+1),g_tr_a,'-x',label=title2, alpha=1,ms=8,zorder=0)
    if(test):
        ax[1].plot(range(1,len(d_te_a)+1),d_te_a,':v',label="Test "+title1, alpha=0.7,ms=5,linewidth=1.2,zorder=1)
        ax[1].plot(range(1,len(g_te_a)+1),g_te_a,':v',label="Test "+title2, alpha=0.7,ms=5,linewidth=1.2,zorder=2)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylim([0.3,1.05])
    ax[1].legend(bbox_to_anchor=(0, -0.42, 1., .102),
          fancybox=True, ncol=2,mode="expand")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid()
    fig.suptitle("Evolution Loss function and Accuracy\nduring the learning on "+database+" database",fontsize=12)
    plt.show()
    
def plot_comp_arg(dtrl,dtra,gtrl,gtra,arg,argname,epochs):
    print_sep("Comparaison between algorithms on "+argname)
    markers=['v','x','P','d']
    linetypes=['--','-.',':','-']
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True,figsize=(6,6))
    for i,j,k in zip(range(len(arg)),dtrl,dtra):
        ax[0].plot(range(1,epochs+1),j,linetypes[i]+markers[i],label=argname+" n°"+str(arg[i]), alpha=1,ms=6,linewidth=1.5)
        ax[1].plot(range(1,epochs+1),k,linetypes[i]+markers[i],label=argname+" n°"+str(arg[i]), alpha=1,ms=6,linewidth=1.5)
    ax[0].set_ylabel("Loss")
    ax[0].grid()
    ax[0].set_title("Quasi-Deterministic",fontsize=10)
    ax[0].set_yscale('log')
    
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim([0,1.05])
    ax[1].grid()
    
    for i,j,k in zip(range(len(arg)),gtrl,gtra):
        ax[2].plot(range(1,epochs+1),j,linetypes[i]+markers[i],label=argname+" n°"+str(arg[i]), alpha=1,ms=6,linewidth=1.5)
        ax[3].plot(range(1,epochs+1),k,linetypes[i]+markers[i],label=argname+" n°"+str(arg[i]), alpha=1,ms=6,linewidth=1.5)
    ax[2].set_ylabel("Loss")
    ax[2].grid()
    ax[2].set_title("Stochastic-Genetic",fontsize=10)
    ax[2].set_yscale('log')
    ax[0].set_ylim((1e-1,10))
    ax[2].set_ylim((1e-1,10))
    ax[3].set_ylabel("Accuracy")
    ax[3].set_ylim([0,1.05])
    ax[3].set_xlabel("Epochs")
    ax[3].grid()
    ax[3].legend(bbox_to_anchor=(0, -0.52, 1., .102),
          fancybox=True, ncol=2,mode="expand")
    ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.suptitle("Comparaison between algorithms on "+argname,x=0.55, y=1.02, fontsize=14)
    plt.show()

class Optimizers():

    def __init__(self,model,printtype,C,size,learning_rate,beta,gamma):
        self.NN=model
        self.model=model.neuralnet
        self.printtype=printtype
        self.pb_dim=self.model.count_params()
        self.Cons=C
        self.data_size=size
        self.init_learning_rate=learning_rate
        self.learning_rate=learning_rate
        self.beta=beta
        self.gamma=gamma

    def print_params(self):
        print_sep("Parameters Used")
        print("--------------------------------------------")
        print("| Size of training database     | %d"%(self.data_size))
        print("--------------------------------------------")
        print("| C                             | %.1e"%(self.Cons))
        print("--------------------------------------------")
        print("| Initial learning rate         | %.1e"%(self.init_learning_rate))
        print("--------------------------------------------")
        print("| Beta                          | %.1f"%(self.beta))
        print("--------------------------------------------")
        print("| Gamma                         | %.1f"%(self.gamma))
        print("--------------------------------------------")


    def plot_epoch_loss_acc(self,loss,acc,learning_rates,epoch,tdeb,tfin):
        if(learning_rates[0]==-1):
            fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True,figsize=(6,4))
        else:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,figsize=(6,4))
            ax1=ax[0]
            ax[1].plot(range(1,len(loss)+1),learning_rates,'c-o')
            ax[1].set_ylabel("Learning Rate")
        ax1.plot(range(1,len(loss)+1),loss,'r-o')
        ax1.set_ylabel("Loss",color="r")
        ax1.set_xlabel("Batch iterations")
        ax1.tick_params(axis='y',labelcolor="r")
        ax1.set_ylim([0,ax1.get_ylim()[1]*1.2])
        ax2=ax1.twinx()
        ax2.plot(range(1,len(loss)+1),acc,'m-o')
        ax2.set_ylabel("Accuracy",color="m")
        ax2.tick_params(axis='y',labelcolor="m")
        ax2.set_ylim([0,ax2.get_ylim()[1]*1.2])
        fig.suptitle("Evolution per epoch n°"+str(epoch+1))
        plt.show()
        print("\n")
        print("Epoch n°%d took %.3f seconds to process" % (epoch+1,tfin-tdeb))

    def plot_global_loss_acc(self,global_training_loss,global_training_acc,global_test_loss,global_test_acc,tdeb,tfin):
        print_sep("Training Evolution")
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,figsize=(6,4))
        ax[0].plot(range(1,len(global_training_loss)+1),global_training_loss,'r-o',label="Train", alpha=0.7)
        ax[0].plot(range(1,len(global_test_loss)+1),global_test_loss,"b-x",label="Test", alpha=0.5)
        ax[0].set_ylabel("Loss")
        ax[0].legend(loc=1,fontsize=8)
        ax[1].plot(range(1,len(global_training_acc)+1),global_training_acc,'m-o',label="Train", alpha=0.7)
        ax[1].plot(range(1,len(global_test_acc)+1),global_test_acc,'c-+',label="Test", alpha=0.5)
        ax[1].set_ylabel("Accuracy")
        ax[1].set_xlabel("Epochs")
        ax[1].legend(loc=1,fontsize=8)
        #ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle("Evolution Loss function and Accuracy during the learning")
        plt.show()
        print("\n")
        print("The whole learning took %.3f seconds to process" % (tfin-tdeb))

    def optimize_weights(self,batch_size, epochs):
        #put the initial weights into the variable current_params
        self.current_params=self.model.get_weights()
        #in the case of the deterministic DFO, process the base one time if it
        #is the direct one that is used
        if(type(self) is DeterministicDFO):
            if(self.base==0):
                self.PSS=self.generatePSS(self.current_params)
        #initialize various things
        self.x,self.y=shuffle(self.NN.X_train,self.NN.y_train)
        self.x_in=self.x[0:batch_size]
        self.y_in=self.y[0:batch_size]
        self.current_f,self.current_acc=self.model.test_on_batch(self.x_in,self.y_in)
        self.success=False
        self.last_dir=None
        self.init_glob()
        postfix={}
        global_training_loss=[]
        global_training_acc=[]
        global_test_loss=[]
        global_test_acc=[]
        #if we want a progressbar whosing epochs evolution, intialize it
        if(1 in self.printtype):
            pbarglob=tqdm(range(epochs),ascii=True,
                      unit="e",leave=True, position=0,
                      file=sys.stdout,
                      bar_format='{l_bar}{bar}|Epoch n°{n_fmt}/{total_fmt}'+
                      '[Elapsed:{elapsed},{rate_fmt}]')
        #else, just initialize a simple iterative range array to go through epochs
        else:
            pbarglob=range(epochs)
        #start the epoch chrono
        tdeb_glob=time.time()
        for epoch in pbarglob:
            self.init_epoch()
            if(0 in self.printtype):
                #print the current information on epoch
                print_sep("Epoch n°"+str(epoch+1)+"/"+str(epochs))
            #shuffle the database
            self.x,self.y=shuffle(self.NN.X_train,self.NN.y_train)
            #initialize things
            batch=0
            self.counteval=0
            self.stagnation=0
            epoch_loss=[]
            epoch_acc=[]
            learning_rates=[]
            #if we want a progressbar whosing batches evolution, intialize it
            if(2 in self.printtype):
                pbar=tqdm(range(0,self.data_size,batch_size),unit="b",
                          file=sys.stdout, leave=True, position=0,
                          bar_format='Epoch n°%2.0d/%2.0d {l_bar}{bar}|Batch n°{n_fmt}/{total_fmt}[ET:{elapsed}, {rate_fmt}{postfix}]'%(epoch+1,epochs))
                postfix["Loss"]="?"
                postfix["Acc"]="?"
                pbar.set_postfix(postfix)
            #else, just initialize a simple iterative range array to go through
            #batches
            else:
                pbar=range(0,self.data_size,batch_size)
            #start the batch chrono
            tdeb=time.time()
            for b in pbar:
                batch+=1
                #select a random batch in the data
                batch_start=np.random.randint(0,self.data_size-batch_size)
                self.x_in=self.x[batch_start:batch_start+batch_size]
                self.y_in=self.y[batch_start:batch_start+batch_size]
                #do a single jump from the current xk to a xk+1
                self.optimization()
                #print information about the batch progression
                if(0 in self.printtype):
                        print("B n°%2.0d: loss = %.3f | lr = %.3e | eval = %d | stag = %d | succ = " % (batch,self.current_f,self.learning_rate,self.counteval,self.stagnation),self.success)
                if(2 in self.printtype):
                    postfix["Loss"]=self.current_f
                    postfix["Acc"]=self.current_acc
                    pbar.set_postfix(postfix)
                #store information about the epoch progression
                epoch_loss+=[self.current_f]
                epoch_acc+=[self.current_acc]
                learning_rates+=[self.learning_rate]
            #stop the batch chrono
            tfin=time.time()
#            if(self.current_acc>0.99):
#                break
            #plot information about the epoch progression
            if (3 in self.printtype):
                self.plot_epoch_loss_acc(epoch_loss,epoch_acc,learning_rates,epoch,tdeb,tfin)
            #store information about the global progression
            global_training_loss+=[epoch_loss[-1]]
            global_training_acc+=[epoch_acc[-1]]
            test_loss,test_acc=self.model.test_on_batch(self.NN.X_test,self.NN.y_test)
            global_test_loss+=[test_loss]
            global_test_acc+=[test_acc]
        #stop the epoch chrono
        tfin_glob=time.time()
        #plot information about the global progression
        if(4 in self.printtype):
            self.plot_global_loss_acc(global_training_loss,global_training_acc,
                                      global_test_loss,global_test_acc,
                                      tdeb_glob,tfin_glob)
        if(5 in self.printtype):
            self.print_params()
        return global_training_loss, global_training_acc, global_test_loss, global_test_acc

class GBO(Optimizers):

    def __init__(self,printtype,model,size):
        self.NN=model
        self.model=model.neuralnet
        self.printtype=printtype
        self.pb_dim=self.model.count_params()
        self.data_size=size
        self.init_learning_rate=-1

    def init_glob(self):
        self.learning_rate=self.init_learning_rate
        
    def init_epoch(self):
        return

    def optimization(self):
        self.current_f, self.current_acc=self.model.train_on_batch(self.x_in,self.y_in)

    def print_params(self):
        print_sep("Parameters Used")
        print("--------------------------------------------")
        print("| Size of training database     | %d"%(self.data_size))
        print("--------------------------------------------")

class DeterministicDFO(Optimizers):

    def __init__(self,learning_rate,beta,gamma,C,base,model,
                 size,printtype,lambd):
        """ Global parameters """
        super().__init__(model,printtype,C,size,learning_rate,beta,gamma)
        if(lambd):
            self.lambd=lambd
        else:
            self.lambd=4+int(3*np.log(self.pb_dim))
        self.base_size=int(self.lambd/2)
        self.base=base
        if(base==0):
            self.base_name="Direct"
        elif(base==2):
            self.base_name="Random"
        else:
            self.base_name="Random Normal"

    def init_glob(self):
        return
    
    def init_epoch(self):
        #learning rate
        self.learning_rate*=(self.gamma+self.beta)

    def print_params(self):
        super().print_params()
        print("| Base Type                     | %s"%(self.base_name))
        print("--------------------------------------------")
        print("| Base Size                     | %d"%(self.base_size*2))
        print("--------------------------------------------")

    """
    This function generates a basis, direct one if base=0, normal distribued
    random one if base=1, and a uniform over [0,1] random one if base=2,
    and return it.
    """

    def generatePSS(self,params):
        if(self.base==0):
            base=np.eye(self.pb_dim)
        elif(self.base==1):
            base=np.random.normal(size=(self.base_size,self.pb_dim))
        else:
            base=np.random.rand(self.base_size,self.pb_dim)
        PSS=[]
        for i in range(len(base)):
            PSS+=[to_weights_shape(np.reshape(base[i],(-1,1)),params)]
            PSS+=[to_weights_shape(np.reshape(-base[i],(-1,1)),params)]
        return PSS

    def optimization(self):
        #set success to False
        self.success=False
        #if the base is not the direct one, we need to generate a new one at
        #each iteration
        if(base!=0):
            self.PSS=self.generatePSS(self.current_params)
        #then we go through the base, and if we have a last successful
        #direction we go this way first
        for i in ([self.last_dir]+self.PSS if self.last_dir else self.PSS):
            #we create a temporary point in the current direction
            temp_params=add_two_weights(self.current_params,[self.learning_rate*j for j in i])
            self.model.set_weights(temp_params)
            #we evaluate this point
            temp_f,temp_acc=self.model.test_on_batch(self.x_in,self.y_in)
            self.counteval+=1
            #if the choice of this new point decreases the function
            if(temp_f<self.current_f-self.Cons*self.learning_rate**2):
                #set xk+1 as this new point
                self.current_params=temp_params
                #and f(xk+1) as the evaluation of this new point
                self.current_f=temp_f
                self.current_acc=temp_acc
                #increase the learning rate
                self.learning_rate=self.gamma*self.learning_rate
                #set the iteration as successful
                self.success=True
                self.stagnation=0
                #keep the last direction in mind
                self.last_dir=i
                #leave the for loop
                break
        #if the iteration has not been successful
        if(self.success!=True):
            #come back to the last weights
            self.model.set_weights(self.current_params)
            #forget the last direction
            self.last_dir=None
            self.stagnation+=1
            #decrease the learning rate
            self.learning_rate=self.beta*self.learning_rate

class GeneticDFO(Optimizers):

    def __init__(self,learning_rate,beta,gamma,C,model,size,printtype,
                 decrease,lambd):
        """ Global parameters """
        super().__init__(model,printtype,C,size,learning_rate,beta,gamma)
        self.decrease=decrease
        #population size
        if(lambd):
            self.lambd=lambd
        else:
            self.lambd=4+int(3*np.log(self.pb_dim))

    def init_epoch(self):
        #learning rate
        #self.learning_rate=self.learning_rate*(self.beta+self.gamma)
        #self.learning_rate=self.init_learning_rate/(self.beta+self.gamma)
        self.learning_rate=self.init_learning_rate
        #self.init_glob()

    def init_glob(self):

        """ Strategy parameter setting: Selection """
        #number of parents/points for recombination
        self.mu=self.lambd/2
        #weights
        self.weights=np.log(self.mu+0.5)-np.array([np.log(range(1,int(self.mu)+1))]).T
        #take int part of mu
        self.mu=int(np.floor(self.mu))
        #normalize weights
        self.weights = self.weights/sum(self.weights)
        #variance-effectiveness of sum w_i x_i
        self.mueff=float(sum(self.weights)**2/sum(self.weights*self.weights))

        """ Strategy parameter setting: Adaptation """
        #time constant for cumulation for C
        self.cc = (4 + self.mueff/self.pb_dim) / (self.pb_dim+4 + 2*self.mueff/self.pb_dim)
        #time constant for cumulation for learning_rate control
        self.cs = (self.mueff+2) / (self.pb_dim+self.mueff+5)
        #learning rate for rank-one update of C
        self.c1 = 2 / ((self.pb_dim+1.3)**2+self.mueff)
        #and for rank-mu update
        self.cmu = min(1-self.c1, 2 * (self.mueff-2+1/self.mueff) / ((self.pb_dim+2)**2+self.mueff))
        #damping for learning_rate
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.pb_dim+1))-1) + self.cs

        """ Initialize dynamic (internal) strategy parameters and constants """

        self.pc = np.zeros((self.pb_dim,1))
        #evolution paths for C and learning_rate
        self.ps = np.zeros((self.pb_dim,1))
        #B defines the coordinate system
        self.B = np.eye(self.pb_dim)
        #diagonal D defines the scaling
        self.D = np.ones((self.pb_dim,1))
        #covariance matrix C
        self.C = np.dot(np.dot(self.B,np.diag((self.D*self.D).T[0])),self.B.T)
        #C^-1/2
        self.invsqrtC = np.dot(np.dot(self.B,np.diag((1/self.D).T[0])),self.B.T)
        #track update of B and D
        self.eigeneval = 0
        #•expectation of ||N(0,I)|| == norm(randn(N,1))
        self.chiN=np.sqrt(self.pb_dim)*(1-1/(4*self.pb_dim)+1/(21*self.pb_dim^2))

    def update_params(self):
        """ Update parameters """
        self.ps=(1-self.cs)*self.ps+np.sqrt(self.cs*(2-self.cs)*self.mueff) * np.dot(self.invsqrtC,self.xmean-self.xold) / self.learning_rate
        hsig=int(bool(sum(self.ps*self.ps)/(1-(1-self.cs)**(2*self.counteval/self.lambd))/self.pb_dim < 2+4/(self.pb_dim+1)))
        self.pc=(1-self.cc)*self.pc+hsig*np.sqrt(self.cc*(2-self.cc)*self.mueff)*(self.xmean-self.xold)/self.learning_rate

        """ Adapt covariance matrix """
        artmp = (1/self.learning_rate)*self.arx-np.tile(self.xold,(1,self.mu))
        self.C=(1-self.c1-self.cmu)*self.C+\
                self.c1*(np.dot(self.pc,self.pc.T)+\
                (1-hsig)*self.cc*(2-self.cc)*self.C)+\
                self.cmu*np.dot(np.dot(artmp,np.diag((self.weights).T[0])),artmp.T)

        """ Adapt learning_rate """
        learning_rate_ES=self.learning_rate*np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN-1))
        if(self.decrease):    
            self.learning_rate=max(learning_rate_ES,self.learning_rate)
        else:
            self.learning_rate=learning_rate_ES


        """ Update B and D """
        if(self.counteval-self.eigeneval>self.lambd/(self.c1+self.cmu)/self.pb_dim/10):
            self.eigeneval=self.counteval
            self.C=np.triu(self.C)+np.triu(self.C,1).T
            self.D,self.B=np.linalg.eig(self.C)
            self.D=abs(np.real(self.D))
            self.D=np.reshape(self.D,(-1,1))
            self.D=np.sqrt(self.D)
            self.invsqrtC=np.dot(np.dot(self.B,np.diag((1/self.D).T[0])),self.B.T)
            

    """
    This function generates a new point around xmean accroding to the
    distribution associated to the Covariance Matrix C.
    """

    def generate_dev(self,xmean):
        prod=self.D*np.random.normal(size=np.shape(xmean))
        new_x=xmean+self.learning_rate*np.dot(self.B,prod)
        return new_x

    def print_params(self):
        super().print_params()
        print("| Offspring size                | %d"%(self.lambd))
        print("--------------------------------------------")
        print("| Decrease required             | %s"%(str(self.decrease)))
        print("--------------------------------------------")

    def optimization(self):
        #set success to False
        self.success=False
        #initialisation of x and f(x) value arrays
        self.arx=[]
        self.arfitness=[]
        #defining current xmean on current params in 1D
        self.xmean=to_1D(self.current_params)
        #generating new points y_k and store them and f(y_k) in arrays
        for k in range(self.lambd):
            self.arx+=[self.generate_dev(self.xmean)]
            current_weights=to_weights_shape(self.arx[k],self.current_params)
            self.model.set_weights(current_weights)
            self.arfitness+=[self.model.test_on_batch(self.x_in,self.y_in)[0]]
            self.counteval+=1
        #sort arx according to arfitness
        arindex=np.argsort(self.arfitness)
        self.arx=list(np.array(self.arx)[arindex])
        #keep only the first mu elements
        self.arx=self.arx[:self.mu]
        self.arx=np.squeeze(self.arx)
        self.arx=self.arx.T
        #keep trace of xk
        self.xold=self.xmean
        old_params=self.current_params
        #calculate tempxk+1
        self.xmean=np.dot(self.arx,self.weights)
        self.current_params=to_weights_shape(self.xmean,self.current_params)
        #calculate f(tempxk+1)
        self.model.set_weights(self.current_params)
        new_f,new_acc=self.model.test_on_batch(self.x_in,self.y_in)
        #if we want a decrease
        if(self.decrease):
            #test the decrease
            if(self.current_f-self.Cons*self.learning_rate**2>new_f):
                #if the function has decreased, declare the iteration as
                #successful
                self.success=True
                #set xk+1 equal to tempxk+1
                #update f(xk+1) equal to f(tempxk+1)
                self.current_acc=new_acc
                self.current_f=new_f
                self.stagnation=0
                #update parameters
                self.update_params()
            else:
                #if the function has no decreased, declare the iteration as
                #unsuccessful
                self.success=True
                #reduce the learning rate
                self.learning_rate*=self.beta
                #let xk+1 equal to xk
                self.current_params=old_params
                self.stagnation+=1
        #if we don't want it, don't check for it and do the same as
        #when the function is decreasing
        else:
            self.success=True
            self.current_acc=new_acc
            self.current_f=new_f
            self.stagnation=0
            self.update_params()

class NN():

    def __init__(self,size=10000,database="blops",initialisation=0,nb_init_points=1):
        self.database=database
        self.generatedata(size)
        print_sep("Model Summary")
        self.neuralnet=self.buildnet()
        self.neuralnet.summary()
        OPT=Adam(lr=0.1,clipnorm=1)
        self.neuralnet.compile(loss='binary_crossentropy',
            optimizer=OPT, metrics=['binary_accuracy'])
        print("\n")
        if(initialisation!=-1):
            self.initial_weights=self.init_weights(initialisation,nb_init_points)
            self.neuralnet.set_weights(self.initial_weights[0])
        else:
            print("Weights are those saved.")
        print("Using %s database." % (str(self.database)))

    def define_optimizer(self,opt,beta=0.9,gamma=1.1,
                         base=0,C=1e-4,learning_rate=1,
                         printtype=[0,1],decrease=True,
                         lambd=20):
        self.opt=opt
        if(opt=="gen"):
            self.optimizer=GeneticDFO(learning_rate,beta,gamma,C,self,self.size,
                                      printtype,decrease,lambd)
        elif(opt=="det"):
            self.optimizer=DeterministicDFO(learning_rate,beta,gamma,C,base,
                                            self,self.size,printtype,lambd)
        else:
            self.optimizer=GBO(printtype,self,self.size)

    def buildnet(self):
        try:
            model = load_model('net'+self.database+'.h5')
            print("Network model loaded !")
        except OSError :
            model = Sequential()

            if(self.database in ["eyes","mnist"]):
                model.add(Conv2D(1, (5,5),strides=2, input_shape=self.input_shape,
                                 padding='same', activation='relu'))
                model.add(Conv2D(2, (5,5),strides=2, padding='same',
                                 activation='relu'))
                if(self.database=="eyes"):
                    model.add(Conv2D(4, (5,5),strides=2, padding='same',
                                     activation='relu'))
                    model.add(Flatten())
                    model.add(Dense(10,activation="relu"))
                else:
                    model.add(Flatten())
            else:
                model.add(Dense(10,input_dim=self.input_dim,activation="relu"))
                model.add(Dense(10,activation="relu"))
            model.add(Dense(1,activation="sigmoid"))
        return model

    def generatedata(self,size):
        self.X_train=np.load('./datasets/Binary'+self.database.upper()+'_X_train.npy')
        self.y_train=np.load('./datasets/Binary'+self.database.upper()+'_y_train.npy')
        self.X_test=np.load('./datasets/Binary'+self.database.upper()+'_X_test.npy')
        self.y_test=np.load('./datasets/Binary'+self.database.upper()+'_y_test.npy')
        (self.X_train,self.y_train)=shuffle(self.X_train,self.y_train)
        (self.X_test,self.y_test)=shuffle(self.X_test,self.y_test)
        self.X_train=self.X_train[:size]
        self.y_train=self.y_train[:size]
        self.size=len(self.y_train)
        self.input_shape=np.shape(self.X_train[0])
        self.im_shape=self.input_shape[:2]
        self.input_dim=2
        
    def test_display(self):
        print_sep("Sample of the database")
        if(self.database in ["moons","circles","blops"]):
            plt.figure()
            for i in range(len(self.X_test)):
                x=self.X_test[i][0]
                y=self.X_test[i][1]
                if(self.y_test[i]):
                    plt.scatter(x,y,c='r',s=5)
                else:
                    plt.scatter(x,y,c='b',s=5)
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.title("Sample of the "+self.database+" database")
        else:
            nrows=3
            ncols=3
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(6,6))
            fig.suptitle("Sample of the "+self.database+" database")
            imgs=np.random.randint(0,len(self.X_test),size=(ncols,nrows))
            for i in range(nrows):
                for j in range(ncols):
                    img_2d = np.reshape(255*self.X_test[imgs[i][j],:,:,0],self.im_shape)
                    ax[i][j].get_xaxis().set_visible(False)
                    ax[i][j].get_yaxis().set_visible(False)
                    ax[i][j].imshow(img_2d, cmap=plt.get_cmap('gray'))
                    if(self.database=="mnist"):
                        label=self.y_test[imgs[i][j]]
                    else:
                        label=("Open" if(self.y_test[imgs[i][j]]==1) else "Closed")
                    ax[i][j].set_title(str(label))

        plt.show()

    def test_learning(self,plot):
        print_sep("Results on the test database")
        loss,acc=self.neuralnet.test_on_batch(self.X_test,self.y_test)
        print("On the test database, loss = %.3f, accuracy = %.3f" % (loss,acc))
        if(plot):
            if(self.database in ["moons","circles","blops"]):
                plt.figure()
                N=100
                xx=np.linspace(0,1,N)
                yy=np.linspace(0,1,N)
                YY, XX = np.meshgrid(yy, xx)
                xy = np.vstack([XX.ravel(), YY.ravel()]).T
                res=self.neuralnet.predict_classes(xy)
                for i in range(len(self.X_test)):
                    x=self.X_test[i][0]
                    y=self.X_test[i][1]
                    if(self.y_test[i]):
                        plt.scatter(x,y,c='r',s=5)
                    else:
                        plt.scatter(x,y,c='b',s=5)
                for i in range(1,len(xy)-N):
                    x=xy[i][0]
                    y=xy[i][1]
                    if(res[i]!=res[i-1] or res[i]!=res[i+N]):
                        plt.scatter(x,y,c='k',s=5)
                plt.xlabel("X-coordinate")
                plt.ylabel("Y-coordinate")
                plt.title("Test of Learning on the test database")
            else:
                nrows=3
                ncols=3
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(6,6))
                fig.suptitle("Test of Learning")
                imgs=np.random.randint(0,len(self.X_test),size=(ncols,nrows))
                for i in range(nrows):
                    for j in range(ncols):
                        img_2d = np.reshape(255*self.X_test[imgs[i][j],:,:,0],self.im_shape)
                        ax[i][j].get_xaxis().set_visible(False)
                        ax[i][j].get_yaxis().set_visible(False)
                        ax[i][j].imshow(img_2d, cmap=plt.get_cmap('gray'))
                        label=self.neuralnet.predict_on_batch(np.array([self.X_test[imgs[i][j]]]))
                        if(self.database=="mnist"):
                            label=int(label[0][0])
                        else:
                            label="Open" if(label[0][0]==1) else "Closed"
                        ax[i][j].set_title(str(label))

            plt.show()

    def plot_loss_dir2D(self,direction,liminf,limup,nb_point=100,batch_size=[50]):
        if(type(batch_size)!=list):
            batch_size=[batch_size]
        if(type(direction)!=list):
            direction=[direction]
        fig=plt.figure(figsize=(8,6))
        ax=fig.add_subplot(111)
        title=''
        for i in direction:
            title+=str(i)
            if(i!=direction[-1]):
                title+=","
        print_sep("2D Plot of the loss function")
        init=self.neuralnet.get_weights()
        base=np.eye(self.neuralnet.count_params())
        alphas=np.linspace(liminf,limup,nb_point)
        for direct in direction:
            d=to_weights_shape(base[direct],init)
            for b in batch_size:
                losses=[]
                if(b!="max"):
                    X_plot=self.X_test[:b]
                    y_plot=self.y_test[:b]
                else:
                    X_plot=self.X_test
                    y_plot=self.y_test
                progbar=tqdm(alphas,unit="point",ascii=True,
                              leave=False, position=0,file=sys.stdout,desc="Plotting 2D graph")
                for alpha in progbar:
                    tempd=[alpha*i for i in d]
                    temp=add_two_weights(init,tempd)
                    self.neuralnet.set_weights(temp)
                    res=self.neuralnet.test_on_batch(X_plot,y_plot)[0]
                    losses+=[res]
                    
                ax.plot(alphas,losses,"-",label="batch-size="+str(b))
                #ax.plot(alphas,losses,"-",label="BS"+str(b)+" D"+str(direct))
            self.neuralnet.set_weights(init)
        #ax.legend(loc=2,bbox_to_anchor=(1, 1))
        ax.legend(loc=1)
        ax.set_title(r"$\widehat{L_\mathcal{D}}(x_0+\alpha d_{"+title+"}$)")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Loss")
        plt.show()

    def plot_loss_dir3D(self,dir1,dir2,liminfA,limupA,liminfB=None,limupB=None,nb_point=20,types="contour",batch_size=50):
        print_sep(r"3D Plot of the loss function along d_"+str(dir1)+" and d_"+str(dir2))
        init=self.neuralnet.get_weights()
        base=np.eye(self.neuralnet.count_params())
        d1=to_weights_shape(base[dir1],init)
        d2=to_weights_shape(base[dir2],init)
        if(liminfB==None or limupB==None):
            liminfB=liminfA
            limupB=limupA
        alphaX=np.linspace(liminfA,limupA,nb_point)
        alphaY=np.linspace(liminfB,limupB,nb_point)
        losses=np.zeros((nb_point,nb_point))
        maxi=0
        mini=100
        if(batch_size!="max"):
            X_plot=self.X_test[:batch_size]
            y_plot=self.y_test[:batch_size]
        else:
            X_plot=self.X_test
            y_plot=self.y_test
        progbar=tqdm(range(len(alphaX)),unit="line",ascii=True,
                      leave=True, position=0,file=sys.stdout,desc="Plotting 3D graph")
        for x in progbar:
            alphax=alphaX[x]
            temp=add_two_weights(init,[alphax*i for i in d1])
            for y in range(len(alphaY)):
                alphay=alphaY[y]
                self.neuralnet.set_weights(add_two_weights(temp,[alphay*j for j in d2]))
                losses[x][y]=self.neuralnet.test_on_batch(X_plot,y_plot)[0]
                if(maxi<losses[x][y]):
                    maxi=losses[x][y]
                if(mini>losses[x][y]):
                    mini=losses[x][y]
        self.neuralnet.set_weights(init)
        alphaX,alphaY=np.meshgrid(alphaX, alphaY)
        fig=plt.figure(figsize=(6,4))
        cmap=cm.terrain
        if(types=="triangle"):
            ax = Axes3D(fig)
            alphaX,alphaY=alphaX.flatten(),alphaY.flatten()
            tri = mtri.Triangulation(alphaX,alphaY)
            losses=losses.flatten()
            surf = ax.plot_trisurf(alphaX, alphaY, losses,
                                                       triangles=tri.triangles,
                                                       cmap=cmap)
            ax.set_zlim(mini, maxi)
            ax.set_zlabel("Loss")
        elif(types=="contour"):
            ax = fig.add_subplot(111)
            levels = np.linspace(mini, maxi, 50)
            surf = ax.contour(alphaX, alphaY, losses,levels=levels,cmap=cm.nipy_spectral, linewidths=1, antialiased=False)
        else:
            ax = Axes3D(fig)
            surf = ax.plot_surface(alphaX, alphaY, losses,cmap=cmap,
                                                    linewidth=0.1, antialiased=False)
            ax.set_zlim(mini, maxi)
            ax.set_zlabel("Loss")
        cb=fig.colorbar(surf,shrink=0.7)
        cb.ax.get_children()[0].set_linewidths(5)
        ax.set_ylabel(r"$\alpha$")
        ax.set_xlabel(r"$\beta$")
        ax.set_title(r"$\widehat{L_\mathcal{D}}(x_0+\alpha d_{"+str(dir1)+r"}+\beta d_{"+str(dir2)+"}$)")
        plt.show()

    def savenet(self):
        self.neuralnet.save('net'+self.database+'.h5')
        print_sep("Model Saved")

    def init_weights(self,initialisation,nb_init_points):
        print_sep("New Initialization")
        nb_var=self.neuralnet.count_params()
        if(initialisation==0):
            loc=0
            scale=0
            initial_weights=np.random.normal(loc=loc,scale=scale,size=(nb_var,nb_init_points))
        if(initialisation==1):
            loc=0
            scale=1
            initial_weights=np.random.normal(loc=loc,scale=scale,size=(nb_var,nb_init_points))
        if(initialisation==2):
            loc=(np.random.rand()-0.5)*10
            scale=(np.random.rand())*10
            initial_weights=np.random.normal(loc=loc,scale=scale,size=(nb_var,nb_init_points))
        if(initialisation==3):
            loc=0
            scale=10
            initial_weights=np.array((lhsmdu.sample(nb_var, nb_init_points)-(0.5+loc))*scale)
        if(initialisation in [0,1,2]):
            print("Weights've been initialized on a normal distribution of mean=%.3f and scale=%.3f." % (loc,scale))
        else:
            print("Weights've been initialized inside a LH of size=%.3f and center=%.3f."% (scale,loc))
        to_return=[]
        for i in initial_weights.T:
            i=np.reshape(i,(-1,1))
            i=to_weights_shape(i,self.neuralnet.get_weights())
            to_return+=[i]
        return to_return

    def train(self, epochs=10, batch_size=50, multistart=False):
        if(not multistart):
            print_sep("Learning with "+self.opt+" and batch size "+str(batch_size))
            return self.optimizer.optimize_weights(batch_size,epochs)
        else:
            count=0
            for i in self.initial_weights:
                count+=1
                print_sep("Learning with initial weights n°"+str(count))
                self.neuralnet.set_weights(i)
                return self.optimizer.optimize_weights(batch_size,epochs)

if __name__ == '__main__':
    print_sep("Session Information")
    seed=10
    np.random.seed(seed)
    print("Randomness has been fixed on seed "+str(seed)+".")

    """Database parameters"""
    possible_databases=['circles','moons','blops','eyes','mnist']
    database=possible_databases[3]
    size=10000  #eyes will only have 4246 data, not 10000

    """Optimizer parameters"""
    # "gen" will use stochastic Optimizer
    # "det" will use deterministic Opimizer
    # "GB" will use gradient-based Optimizer
    opt="gen"
    
    """Initialization"""
    
    #-1 will initialize with saved weights, need to uncomment savenet at 
    #the end of the code in order to save weights
    #0 will initialize all weights at 0
    #1 will use a normal dist (mean 0 and scale 1)
    #2 will use a normal dist (random mean in [-5,5] random scale in [0,10])
    #3 will use a latin hypercube to return initial weights
    initialisation=1
    if(initialisation!=-1):
        for i in possible_databases:
            reinit_training('net'+i+'.h5')
    #number of initial set of weights we want to test
    nb_init_points=20
    #if multi is false, we will only train starting with the first set of the 
    #generated initial sets of weights
    multi=False
    
    """Epochs and Batch size"""
    
    epochs=5
    batch_size=1000

    """Common parameters"""
    
    beta=0.9
    gamma=1.1
    #0 will print information about each iteration
    #1 will show a global PB
    #2 will show a progressbar for each epoch
    #3 will show the evolution of the loss function and the accuracy during each epoch
    #4 will show the evolution of training loss and test loss during the learning
    #5 will show the parameters
    printtype=[2,4,5]
    #Decrease constant
    C=0
    #Initial learning rate
    learning_rate=1
    #Number of points generated at each epochs, if equal to None, its value
    #will be 4+int(3*np.log(self.pb_dim))
    lambd=20

    """Deterministic parameters"""
    #0 will use a direct basis
    #1 will use a normal random generated basis
    #2 will use a random basis
    base=1

    """Stochastic parameters"""
    #whether we want the stochastic algorithm to be convergent or not
    decrease=False

    """Start of the session"""

    nn=NN(size=size,database=database,initialisation=initialisation,
              nb_init_points=nb_init_points)
    nn.define_optimizer(opt=opt,beta=beta,
            gamma=gamma,base=base,C=C,learning_rate=learning_rate,
            printtype=printtype,decrease=decrease,
            lambd=lambd)
    nn.train(epochs=epochs,batch_size=batch_size,
            multistart=multi)
    nn.test_learning(plot=True)
    #nn.plot_loss_dir2D(direction=0,liminf=-10,limup=10,nb_point=100,batch_size=[600,300,100,50])
    #nn.plot_loss_dir3D(dir1=5,dir2=10,liminfB=None,limupB=None,liminfA=-10,limupA=10,nb_point=31,types="triangle",batch_size="max")
    #nn.savenet()