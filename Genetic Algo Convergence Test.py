import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

class GeneticDFO():
    
    def __init__(self,sigma,x0):
        self.plotf()
        """ Global parameters """
        self.pb_dim=np.shape(x0)[0]
        self.sigma=sigma
        self.x0=x0
        
        """ Strategy parameter setting: Selection """
        self.lambd=int(4+np.floor(3*np.log(self.pb_dim))) #Population size
        self.mu=self.lambd/2 #Number of parents/points for recombination
        self.weights=np.log(self.mu+0.5)-np.array([np.log(range(1,int(self.mu)+1))]).T #muXone array for weighted recombination
        self.mu=int(np.floor(self.mu))
        self.weights = self.weights/sum(self.weights) #Normalize recombination weights array
        self.mueff=float(sum(self.weights)**2/sum(self.weights*self.weights)) #variance-effectiveness of sum w_i x_i
        
        """ Strategy parameter setting: Adaptation """
        self.cc = (4 + self.mueff/self.pb_dim) / (self.pb_dim+4 + 2*self.mueff/self.pb_dim) # time constant for cumulation for C
        self.cs = (self.mueff+2) / (self.pb_dim+self.mueff+5)  # t-const for cumulation for sigma control
        self.c1 = 2 / ((self.pb_dim+1.3)**2+self.mueff)   # learning rate for rank-one update of C
        self.cmu = min(1-self.c1, 2 * (self.mueff-2+1/self.mueff) / ((self.pb_dim+2)**2+self.mueff))  # and for rank-mu update
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.pb_dim+1))-1) + self.cs # damping for sigma 
        
        """ Initialize dynamic (internal) strategy parameters and constants """
        self.pc = np.zeros((self.pb_dim,1)) 
        self.ps = np.zeros((self.pb_dim,1))   # evolution paths for C and sigma
        self.B = np.eye(self.pb_dim)                       # B defines the coordinate system
        self.D = np.ones((self.pb_dim,1))                      # diagonal D defines the scaling
        self.C = np.dot(np.dot(self.B,np.diag((self.D*self.D).T[0])),self.B.T)           # covariance matrix C
        self.invsqrtC = np.dot(np.dot(self.B,np.diag((1/self.D).T[0])),self.B.T)    # C^-1/2 
        self.eigeneval = 0                      # track update of B and D
        self.chiN=np.sqrt(self.pb_dim)*(1-1/(4*self.pb_dim)+1/(21*self.pb_dim^2))  # expectation of ||N(0,I)|| == norm(randn(N,1))

    def function(self,x):
        return np.sqrt(1+x[0]**2)+np.sqrt(1+x[1]**2)
    
    def plotf(self):
        x=np.linspace(-10, 10, 100)
        y=np.linspace(-10, 10, 100)
        X,Y=np.meshgrid(x,y)
        Z=self.function([X,Y]).T
        self.fig, self.ax=plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6,6))
        for i in range(2):
            for j in range(2):
                self.ax[i][j].contour(X,Y,Z,levels=50, cmap=plt.cm.RdBu,vmin=abs(Z).min(), vmax=abs(Z).max(),zorder=0)
                if(j==0):
                    self.ax[i][j].set_ylabel("Y-coordinate")
                if(i==1):
                    self.ax[i][j].set_xlabel("X-coordinate")
                    
    def optimize(self,fitness):
        self.xmean=self.x0
        self.sigmas=[self.sigma]
        counteval=0
        count=0
        for i in range(2):
            for l in range(2):
                count+=1
                self.ax[i][l].plot(self.xmean[0],self.xmean[1],'bo',label="Last point",ms=4)
                arx=[]
                arfitness=[]
                for k in range(self.lambd):
                    arx+=[self.xmean+self.sigma*np.dot(self.B,self.D*np.random.normal(size=np.shape(self.xmean)))]
                    if(k==0):
                        self.ax[i][l].plot(arx[-1][0],arx[-1][1],'kx',label="Offspring points",ms=4)
                    self.ax[i][l].plot(arx[-1][0],arx[-1][1],'kx',ms=4)
                    arfitness+=[self.function(arx[k])[0]]
                    counteval+=1
                arindex=np.argsort(arfitness)
                xold=self.xmean
                arx=np.array(arx)[arindex]
                arx=np.squeeze(arx)
                arx=arx[:self.mu]
                arx=arx.T
                self.xmean=np.dot(arx,self.weights)
                
                """ Update parameters """
                self.ps=(1-self.cs)*self.ps+np.sqrt(self.cs*(2-self.cs)*self.mueff) * np.dot(self.invsqrtC,self.xmean-xold) / self.sigma
                hsig=bool(sum(self.ps*self.ps)/(1-(1-self.cs)**(2*counteval/self.lambd))/self.pb_dim < 2+4*(self.pb_dim+1))
                self.pc=(1-self.cc)*self.pc+hsig*np.sqrt(self.cc*(2-self.cc)*self.mueff)*(self.xmean-xold)/self.sigma
                
                """ Adapt covariance matrix """
                artmp = (1/self.sigma)*arx-np.tile(xold,(1,self.mu))
                self.C=(1-self.c1-self.cmu)*self.C+self.c1*(np.dot(self.pc,self.pc.T)+\
                        (1-hsig)*self.cc*(2-self.cc)*self.C)+self.cmu*np.dot(np.dot(artmp,np.diag((self.weights).T[0])),artmp.T)
                """ Adapt sigma """
                self.sigma=self.sigma*np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN-1))
                
                """ Update B and D """
                if(counteval-self.eigeneval>self.lambd/(self.c1+self.cmu)/self.pb_dim/10):
                    self.eigeneval=counteval
                    self.C=np.triu(self.C)+np.triu(self.C,1).T
                    self.D,self.B=np.linalg.eig(self.C)
                    self.D=np.reshape(self.D,(-1,1))
                    self.D=np.sqrt(self.D)
                    self.invsqrtC=np.dot(np.dot(self.B,np.diag((1/self.D).T[0])),self.B.T)
                self.ax[i][l].plot(self.xmean[0],self.xmean[1],'ro',label="New point",ms=4)
                self.ax[i][l].set_title("Itération n°"+str(count))
                self.ax[i][l].legend(loc=2,fontsize=8)
                self.sigmas+=[self.sigma]
if __name__ == '__main__': 
    Opti=GeneticDFO(1,np.array([[5],[5]]))
    Opti.optimize(1e-5)
            
    