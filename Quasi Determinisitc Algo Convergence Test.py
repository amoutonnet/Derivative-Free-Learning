import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
plt.close()
class DeterministicDFO():
    
    def __init__(self,alpha,beta,gamma,x0):
        self.plotf()
        """ Global parameters """
        self.pb_dim=np.shape(x0)[0]
        self.x0=x0
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        
    def function(self,x):
        return np.sqrt(1+x[0]**2)+np.sqrt(1+x[1]**2)
    
    def generatePSS(self,dim):
        base=np.random.normal(size=(1,self.pb_dim))
        PSS=[]
        for i in range(1):
            PSS+=[np.reshape(base[i],(-1,1))]
            PSS+=[-np.reshape(base[i],(-1,1))]
        return PSS
    
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
        self.current_x=self.x0
        self.alphas=[self.alpha]
        count=0
        for k in range(2):
            for l in range(2):
                count+=1
                self.ax[k][l].plot(self.current_x[0],self.current_x[1],'bo',label="Last point")
                self.PSS=self.generatePSS(self.pb_dim)
                current_f=self.function(self.current_x)[0]
                ite=False
                for i in self.PSS:
                    self.ax[k][l].arrow(self.current_x[0][0],
                               self.current_x[1][0],self.alpha*i[0][0],
                               self.alpha*i[1][0],width=0.01,head_width=0.5,zorder=1)
                for i in self.PSS:
                    temp_x=self.current_x+self.alpha*i
                    if(self.function(temp_x)[0]<current_f):
                        self.current_x=temp_x
                        ite=True
                        break
                if(ite):
                    self.alpha=self.gamma*self.alpha
                else:
                    self.alpha=self.beta*self.alpha
                self.ax[k][l].plot(self.current_x[0],self.current_x[1],'ro',label="New point")
                self.ax[k][l].set_title("Itération n°"+str(count))
                self.ax[k][l].legend(loc=2,fontsize=8)
                self.alphas+=[self.alpha]
        
if __name__ == '__main__': 
    Opti=DeterministicDFO(1,0.5,1.5,np.array([[5],[5]]))
    Opti.optimize(1e-5)
            
    