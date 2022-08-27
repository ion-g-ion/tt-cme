"""
Implements the basic univariate bases.    

"""
import numpy as np
import matplotlib.pyplot 
import scipy
import scipy.interpolate
from scipy.interpolate import BSpline


def points_weights(a,b,nl):
    pts,ws = np.polynomial.legendre.leggauss(nl)
    pts = 0.5 * (b-a) * (pts+1) + a
    ws = (b-a) / 2  *ws
    return pts, ws

class UnivariateBasis():
    pass

class LegendreBasis(UnivariateBasis):
    def __init__(self, dim, domain = [-1,1]):
        self.__dim = dim
        self.__domain = domain
        self.__basis = [ np.polynomial.legendre.Legendre.basis(i,domain) for i in range(dim) ]
        self.__stiff = np.zeros((dim,dim))
        self.__mass = np.zeros((dim,dim))
        self.__ints = np.zeros(dim)
        

        for i in range(dim):
            pint = self.__basis[i].integ()
            self.__ints[i] = pint(domain[1]) - pint(domain[0])
            for j in range(dim):
                pint = (self.__basis[i] * self.__basis[j]).integ()
                self.__mass[i,j] = pint(domain[1]) - pint(domain[0])
                pint = (self.__basis[i] * self.__basis[j].deriv()).integ()
                self.__stiff[i,j] = pint(domain[1]) - pint(domain[0])
                
                # pint = (self.basis[i] * self.basis[j].deriv()).integ()
                # self.stiff[i,j] = pint(domain[1]) - pint(domain[0])
                
                
                
    @property
    def dim(self):
        """
        Get the dimension of the basis instance.

        Returns:
            int: the dimension.
        """
        return self.__dim

    @property
    def int(self):
        """
        Return 

        Returns:
            _type_: _description_
        """
        return self.__ints

    def __repr__(self):
        return 'Legendre basis defined on ['+str(self.__domain[0])+', '+str(self.__domain[1])+'] and dimension '+str(self.__dim)

    def __call__(self,x,deriv = 0):
        result = []
        for b in self.__basis:
            result.append(b.deriv(deriv)(x))
        return np.array(result)
    
    def get_integral(self):
        return self.__ints
    
    def get_dimension(self):
        return self.__dim
    
    @property
    def stiff(self):
        return self.__stiff
    
    @property
    def mass(self):
        return self.__mass


    
    
    def interpolate(self,fun):
        
        pts,ws = np.polynomial.legendre.leggauss(self.__dim*4)
        pts = 0.5 * (self.domain[1]-self.domain[0]) * (pts+1) + self.domain[0]
        ws = (self.domain[1]-self.domain[0]) / 2  *ws
        
        vals = fun(pts)*ws
        
        b = np.sum(self(pts) * vals,1).reshape([-1,1])
       
        return np.linalg.solve(self.mass,b).flatten()
       
    def integration_points(self,mult = 3):
        
            
        p, w = points_weights(self.__domain[0],self.__domain[1], self.__dim*mult)
        
        return p, w
    
    
class ChebyBasis(UnivariateBasis):
    def __init__(self, dim, domain = [-1,1]):
        self.__dim = dim
        self.domain = domain
        self.basis = [ np.polynomial.chebyshev.Chebyshev.basis(i,domain) for i in range(dim) ]
        self.__stiff = np.zeros((dim,dim))
        self.__mass = np.zeros((dim,dim))
        self.__ints = np.zeros(dim)
        
        for i in range(dim):
            pint = self.basis[i].integ()
            self.__ints[i] = pint(domain[1]) - pint(domain[0])
            for j in range(dim):
                pint = (self.basis[i] * self.basis[j]).integ()
                self.__mass[i,j] = pint(domain[1]) - pint(domain[0])
                pint = (self.basis[i] * self.basis[j].deriv()).integ()
                self.__stiff[i,j] = pint(domain[1]) - pint(domain[0])
        
    def __repr__(self):
        return 'Chebyshev basis defined on ['+str(self.__domain[0])+', '+str(self.__domain[1])+'] and dimension '+str(self.__dim)

    def __call__(self,x,deriv = 0):
        result = []
        for b in self.basis:
            result.append(b.deriv(deriv)(x))
        return np.array(result)
    
    def get_integral(self):
        return self.__ints
    
    @property
    def dim(self):
        """
        Get the dimension of the basis instance.

        Returns:
            int: the dimension.
        """
        return self.__dim
        
    @property
    def stiff(self):
        return self.__stiff
    
    @property
    def mass(self):
        return self.__mass
    
    def plot(self):
        x = np.linspace(self.domain[0],self.domain[1],self.dim*32)
        for b in self.basis:
            matplotlib.pyplot.plot(x,b(x))
            
    def interpolate(self,fun):
        
        pts,ws = np.polynomial.legendre.leggauss(self.dim*4)
        pts = 0.5 * (self.domain[1]-self.domain[0]) * (pts+1) + self.domain[0]
        ws = (self.domain[1]-self.domain[0]) / 2  *ws
        
        vals = fun(pts)*ws
        
        b = np.sum(self(pts) * vals,1).reshape([-1,1])
       
        return np.linalg.solve(self.mass,b).flatten()


class LagrangeBasis(UnivariateBasis):
    def __init__(self, knots, domain = None):
        self.__dim = len(knots)
        if domain is None:
            domain = (min(knots), max(knots))
        
        self.__domain = domain
       
        self.__pts = knots

        self.__basis = [ scipy.interpolate.lagrange(self.__pts,np.eye(self.__dim)[:,i]) for i in range(self.__dim) ]
      
        self.__stiff = np.zeros((self.__dim, self.__dim))
        self.__mass = np.zeros((self.__dim, self.__dim))
        self.__ints = np.zeros(self.__dim)
        
        for i in range(self.__dim):
            pint = self.__basis[i].integ()
            self.__ints[i] = pint(domain[1]) - pint(domain[0])
            for j in range(self.__dim):
                pint = (self.__basis[i] * self.__basis[j]).integ()
                self.__mass[i,j] = pint(domain[1]) - pint(domain[0])
                pint = (self.__basis[i] * self.__basis[j].deriv()).integ()
                self.__stiff[i,j] = pint(domain[1]) - pint(domain[0])

    def __repr__(self):
        return 'Lagrange basis defined on ['+str(self.__domain[0])+', '+str(self.__domain[1])+'] and dimension '+str(self.__dim)

    def __call__(self,x,deriv = 0):
        result = []
        for b in self.__basis:
            result.append(b.deriv(deriv)(x))
        return np.array(result)

    @property
    def dim(self):
        """
        Get the dimension of the basis instance.

        Returns:
            int: the dimension.
        """
        return self.__dim
    
    @property
    def stiff(self):
        return self.__stiff
    
    @property
    def int(self):
        return self.__ints.flatten()
    
    @property
    def mass(self):
        return self.__mass

    @property
    def interpolation_pts(self):
        return self.__pts, np.eye(self.__dim)

    def integration_points(self,mult = 2):
        return points_weights(self.domain[0], self.domain[1], self.deg*mult)
        
class DiracDeltaBase(UnivariateBasis):
    def __init__(self, n, pts = None):
        self.__dim = n
        if pts is None:
            self.__pts = np.arange(n) 
        else:
            self.__pts = pts 

    def __call__(self,x,deriv = 0):
        result = []
        for p in self.__pts:
            result.append(1.0*(p == x))
        return np.array(result)

    def __repr__(self):
        return 'Dirac-deltas basis defined on {'+str(self.__pts[0])+', ... ,'+str(self.__pts[-1])+'} and dimension '+str(self.__dim)

    @property
    def dim(self):
        """
        Get the dimension of the basis instance.

        Returns:
            int: the dimension.
        """
        return self.__dim
    
    @property
    def stiff(self):
        pass
    
    @property
    def int(self):
        return np.ones((self.__dim,))
        
    @property
    def mass(self):
        return np.eye(self.__dim)

    @property
    def interpolation_pts(self):
        return self.__pts, np.eye(self.__dim)

    def integration_points(self,mult = 2):
        return self.__pts, np.ones((self.__dim,))/self.__dim
        
        
class BSplineBasis(UnivariateBasis):
    
    def __init__(self, dim, domain = [-1,1],deg = 1):
        """
        

        Args:
            dim (int): the dimension of the basis.
            domain (list[int], optional): the interval where the basis functions are defined. Defaults to [-1,1].
            deg (int, optional): the degree of the B-spline basis. Defaults to 1.
        """
        self.__dim = dim
        self.__deg = deg
        self.__domain = domain
        knots = np.linspace(domain[0],domain[1],dim+1-deg)
        self.__N=knots.size+deg-1
        self.__deg=deg
        self.__knots=np.hstack( ( np.ones(deg)*knots[0] , knots , np.ones(deg)*knots[-1] ) )
        self.__spl = []
        self.__dspl = []
        for i in range(self.__N):
            c=np.zeros(self.__N)
            c[i]=1
            self.__spl.append(BSpline(self.__knots,c,self.__deg))
            self.__dspl.append(scipy.interpolate.splder( BSpline(self.__knots,c,self.__deg) ))
        
        self.__compact_support_bsp = np.zeros((self.__N,2))
        for i in range(self.__N):
            self.__compact_support_bsp[i,0] = self.__knots[i]
            self.__compact_support_bsp[i,1] = self.__knots[i+self.__deg+1]
            
        int_bsp_bsp = np.zeros((self.__N,self.__N))
        int_bsp = np.zeros((self.__N,1))
        # int_bsp_v = np.zeros((self.___Nz,1))
        
        Pts, Ws =np.polynomial.legendre.leggauss(20)
        for i in range(self.__N):
            a=self.__compact_support_bsp[i,0]
            b=self.__compact_support_bsp[i,1]

            for k in range(self.__knots.size-1):
                if self.__knots[k]>=a and self.__knots[k+1]<=b:
                    pts = self.__knots[k]+(Pts+1)*0.5*(self.__knots[k+1]-self.__knots[k])
                    ws = Ws*(self.__knots[k+1]-self.__knots[k])/2
                    int_bsp[i,0] += np.sum( self.__call__(pts,i) * ws )
                    
            for j in range(i,self.__N):
                a=self.__compact_support_bsp[j,0]
                b=self.__compact_support_bsp[i,1]
                if b>a:
                    for k in range(self.__knots.size-1):
                        if self.__knots[k]>=a and self.__knots[k+1]<=b:
                            pts = self.__knots[k]+(Pts+1)*0.5*(self.__knots[k+1]-self.__knots[k])
                            ws = Ws*(self.__knots[k+1]-self.__knots[k])/2
                            int_bsp_bsp[i,j] += np.sum(  self.__call__(pts,i) *self.__call__(pts,j) * ws )
                            # int_bspp[i,j] += np.sum( self.___bspp(pts)[i,:]* self.bspp(pts)[j,:]*ws )
                    if i!=j:
                        int_bsp_bsp[j,i] = int_bsp_bsp[i,j]
                        # int_bspp[j,i] = int_bspp[i,j]
                    
        
        self.__int_bsp_bsp = int_bsp_bsp
        # self.___int_bspp_bspp = int_bspp
        self.__int_bsp = int_bsp
        
        
    @property 
    def dim(self):
        """
        Get the dimension of the basis instance.

        Returns:
            int: the dimension.
        """
        return self.__dim

    @property
    def mass(self):
        return self.__int_bsp_bsp
        
    @property
    def int(self):
        return self.__int_bsp.flatten()

    @property 
    def stiff(self):
        return None
    
    @property
    def interpolation_pts(self):
        xg = self.__greville().flatten()
        yg = self(xg)
        return xg, yg
    
    def __repr__(self):
        return 'B-spline basis defined on ['+str(self.__domain[0])+', '+str(self.__domain[1])+'] and dimension '+str(self.__dim)

    def __call__(self,x,i=None,derivative=False):
        if i==None:
            if derivative:
                ret = np.array([self.__dspl[i](x) for i in range(self.__N)])
                return ret
            else:
                ret = np.array([self.__spl[i](x) for i in range(self.__N)])
                return ret
        else:
            if derivative:
                 return self.__dspl[i](x)
            else:
                return self.__spl[i](x)
                
    def __greville(self):
        return np.array([np.sum(self.__knots[i+1:i+self.__deg+1]) for i in range(self.__N)])/(self.__deg)
        
    def eval_all(self,c,x):
        c=np.hstack((c,np.zeros(self.__deg-2)))
        return BSpline(self.__knots,c,self.__deg)(x)
    
    

    
    def derivative(self):
        bd = scipy.interpolate.splder(BSpline(self.__knots,np.zeros(self.__N+self.__deg-1)+1,self.__deg))
        return BSplineBasis(np.unique(bd.t), bd.k)

    
    def integration_points(self,mult = 2):
        pts = []
        ws = []
        
        ku = np.unique(self.__knots)
        for i in range(ku.size-1):
            
            p, w = points_weights(ku[i], ku[i+1], self.__deg*mult)
            
            pts += list(p)
            ws += list(w)
            
        return np.array(pts), np.array(ws)
            
    