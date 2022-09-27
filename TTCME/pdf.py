"""
This contains the basic probability density function `pdfTT` represented using tensor product basis and TT DoFs.

"""
import torchtt as tntt
import torch as tn
import torch
import numpy as np
from .basis import BSplineBasis, LagrangeBasis, LegendreBasis, ChebyBasis, DiracDeltaBase, UnivariateBasis
import copy
import TTCME

def GammaPDF(alphas, betas, basis, variable_names = []):
    """
    Compute the PDF for multivariate distribution of independent gammas. 

    Args:
        alphas (numpy.array): the alphas of the distribution.
        betas (numpy.array): the betas of the distribution. 
        basis (list[TTCME.basis.UnivariateBasis]): list of the bases.
        variable_names (list[str], optional): the variable names. Defaults to [].

    Returns:
        pdfTT: the PDF instance.
    """
    pdf = pdfTT(basis, variable_names = variable_names)
    # print(basis.interpolate(lambda x : x**(alpha-1) * np.exp(-beta*x)))
    tts = []
    for i in range(len(basis)):
        p, M = basis[i].interpolation_pts
        f = lambda x : x**(alphas[i]-1) * np.exp(-betas[i]*x)
        tts.append(tn.tensor(np.linalg.solve(M,f(p))))
    pdf.dofs = tntt.rank1TT(tts)
    pdf.normalize()        
    return pdf

def UniformPDF( basis, variable_names = []):
    """
    Compute the PDF for multivariate uniform RVs. 

    Args:
        basis (list[TTCME.basis.UnivariateBasis]): list of the bases.
        variable_names (list[str], optional): the variable names. Defaults to [].

    Returns:
        pdfTT: the PDF instance.
    """
    
    pdf = pdfTT(basis, variable_names = variable_names)
    # print(basis.interpolate(lambda x : x**(alpha-1) * np.exp(-beta*x)))
    tts = []
    for i in range(len(basis)):
        p, M = basis[i].interpolation_pts
        f = lambda x : x*0+1.0
        tts.append(tn.tensor(np.linalg.solve(M,f(p))))
    pdf.update(tntt.rank1TT(tts))
    pdf.normalize()        
    return pdf


def SingularPMF(N,I, variable_names = []):
    
    dofs =  tntt.rank1TT([tn.eye(N[i])[I[i],:] for i in range(len(N))])
    basis = [DiracDeltaBase(n) for n in N]

    return pdfTT(basis, [], variable_names, [], dofs)
    
def BetaPdfTT(N, alphas, betas):
    """
    Compute the PDF for independent beta distributed RVs. 

    Args:
        N (list[int]): the size of the univariate bases.
        alphas (list[float]): list of the values of alpha.
        betas (list[float]): list of values of beta.

    Returns:
        pdfTT: the pdf instance.
    """
    
    basis = [BSplineBasis(n,[0,1],2) for n in N]
    vects = []
    pdf_g = lambda x,alpha,beta: x**(alpha-1)*(1-x)**(beta-1)
    for i in range(len(N)):
        p, M = basis[i].interpolation_pts
        vects.append(tn.tensor(np.linalg.solve(M,pdf_g(p,alphas[i],betas[i]))))

    pdf = pdfTT(basis,[],[],[],tntt.rank1TT(vects)) 
    pdf.normalize()
    return pdf

class LogNormalObservation:

    def __init__(self, N, sigmas):
        """
        Implements the Gaussian observation operator class for the CME:

        $$ p(\\mathbf{y} | \\mathbf{x}) \sim \prod\limits_{k=1}^d \\frac{1}{y_k \sigma_k \sqrt{2 \pi}} \exp(-\\frac{1}{2}\\frac{(\log{y_k}-\log{(x_k+1)})^2}{\sigma_k^2})$$
        
        Args:
            N (list[int]): the state truncation.
            sigmas (list[float]): the variances of the independent Gaussians.
        """
        self.__N = N
        self.__sigmas = sigmas
        
    def likelihood(self, observation):
        """
        Computes the likelihood given an observation and returns it a a tensor in the TT format.
        The returned tensor is
        
        $$ \mathsf{p}^{(\\text{obs})}_{i_1...i_d} = p(\\mathbf{y} | \\mathbf{x} = (i_1,...,i_d) ),$$ 
        
        where \( \\mathbf{y} \) is the given observation.

        Args:
            observation (numpy.array): the observation vector.

        Returns:
            torchtt.TT: the likelihood.
        """
        noise_model = lambda x,y,s : 1/(y*s*np.sqrt(2*np.pi)) * np.exp(-(np.log(y)-np.log(x+1))**2/(2*s**2))

        tens = tntt.rank1TT([tn.tensor(noise_model(np.arange(self.__N[i]),observation[i],self.__sigmas[i])) for i in range(len(self.__N))])

        return tens

    @staticmethod
    def add_noise(sample, sigmas):
        """
        Adds noise to the given sample according to the pdf.

        Args:
            sample (np.array): m x d array containing the number of eact species at disrete time steps. d is the number of species and m is the number of observations.
            sigmas (list[float]): the sigmas of the normal random number generator.

        Returns:
            np.array: the resulting sample with noise.
        """
        lst = [ np.random.lognormal(np.log(sample[:,i]+1),sigmas[i]).reshape([-1,1]) for i in range(len(sigmas)) ]
        
        sample = np.hstack(tuple(lst))
        
        return sample

class GaussianObservation:

    def __init__(self, N, sigmas):
        """
        Implements the Gaussian observation operator class for the CME:

        $$ p(\mathbf{y} | \mathbf{x}) \sim \prod\limits_{k=1}^d \exp(-\\frac{1}{2}\\frac{(y_k-x_k)^2}{\sigma_k^2})$$
        
        Args:
            N (list[int]): the state truncation.
            sigmas (list[float]): the variances of the independent Gaussians.
        """
        self.__N = N
        self.__sigmas = sigmas
        
    def likelihood(self, observation):
        """
        Computes the likelihood given an observation and returns it a a tensor in the TT format.
        The returned tensor is
        
        $$ \mathsf{p}^{(\\text{obs})}_{i_1...i_d} = p(\\mathbf{y} | \\mathbf{x} = (i_1,...,i_d) ),$$ 
        
        where \( \\mathbf{y} \) is the given observation.

        Args:
            observation (numpy.array): the observation vector.

        Returns:
            torchtt.TT: the likelihood.
        """
        noise_model = lambda x,y,s :  np.exp(-(y-x)**2/(2*s**2))

        tens = tntt.rank1TT([tn.tensor(noise_model(np.arange(self.__N[i]),observation[i],self.__sigmas[i])) for i in range(len(self.__N))])

        return tens

    @staticmethod
    def add_noise(sample, sigmas ):
        """
        Adds noise to the given sample accordinf to the pdf.

        Args:
            sample (numpy.array): m x d array containing the number of eact species at disrete time steps. d is the number of species and m is the number of observations.
            sigmas (list[float]): the sigmas of the normal random number generator.

        Returns:
            np.array: the resulting sample with noise.
        """
        sample += np.random.normal(scale = sigmas, size=sample.shape)
        return sample

class pdfTT():
    def __init__(self, basis, basis_conditioned = [], variable_names = [], conditioned_variable_names = [], dofs = None):
        """
        Probability density function approximation using a tensor-product basis:   
        $$ p(x_1,...,x_d|y_1,...,y_{d'})  = \\sum\\limits_{k_1,...,k_d,n_1,...,n_{d'}} \\mathsf{d}_{k_1...k_dn_1...n_d} b^{(1)}(x_1) \\cdots b^{(d)}(x_d) b'^{(1)}(y_1) \\cdots b'^{(d')}(y_{d'}),$$
        where \( \\mathsf{d}\) is the DoF tensor and is represented in the TT format.
        
        Using the subscript operator, the pdf can be evaluated at given values. Following possible ways of using the subscript operator are possible:
         
         - For a conditional PDF, the `Ellipsis` can be used to evaluate the variables we condition on. As an example, for the `pdfTT` instance `p` representing the conditional `p(a,b|c,d)`, the expression `p[...,1,2]` represents the PDF `p(a,b|c=1,d=2)`.
         - If only torcht.tensor of float instances are provided, the result is the tensor that results when the PDF is evaluated on the given tensor-product grid.


        Args:
            basis (list[UnivariateBasis]): the univariate bases used to construct the tensor-product basis (\( b^{(k)} \) in the formula above). 
            basis_conditioned (list[UnivariateBasis], optional): the univariate bases in case a conditional is used (corresponding to the variables right of the conditioned sign). In the equation above are denoted by \( b'^{(k)} \). Defaults to [].
            variable_names (list[str], optional): the name of variables as a list of strings. Defaults to [].
            conditioned_variable_names (list[str], optional): the names of the bariables that we condition on. Defaults to [].
            dofs (torchtt.TT, optional): the dofs in the TT format. If None is provided an uniform PDF is created. Defaults to None.
        """
        self.__d = len(basis)
        self.__dc = len(basis_conditioned) 
        self.__variable_names = variable_names
        self.__conditioned_variable_names = conditioned_variable_names
        self.__N = [b.dim for b in basis]
        self.__Nc = [b.dim for b in basis_conditioned]
        self.__basis = basis.copy()
        self.__basis_cond = basis_conditioned.copy()

        if dofs == None:
            self.__tt = tntt.ones(self.__N+self.__Nc)
        else:
            self.__tt = dofs.clone()


    def copy(self):
        """
        Create a copy of the `pdfTT` instance.

        Returns:
            pdfTT: the copy.
        """
        return copy.deepcopy(self)

    @property
    def basis(self):
        """
        list[UnivariateBasis]: the basis used.
        """
        return self.__basis.copy()

    @property
    def basis_conditioned(self):
        """
        list[UnivariateBasis]: the basis used for the conditioned variables.
        """
        return self.__basis_cond.copy()

    @property
    def variable_names(self):
        """
        list[str]: the name of the variables.
        """
        return self.__variable_names.copy()

    @property
    def conditioned_variable_names(self):
        """ 
        list[str]: the name of the variables we condition on.
        """
        return self.__conditioned_variable_names.copy()

    @property
    def dofs(self):
        """
        torchtt.TT: the dofs tensor in the TT format.
        """
        return self.__tt

    @dofs.setter 
    def dofs(self, value):
        if not isinstance(value, tntt.TT):
            raise Exception("DOFS have to be in the TT format")

        if value.is_ttm or value.N != self.__tt.N:
            raise Exception("Dimensions must match")

        self.__tt = value
        
    @classmethod
    def interpoalte(cls, pdf, basis, basis_conditioned = [], variable_names = [], conditioned_variable_names = [], eps = 1e-10):
        """
        Interpolate a pdf using the given basis.
        
        Example:
            ```
            import TTCME
            basis = [TTCME.basis.BSplineBasis(64,[0,1],2), TTCME.basis.BSplineBasis(64,[0,1],2)]
            basis_cond = [TTCME.basis.BSplineBasis(32,[2,3],2), TTCME.basis.BSplineBasis(43,[2, 3],2)]
            pdf_c = TTCME.pdf.pdfTT.interpoalte(pdf = lambda x: x[...,0]**(2-1)*(1-x[...,0])**(x[...,2]-1) * x[...,1]**(2-1)*(1-x[...,1])**(x[...,3]-1), basis = basis, basis_conditioned= basis_cond, variable_names=['x1','x2'], conditioned_variable_names=['beta1','beta2'] )
            ```
        

        Args:
            pdf (_type_): _description_
            basis (_type_): _description_
            basis_conditioned (list, optional): _description_. Defaults to [].
            variable_names (list, optional): _description_. Defaults to [].
            conditioned_variable_names (list, optional): _description_. Defaults to [].
            eps (_type_, optional): _description_. Defaults to 1e-10.

        Returns:
            _type_: _description_
        """
        xs = tntt.meshgrid([tn.tensor(b.interpolation_pts[0]) for b in basis]+[tn.tensor(b.interpolation_pts[0]) for b in basis_conditioned])
        Ms = tntt.rank1TT([tn.tensor(np.linalg.inv(b.interpolation_pts[1])) for b in basis]+[tn.tensor(np.linalg.inv(b.interpolation_pts[1])) for b in basis_conditioned])
        dofs = Ms @  tntt.interpolate.function_interpolate(pdf, xs, eps=eps)

        p = pdfTT(basis, basis_conditioned = basis_conditioned, variable_names = variable_names, conditioned_variable_names = conditioned_variable_names, dofs = dofs)
        p.normalize()

        return p

    def __repr__(self):
        """
        Offer a srtring representation of the instance.        

        Returns:
            str: the representation.
        """
        s = "Probability density function:\n"
        s+= "p(" + ",".join(self.__variable_names) 
        if self.__dc == 0 :
            s += ")\n"
        else:
            s +="|" + ",".join(self.__conditioned_variable_names)+")\n"
        s+= "\nBasis:\n" + "\n".join([str(b) for b in self.__basis]+[str(b) for b in self.__basis_cond])
        s+= "\n\nDoF:\n"+repr(self.__tt)

        return s

        
    def normalize(self):
        """
        Normalize the PDF.
        """
        int_tt = tntt.rank1TT([tn.tensor(b.int) for b in self.__basis ])

        if self.__dc>0:
            int_tt = int_tt ** tntt.ones(self.__Nc)

        Z = (self.__tt * int_tt).sum(list(range(self.__d)))
        if self.__dc>0: 
            Z = tntt.ones(self.__N) ** Z

        self.__tt = self.__tt/Z

    @property
    def Z(self):
        """
        Computes the normalization constant. No normalization is performed.

        Returns:
            Union[torchtt.TT,float]: the normalization constant. In case of conditioned RVs, a `torchtt.TT` instance is returned.
        """
        int_tt = tntt.rank1TT([tn.tensor(b.int) for b in self.__basis ])

        if self.__dc>0:
            int_tt = int_tt ** tntt.ones(self.__Nc)

        Z = (self.__tt * int_tt).sum(list(range(self.__d)))
        if self.__dc>0: 
            Z = tntt.ones(self.__N) ** Z
        
        return Z


    def expected_value(self):
        """
        Compute the expected value

        Returns:
            Union[list[torchtt.TT],list[float]]: _description_
        """
        E = []
        for i in range(self.__d):
            pts, ws = self.__basis[i].integration_points(2)
            temp = tn.einsum('ij,j->i',tn.tensor(self.__basis[i](pts) * pts) ,tn.tensor(ws))

            tmp = tntt.rank1TT([tn.tensor(self.__basis[k].int) if k!=i else temp for k in range(self.__d)])
            if self.__dc > 0:
                tmp = tmp ** tntt.ones(self.__Nc)
                E.append( pdfTT([], self.__basis_cond, [], self.__conditioned_variable_names, (tmp*self.__tt).sum(list(range(self.__d))) ) )
            else:
                E.append( (tmp*self.__tt).sum() )
            
        return E
    
    def covariance_matrix(self):
        """
        Compute the covariance matrix.
        Currently no conditioned RVs are accepted!!!
        
        Returns:
            torch.tensor: the covariance matrix.
        """
        C = tn.zeros((self.__d,self.__d))
        E = self.expected_value()
        
        Pts = [b.integration_points(4)[0] for b in self.__basis]
        Ws = [b.integration_points(4)[1] for b in self.__basis]
        Bs = [self.__basis[k](Pts[k]) for k in range(self.__d)]
        w_tt = tntt.rank1TT([tn.tensor(w) for w in Ws])
        
        for i in range(self.__d):
            
            for j in range(i,self.__d):
                
                Iop = tntt.rank1TT([ tn.tensor(Bs[k]*(Pts[k] if k==i else 1)*(Pts[k] if k==j else 1)) for k in range(self.__d)])
                
                C[i,j] = tntt.dot( Iop.t() @ self.__tt , w_tt ) - E[i]*E[j]
            
        return C
    
    def marginal(self,mask):
        """
        Compute the marginal w.r.t. dimensions given as the mask.
        Currently does not work for conditioned RVs.
        
        Args:
            mask (list[int]): the dimensions. The dimensions are numbered from 0.

        Returns:
            pdfTT: the resultimg pdf.
        """
        ints = [tn.tensor(self.basis[k].int) if k in mask else tn.ones([self.basis[k].dim]) for k in range(self.__d)]
        
        basis_new = [self.__basis[k] for k in range(self.__d) if not k in mask]
        variable_names_new = [self.__variable_names[k] for k in range(self.__d) if not k in mask]
        tt_new = (self.dofs * tntt.rank1TT(ints)).sum(mask)
        pdf_new = pdfTT(basis_new, variable_names=variable_names_new, dofs = tt_new)
        pdf_new.normalize()
        return pdf_new
    
     
    
    def round(self,eps=1e-12,rmax=9999):
        """
        Round the TT degrees of freedom.
        

        Args:
            eps (float, optional): the epsilon accuracy. Defaults to 1e-12.
            rmax (int, optional): the maximum rank. Defaults to 9999.
        """
        self.__tt = self.__tt.round(eps,rmax)
    
    def __call__(self,x):
        
        beval = tntt.rank1TT([tn.tensor(self.__basis[i](x[...,i])).T for i in range(self.__d)]+[tn.tensor(self.__basis_cond[i](x[...,i+self.__d])).T for i in range(self.__dc)])             

        return beval @ self.__tt if beval.is_ttm else tntt.dot(beval, self.__tt)
        
    def __getitem__(self, items):
        """
        Evaluate the PDF instance. Following possible ways of using the subscript operator are possible:
         
         - For a conditional PDF, the `Ellipsis` can be used to evaluate the variables we condition on. As an example, for the `pdfTT` instance `p` representing the conditional `p(a,b|c,d)`, the expression `p[...,1,2]` represents the PDF `p(a,b|c=1,d=2)`.
         - If only torcht.tensor of float instances are provided, the result is the tensor that results when the PDF is evaluated on the given tensor-product grid.

        Args:
            items (tuple): _description_

        Returns:
            _type_: _description_
        """
        
#         if items[0] == Ellipsis:
#             idx = [slice(None,None,None)]*(self.__d+self.__dc-len(items))
#         else:
#             idx = items
# 
#         if len(idx) != self.__d+self.__dc:
#             raise Exception("The number of dimensions does not match with the number of inputs.")
# 
#         for i in range(len(idx)):
#             if isinstance(idx[i], slice) and idx[i].start == None and idx[i].stop==None and idx[i].step == None:
        if not isinstance(items, tuple):
            items = (items,)
        if items[0] is Ellipsis:
            bevals = tntt.ones(self.__N) ** tntt.rank1TT([b(i) for i,b in zip(items[1:], self.__basis_cond)])
            dofs = (self.dofs*bevals).sum(list(range(self.__d,self.__d+self.__dc)))
            return pdfTT(self.__basis, [], self.__variable_names, [], dofs)
        else:
            bt = self.__basis + self.__basis_cond
            bevals = tntt.rank1TT([tn.tensor(bt[i](items[i]).T.reshape([-1,bt[i].dim])) for i in range(len(bt))])
            return bevals @ self.dofs

                

    def __pow__(self,other):
        """
        Joint of 2 independent PDFs:
        p(x1,...,xn,y1,...,ym) = p(x1,...,xn) * p(y1,...,ym)

        Args:
            other (pdfTT): the second pdf

        Returns:
            pdfTT: the resulting pdf
        """
        basis_new = self.basis + other.basis
        variable_names = self.variable_names + other.variable_names

        pdf = pdfTT(basis_new, variable_names=variable_names, dofs = self.dofs ** other.dofs)
        pdf.normalize()

        return pdf
                
