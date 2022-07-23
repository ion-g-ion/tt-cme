import torchtt as tntt
import torch as tn
import numpy as np
from .basis import BSplineBasis, LagrangeBasis, LegendreBasis, ChebyBasis, DiracDeltaBase
import copy

def GammaPDF(alphas, betas, basis, variable_names = []):

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
    
def BetaPdfTT(N : list[int], alphas: list[float], betas: list[float]):

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
        self.__N = N
        self.__sigmas = sigmas
        
    def likelihood(self, observation):

        noise_model = lambda x,y,s : 1/(y*s*np.sqrt(2*np.pi)) * np.exp(-(np.log(y)-np.log(x+1))**2/(2*s**2))

        tens = tntt.rank1TT([tn.tensor(noise_model(np.arange(self.__N[i]),observation[i],self.__sigmas[i])) for i in range(len(self.__N))])

        return tens

class GaussianObservation:

    def __init__(self, N, sigmas):
        self.__N = N
        self.__sigmas = sigmas
        
    def likelihood(self, observation):

        noise_model = lambda x,y,s :  np.exp(-(y-x)**2/(2*s**2))

        tens = tntt.rank1TT([tn.tensor(noise_model(np.arange(self.__N[i]),observation[i],self.__sigmas[i])) for i in range(len(self.__N))])

        return tens


class pdfTT():
    def __init__(self, basis, basis_conditioned = [], variable_names = [], conditioned_variable_names = [], dofs = None):
        
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
        return copy.deepcopy(self)

    @property
    def basis(self):
        return self.__basis.copy()

    @property
    def basis_conditioned(self):
        return self.__basis_cond.copy()

    @property
    def variable_names(self):
        return self.__variable_names.copy()

    @property
    def conditioned_variable_names(self):
        return self.__conditioned_variable_names.copy()

    @property
    def dofs(self):
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
        xs = tntt.meshgrid([tn.tensor(b.interpolation_pts[0]) for b in basis]+[tn.tensor(b.interpolation_pts[0]) for b in basis_conditioned])
        Ms = tntt.rank1TT([tn.tensor(np.linalg.inv(b.interpolation_pts[1])) for b in basis]+[tn.tensor(np.linalg.inv(b.interpolation_pts[1])) for b in basis_conditioned])
        dofs = Ms @  tntt.interpolate.function_interpolate(pdf, xs, eps=eps)

        p = pdfTT(basis, basis_conditioned = basis_conditioned, variable_names = variable_names, conditioned_variable_names = conditioned_variable_names, dofs = dofs)
        p.normalize()

        return p

    def __repr__(self):
        s = "Probability density function:\n"
        s+= "p(" + ",".join(self.__variable_names) 
        if self.__dc == 0 :
            s += ")\n"
        else:
            s +="|" + ",".join(self.__conditioned_variable_names)+")\n"
        s+= "\nBasis:\n" + "\n".join([str(b) for b in self.__basis]+[str(b) for b in self.__basis_cond])
        s+= "\n\nDoF:\n"+repr(self.__tt)

        return s

    def update(self,tensor):
        self.tt = tensor
        
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
        int_tt = tntt.rank1TT([tn.tensor(b.int) for b in self.__basis ])

        if self.__dc>0:
            int_tt = int_tt ** tntt.ones(self.__Nc)

        Z = (self.__tt * int_tt).sum(list(range(self.__d)))
        if self.__dc>0: 
            Z = tntt.ones(self.__N) ** Z
        
        return Z


    def expected_value(self):

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
        '''
        Compute the expected value of the pdf.

        Returns
        -------
        E : np array
            the expected value.

        '''
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
        
        ints = [tn.tensor(self.basis[k].int) if k in mask else tn.ones([self.basis[k].dim]) for k in range(self.__d)]
        
        basis_new = [self.__basis[k] for k in range(self.__d) if not k in mask]
        variable_names_new = [self.__variable_names[k] for k in range(self.__d) if not k in mask]
        tt_new = (self.dofs * tntt.rank1TT(ints)).sum(mask)
        pdf_new = pdfTT(basis_new, variable_names=variable_names_new, dofs = tt_new)
        pdf_new.normalize()
        return pdf_new
    
     
    
    def round(self,eps=1e-12,rmax=9999):
        self.__tt = self.__tt.round(eps,rmax)
    
    def __call__(self,x):
        
        beval = tntt.rank1TT([tn.tensor(self.__basis[i](x[...,i])).T for i in range(self.__d)]+[tn.tensor(self.__basis_cond[i](x[...,i+self.__d])).T for i in range(self.__dc)])             

        return beval @ self.__tt if beval.is_ttm else tntt.dot(beval, self.__tt)
        
    def __getitem__(self, items):
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
        
        basis_new = self.basis + other.basis
        variable_names = self.variable_names + other.variable_names

        pdf = pdfTT(basis_new, variable_names=variable_names, dofs = self.dofs ** other.dofs)
        pdf.normalize()

        return pdf
                
def get_mass(basis):
    lst = [b.get_mass().reshape([1,b.get_dimension(),b.get_dimension(),1]) for b in basis]
    lst_inv = [np.linalg.inv(b.get_mass()).reshape([1,b.get_dimension(),b.get_dimension(),1]) for b in basis]
    return tt.matrix().from_list(lst), tt.matrix().from_list(lst_inv)

def get_stiff(Att_extended,N,pts_list,ws_list,basis):
    Np = len(basis)
    
    lst_cores = Att_extended.to_list(Att_extended)
    
    for i in range(len(N),len(N)+Np):
        
        coreA = lst_cores[i]
        coreAA = lst_cores[i]
        P = basis[i-len(N)](pts_list[i-len(N)])
        # print(i, i-len(N),P.shape,ws_list[i-len(N)].shape,np.sum(ws_list[i-len(N)]),basis[i-len(N)].domain)
        coreA = np.einsum('abcd,bc->abcd',coreA,np.diag(ws_list[i-len(N)]))
        coreA = np.einsum('abcd,nb->ancd',coreA,P)
        coreA = np.einsum('ancd,lc->anld',coreA,P)
        
        core_new = np.zeros((coreAA.shape[0],coreAA.shape[1],coreAA.shape[3]))
        for p in range(basis[i-len(N)].get_dimension()):
            core_new[:,p,:] = coreAA[:,p,p,:]
            
        core_new = np.einsum('apb,p,mp,lp->amlb',core_new,ws_list[i-len(N)],P,P)
            
        # print(np.linalg.norm(core_new-coreA)/np.linalg.norm(core_new))
        # coreA = np.einsum('anld,nl->anld',coreA,P)
        
        # coreA = np.einsum('abcd,bc->abcd',coreA,np.diag(ws_list[len(N)-i]))
        # print(coreAA[-1,:,:,-1])
        lst_cores[i] = coreA

    Aext = tt.matrix().from_list(lst_cores)
    return Aext
 
def interpolate_cme_tt(Att_extended,N,pts_list,ws_list,basis):
    Np = len(basis)
    
    lst_cores = Att_extended.to_list(Att_extended)
    
    for i in range(len(N),len(N)+Np):
        
        coreA = lst_cores[i]
        
        
        core_new = np.zeros((coreA.shape[0],coreAA.shape[1],coreAA.shape[3]))
        for p in range(basis[len(N)-i].get_dimension()):
            core_new[:,p,:] = coreAA[:,p,p,:]
            
        core_new = np.einsum('apb,p,mp->amb',core_new,ws_list[len(N)-i],P)
            
        print(np.linalg.norm(core_new-coreA)/np.linalg.norm(core_new))
        # coreA = np.einsum('anld,nl->anld',coreA,P)
        
        # coreA = np.einsum('abcd,bc->abcd',coreA,np.diag(ws_list[len(N)-i]))
        lst_cores[i] = core_new

    Aext = tt.matrix().from_list(lst_cores)
    return Aext



def extend_cme(Alist, pts_rates):
    n = len(Alist)
    
    for i,pts in enumerate(pts_rates):
        if pts.size!=1:
            for k in range(n):
                Alist[k] = tt.kron(Alist[k],tt.eye([pts.size]) if k!=i else tt.matrix(np.diag(pts)))
        else:
            Alist[i] = Alist[i]*pts[0]
            
    Att = Alist[0]*0
    for A in Alist: Att += A
    
    return Att