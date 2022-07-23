import torch as tn
import torchtt as tntt
import numpy as np
from .basis import *

            
class TTInt():
    
    def  __init__(self, Operator, epsilon = 1e-6, N_max = 64, dt_max = 1e-1, method = 'implicit-euler'):
        
        self.__A_tt = Operator
        self.__epsilon = epsilon
        self.__N_max = N_max
        self.__method = method
        
    def get_SP(self,T,N):
        if self.__method == 'implicit-euler':
            S = np.eye(N)-np.diag(np.ones(N-1),-1)
            P = T*np.eye(N)/N
            ev = (S@np.ones((N,1))).flatten()
            basis = None
        elif self.__method == 'crank–nicolson':
            S = np.eye(N)-np.diag(np.ones(N-1),-1)
            P = np.eye(N)+np.diag(np.ones(N-1),-1)
            P[0,:] = 0
            P = P * T / (2*(N-1))
            ev = (S@np.ones((N,1))).flatten()
            basis = None
        elif self.__method == 'cheby':
            basis = ChebyBasis(N,[0,T])
            S = basis.stiff+np.outer(basis(np.array([0])).flatten(),basis(np.array([0])).flatten())
            P = basis.mass
            ev = basis(np.array([0])).flatten()
        elif self.__method == 'legendre':
            basis = LegendreBasis(N,[0,T])
            S = basis.stiff+np.outer(basis(np.array([0])).flatten(),basis(np.array([0])).flatten())
            P = basis.mass
            ev = basis(np.array([0])).flatten()
            
        return S,P,ev,basis
        
    def solve(self, initial_tt, T, intervals = None, return_all = False,nswp = 40,qtt = False,verb = False,rounding = True, device = None):
        
        dev = self.__A_tt.cores[0].device
       
        if intervals == None:
            pass
        else:
            x_tt = initial_tt
            dT = T / intervals
            Nt = self.__N_max
            
            
                
            S,P,ev,basis = self.get_SP(dT,Nt)
            
            S = tn.tensor(S).to(dev)
            P = tn.tensor(P).to(dev)
            ev= tn.tensor(ev).to(dev)

            if qtt:
                nqtt = int(np.log2(Nt))
                S = tntt.rank1TT([S]).to_qtt()
                P = tntt.rank1TT([P]).to_qtt()
                I_tt = tntt.eye(self.__A_tt.N).to(dev)
                B_tt = I_tt ** S - (I_tt ** P) @ (self.__A_tt ** tntt.eye([Nt]).to(dev).to_qtt())

            else: 
                nqtt = 1
                S = tntt.rank1TT([S])
                P = tntt.rank1TT([P])
                I_tt = tntt.eye(self.__A_tt.N).to(dev)
                B_tt = I_tt ** S - (I_tt ** P) @ (self.__A_tt ** tntt.eye([Nt]).to(dev))

            # print(dT,T,intervals)
            returns = []
            for i in range(intervals):
                # print(i)
                if qtt:
                    f_tt = x_tt ** tntt.TT(ev).to(dev).to_qtt()
                else: 
                    f_tt = x_tt ** tntt.TT(ev).to(dev)
                # print(B_tt.n,f_tt.n)
                try:
                    # xs_tt = xs_tt.round(1e-10,5)
                    # tme = datetime.datetime.now()
                    if device != None:
                        xs_tt = tntt.solvers.amen_solve(B_tt.to(device), f_tt.to(device), x0 = self.xs_tt.to(device), eps = self.__epsilon, verbose = verb, nswp = nswp, kickrank = 8, rmax = 2000, preconditioner=None ).cpu()
                    else:
                        xs_tt = tntt.solvers.amen_solve(B_tt, f_tt, x0 = self.xs_tt, eps = self.__epsilon, verbose = verb, nswp = nswp, kickrank = 8, rmax = 2000, preconditioner=None )
                    # tme = datetime.datetime.now() - tme
                    # print(tme)
                    
                    self.xs_tt = xs_tt
                except:
                    # tme = datetime.datetime.now()
                    if device != None:
                        xs_tt = tntt.solvers.amen_solve(B_tt.to(device), f_tt.to(device), eps = self.__epsilon, verbose = verb, nswp = nswp, kickrank = 8, rmax=2000, preconditioner=None ).cpu()
                    else:
                        xs_tt = tntt.solvers.amen_solve(B_tt, f_tt, eps = self.__epsilon, verbose = verb, nswp = nswp, kickrank = 8, rmax=2000, preconditioner=None ) 
                    # tme = datetime.datetime.now() - tme
                    # print(tme)
                    
                    self.xs_tt = xs_tt
                # print('SIZE',tt_size(xs_tt)/1e6)
                # print('PLMMM',tt.sum(xs_tt),xs_tt.r)
                if basis == None:
                    if return_all: returns.append(xs_tt)
                    x_tt = xs_tt[tuple([slice(None,None,None)]*len(self.__A_tt.N)+[-1]*nqtt)]
                    x_tt = x_tt.round(self.__epsilon/10)
                else:
                    
                    if return_all:
                        if qtt:
                            beval = basis(np.array([0])).flatten()
                            temp1 = xs_tt* ( tntt.ones(self.__A_tt.N) ** tntt.TT(beval).to_qtt())
                            for l in range(nqtt): temp1 = temp1.sum(len(temp1.N)-1)
                            beval = basis(np.array([dT])).flatten()
                            temp2 = xs_tt *(tntt.ones(self.__A_tt.N) ** tntt.TT(beval).to_qtt())        
                            for l in range(nqtt): temp2 = temp2.sum(len(temp2.N)-1)
                            returns.append((temp1 ** tntt.TT(np.array([1.0,0.0])))+(temp2 ** tntt.TT(np.array([0.0,1.0]))) ) 
                        else:
                            beval = basis(np.array([0])).flatten()
                            temp1 = xs_tt * (tntt.ones(self.__A_tt.N) ** tntt.TT(beval))
                            temp1 = temp1.sum(len(temp1.n)-1)
                            beval = basis(np.array([dT])).flatten()
                            temp2 = xs_tt * (tntt.ones(self.__A_tt.N) ** tntt.TT(beval))
                            temp2 = temp2.sum(len(temp2.n)-1)
                            returns.append((temp1 ** tntt.TT(np.array([1.0,0.0])))+(temp2 ** tntt.TT(np.array([0.0,1.0]))))
 
                    beval = basis(np.array([dT])).flatten()
                    if qtt:
                        x_tt = xs_tt * (tntt.ones(self.__A_tt.N) ** tntt.TT(beval).to_qtt())
                        for l in range(nqtt): x_tt = x_tt.sum(len(x_tt.N)-1)
                        if rounding: x_tt = x_tt.round(self.__epsilon/10)
                    else:
                        x_tt = (xs_tt * (tntt.ones(self.__A_tt.N) ** tntt.TT(beval))).sum(len(xs_tt.N)-1)
                        if rounding: x_tt = x_tt.round(self.__epsilon/10)
                # print('SIZE 2 ',tt_size(x_tt)/1e6)
            if not return_all: returns = x_tt 
            return returns
        