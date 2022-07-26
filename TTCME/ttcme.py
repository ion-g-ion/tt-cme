import torch as tn 
import torchtt as tntt
import numpy as np
import math
import scipy.sparse as sps
from ._ssa import GillespieMultiple, Gillespie, Observations_grid

class ChemicalReaction:

    def __init__(self, species, formula, constant, decomposable_propensity=[], params = []):

        self.__species = species.copy()
        self.__formula = formula
        self.__params = params.copy()

        
        
        self.__const = constant
        d = len(species)

        before, after = formula.split("->")
        
        pre = np.zeros((d,),np.int64)
        for b in before.split('+'):
            b = b.strip()
            b = b.split('*')
            if(len(b)==1):
                if len(b[0].strip())>0: pre[self.__species.index(b[0].strip())] = 1
            else:
                if len(b[1].strip())>0: pre[self.__species.index(b[1].strip())] = int(b[0].strip())

        self.__pre = pre

        post = np.zeros((d,), np.int64)
        for a in after.split('+'):
            a = a.strip()
            a = a.split('*')

            if(len(a)==1):
                if len(a[0].strip())>0: post[self.__species.index(a[0].strip())] = 1
            else:
                if len(a[1].strip())>0: post[self.__species.index(a[1].strip())] = int(a[0].strip())

        self.__post = post

        if len(decomposable_propensity)>0:
            self.__propensities = decomposable_propensity
        else:
            self.__propensities = []
            for k in range(len(species)):
                if len(self.__params)>0 :
                    prop = lambda q: lambda x,*params: math.comb(x,q)
                else:
                    prop = lambda q: lambda x: math.comb(x,q)
                self.__propensities.append(prop(self.__pre[k]))

    def __repr__(self):
        """

        Returns:
            str: _description_
        """
        s = 'Chemical reaction: '+self.__formula + ' with parameters: '+str(self.__params) 
        return s

    @property
    def pre(self):
        """
        
        """
        return self.__pre.copy()

    @property
    def post(self):
        return self.__post.copy()
    
    @property
    def propensity(self):
        """
        Return the propensities

        Returns:
            list[callable]: the propensities.
        """
        return self.__propensities

    @property
    def params(self):
        return self.__params.copy()
    
    @property
    def const(self):
        if isinstance(self.__const,str):
            return None
        else:
            return self.__const

    def cme_operator_tt(self, N , parameter_grid):
        Att = None


        if len(self.__params)==0 or len(self.__params)==1 and self.__const == self.__params[0]:
            A1 = []
            A2 = []

            for k in range(len(N)):
                core = tn.zeros((N[k],N[k]))
                for j in range(N[k]):
                    core[j,j] = self.__propensities[k](j) if j+(self.__post[k]-self.__pre[k])>=0 and  j+(self.__post[k]-self.__pre[k])<N[k] else 0.0           
                A1.append(core)

                core = tn.zeros((N[k],N[k]))
                for j in range(N[k]):
                    if j+(self.__post[k]-self.__pre[k])>=0 and  j+(self.__post[k]-self.__pre[k])<N[k]:
                        core[j+(self.__post[k]-self.__pre[k]),j] = self.__propensities[k](j)          
                A2.append(core)
            
            Att = (tntt.rank1TT(A2) - tntt.rank1TT(A1))
            if len(self.__params)==0:
                Att = self.__const * Att
            else:
                Att = Att ** tntt.rank1TT([tn.diag(parameter_grid[0])])

            Att = Att.round(1e-18)
        else:
            ## TODO : more comp;icated stuff
            pass

        return Att
    
    def construct_generator(self, N, params = None):
        idx_row = None
        idx_col = None
        vals = None

        num_states = np.prod(N)

        I = list(range(num_states))
        Xk = np.array(np.unravel_index(np.arange(num_states), N)).transpose()

        Xp = Xk + (self.__post - self.__pre)
        idx_keep = np.logical_and(np.all(Xp >= 0, axis=1), np.all(Xp < N, axis=1))

        # print(Xk)
        # print(Xp)
        # add diagonal 
        tmp = np.arange(num_states)
        tmp = tmp[idx_keep]
           
        idx_row = tmp
        idx_col = tmp

        tmp = Xk[idx_keep, :]

        vals = np.ones((tmp.shape[0],))
        if params != None:  
            if isinstance(self.__const,str):
                vals = -vals * params[self.__params.index(self.__const)]
            else:
                vals = -self.__const * vals
            for k in range(len(self.__species)):
                vals *= np.array(list(map( lambda x: self.__propensities[k](x,*params), tmp[:,k])))
        else:
            vals = -self.__const * vals
            for k in range(len(self.__species)):
                vals *= np.array(list(map(lambda x: self.__propensities[k](x), tmp[:,k])))
            
        # add non diagonal
        tmp_col = np.arange(num_states)
        tmp_col = tmp_col[idx_keep]

        Xp = Xp[idx_keep, :]

        tmp_row = np.ravel_multi_index(Xp.transpose(), N)
        
        tmp = Xk[idx_keep, :]
        tmp_val = np.ones((tmp.shape[0],))
        if params != None:  
            if isinstance(self.__const,str):
                tmp_val = tmp_val * params[self.__params.index(self.__const)]
            else:
                tmp_val = self.__const * tmp_val
            for k in range(len(self.__species)):
                tmp_val *= np.array(list(map(lambda x: self.__propensities[k](x,*params), tmp[:,k])))
        else:
            tmp_val = self.__const * tmp_val
            for k in range(len(self.__species)):
                tmp_val *= np.array(list(map(lambda x: self.__propensities[k](x), tmp[:,k])))

        idx_row = np.concatenate((idx_row,tmp_row))
        idx_col = np.concatenate((idx_col,tmp_col))
        vals = np.concatenate((vals,tmp_val))
        
        #print(np.array(vals), np.array(idx_row), np.array(idx_col))
            
        
        vals = np.array(vals)
        idx_row = np.array(idx_row)
        idx_col = np.array(idx_col)
            
        return  sps.csr_matrix((vals, (idx_row, idx_col)), shape=(num_states, num_states))

class ReactionSystem:

    def __init__(self, species, reactions, params = []):
        """
        Reactio system class. 

        Args:
            species (list[str]): the names of the species.
            reactions (list[ChemicalReaction]): _description_
            params (list, optional): _description_. Defaults to [].
        """
        self.__species = species.copy()
        self.__reactions = reactions.copy()
        self.__d = len(species)

        self.__params = params.copy()

    @property
    def reactions(self):
        """
        list[ChemicalReaction]: the reactions.
        """
        return self.__reactions

    def __repr__(self):
        """
        Represent the instance as a string.

        Returns:
            str: the representation.
        """
        s =  "Chemical reaction system\nSpecies involved: "+",".join(self.__species)
        s += "\nReactions:\n" + "\n".join([ str(r) for r in self.__reactions]) 
        return s

    def __call__(self):
        pass

    def add_reaction(self, reaction):
        """
        Add a chemical reaction to the system.

        Args:
            reaction (ChemicalReaction): the reaction to be added.
        """
        self.__reactions.append(reaction)

    def generator_TT_parameters(self, N, params = [], eps = 1e-14):

        num_r = len(self.__reactions)


        Att = tntt.eye(N+[p.shape[0] for p in params])*0

        for i in range(num_r):
            index_params = [self.__params.index(p) for p in self.__reactions[i].params]
            if sorted(index_params) != index_params:
                raise Exception("Parameters of the individual reactions should not appear in other order than given for the entire reaction system.")

            Atmp = self.__reactions[i].cme_operator_tt(N,[params[i] for i in index_params])

            cores = Atmp.cores[:len(N)].copy()
            
            kk = len(N)
            for k in range(len(params)):
                if self.__params[k] in self.__reactions[i].params:
                    cores.append(Atmp.cores[kk])
                    kk += 1
                else:
                    cores.append(tn.einsum('ij,kl->iklj',tn.eye(cores[-1].shape[-1]), tn.eye(params[k].shape[0])))
            Atmp = tntt.TT(cores)

            Att = Att + Atmp
            Att = Att.round(eps)

        return Att
    
    def generatorTT(self, N,basis_params = [], eps = 1e-14):

        num_r = len(self.__reactions)


        Att = tntt.eye(N+[b.dim for b in basis_params])*0

        pts = [tn.tensor(b.interpolation_pts[0]) for b in basis_params]
        Ms = [tn.linalg.inv(tn.tensor(b.interpolation_pts[1]).T) for b in basis_params]

        for i in range(num_r):
            index_params = [self.__params.index(p) for p in self.__reactions[i].params]
            if sorted(index_params) != index_params:
                raise Exception("Parameters of the individual reactions should not appear in other order than given for the entire reaction system.")

            Atmp = self.__reactions[i].cme_operator_tt(N,[pts[i] for i in index_params])

            cores = Atmp.cores[:len(N)].copy()
            
            kk = len(N)
            for k in range(len(basis_params)):
                if self.__params[k] in self.__reactions[i].params:
                    cores.append(Atmp.cores[kk])
                    kk += 1
                else:
                    cores.append(tn.einsum('ij,kl->iklj',tn.eye(cores[-1].shape[-1]), tn.eye(basis_params[k].dim)))
            Atmp = tntt.TT(cores)

            Att = Att + Atmp
            Att = Att.round(eps)

        cores = Att.cores

        for k in range(len(basis_params)):
            ctemp = tn.diagonal(cores[len(N)+k],dim1=1,dim2=2)
            ctemp = tn.einsum('ij,klj->kil',Ms[k],ctemp)
            cores[len(N)+k] = tn.einsum('kil,ij->kijl',ctemp,tn.eye(basis_params[k].dim))

        Att = tntt.TT(cores)
        return Att
    
    def generator_tt_galerkin(self, N, basis_params, eps = 1e-12):

        pts = [tn.tensor(b.integration_points(4)[0]) for b in basis_params]
        ws  = [tn.tensor(b.integration_points(4)[1]) for b in basis_params]
        
        Att = self.generator_TT_parameters(N, pts, eps)
        cores = Att.cores

        for i in range(len(N), len(Att.N)):
            core = cores[i]

            P = tn.tensor(basis_params[i-len(N)](pts[i-len(N)]))

            core = tn.einsum('abcd,bc->abcd',core,tn.diag(ws[i-len(N)]))
            core = tn.einsum('abcd,nb->ancd',core,P)
            core = tn.einsum('ancd,lc->anld',core,P)

            core_new = np.zeros((cores[i].shape[0],cores[i].shape[1],cores[i].shape[3]))
            for p in range(basis_params[i-len(N)].dim):
                core_new[:,p,:] = cores[i][:,p,p,:]
                
            core_new = np.einsum('apb,p,mp,lp->amlb',core_new,ws[i-len(N)],P,P)
                
            # print(np.linalg.norm(core_new-coreA)/np.linalg.norm(core_new))
            # coreA = np.einsum('anld,nl->anld',coreA,P)
            
            # coreA = np.einsum('abcd,bc->abcd',coreA,np.diag(ws_list[len(N)-i]))
            # print(coreAA[-1,:,:,-1])
            cores[i] = core.clone()

        Stt = tntt.TT(cores)

        Mtt_inv = tntt.eye(N) ** tntt.rank1TT([tn.tensor(np.linalg.inv(b.mass)) for b in basis_params])
        Mtt = tntt.eye(N) ** tntt.rank1TT([tn.tensor(b.mass) for b in basis_params])

        return Stt, Mtt, Mtt_inv
    
    def generator_sparse(self, N, params = None):
        """
        Return the CME generator in `scipy.sparse.csr_matrix` for a fixed parameter passed as an argument.
        

        Args:
            N (list[int]): the trucnatikn of the CME in every direction.
            params (Union[list[float], numpy.array, None], optional): The parameter for which the CME operator should be computed. None means that the system depends on no parameter. Defaults to None.

        Raises:
            Exception: _description_Parameters of the individual reactions should not appear in other order than given for the entire reaction system.

        Returns:
            scipy.sparse.csr_matrix: the generator in sparse format.
        """
        num_r = len(self.__reactions)

        Gen = None

        for i in range(num_r):
            if not params is None:
                index_params = [self.__params.index(p) for p in self.__reactions[i].params]
                if sorted(index_params) != index_params:
                    raise Exception("Parameters of the individual reactions should not appear in other order than given for the entire reaction system.")

                tmp = self.__reactions[i].construct_generator(N,params[index_params])



            else:
                tmp = self.__reactions[i].construct_generator(N)

            if Gen != None:
                Gen += tmp
            else:
                Gen = tmp

        return Gen

    def ssa(self,x0,time,Ns = 1):
        """
        Run the `SSA` algorithm to obtain a sample of size Ns.        

        Args:
            x0 (numpy.array): the initial state (length d). 
            time (numpy.array): the time grid for observing the states.
            Ns (int, optional): sample size. Defaults to 1.

        Returns:
            numpy.array: the states Ns x d.
        """

        if x0.ndim==1 :
            x0 = np.tile(x0.reshape([-1,1]),Ns).transpose()
        Pre = []
        nu = []
        C = []
        for r in self.__reactions:
            Pre.append(r.pre)
            nu.append(r.post-r.pre)
            C.append(r.const)

        Sample = GillespieMultiple(x0.astype(np.int64),Ns,time.astype(np.float64), np.array(Pre).astype(np.int64), np.array(nu).astype(np.int64), np.array(C).astype(np.float64))
        return Sample
    
    def ssa_single(self, x0, time_max):
        """
        Compute a single trajectory using the `SSA` algorithm.

        Args:
            x0 (list[int]): the initial state.
            time_max (float): the maximum time for the simulation.

        Returns:
            (numpy.array, numpy.array, numpy.array): the eaction times, the states after every reaction times and the indices of the reactions.
        """
        Pre = []
        nu = []
        C = []
        for r in self.__reactions:
            Pre.append(r.pre)
            nu.append(r.post-r.pre)
            C.append(r.const)
        return Gillespie(np.array(x0),time_max, np.array(Pre),np.array(nu),np.array(C))
    
    def jump_process_to_states(self, time_grid, reaction_time, reaction_jumps):
        """
        Discretize the output of `TTCME.ssa_single()` on the given time grid.

        Args:
            time_grid (numpy.array): the time grid (time steps must be sorted in ascending order).
            reaction_time (numpy.array): the reaction times.
            reaction_jumps (numpy.array): the reaction jumps.

        Returns:
            _type_: _description_
        """
        states = Observations_grid(time_grid, reaction_time, reaction_jumps)
        return states




