import torch as tn
import torchtt as tntt
import TTCME
import matplotlib.pyplot as plt 
import scipy.integrate
import numpy as np
import unittest

tn.set_default_tensor_type(tn.DoubleTensor)

class ParameterDependentGenerator(unittest.TestCase):

    def checkSparse(self):


        r1 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 0.015)
        r2 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 0.002)
        r3 = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 0.1)
        r4 = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 0.01)

        mdl_single = TTCME.ReactionSystem(['mRNA','protein'],[r1, r2, r3, r4])

        r1m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 'c1', params = ['c1'])
        r2m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 'c2', params = ['c2'])
        r3m = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 'c3', params = ['c3'])
        r4m = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 'c4', params = ['c4'])

        mdl_multi = TTCME.ReactionSystem(['mRNA','protein'],[r1m, r2m, r3m, r4m], params= ['c1','c2','c3','c4'])

        N = [80,120] 

        Asp = mdl_single.generator_sparse(N)

        Asp2 = mdl_multi.generator_sparse(N,np.array([0.015,0.002,0.1,0.01]))

        self.assertLess(scipy.sparse.linalg.norm(Asp-Asp2)/scipy.sparse.linalg.norm(Asp), 1e-15, "Sparse construction error: parameter dependence.")

    def checkTT(self):
        
        r1 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 0.015)
        r2 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 0.002)
        r3 = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 0.1)
        r4 = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 0.01)

        mdl_single = TTCME.ReactionSystem(['mRNA','protein'],[r1, r2, r3, r4])

        r1m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 'c1', params = ['c1'])
        r2m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 'c2', params = ['c2'])
        r3m = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 'c3', params = ['c3'])
        r4m = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 'c4', params = ['c4'])

        mdl_multi = TTCME.ReactionSystem(['mRNA','protein'],[r1m, r2m, r3m, r4m], params= ['c1','c2','c3','c4'])

        N = [80,120] 

        Att = mdl_single.generatorTT(N)

        # basis_params = [TTCME.basis.BSplineBasis(32, [0.005,0.025], 2), TTCME.basis.BSplineBasis(32, [0.001,0.004], 2), TTCME.basis.BSplineBasis(32, [0.05,0.2], 2), TTCME.basis.BSplineBasis(32, [0.005,0.02], 2)]
        Att2 = mdl_multi.generator_TT_parameters(N,params=[tn.tensor([0.01, 0.015, 0.02]), tn.tensor([0.001, 0.002, 0.004]), tn.tensor([0.05, 0.1, 0.2]), tn.tensor([0.005, 0.01, 0.02])])

        self.assertLess((Att2[:,:,1,1,1,1,:,:,1,1,1,1] - Att).norm()/Att.norm(), 1e-13, "TT construction error: parameter dependence.")

    def checkTT2(self):
        r1 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 0.015)
        r2 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 0.002)
        r3 = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 0.1)
        r4 = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 0.01)

        mdl_single = TTCME.ReactionSystem(['mRNA','protein'],[r1, r2, r3, r4])

        r1m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 'c1', params = ['c1'])
        r2m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 'c2', params = ['c2'])
        r3m = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 'c3', params = ['c3'])
        r4m = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 'c4', params = ['c4'])

        mdl_multi = TTCME.ReactionSystem(['mRNA','protein'],[r1m, r2m, r3m, r4m], params= ['c1','c2','c3','c4'])

        N = [80,120] 

        Att = mdl_single.generatorTT(N)

        basis_params = [TTCME.basis.LagrangeBasis(np.array([0.01, 0.015, 0.02])), TTCME.basis.LagrangeBasis(np.array([0.001, 0.002, 0.004])), TTCME.basis.LagrangeBasis(np.array([0.05, 0.1, 0.2])), TTCME.basis.LagrangeBasis(np.array([0.005, 0.01, 0.02])),]
        Att2 = mdl_multi.generatorTT(N, basis_params)

        self.assertLess((Att2[:,:,1,1,1,1,:,:,1,1,1,1] - Att).norm()/Att.norm(), 1e-13, "TT construction error: parameter dependence.")



r1 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 0.015)
r2 = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 0.002)
r3 = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 0.1)
r4 = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 0.01)

mdl_single = TTCME.ReactionSystem(['mRNA','protein'],[r1, r2, r3, r4])

r1m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 'c1', params = ['c1'])
r2m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 'c2', params = ['c2'])
r3m = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 'c3', params = ['c3'])
r4m = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 'c4', params = ['c4'])

mdl_multi = TTCME.ReactionSystem(['mRNA','protein'],[r1m, r2m, r3m, r4m], params= ['c1','c2','c3','c4'])

N = [80,120] 

Att = mdl_single.generatorTT(N)

basis_params = [TTCME.basis.BSplineBasis(32, [0.005,0.025], 2), TTCME.basis.BSplineBasis(32, [0.001,0.004], 2), TTCME.basis.BSplineBasis(32, [0.05,0.2], 2), TTCME.basis.BSplineBasis(32, [0.005,0.02], 2)]
Att3 = mdl_multi.generatorTT(N,basis_params=basis_params)