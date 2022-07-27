import torch as tn
import torchtt as tntt
import TTCME

tn.set_default_tensor_type(tn.DoubleTensor)

r1 = TTCME.ChemicalReaction(['A','B'],'A->A+B', 0.015)
r2 = TTCME.ChemicalReaction(['A','B'],'A->', 0.002)
r3 = TTCME.ChemicalReaction(['A','B'],'->A', 0.1)
r4 = TTCME.ChemicalReaction(['A','B'],'B->', 0.01)

rs = TTCME.ReactionSystem(['A','B'],[r1, r2, r3, r4])


N = [80,120] 

Att = rs.generatorTT(N)

integrator = TTCME.TimeIntegrator.TTInt(Att, method="legendre")

p0 = tn.zeros(N)
p0[2,4] = 1.0
p0 = tntt.TT(p0)
p = p0.clone()

for i in range(20):
    print(i)
    p = integrator.solve(p, 50, intervals = 1)

