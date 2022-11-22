import torch as tn
import torchtt as tntt
import TTCME
import matplotlib.pyplot as plt 
import datetime
import numpy as np

tn.set_default_tensor_type(tn.DoubleTensor)

r1 = TTCME.ChemicalReaction(['Prey', 'Predator'],'Prey->2*Prey', 5e-4)
r2 = TTCME.ChemicalReaction(['Prey', 'Predator'],'Prey+Predator->Predator', 1e-4)
r3 = TTCME.ChemicalReaction(['Prey', 'Predator'],'Prey+Predator->Prey+2*Predator', 1e-4)
r4 = TTCME.ChemicalReaction(['Prey', 'Predator'],'Predator->', 5e-4)


mdl = TTCME.ReactionSystem(['Prey', 'Predator'],[r1, r2 ,r3, r4])

print(mdl)

N = [128,128]
x0 = [20,5]

T = 5000

np.random.seed(424234667)

time_grid = np.linspace(0,T,3000)

reaction_time,reaction_jumps,reaction_indices = mdl.ssa_single(x0,T)
states_all = mdl.jump_process_to_states(time_grid, reaction_time, reaction_jumps)

s1 = 0.1
s2 = 0.1
s3 = 0.1
s4 = 0.05

plt.figure()
plt.plot(time_grid, states_all[:,0],'g')
plt.plot(time_grid, states_all[:,1],'r')
plt.xlabel(r'$t$ [units]')
plt.ylabel(r'#individuals')
plt.legend(['Prey','Predator'])

# import tikzplotlib
# tikzplotlib.save('./predator_prey_sample.tex')