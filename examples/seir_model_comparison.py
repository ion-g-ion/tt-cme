#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as tn
import torchtt as tntt
import TTCME
import matplotlib.pyplot as plt 
import datetime
import numpy as np
import scipy.integrate
import tikzplotlib

tn.set_default_tensor_type(tn.DoubleTensor)
qtt = False


# In[2]:


r1 = TTCME.ChemicalReaction(['S','E','I','R'],'S+I->E+I', 0.1)
r2 = TTCME.ChemicalReaction(['S','E','I','R'],'E->I',     0.5)
r3 = TTCME.ChemicalReaction(['S','E','I','R'],'I->S',     1.0)
r4 = TTCME.ChemicalReaction(['S','E','I','R'],'S->',      0.01)
r5 = TTCME.ChemicalReaction(['S','E','I','R'],'E->',      0.01)
r6 = TTCME.ChemicalReaction(['S','E','I','R'],'I->R',     0.01)
r7 = TTCME.ChemicalReaction(['S','E','I','R'],'->S',      0.4)

mdl = TTCME.ReactionSystem(['S','E','I','R'],[r1, r2 ,r3, r4, r5 ,r6, r7])
N = [128,64,64,64]

Att = mdl.generatorTT(N)
Aqtt = Att.to_qtt().round(1e-13)
print('Storage generator     ', tntt.numel(Att)*8/1e6,' MB')
print('Storage generator QTT ', tntt.numel(Aqtt)*8/1e6,' MB')
print('Rank generator        ', Att.R)
print('Rank generator QTT    ', Aqtt.R)


# In[3]:


mu0 = [50,4,0,0]
sigma = 1
p0 = TTCME.pdf.SingularPMF(N,mu0).dofs
p0 = p0 / p0.sum()


# In[4]:


qtt = True
fwd_int = TTCME.TimeIntegrator.TTInt(Att if not qtt else Aqtt, epsilon=1e-5, N_max = 8, dt_max=1, method='cheby')

Nt = 4
Tend = 8


# In[ ]:

tme_total = datetime.datetime.now()
if qtt:
    p = p0.clone()
    time = 0.0
    Ps = [p0.clone()]
    p = p.to_qtt()
    for i in range(Nt):
        dt = Tend/Nt
        tme = datetime.datetime.now()
        p = fwd_int.solve(p, dt, intervals = 4, qtt = True, verb=True, rounding = False) 
        tme = datetime.datetime.now() - tme
        time += dt
        Ps.append(tntt.reshape(p.clone(),N))
        p = p.round(1e-10)
        print('Time ', time, ', rank ',p.R,', solver time ', tme)
    p = tntt.reshape(p,N)
else:
    p = p0.clone()
    time = 0.0
    Ps = [p0.clone()]

    for i in range(Nt):
        dt = Tend/Nt
        tme = datetime.datetime.now()
        p = fwd_int.solve(p, dt, intervals = 4, verb = True) 
        tme = datetime.datetime.now() - tme
        time += dt
        Ps.append(p.clone())
        print('Time ', time, ', rank ',p.R,', solver time ', tme)
tme_total = datetime.datetime.now()-tme_total
print('Time TT-IGA ', tme_total)

# In[ ]:


Gen = mdl.generator_sparse([128,64,64,64])


# In[ ]:



P0 = p0[:80,:64,:64,:64].numpy()
P0 = p0.numpy()
# Gen = mdl.generator_sparse([80,64,64,64])
Gen = mdl.generator_sparse([128,64,64,64])
def func(t,y):
    # print(t)
    return Gen.dot(y)

# solve CME

tme_ode45 = datetime.datetime.now()
# res = scipy.integrate.solve_ivp(func,[0,time],P0.flatten(),t_eval=[0,time])
res = scipy.integrate.solve_ivp(func,[0,time],P0.flatten(),t_eval=[0,time],max_step=dt/500)
Pt = res.y.reshape([128,64,64,64]+[-1])
tme_ode45 = datetime.datetime.now() - tme_ode45

print('Time ODEINT ',tme_ode45)
P_ref = Pt[:,:,:,:,-1]


# In[ ]:


Pend = p[:80,:64,:64,:64].numpy()
P_ref = P_ref[:80,:64,:64,:64]

plt.figure()
plt.imshow(Pend.sum(2).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(2).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.figure()
plt.imshow(Pend.sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')

plt.figure()
plt.imshow(Pend.sum(0).sum(0).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_3$')
plt.ylabel(r'$x_4$')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(0).sum(0).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_3$')
plt.ylabel(r'$x_4$')

plt.figure()
plt.imshow(Pend.sum(1).sum(1).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_4$')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(1).sum(1).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_4$')

plt.figure()
plt.imshow(Ps[0].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')

plt.figure()
plt.imshow(Ps[1].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')

plt.figure()
plt.imshow(Ps[2].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')

plt.figure()
plt.imshow(Ps[3].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')


# In[ ]:


plt.figure()
plt.imshow(Ps[0].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('EI_marginal_0.tex')

plt.figure()
plt.imshow(Ps[1].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('EI_marginal_2.tex')

plt.figure()
plt.imshow(Ps[2].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('EI_marginal_4.tex')

plt.figure()
plt.imshow(Ps[3].numpy().sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('EI_marginal_6.tex')

plt.figure()
plt.imshow(Pend.sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('EI_marginal_8.tex')

plt.figure()
plt.imshow(np.abs(Pend-P_ref).sum(0).sum(2).transpose(),origin='lower',cmap='gray_r')
plt.colorbar()
plt.xlabel(r'$x_2$')
plt.ylabel(r'$x_3$')
tikzplotlib.save('EI_marginal_err.tex')


