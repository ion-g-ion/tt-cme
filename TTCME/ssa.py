import numpy as np
import numba

@numba.jit('float64[:](int64[:,:],int64[:])',nopython=True)
def Propensity(M,x):
    props = np.ones(M.shape[0],dtype=np.float64)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]!=0:
                props[i] *= M[i,j]*x[j]
    return props

@numba.jit('int64[:,:,:](int64[:,:],int64,float64[:],int64[:,:],int64[:,:],float64[:])',parallel=False,nopython=True)
def GillespieMultiple(X0,Ns,time,Pre,nu,C):
   
    num_r = Pre.shape[0]
    num_s = Pre.shape[1]
    Nt = time.shape[0]
    tmax = np.max(time)
    
    xs = np.zeros((Nt,nu.shape[1],Ns),dtype=np.int64)
    
    
    for i in numba.prange(Ns):
        # print(i)
        
        x = X0[i,:]
            
        counter = 1
        state_temp = np.zeros((Nt,nu.shape[1]),dtype=np.int64)
        state_temp[0,:] = x
        
        total_time = 0.0
        while total_time<=tmax:
            
            
                
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            a = C * Propensity(Pre, x)
            
            a0 = np.sum(a)    
           
            if a0 != 0:
                treact = np.abs(-np.log(r1)/a0)
                
                cum = np.cumsum(a/a0)
                    
                choice = num_r-1
                for k in range(num_r):
                    if cum[k]>r2:
                        choice = k
                        break
                    
                while counter < Nt and time[counter] < total_time :
                    state_temp[counter,:] = x.copy()
                    counter += 1
                    
                total_time += treact
                # print (total_time,treact,a,choice,nu[choice,:],x)
                
                if total_time <= tmax:
                    x += nu[choice,:]

            else: 
                total_time = tmax+1
                
        while counter < Nt and time[counter] <= total_time :
            state_temp[counter,:] = x.copy()
            counter += 1    
    
        xs[:,:,i] = state_temp
                
    return xs

def Observations_grid(time_grid,reaction_time,reaction_jumps):
    
    observation = np.zeros((time_grid.size,reaction_jumps.shape[1]),dtype=np.int64)
    
    index = 1
    
    for i in range(time_grid.size):    
        while time_grid[i] >= reaction_time[index] and index < reaction_time.size-1:
            index += 1
        
        observation[i,:] = reaction_jumps[index-1]
        
    return observation
    
    
def Gillespie(X0,tmax,Pre,nu,C,props = None):
   
    num_r = Pre.shape[0]
    num_s = Pre.shape[1]

            
    t = [0.0]
    xs = [ list(X0.copy()) ] 
    reaction_indices = []
    x = X0.copy()
    
    total_time = 0.0
    while True:
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        # print(Pre.shape,x.shape)
        if props == None:
            a = C * Propensity(Pre, x)
        else:
            a = np.array([p(x.reshape([1,-1])) for p in props]).flatten() 
            # print(x,a)
            a = C*a
        a0 = np.sum(a)    
       
        if a0 != 0:
            treact = -np.log(r1)/a0
            cum = np.cumsum(a/a0)
                
            choice = num_r-1
            for k in range(num_r):
                if cum[k]>r2:
                    choice = k
                    break
            
            total_time += treact
            if total_time > tmax:
                t.append(tmax)
                xs.append(list(x))
                break
            else:
                x += nu[choice,:]
                
                reaction_indices.append(choice)
                xs.append(list(x))
                t.append(total_time)
        else: 
            t.append(tmax)
            xs.append(list(x))
            break
    return np.array(t),np.array(xs),np.array(reaction_indices)