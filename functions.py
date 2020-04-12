import numpy as np
from numpy.linalg import inv

def pyramid(flo):   
  flow_level = []
  velocity = []
  
  for i in range(flo)-1:
    
    flow = np.subtract(flo[i+1],flo[i])
    flow_level.append(flow)
  
  flow_level.append(flo[2])
  
  for i in range(flow_level)-1:
    
    vel = np.subtract(flow_level[i+1],flow_level[i])
    velocity.append(vel)
   
  acceleration = np.subtract(velocity[1],velocity[0])
  
  velocity.append(np.add(velcoity[1],acceleration))
  
  flow_level.append(np.add(velocity[2],flow_level[2]))
  
  inverse_flow = inv(flow_level[3])
  
  return np.array(inverse_flow)
