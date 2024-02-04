from torch import nn
from lidcavity import LidCavity 
import torch
from tqdm import trange
import numpy as np
import time as time_func
from sklearn.ensemble import RandomForestRegressor
data=LidCavity(10)
torch.manual_seed(0)
train_loader=data.train_loader
test_loader=data.test_loader
params_train,y_train=data.train_loader.dataset.tensors
params_train=params_train.numpy()
y_train=y_train.numpy()
time_size=data.V_train.shape[1]
x_size=data.V_train.shape[2]
u_train=y_train.reshape(-1,3)[:,0]
v_train=y_train.reshape(-1,3)[:,1]
p_train=y_train.reshape(-1,3)[:,2]
params_train=params_train.reshape(-1,4)
params_test,y_test=data.test_loader.dataset.tensors
params_test=params_test.numpy()
y_test=y_test.numpy()
params_test=params_test.reshape(-1,4)
u_test=y_test.reshape(-1,3)[:,0]
v_test=y_test.reshape(-1,3)[:,1]
p_test=y_test.reshape(-1,3)[:,2]


weights_space=data.weights_space
weights_time=data.weights_time
diff_x=data.diff_x.T
diff_y=data.diff_y.T

def loss_function(x,y):
    err=x-y
    weights_space_tmp=weights_space.reshape(1,1,-1)
    return np.mean(np.sqrt(np.max(np.sum(weights_space_tmp*(err**2+(err@diff_x)**2+(err@diff_y)**2),axis=2),axis=1)/np.max(np.sum(weights_space_tmp*(y**2+(y@diff_x)**2+(y@diff_y)**2),axis=2),axis=1)))


tree_u=RandomForestRegressor(max_depth=100,n_estimators=2)
tree_u.fit(params_train,u_train)
rec_u_train=tree_u.predict(params_train)
rec_u_test=tree_u.predict(params_test)
u_train=u_train.reshape(-1,time_size,x_size)
rec_u_train=rec_u_train.reshape(-1,time_size,x_size)
u_test=u_test.reshape(-1,time_size,x_size)
rec_u_test=rec_u_test.reshape(-1,time_size,x_size)
rel_u_train_error=loss_function(rec_u_train,u_train)
rel_u_test_error=loss_function(rec_u_test,u_test)

print("rel train error of u_1 is", "{:.1E}".format(rel_u_train_error))
print("rel test error of u_1 is", "{:.1E}".format(rel_u_test_error))

del tree_u

tree_v=RandomForestRegressor(max_depth=100,n_estimators=2)
tree_v.fit(params_train,v_train)
rec_v_train=tree_v.predict(params_train)
rec_v_test=tree_v.predict(params_test)
v_train=v_train.reshape(-1,time_size,x_size)
rec_v_train=rec_v_train.reshape(-1,time_size,x_size)
v_test=v_test.reshape(-1,time_size,x_size)
rec_v_test=rec_v_test.reshape(-1,time_size,x_size)
rel_v_train_error=loss_function(rec_v_train,v_train)
rel_v_test_error=loss_function(rec_v_test,v_test)

print("rel train error of u_2 is", "{:.1E}".format(rel_v_train_error))
print("rel test error of u_2 is", "{:.1E}".format(rel_v_test_error))

del tree_v

tree_p=RandomForestRegressor(max_depth=100,n_estimators=2)
tree_p.fit(params_train,p_train)
rec_p_train=tree_p.predict(params_train)
rec_p_test=tree_p.predict(params_test)
p_train=p_train.reshape(-1,time_size,x_size)
rec_p_train=rec_p_train.reshape(-1,time_size,x_size)
p_test=p_test.reshape(-1,time_size,x_size)
rec_p_test=rec_p_test.reshape(-1,time_size,x_size)
rel_p_train_error=loss_function(rec_p_train,p_train)
rel_p_test_error=loss_function(rec_p_test,p_test)

del tree_p

print("rel train error of p is", "{:.1E}".format(rel_p_train_error))
print("rel test error of p is", "{:.1E}".format(rel_p_test_error))


#### ANIMATION PART
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
nSeconds = 4
fps=25
times=data.time
name=os.path.splitext(os.path.basename(sys.argv[0]))[0]
u_all=np.sqrt(u_train[-1]**2+v_train[-1]**2).reshape(-1,21,21)
u_rec_all=np.sqrt(rec_u_train[-1]**2+rec_v_train[-1]**2).reshape(-1,21,21)

err=u_all-u_rec_all
print(np.max(np.abs(err)))
print(np.max(np.abs(u_all)))
print(np.max(np.abs(u_rec_all)))

fig, axarr = plt.subplots(1,3,figsize=(24,8))
a=axarr[0].imshow(u_all[0],interpolation='none', aspect='auto', vmin=0, vmax=0.5)
b=axarr[1].imshow(u_rec_all[0],interpolation='none', aspect='auto', vmin=0, vmax=0.5)
c=axarr[2].imshow(err[0],interpolation='none', aspect='auto', vmin=0, vmax=0.5)
axarr[0].set(xlabel='orig')
axarr[1].set(xlabel='rec')
axarr[2].set(xlabel='err')




def animate_func(i):
    a.set_array(u_all[i])
    b.set_array(u_rec_all[i])
    fig.suptitle('t='+str(times[i]))
    c.set_array(err[i])
    return [a,b,c]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('videos/u_'+name+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

p_all=p_train[-1].reshape(-1,21,21)
p_rec_all=rec_p_train[-1].reshape(-1,21,21)
err=p_all-p_rec_all
print(np.max(np.abs(err)))
print(np.max(np.abs(p_all)))
print(np.max(np.abs(p_rec_all)))


fig, axarr = plt.subplots(1,3,figsize=(24,8))
a=axarr[0].imshow(p_all[0],interpolation='none', aspect='auto', vmin=0, vmax=0.05)
b=axarr[1].imshow(p_rec_all[0],interpolation='none', aspect='auto', vmin=0, vmax=0.05)
c=axarr[2].imshow(err[0],interpolation='none', aspect='auto', vmin=0, vmax=0.05)
axarr[0].set(xlabel='orig')
axarr[1].set(xlabel='rec')
axarr[2].set(xlabel='err')




def animate_func(i):
    a.set_array(p_all[i])
    b.set_array(p_rec_all[i])
    fig.suptitle('t='+str(times[i]))
    c.set_array(err[i])
    return [a,b,c]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('videos/p_'+name+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
