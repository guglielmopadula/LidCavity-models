from lidcavity import LidCavity 
from pydmd import ParametricDMD, DMD
from ezyrb import POD
import GPy
import numpy as np

class GPR():
    def __init__(self):
        self.kernel=GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    
    def fit(self,x,y):
        x_r=np.real(x)
        y_r=np.real(y)
        self.model=GPy.models.GPRegression(x_r,y_r,self.kernel)
        self.model.optimize()

    def predict(self,x):
        x_r=np.real(x)
        return self.model.predict(x_r)[0]


class Model():
    def __init__(self,num_params):
        self.model=ParametricDMD([DMD(svd_rank=-1) for _ in range(num_params)],POD('svd',rank=34),GPR())
    
    def fit(self,x,y):
        self.model.fit(np.transpose(y,(0,2,1)),x)
    
    def predict(self,x):
        self.model.parameters=x
        return np.transpose(self.model.reconstructed_data,(0,2,1))

data=LidCavity(10)

weights_space=data.weights_space
weights_time=data.weights_time

def loss_function(x,y):
    weights_time_tmp=weights_time.reshape(1,-1)
    weights_space_tmp=weights_space.reshape(1,1,-1)
    return  np.mean(np.sum(weights_time_tmp*np.sum(weights_space_tmp*np.abs(x-y),axis=2)**2,axis=1)/np.sum(weights_time_tmp*np.sum(weights_space_tmp*np.abs(y),axis=2)**2,axis=1))


params_train=data.params_train.reshape(-1,1)
params_test=data.params_test.reshape(-1,1)

u_train=data.V_train[:,:,:,0].numpy()


u_train=data.V_train[:,:,:,0].numpy()
u_test=data.V_test[:,:,:,0].numpy()
u_model=Model(len(params_train))
u_model.fit(params_train,u_train)
rec_u_train=u_model.predict(params_train)
rel_u_train_error=loss_function(rec_u_train,u_train)
rec_u_test=u_model.predict(params_test)
rel_u_test_error=loss_function(rec_u_test,u_test)


v_train=data.V_train[:,:,:,1].numpy()
v_test=data.V_test[:,:,:,1].numpy()
v_model=Model(len(params_train))
v_model.fit(params_train,v_train)
rec_v_train=v_model.predict(params_train)
rel_v_train_error=loss_function(rec_v_train,v_train)
rec_v_test=v_model.predict(params_test)
rel_v_test_error=loss_function(rec_v_test,v_test)

p_train=data.V_train[:,:,:,2].numpy()
p_test=data.V_test[:,:,:,2].numpy()
p_model=Model(len(params_train))
p_model.fit(params_train,p_train)
rec_p_train=p_model.predict(params_train)
rel_p_train_error=loss_function(rec_p_train,p_train)
rec_p_test=p_model.predict(params_test)
rel_p_test_error=loss_function(rec_p_test,p_test)


print("rel train error of u_1 is", "{:.1E}".format(rel_u_train_error))
print("rel test error of u_1 is", "{:.1E}".format(rel_u_test_error))
print("rel train error of u_2 is", "{:.1E}".format(rel_v_train_error))
print("rel test error of u_2 is", "{:.1E}".format(rel_v_test_error))
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
u_all=np.sqrt(u_train[-1]**2+v_train[-1]**2).reshape(-1,20,20)
u_rec_all=np.sqrt(rec_u_train[-1]**2+rec_v_train[-1]**2).reshape(-1,20,20)

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

p_all=p_train[-1].reshape(-1,20,20)
p_rec_all=rec_p_train[-1].reshape(-1,20,20)
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
