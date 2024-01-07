from lidcavity import LidCavity 
from pydmd import ParametricDMD, DMD
from ezyrb import POD, RBF
import numpy as np


class Model():
    def __init__(self,num_params):
        self.model=ParametricDMD([DMD(svd_rank=-1) for _ in range(num_params)],POD('svd',rank=34),RBF())
    
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
