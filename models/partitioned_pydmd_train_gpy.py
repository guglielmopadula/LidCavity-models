from lidcavity import LidCavity 
from pydmd import ParametricDMD, DMD
from ezyrb import POD
import GPy
import numpy as np
data=LidCavity(10)
params_train=data.params_train
params_test=data.params_test
u_train=data.V_train[:,:,:,0].transpose(1,2).numpy()
u_test=data.V_test[:,:,:,0].transpose(1,2).numpy()
dmd=[DMD(svd_rank=-1) for _ in range(len(params_train))]
rom=POD('svd',rank=34)

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


interpolator=GPR()

pdmd_partitioned_velocity_u = ParametricDMD(dmd, rom, interpolator)
pdmd_partitioned_velocity_u.fit(
    u_train, params_train
)


pdmd_partitioned_velocity_u.parameters=params_train.reshape(-1,1)
rel_u_train_error=np.linalg.norm(pdmd_partitioned_velocity_u.reconstructed_data-u_train)/np.linalg.norm(u_train)

pdmd_partitioned_velocity_u.parameters=params_test.reshape(-1,1)
rel_u_test_error=np.linalg.norm(pdmd_partitioned_velocity_u.reconstructed_data-u_test)/np.linalg.norm(u_test)

v_train=data.V_train[:,:,:,1].transpose(1,2).numpy()
v_test=data.V_test[:,:,:,1].transpose(1,2).numpy()
dmd=[DMD(svd_rank=-1) for _ in range(len(params_train))]
rom=POD('svd',rank=34)
interpolator=GPR()

pdmd_partitioned_velocity_v = ParametricDMD(dmd, rom, interpolator)
pdmd_partitioned_velocity_v.fit(
    v_train, params_train
)

pdmd_partitioned_velocity_v.parameters=params_train.reshape(-1,1)
rel_v_train_error=np.linalg.norm(pdmd_partitioned_velocity_v.reconstructed_data-v_train)/np.linalg.norm(v_train)

pdmd_partitioned_velocity_v.parameters=params_test.reshape(-1,1)
rel_v_test_error=np.linalg.norm(pdmd_partitioned_velocity_v.reconstructed_data-v_test)/np.linalg.norm(v_test)


p_train=data.V_train[:,:,:,2].transpose(1,2).numpy()
p_test=data.V_test[:,:,:,2].transpose(1,2).numpy()
dmd=[DMD(svd_rank=-1) for _ in range(len(params_train))]
rom=POD('svd',rank=34)
interpolator=GPR()

pdmd_partitioned_velocity_p = ParametricDMD(dmd, rom, interpolator)
pdmd_partitioned_velocity_p.fit(
    p_train, params_train
)

pdmd_partitioned_velocity_p.parameters=params_train.reshape(-1,1)
rel_p_train_error=np.linalg.norm(pdmd_partitioned_velocity_p.reconstructed_data-p_train)/np.linalg.norm(p_train)

pdmd_partitioned_velocity_p.parameters=params_test.reshape(-1,1)
rel_p_test_error=np.linalg.norm(pdmd_partitioned_velocity_p.reconstructed_data-p_test)/np.linalg.norm(p_test)

print("rel train error of u_1 is", "{:.1E}".format(rel_u_train_error))
print("rel test error of u_1 is", "{:.1E}".format(rel_u_test_error))
print("rel train error of u_2 is", "{:.1E}".format(rel_v_train_error))
print("rel test error of u_2 is", "{:.1E}".format(rel_v_test_error))
print("rel train error of p is", "{:.1E}".format(rel_p_train_error))
print("rel test error of p is", "{:.1E}".format(rel_p_test_error))