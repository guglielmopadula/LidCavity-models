from torch import nn
from lidcavity import LidCavity 
import torch
from tqdm import trange
import numpy as np

data=LidCavity(10)
torch.manual_seed(0)
train_loader=data.train_loader
test_loader=data.test_loader
l_t=len(train_loader)
hidden_size=200
output_size=data.V_train.shape[2]
num_times=data.V_train.shape[1]
num_x=data.V_train.shape[2]
len_trainloader=data.V_train.shape[0]
len_testloader=data.V_test.shape[0]
num_layers=1
h0=torch.zeros(num_layers,hidden_size)
c0=torch.zeros(num_layers,hidden_size)
input_size=1

params_train=torch.tensor(data.params_train)
time=torch.tensor(data.time)

weights_space=torch.tensor(data.weights_space)
weights_time=torch.tensor(data.weights_time)


max_par=torch.max(params_train,axis=0)[0].item()
max_t=torch.max(time,axis=0)[0].item()



class RNNModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(RNNModel,self).__init__()
        self.rnn=nn.GRU(input_size,hidden_size,num_layers=num_layers,batch_first=True)
        self.seq=nn.Sequential(
                nn.Linear(hidden_size,hidden_size),
                #nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,hidden_size),
                #nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,hidden_size),
                #nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,hidden_size),
                #nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,output_size)
                )
        
    def forward(self,x,h0):
        h0_=h0
        x=x[:,:,0,3].unsqueeze(-1)
        x=x/max_par
        h0_=h0_.unsqueeze(1).repeat(1,x.shape[0],1)
        x,_=self.rnn(x,h0_)
        x=self.seq(x)
        return x
    

def train_loss(y,y_hat):
    return torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))

def rel_train_loss(y,y_hat):
    return torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))/torch.linalg.norm(y.reshape(y.shape[0],-1))

model=RNNModel(input_size,hidden_size,output_size,num_layers)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



NUM_TIMES=500
lr=0.0001



def train(i,num_times=NUM_TIMES,lr=lr,clip=True,step=True):
    model=RNNModel(input_size,hidden_size,output_size,num_layers)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10*len(train_loader))
    for epoch in trange(num_times):
        num_s=0
        den_s=0
        norm=0
        for x,y in train_loader:
            y=y[:,:,:,i]
            y_hat=model(x,h0)
            loss=train_loss(y,y_hat)
            optimizer.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()
            if step:
                scheduler.step(loss)
            grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
            ]


            with torch.no_grad():
                num_s+=torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))**2
                den_s+=torch.linalg.norm(y.reshape(y.shape[0],-1))**2
                norm+=torch.linalg.norm(torch.cat(grads)).item()
        with torch.no_grad():
            rel_train_error=torch.sqrt(num_s/den_s).item()
            print(rel_train_error)
            print(norm/l_t)

    model=model.eval()

    with torch.no_grad():
        q_list=[]
        for x,y in train_loader:
            q=model(x,h0)
            q_list.append(q)
        rec_q_train=torch.cat(q_list,dim=0).numpy()
        q_list=[]
        for x,y in test_loader:
            q=model(x,h0)
            q_list.append(q)
        rec_q_test=torch.cat(q_list,dim=0).numpy()
        q_train=data.V_train[:,:,:,i].numpy()
        q_test=data.V_test[:,:,:,i].numpy()
    return rec_q_train,rec_q_test,q_train,q_test

rec_u_train,rec_u_test,u_train,u_test=train(0)
rec_v_train,rec_v_test,v_train,v_test=train(1,100,0.0001,False,False)
rec_p_train,rec_p_test,p_train,p_test=train(2)


weights_space=data.weights_space
weights_time=data.weights_time

diff_x=data.diff_x.T
diff_y=data.diff_y.T

def loss_function(x,y):
    err=x-y
    weights_space_tmp=weights_space.reshape(1,1,-1)
    return np.mean(np.sqrt(np.max(np.sum(weights_space_tmp*(err**2+(err@diff_x)**2+(err@diff_y)**2),axis=2),axis=1)/np.max(np.sum(weights_space_tmp*(y**2+(y@diff_x)**2+(y@diff_y)**2),axis=2),axis=1)))

rel_u_train_error=loss_function(rec_u_train,u_train)
rel_u_test_error=loss_function(rec_u_test,u_test)
rel_v_train_error=loss_function(rec_v_train,v_train)
rel_v_test_error=loss_function(rec_v_test,v_test)
rel_p_train_error=loss_function(rec_p_train,p_train)
rel_p_test_error=loss_function(rec_p_test,p_test)

print(rel_v_train_error)
print(rel_v_test_error)

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
