from torch import nn
from ezyrb import POD
from lidcavity import LidCavity
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import math
import numpy as np
from tqdm import trange
a=LidCavity(10)

y_train=a.V_train
y_test=a.V_test
parameters_train=a.params_train
parameters_test=a.params_test

data=LidCavity(10)
torch.manual_seed(0)
train_loader=data.train_loader
test_loader=data.test_loader
l_t=len(train_loader)
hidden_size=500
output_size=1
num_times=data.V_train.shape[1]
num_x=data.V_train.shape[2]
len_trainloader=data.V_train.shape[0]
len_testloader=data.V_test.shape[0]
num_layers=1
input_size=441

params_train=torch.tensor(data.params_train).float()
params_test=torch.tensor(data.params_test).float()
params_train=params_train.unsqueeze(1)
params_test=params_test.unsqueeze(1)
time=torch.tensor(data.time)

weights_space=torch.tensor(data.weights_space)
weights_time=torch.tensor(data.weights_time)


max_par=torch.max(params_train,axis=0)[0].item()
min_par=torch.min(params_train,axis=0)[0].item()
max_t=torch.max(time,axis=0)[0].item()
min_t=torch.min(time,axis=0)[0].item()

params_train=(params_train-min_par)/(max_par-min_par)
params_test=(params_test-min_par)/(max_par-min_par)

print(min_t)

max_grid=torch.max(data.grid_train[:,:,:,1]).item()
min_grid=torch.min(data.grid_train[:,:,:,1]).item()

NUM_TIMES=500
lr=0.0001

class TorchedPOD():
    def __init__(self,rank):
        self.pod=POD(method='randomized_svd',rank=rank)

    
    def fit(self,X):
        self.pod.fit(X.numpy())
        return torch.tensor(self.pod._modes).float(), torch.tensor(self.pod.transform(X)).float()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:-1]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers =  TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    

def train_loss(y,y_hat):
    return torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))+torch.var(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1),axis=1).mean()

def rel_train_loss(y,y_hat):
    return torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))/torch.linalg.norm(y.reshape(y.shape[0],-1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(i,num_epochs=NUM_TIMES,lr=lr):
    y=y_train[:,:,:,i]
    y=y.reshape(y.shape[0],-1)
    torchpod=TorchedPOD(30)
    modes,basis=torchpod.fit(y)
    basis=basis.reshape(-1,num_times,num_x)
    basis=basis.transpose(0,1)
    model=TransformerModel(input_size,d_hid=hidden_size,nhead=21,nlayers=num_layers)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len_trainloader*50, gamma=0.1)
    for epoch in trange(num_epochs):
        num_s=0
        den_s=0
        norm=0
        for j in range(30):
            tmp=basis[:,j,:].unsqueeze(1)
            x=tmp[:99]
            y=tmp[1:]
            y_hat=model(x)
            loss=train_loss(y,y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
            ]
            #scheduler.step()
            with torch.no_grad():
                num_s+=torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))**2
                den_s+=torch.linalg.norm(y.reshape(y.shape[0],-1))**2
                norm+=torch.linalg.norm(torch.cat(grads)).item()
        with torch.no_grad():
            rel_train_error=torch.sqrt(num_s/den_s).item()
            print(rel_train_error)
            print(norm/l_t)

    model_param=nn.Sequential(nn.Linear(1,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,30))
    optimizer=torch.optim.Adam(model_param.parameters(),lr=lr)

    for epoch in trange(num_epochs):
        y_hat=model_param(params_train)
        loss=train_loss(modes,y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            rel_train_error=torch.linalg.norm(modes-y_hat)/torch.linalg.norm(modes+1)
            print(rel_train_error)


    model=model.eval()
    model_param=model_param.eval()
    with torch.no_grad():
        q_list=[]
        for j in range(30):
            tmp=basis[:,j,:].unsqueeze(1)
            x=tmp[:99]
            y=tmp[1:]
            y_hat=model(x)
            q_list.append(torch.cat((x.transpose(0,1)[:,0,:].unsqueeze(1),y_hat.transpose(0,1)),axis=1))
        rec_q=torch.cat(q_list,dim=0).numpy().reshape(30,-1)
        
        rec_modes_train=model_param(params_train)
        rec_q_train=rec_modes_train@rec_q
        rec_q_train=rec_q_train.reshape(-1,num_times,num_x)

        rec_modes_test=model_param(params_test)

        rec_q_test=rec_modes_test@rec_q
        rec_q_test=rec_q_test.reshape(-1,num_times,num_x)

        q_train=y_train[:,:,:,i]
        q_test=y_test[:,:,:,i]

    return rec_q_train.numpy(),rec_q_test.numpy(),q_train.numpy(),q_test.numpy()


rec_u_train,rec_u_test,u_train,u_test=train(0,NUM_TIMES,lr)
rec_v_train,rec_v_test,v_train,v_test=train(1,NUM_TIMES,lr)
rec_p_train,rec_p_test,p_train,p_test=train(2,NUM_TIMES,lr)

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
