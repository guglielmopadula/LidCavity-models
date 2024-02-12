from torch import nn
from lidcavity import LidCavity 
import torch
from tqdm import trange
import numpy as np
import math
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
num_layers=5
h0=torch.zeros(num_layers,hidden_size)
c0=torch.zeros(num_layers,hidden_size)
input_size=1

params_train=torch.tensor(data.params_train)
time=torch.tensor(data.time)
weights_space=torch.tensor(data.weights_space)
weights_time=torch.tensor(data.weights_time)
max_par=torch.max(params_train,axis=0)[0].item()
max_t=torch.max(time,axis=0)[0].item()
min_par=torch.min(params_train,axis=0)[0].item()
min_t=torch.min(time,axis=0)[0].item()


'''
class Transformer2D_Layer(nn.Module):

    def __init__(self, embed_dim=128, num_heads=8,
                 kdim=None, vdim=None, hidden_dim=256):

        super().__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         kdim=kdim, vdim=vdim,
                                         batch_first=True)
        self.FF = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, 1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, embed_dim, 1)
        )

    def forward(self, x):

        #  x has shape (B, C, N) where N=HW
        x = x.permute(0, 2, 1)
        # now, x has shape (B, N, C) where C=embed_dim
        x = x + self.MHA(x, x, x, need_weights=False)[0]
        x = x.permute(0, 2, 1)
        x = x + self.FF(x)
        return x


class Transformer2D(nn.Module):

    def __init__(self, shape=(4, 4), n_layers=6,
                 MHA_kwargs=dict(embed_dim=128, num_heads=8,
                                 kdim=None, vdim=None, hidden_dim=256),
                 periodic=True):

        super().__init__()
        self.spatial_shape = shape  # (nx, ny)
        nx, ny = shape

        if periodic:
            x, y = torch.meshgrid(torch.arange(nx), torch.arange(ny))
            x_freq = torch.fft.rfftfreq(nx)[1:, None, None]
            y_freq = torch.fft.rfftfreq(ny)[1:, None, None]
            x_sin = torch.sin(2*np.pi*x_freq*x)
            x_cos = torch.cos(2*np.pi*x_freq*x)
            y_sin = torch.sin(2*np.pi*y_freq*y)
            y_cos = torch.cos(2*np.pi*y_freq*y)
            pos_info = torch.cat([x_sin, x_cos, y_sin, y_cos])
        else:
            x, y = torch.meshgrid(torch.arange(1, nx+1)/nx,
                                  torch.arange(1, ny+1)/ny)
            pos_info = torch.stack([x, y])

        dim_pos = pos_info.shape[0]
        self.pos_info = pos_info.unsqueeze(0) # for the batch dimension

        self.pos_embedder = nn.Sequential(
            nn.Conv2d(dim_pos, dim_pos*4, 1), nn.LeakyReLU(),
            nn.Conv2d(dim_pos*4, MHA_kwargs['embed_dim'], 1)
            )

        layers = [Transformer2D_Layer(**MHA_kwargs) for i in range(n_layers)]
        self.transformer = nn.Sequential(*layers)

    def forward(self, x):

        x += self.pos_embedder(self.pos_info.to(x.device))
        x = x.flatten(-2)
        x = self.transformer(x)
        x = x.reshape(*x.shape[:-1], *self.spatial_shape)
        return x

'''
class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.max_len=99
        self.d_model=441
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        self.pe=pe.unsqueeze(0)
    def forward(self, x,pos):
        return x+self.pe

class Transformer(nn.Module):
    def __init__(self,input_size,num_layers):
        super(Transformer,self).__init__()
        self.pos_enc=PositionalEmbedding()
        self.transformer_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_size, nhead=21, dim_feedforward=2048,batch_first=True,dropout=0.), num_layers=num_layers,enable_nested_tensor=False)
        self.lin=nn.Linear(input_size,input_size)
    def forward(self,x,pos):
        x=self.pos_enc(x,pos)
        x=self.transformer_encoder(x)
        x=self.lin(x)
        return x
    

def train_loss(y,y_hat):
    return torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))+torch.var(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1),axis=1).mean()

def rel_train_loss(y,y_hat):
    return torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y.reshape(y.shape[0],-1))/torch.linalg.norm(y.reshape(y.shape[0],-1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model=Transformer(num_x,num_layers)


NUM_TIMES=5
lr=0.001



def train(i,num_times=NUM_TIMES,lr=lr,clip=True,step=True):
    model=Transformer(num_x,num_layers)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10*len(train_loader))
    for epoch in trange(num_times):
        num_s=0
        den_s=0
        norm=0
        for x,y in train_loader:
            y=y[:,:,:,i]
            y_start=y[:,:99]
            y_end=y[:,1:]
            x=x[:,:99,:,:]
            x[:,:,0,0]=(x[:,:,0,0]-min_par)/(max_par-min_par)
            x[:,:,0,3]=(x[:,:,0,3]-min_t)/(max_t-min_t)
            pos=torch.concatenate((x[:,:,0,0].unsqueeze(-1),x[:,:,0,3].unsqueeze(-1)),axis=-1)
            y_hat=model(y_start,pos)
            loss=train_loss(y_end,y_hat)
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
                num_s+=torch.linalg.norm(y_hat.reshape(y_hat.shape[0],-1)-y_end.reshape(y_end.shape[0],-1))**2
                den_s+=torch.linalg.norm(y_end.reshape(y.shape[0],-1))**2
                norm+=torch.linalg.norm(torch.cat(grads)).item()
        with torch.no_grad():
            rel_train_error=torch.sqrt(num_s/den_s).item()
            print(rel_train_error)
            print(torch.var(y_end[-1,:,10]))
            print(torch.var(y_hat[-1,1:,10]))


    with torch.no_grad():
        q_list=[]
        for x,y in train_loader:
            y=y[:,:,:,i]
            y_start=y[:,:99]
            y_end=y[:,1:]
            x=x[:,:99,:,:]
            x[:,:,0,0]=(x[:,:,0,0]-min_par)/(max_par-min_par)
            x[:,:,0,3]=(x[:,:,0,3]-min_t)/(max_t-min_t)
            pos=torch.concatenate((x[:,:,0,0].unsqueeze(-1),x[:,:,0,3].unsqueeze(-1)),axis=-1)
            q=model(y_start,pos)
            q_list.append(q)
        rec_q_train=torch.cat(q_list,dim=0).numpy()
        q_list=[]
        for x,y in test_loader:
            y=y[:,:,:,i]
            y_start=y[:,:99]
            y_end=y[:,1:]
            x=x[:,:99,:,:]
            x[:,:,0,0]=(x[:,:,0,0]-min_par)/(max_par-min_par)
            x[:,:,0,3]=(x[:,:,0,3]-min_t)/(max_t-min_t)
            pos=torch.concatenate((x[:,:,0,0].unsqueeze(-1),x[:,:,0,3].unsqueeze(-1)),axis=-1)
            q=model(y_start,pos)
            q_list.append(q)
        rec_q_test=torch.cat(q_list,dim=0).numpy()
        q_train=data.V_train[:,:,:,i].numpy()
        q_test=data.V_test[:,:,:,i].numpy()
    return rec_q_train,rec_q_test,q_train,q_test

rec_u_train,rec_u_test,u_train,u_test=train(0,NUM_TIMES,lr,False,False)
rec_u_train=np.concatenate((u_train[:,0,:].reshape(-1,1,num_x),rec_u_train),axis=1)
rec_u_test=np.concatenate((u_test[:,0,:].reshape(-1,1,num_x),rec_u_test),axis=1)
rec_v_train,rec_v_test,v_train,v_test=train(1,NUM_TIMES,lr,False,False)
rec_v_train=np.concatenate((v_train[:,0,:].reshape(-1,1,num_x),rec_v_train),axis=1)
rec_v_test=np.concatenate((v_test[:,0,:].reshape(-1,1,num_x),rec_v_test),axis=1)
rec_p_train,rec_p_test,p_train,p_test=train(2,NUM_TIMES,lr,False,False)
rec_p_train=np.concatenate((p_train[:,0,:].reshape(-1,1,num_x),rec_p_train),axis=1)
rec_p_test=np.concatenate((p_test[:,0,:].reshape(-1,1,num_x),rec_p_test),axis=1)


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
