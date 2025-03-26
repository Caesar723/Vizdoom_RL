import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
import numpy as np
import layers.net as net
import layers.simple as simple
import matplotlib.pyplot as plt
import gc
from torch.utils.data import TensorDataset, DataLoader



def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = net.Net(d_t=128, d_r=128, d_state=input_dim, hidden_dim=hidden_dim, num_layers=2, action_dim=output_dim,vocab_size=10,embedding_dim=64)
        self.softmax = nn.Softmax(dim=-1)
        self.critic = nn.Linear(hidden_dim, 1)
        self.actor = nn.Linear(hidden_dim, output_dim)

        
        orthogonal_init(self.critic)
        orthogonal_init(self.actor,gain=0.01)

    def forward(self, images_seq1, images_seq2, state_seq,obj_ids):
        print("images_seq1.shape:",images_seq1.shape,images_seq1)
        print("images_seq2.shape:",images_seq2.shape,images_seq2)
        print("state_seq.shape:",state_seq.shape,state_seq)
        print("obj_ids.shape:",obj_ids.shape,obj_ids)
        x = self.net(images_seq1, images_seq2, state_seq,obj_ids)
        value = self.critic(x)
        
        action_prob = self.softmax(self.actor(x))
        # print(action_prob)
        # print(action_prob.shape)
        # print("action_prob sum:", action_prob.sum(dim=-1))
        dist = Categorical(action_prob)
        return value, action_prob, dist,dist.entropy()

class RunningMeanStd:
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class Normalization:#Trick 2—State Normalization
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        #print(x)
        # Whether to update the mean and std,during the evaluating,update=Flase
        x=np.array(x)
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x
    

class RewardScaling:#Trick 4—Reward Scaling

    def __init__(self, gamma):
        self.shape = 1  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x[0]

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)




class PPO:
    

    def __init__(self,input_dim, output_dim):#input_dim: state
        self.gamma=0.99
        self.lambd=0.95
        self.clip_para=0.2
        self.epochs=5
        self.max_step=3000000
        self.total_step=0
        self.lr=2e-5
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(input_dim=input_dim, hidden_dim=128, output_dim=output_dim).to(self.device)
        #self.load_model("model_complete.pth")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        self.scheduler=StepLR(self.optimizer, step_size=200, gamma=0.99)
        self.MSEloss=nn.MSELoss()
        self.reward_scale=RewardScaling(0.99)

        self.warmup_steps = 4

        self.state = []
        self.obj_ids=[]
        self.images_seq1 = []
        self.images_seq2 = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []
        self.next_images_seq1 = []
        self.next_images_seq2 = []
        self.next_obj_ids = []
        self.init_graph()

    def init_graph(self):
        self.rewards = 0
        self.step=0
        self.rewards_store=[]
        self.best_mean_reward=0
        fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.rewards, label='Total Rewards per Episode', color='b')

        # 设置图表标题和标签
        self.ax.set_title('PPO Training Rewards Over Episodes')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')
        self.ax.legend()
        self.ax.grid(True)

    def normalize_adv(self,adv:torch.Tensor):
        return ((adv - adv.mean()) / (adv.std() + 1e-5))
    
    def choose_act(self,state,obj_ids,images_seq1,images_seq2):
        state = state.to(self.device)
        obj_ids = obj_ids.to(self.device)
        images_seq1 = images_seq1.to(self.device)
        images_seq2 = images_seq2.to(self.device)
        with torch.no_grad():
            value, action_prob, dist, entropy = self.model(images_seq1,images_seq2,state,obj_ids)
            action = dist.sample()
            #action_log_prob = dist.log_prob(action)
        
        return action.item()
    
    def store(self,state,obj_ids,images_seq1,images_seq2,action,reward,next_state,next_obj_ids,next_images_seq1,next_images_seq2,done):
        self.graph_on_step(reward)
        self.state.append(state)
        self.obj_ids.append(obj_ids)
        self.images_seq1.append(images_seq1)
        self.images_seq2.append(images_seq2)
        self.action.append(action)
        #reward=self.reward_scale(reward)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.next_obj_ids.append(next_obj_ids)
        self.done.append(done)
        self.next_images_seq1.append(next_images_seq1)
        self.next_images_seq2.append(next_images_seq2)

        


    def clean(self):
        self.state=[]
        self.images_seq1=[]
        self.images_seq2=[]
        self.obj_ids=[]
        self.action=[]
        self.reward=[]
        self.next_state=[]
        self.next_obj_ids=[]
        self.next_images_seq1=[]
        self.next_images_seq2=[]
        self.done=[]
        self.reward_scale.reset()
        
        gc.collect()
        # if self.device.type == "cuda":
        #     torch.cuda.empty_cache()
        # elif self.device.type == "mps":
        #     torch.mps.empty_cache()
        #torch.mps.empty_cache()

    def advantage_cal(self,delta:torch.Tensor,done:torch.Tensor):
        advantage_list=[]
        advantage=0
        #print(delta.cpu().numpy())
        for reward,done in zip(delta.cpu().numpy()[::-1],done.cpu().numpy()[::-1]):
            if done:
                advantage=0
            advantage=reward+self.lambd*self.gamma*advantage
            # print(advantage)
            # print()
            advantage_list.insert(0,advantage)
        #print(advantage_list)
        return np.array(advantage_list)
    
    def train(self):
        self.graph_on_rollout_end()
        state=torch.FloatTensor(np.array(self.state)).to(self.device).detach()
        obj_ids=torch.IntTensor(np.array(self.obj_ids)).to(self.device).detach()
        images_seq1=torch.FloatTensor(np.array(self.images_seq1)).to(self.device).detach()
        images_seq2=torch.FloatTensor(np.array(self.images_seq2)).to(self.device).detach()
        action=torch.LongTensor(np.array(self.action)).unsqueeze(1).to(self.device).detach()
        done=torch.FloatTensor(np.array(self.done)).unsqueeze(1).to(self.device).detach()
        next_obj_ids=torch.IntTensor(np.array(self.next_obj_ids)).to(self.device).detach()
        next_state=torch.FloatTensor(np.array(self.next_state)).to(self.device).detach()
        next_images_seq1=torch.FloatTensor(np.array(self.next_images_seq1)).to(self.device).detach()
        next_images_seq2=torch.FloatTensor(np.array(self.next_images_seq2)).to(self.device).detach()
        reward=torch.FloatTensor(np.array(self.reward)).unsqueeze(1).to(self.device).detach()

        
        # print(state.shape)
        # print(obj_ids.shape)
        # print(images_seq1.shape)
        # print(images_seq2.shape)
        # print(action.shape)
        # print(done.shape)
        # print(next_state.shape)
        # print(next_images_seq1.shape)
        # print(next_images_seq2.shape)
        # print(next_obj_ids.shape)
        # print(reward.shape)
        with torch.no_grad():
            v, _, dist,_ = self.model(images_seq1,images_seq2,state,obj_ids)
            v_, _, _, _ = self.model(next_images_seq1,next_images_seq2,next_state,next_obj_ids)
            delta=reward+self.gamma*v_*(1-done)-v
            advantage=self.advantage_cal(delta,done)
            advantage=torch.FloatTensor(advantage).detach().to(self.device)
            rewards=advantage+v
            advantage=self.normalize_adv(advantage)
            #print(advantage.shape)
            action_log_prob = dist.log_prob(action.squeeze(-1))
            #print(action_log_prob.shape)

        # dataset = TensorDataset(
        #     state,
        #     obj_ids,
        #     images_seq1,
        #     images_seq2,
        #     action,
        #     done,
        #     next_state,
        #     next_obj_ids,
        #     next_images_seq1,
        #     next_images_seq2,
        #     advantage,action_log_prob,rewards
        #     )
        #dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        batch_size = 64
        data_size=state.size(0)
        for _ in range(self.epochs):
            # indices = torch.randperm(data_size)
            # state=state[indices]
            # obj_ids=obj_ids[indices]
            # images_seq1=images_seq1[indices]
            # images_seq2=images_seq2[indices]
            # action=action[indices]
            
            # advantage=advantage[indices]
            # action_log_prob=action_log_prob[indices]
            # rewards=rewards[indices]
            print("epoch",_)
            #for batch in dataloader:
            for i in range(0,data_size,batch_size):
                state_batch=state[i:i+batch_size]
                obj_ids_batch=obj_ids[i:i+batch_size]
                images_seq1_batch=images_seq1[i:i+batch_size]
                images_seq2_batch=images_seq2[i:i+batch_size]
                action_batch=action[i:i+batch_size]
                
                advantage_batch=advantage[i:i+batch_size]
                action_log_prob_batch=action_log_prob[i:i+batch_size]
                rewards_batch=rewards[i:i+batch_size]



                
                v, _, dist,entropy = self.model(images_seq1_batch,images_seq2_batch,state_batch,obj_ids_batch)
                new_prob_log=dist.log_prob(action_batch.squeeze(-1))
                
                rate=torch.exp(new_prob_log-action_log_prob_batch.detach()).unsqueeze(1)
                surr1=rate*advantage_batch
                surr2=torch.clamp(rate,1-self.clip_para,1+self.clip_para)*advantage_batch


                act_loss=-torch.min(surr1, surr2).mean()
                val_loss=F.mse_loss(v,rewards_batch)
                
                
                loss=act_loss+0.5 *val_loss-0.01*entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()
                print("loss",loss)

        
        self.clean()

    def graph_on_rollout_end(self) -> None:
        if self.best_mean_reward<self.rewards/self.step:
            self.best_mean_reward=self.rewards/self.step
            torch.save(self.model, '/mnt/model_complete.pth')
            #torch.save(self.model_val, 'model_complete_val.pth')
        torch.save(self.model, '/mnt/model_complete_normal.pth')
        self.rewards_store.append(self.rewards)
        self.rewards = 0
        self.step=0
        self.line.set_data(range(len(self.rewards_store)), self.rewards_store)
    
        self.ax.set_xlim(0, len(self.rewards_store))
        self.ax.set_ylim(min(self.rewards_store) - 5, max(self.rewards_store) + 5)
        plt.savefig('/mnt/ppo_training_reward.png')
        
    def graph_on_step(self,reward):
            self.step+=1
            self.rewards+=reward

    def load_model(self,path):
        self.model = torch.load(path,weights_only=False).to(self.device)

if __name__ == "__main__":
    ppo = PPO(input_dim=7, output_dim=10)
    
    for i in range(50):
        state = torch.randn(128, 5, 10).to(ppo.device)
        images_seq1 = torch.randn(128, 5, 1, 64, 64).to(ppo.device)
        images_seq2 = torch.randn(128, 5, 1, 64, 64).to(ppo.device)
        
        print(ppo.model(images_seq1,images_seq2,state))
    #action, action_log_prob = ppo.model(images_seq1,images_seq2,state)
    # print(action, action_log_prob)