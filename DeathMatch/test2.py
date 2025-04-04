if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import vizdoom as vzd
import numpy as np
import cv2
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import DeathMatch.ppo2 as ppo2


def image_process(image):
    image = image[:-75, :]
    image = cv2.resize(image, (128, 128))
    return image


def pad_or_truncate(tensor, fixed_len):
    N, D = tensor.shape
    if N > fixed_len:
        return tensor[:fixed_len]  # truncate
    elif N < fixed_len:
        pad_len = fixed_len - N
        padding = torch.zeros(pad_len, D, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)  # pad
    else:
        return tensor

def get_state(game):
    state = game.get_state()

    keeped_ids={
        # "DoomPlayer":0,
        "Rocket":1,
        "ShotgunGuy":2,
        "MarineChainsawVzd":3,
        "Zombieman":4,
        "ChaingunGuy":5,
        "Demon":6,
        "HellKnight":7,
        "Medikit":8,
        "Stimpack":9,
        "HealthBonus":10,
        "GreenArmor":11,
        "BlueArmor":12,
        "ArmorBonus":13
    }
    class_opponent=[
        "DoomPlayer",
        "Rocket",
        "ShotgunGuy",
        "MarineChainsawVzd",
        "Zombieman",
    ]
    class_buff=[
        "HealthBonus",
        "GreenArmor",
        "BlueArmor",
        "ArmorBonus"
    ]
    class_ammo=[
        "Rocket"
    ]
    all_class=[
        (class_opponent,[],10),
        (class_buff,[],10),
        (class_ammo,[],4)
    ]
    #labels=[[   0,  -82,   62,  163,  100]]
    
    for i in state.labels:
        if i.object_name not in keeped_ids:
            continue
        
        element=[
            
            i.x+i.width//2, #- state.screen_buffer.shape[1] // 2,
            i.y+i.height//2, #- state.screen_buffer.shape[0] // 2,
            
        ]
        for class_tuple in all_class:
            if i.object_name in class_tuple[0]:
                if len(class_tuple[1])<class_tuple[2]:
                    class_tuple[1].append(element[0])
                    class_tuple[1].append(element[1])
                break
        
    for class_tuple in all_class:
        while len(class_tuple[1])<class_tuple[2]:
            class_tuple[1].append(0)
            class_tuple[1].append(0)
        #labels.append(element)
        
        
    
    
    depth_map = state.depth_buffer
    # small_map = state.automap_buffer
    # map=state.screen_buffer

    normalized_depth = image_process(depth_map)/255
    # normalized_map = image_process(map)/255
    # #print(map.shape)
    # cropped_map = cv2.resize(map[:-75, 250:-250], (128, 128))/255

    normal_state=[
        game.get_game_variable(vzd.GameVariable.HEALTH),
        game.get_game_variable(vzd.GameVariable.AMMO2)
    ]

    labels=[]
    for class_tuple in all_class:
        labels+=class_tuple[1]
    labels=torch.FloatTensor(np.array(labels)/500)
    
    return normalized_depth,labels,normal_state


def pad_labels(labels_cache):
    max_len=25
    #print(labels_cache)
    labels_pad=[pad_or_truncate(label, max_len) for label in labels_cache]
    lengths = torch.tensor([min(label.size(0), max_len) for label in labels_cache])
    
    
    labels_pad=torch.stack(labels_pad, dim=0)
    
    B, N_max = labels_pad.shape[0], labels_pad.shape[1]
    mask = torch.arange(N_max)[None, :].expand(B, N_max) < lengths[:, None]  # shape: [B, N_max]
    return labels_pad,mask


def state_iter(game):
    labels_cache=[]
    for i in range(50):
        normalized_depth, labels,normal_state = get_state(game)
        labels_cache.append(labels)
        #game.advance_action()
        #yield None
    
    #labels_pad,mask=pad_labels(labels_cache)

    #normalized_depth, labels,normal_state = get_state(game)
    while True:
        
        
        #labels_pad=pad_or_truncate(labels, 25)
        
        #mask = torch.arange(labels_pad.shape[0])[None, :] < min(labels.size(0),25)
        # print(mask.shape)
        # print(labels_pad.shape)
        # print(mask)
        tensor_labels=torch.stack(labels_cache)
        #print(tensor_labels)
        yield normalized_depth, tensor_labels,normal_state
        normalized_depth, labels,normal_state = get_state(game)
        labels_cache.pop(0)
        labels_cache.append(labels)
        
        #labels_pad,mask=pad_labels(labels_cache)

def get_reward(game,previous_kill_count,previous_health,previous_ammo):
    current_kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
    current_ammo = game.get_game_variable(vzd.GameVariable.AMMO5)
    reward=0
    reward += (current_kill_count - previous_kill_count) * 1000
    reward += -20
    reward += (current_ammo - previous_ammo) * 100
    #if current_health>previous_health:
    reward += (current_health - previous_health) * 1
    previous_kill_count = current_kill_count
    previous_health = current_health
    previous_ammo = current_ammo
    
    done = game.is_episode_finished()
    if done and previous_health<=0:
        reward=-10000
    # elif done and previous_health>0:
    #     reward=1000
    
    reward = reward/1000
    return reward,done,current_kill_count,current_health,current_ammo
# 初始化 DoomGame
game = vzd.DoomGame()

game.load_config(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+"/DeathMatch/myconfig.cfg")

# 启用深度图
game.set_available_game_variables([
    vzd.GameVariable.HEALTH,        # 血量
    vzd.GameVariable.AMMO5, 
    vzd.GameVariable.KILLCOUNT,
    vzd.GameVariable.AMMO2,
])
game.set_screen_format(vzd.ScreenFormat.GRAY8)  # 设置屏幕格式为灰度
game.set_depth_buffer_enabled(True)  # 启用深度缓冲区
game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)

game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_window_visible(False)
game.init()


num_actions=9
agent = ppo2.PPO(input_size=128,label_size=24,output_dim=num_actions)

frame_repeat=10
step=1
while True:
    game.new_episode()  # 重新开始游戏
    game.send_game_command("removeallitems")
    game.send_game_command("give rocketlauncher")
    game.send_game_command("use rocketlauncher")
    game.send_game_command("give allammo")
    game.add_game_args("+sv_nopickup 1")
    # state_list = []
    # obj_ids_list = []
    # image1_list = []
    # image2_list = []
    previous_kill_count = 0
    previous_health = 100
    previous_ammo = game.get_game_variable(vzd.GameVariable.AMMO5)
    first_step = True

    state_getter=state_iter(game)
    next_state=next(state_getter)
    # while next_state is None:
    #     next_state=next(state_iter)

    normalized_depth, labels,normal_state=next_state
    # print(normalized_depth.shape)
    # print(labels_pad.shape)
    # print(normal_state)
    # print(mask.shape)
    # exit()
   
    
    while not game.is_episode_finished():
        game.send_game_command("give ammo")
        
        step+=1
        
        with torch.no_grad():
            if first_step:
                torch_image1_list = torch.tensor(normalized_depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                normal_state_list = torch.tensor(normal_state, dtype=torch.float32).unsqueeze(0)
                labels_list = labels.unsqueeze(0)
                #print(labels_list.shape)
                
                
                
                
        
        
            action =agent.choose_act(torch_image1_list, labels_list, normal_state_list)
            #action = 3
        action_list = np.zeros(num_actions)
        action_list[action] = 1
        reward_path = game.make_action(action_list)/2
        for _ in range(frame_repeat):
            game.advance_action()
        # current_kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        # current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        # current_ammo = game.get_game_variable(vzd.GameVariable.AMMO5)
        #print((current_kill_count - previous_kill_count) * 80)
        #print((current_health - previous_health) * 2)
        # reward=0
        # reward += (current_kill_count - previous_kill_count) * 1000
        # reward += -20
        # reward += (current_ammo - previous_ammo) * 100
        # #reward += (current_health - previous_health) * 1
        # previous_kill_count = current_kill_count
        # previous_health = current_health
        # previous_ammo = current_ammo
        
        # done = game.is_episode_finished()
        # if done and previous_health<=0:
        #     reward=-700
        # elif done and previous_health>0:
        #     reward=1000
        
        # reward = reward/1000

        reward,done,current_kill_count,current_health,current_ammo=get_reward(game,previous_kill_count,previous_health,previous_ammo)
        previous_kill_count = current_kill_count
        previous_health = current_health
        previous_ammo = current_ammo
        # print(reward)
        # print(reward_path)
        # print()
        if done:
            with torch.no_grad():
                agent.store(
                    
                    torch_image1_list.squeeze(0), 
                    labels.squeeze(0), 
                    normal_state_list.squeeze(0),
                    
                    action,
                    reward,
                    
                    torch_image1_list.squeeze(0), 
                    labels.squeeze(0), 
                    normal_state_list.squeeze(0),
                    
                    
                    done
                    )
                
            if step%256==0:
                agent.train()
                step=0
            break
        
        

        next_normalized_depth, next_labels,next_normal_state=next(state_getter)
        
        
        
        with torch.no_grad():
            next_torch_image1_list = torch.tensor(next_normalized_depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            next_normal_state_list = torch.tensor(next_normal_state, dtype=torch.float32).unsqueeze(0)
            next_labels_list = next_labels.unsqueeze(0)
            
            
            agent.store(
                torch_image1_list.squeeze(0), 
                labels_list.squeeze(0), 
                normal_state_list.squeeze(0),
                
                action,
                reward,
                next_torch_image1_list.squeeze(0),
                next_labels_list.squeeze(0),
                next_normal_state_list.squeeze(0),
                
                done
                )
            
        if step%256==0:
            agent.train()
            step=0
            break
            
        normalized_depth=next_normalized_depth
        labels=next_labels
        normal_state=next_normal_state
        
        
        
        
            
            

            
        
        
        
        

    print("Episode finished! Restarting...")

cv2.destroyAllWindows()
game.close()
