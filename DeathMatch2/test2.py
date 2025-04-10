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
import DeathMatch2.ppo2 as ppo2


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
    
    class_opponent=[
        # "DoomPlayer",
        "ShotgunGuy",
        "MarineChainsawVzd",
        "Zombieman",
        "ChaingunGuy",
        "Demon",
        "HellKnight",
        
        
    ]

    class_self=[
        "DoomPlayer",
    ]

    class_buff=[
        "HealthBonus",
        "Medikit",
    ]
    class_rocket=[
        "Rocket"
    ]

    all_class=[
        [class_opponent,[255, 0, 0,0],[]],
        [class_self,[0,  255, 0,0],[]],
        [class_buff,[0, 0, 255,0],[]],
        [class_rocket,[0, 0, 0,255],[]],
    ]

   
    
    
    
    depth_map = state.depth_buffer
    small_map = state.automap_buffer

    labels = state.labels_buffer
    #print(labels)
    color_map = np.zeros((labels.shape[0], labels.shape[1],4), dtype=np.uint8)
    for label in state.labels:
        for i in range(len(all_class)):
            if label.object_name in all_class[i][0]:
                all_class[i][2].append(label.value)
                break
    for i in range(len(all_class)):
        color_map[np.isin(labels,all_class[i][2])]=all_class[i][1]
    #
    color_map_process = color_map[:-75,:, :]

    color_map_process = cv2.resize(color_map_process, (128, 128))
    color_map_process=color_map_process.transpose(2,0,1)/255

    # medi_img=color_map_process[2]
    # color_map_process=color_map_process[:2]
    
    

    
    map=state.screen_buffer
    life_img=map[-75:-25,100:210]
    life_img=cv2.resize(life_img,(128,128))
    life_img=life_img/255
    

    normalized_depth = image_process(depth_map)/255
    normalized_depth = np.expand_dims(normalized_depth, axis=0)
    # normalized_small_map = image_process(small_map)/255
    #normalized_map = image_process(map)/255
    #normalized_map = np.expand_dims(normalized_map, axis=0)
    #print(map.shape)

    #print(color_map[:-75, 250:-250,0].shape)
    cropped_map = cv2.resize(color_map[:-75, 250:-250,0], (128, 128))/255
    cropped_map = np.expand_dims(cropped_map, axis=0)

   
    img_stat=np.concatenate([normalized_depth,color_map_process,cropped_map,],axis=0)
    #print(img_stat.shape)
    return img_stat,life_img


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
    map_cache=[]
    for i in range(8):
        state_img,life_img= get_state(game)
        map_cache.append(state_img)
        game.advance_action()
        #yield None
    
    #labels_pad,mask=pad_labels(labels_cache)

    #normalized_depth, labels,normal_state = get_state(game)
    while True:
        
        
        tensor_map=torch.FloatTensor(np.array(map_cache)).detach()
        #print(tensor_map.shape)
        #print(tensor_labels)
        yield tensor_map,life_img
        state_img,life_img = get_state(game)
        map_cache.pop(0)
        map_cache.append(state_img)
        
        #labels_pad,mask=pad_labels(labels_cache)

def get_reward(game,previous_kill_count,previous_health,previous_ammo):
    current_kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
    current_ammo = game.get_game_variable(vzd.GameVariable.AMMO5)
    reward=0
    reward += (current_kill_count - previous_kill_count) * 1000
    reward += -20
    if current_ammo<previous_ammo:
        reward += (current_ammo - previous_ammo) * 100
    #if current_health>previous_health:
    reward += (current_health - previous_health) * 10
    
    previous_kill_count = current_kill_count
    previous_health = current_health
    previous_ammo = current_ammo
    
    done = game.is_episode_finished()
    
    if done and previous_health<=0:
        reward=-1000
    # elif done and previous_health>0:
    #     reward=1000
    
    reward = reward/1000
    #print(reward)
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
    vzd.GameVariable.PLAYER_NUMBER
])
game.set_screen_format(vzd.ScreenFormat.GRAY8)  # 设置屏幕格式为灰度
game.set_depth_buffer_enabled(True)  # 启用深度缓冲区
game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)

game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_window_visible(False)
game.init()


num_actions=9
agent = ppo2.PPO(input_size=128,output_dim=num_actions)

frame_repeat=10
step=1
while True:
    game.new_episode()  # 重新开始游戏
    game.send_game_command("removeallitems")
    game.send_game_command("give rocketlauncher")
    game.send_game_command("use rocketlauncher")
    game.send_game_command("give allammo")
    game.add_game_args("+sv_nopickup 1")
    
    previous_kill_count = 0
    previous_health = 100
    previous_ammo = 50#game.get_game_variable(vzd.GameVariable.AMMO5)
    first_step = True

    state_getter=state_iter(game)
    next_state=next(state_getter)
    

    state_img,life_img=next_state
   
    
    while not game.is_episode_finished():
        game.send_game_command("give ammo")
        
        step+=1
        
        with torch.no_grad():
            if first_step:
                torch_image1_list = torch.tensor(life_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                torch_image2_list = torch.tensor(state_img, dtype=torch.float32).unsqueeze(0)
                
            action =agent.choose_act(torch_image1_list, torch_image2_list)
            #action = 3
        action_list = np.zeros(num_actions)
        action_list[action] = 1
        reward_path = game.make_action(action_list)/2
        for _ in range(frame_repeat):
            game.advance_action()
        

        reward,done,current_kill_count,current_health,current_ammo=get_reward(game,previous_kill_count,previous_health,previous_ammo)
        previous_kill_count = current_kill_count
        previous_health = current_health
        previous_ammo = current_ammo
        
        if done:
            with torch.no_grad():
                agent.store(
                    
                    torch_image1_list.squeeze(0), 
                    torch_image2_list.squeeze(0), 
                   
                    action,
                    reward,
                    
                    torch_image1_list.squeeze(0), 
                    torch_image2_list.squeeze(0), 
                    
                    done
                    )
                
            if step%256==0:
                agent.train()
                step=0
            break
        
        

        next_state_img,next_life_img=next(state_getter)
        
        
        
        with torch.no_grad():
            next_torch_image1_list = torch.tensor(next_life_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            next_torch_image2_list = torch.tensor(next_state_img, dtype=torch.float32).unsqueeze(0)
            
            
            agent.store(
                torch_image1_list.squeeze(0), 
                torch_image2_list.squeeze(0), 
                
                action,
                reward,
                next_torch_image1_list.squeeze(0),
                next_torch_image2_list.squeeze(0),
                
                done
                )
            
        if step%256==0:
            agent.train()
            step=0
            break
            
        
        life_img=next_life_img
        state_img=next_state_img
        
        
        
        
            
            

            
        
        
        
        

    print("Episode finished! Restarting...")

cv2.destroyAllWindows()
game.close()
