import vizdoom as vzd
import numpy as np
import cv2
import torch
import ppo2


def image_process(image):
    image = image[:-75, :]
    image = cv2.resize(image, (128, 128))
    return image

def get_state(game):
    
    state = game.get_state()
    
   
    
    depth_map = state.depth_buffer
    small_map = state.automap_buffer
    map=state.screen_buffer

    normalized_depth = image_process(depth_map)/255
    normalized_small_map = image_process(small_map)/255
    normalized_map = image_process(map)/255
    #print(map.shape)
    cropped_map = cv2.resize(map[:-75, 250:-250], (128, 128))/255
    return normalized_depth, cropped_map,normalized_map

# 初始化 DoomGame
game = vzd.DoomGame()
print(vzd.scenarios_path)
game.load_config(vzd.scenarios_path + "/deadly_corridor.cfg")

# 启用深度图
game.set_available_game_variables([
    vzd.GameVariable.HEALTH,        # 血量
    vzd.GameVariable.AMMO5, 
    vzd.GameVariable.KILLCOUNT
])
game.set_screen_format(vzd.ScreenFormat.GRAY8)  # 设置屏幕格式为灰度
game.set_depth_buffer_enabled(True)  # 启用深度缓冲区
game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_window_visible(False)
game.init()

num_actions=7
agent = ppo2.PPO(input_size_1=128,input_size_2=128,input_size_3=128,output_dim=num_actions)

frame_repeat=10
step=1
while True:
    game.new_episode()  # 重新开始游戏
    
    # state_list = []
    # obj_ids_list = []
    # image1_list = []
    # image2_list = []
    previous_kill_count = 0
    previous_health = 100
    previous_ammo = game.get_game_variable(vzd.GameVariable.AMMO5)
    first_step = True
    
    while not game.is_episode_finished():
        
        #print(game.get_ticrate())
        # 获取深度图
        #if step%7==0:
        
        #if len(state_list) == agent.warmup_steps:
        step+=1
        normalized_depth, cropped_map,normalized_map = get_state(game)
        # print(normalized_depth.shape)
        # print(cropped_map.shape)
        # print(normalized_map.shape)
        # print(normalized_depth)
        # print(cropped_map)
        #print((cropped_map*255).astype(np.uint8))
        # cv2.imshow("cropped_map",cropped_map)
        # cv2.imshow("cropped_map2",np.round((cropped_map.astype(np.float32)/255*255).astype(np.uint8)))
        # cv2.imshow("normalized_map",normalized_map)
        # cv2.imshow("normalized_depth",normalized_depth)
        # cv2.waitKey(0)
        with torch.no_grad():
            if first_step:
                torch_image1_list = torch.tensor(normalized_depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                torch_image2_list = torch.tensor(cropped_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                torch_image3_list = torch.tensor(normalized_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
        
        
            action = agent.choose_act(torch_image1_list, torch_image2_list, torch_image3_list)
            #action = 0
        action_list = np.zeros(num_actions)
        action_list[action] = 1
        reward_path = game.make_action(action_list)/2
        for _ in range(frame_repeat):
            game.advance_action()
        current_kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_ammo = game.get_game_variable(vzd.GameVariable.AMMO5)
        #print((current_kill_count - previous_kill_count) * 80)
        #print((current_health - previous_health) * 2)
        reward=0
        reward += (current_kill_count - previous_kill_count) * 1000
        reward += reward_path*10
        reward += (current_ammo - previous_ammo) * 100
        #reward += (current_health - previous_health) * 1
        previous_kill_count = current_kill_count
        previous_health = current_health
        previous_ammo = current_ammo
        
        done = game.is_episode_finished()
        if done and previous_health<=0:
            reward=-1000
        elif done and previous_health>0:
            reward=1000
        
        reward = reward/1000
        if done:
            with torch.no_grad():
                agent.store(
                    
                    torch_image1_list.squeeze(0), 
                    torch_image2_list.squeeze(0), 
                    torch_image3_list.squeeze(0),
                    action,
                    reward,
                    
                    torch_image1_list.squeeze(0),
                    torch_image2_list.squeeze(0),
                    torch_image3_list.squeeze(0),
                    done
                    )
                
            if step%256==0:
                agent.train()
                step=0
            break
        
        

        new_normalized_depth, new_cropped_map,new_normalized_map = get_state(game)
        
        
        
        with torch.no_grad():
            next_torch_image1_list = torch.tensor(new_normalized_depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            next_torch_image2_list = torch.tensor(new_cropped_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            next_torch_image3_list = torch.tensor(new_normalized_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            agent.store(
                
                torch_image1_list.squeeze(0), 
                torch_image2_list.squeeze(0), 
                torch_image3_list.squeeze(0),
                action,
                reward,
                next_torch_image1_list.squeeze(0),
                next_torch_image2_list.squeeze(0),
                next_torch_image3_list.squeeze(0),
                done
                )
            
        if step%256==0:
            agent.train()
            step=0
            break
            
        
        
        torch_image1_list = next_torch_image1_list
        torch_image2_list = next_torch_image2_list
        torch_image3_list = next_torch_image3_list
            
            

            
        
        
        
        

    print("Episode finished! Restarting...")

cv2.destroyAllWindows()
game.close()
