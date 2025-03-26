import vizdoom as vzd
import numpy as np
import cv2
import torch
import ppo


def image_process(image):
    image = image[:-35, :]
    image = cv2.resize(image, (128, 128))
    return image

def get_state(game):
    number=10
    state = game.get_state()
    labels = state.labels
    game_vars = list(state.game_variables)[:-2]
    obj_ids=[]
    for label in labels:
        if  len(game_vars)+8>number*8+2:
            break
        obj_ids.append(label.object_id+1)
        
        game_vars.append(label.object_position_x)
        game_vars.append(label.object_position_y)
        game_vars.append(label.object_position_z)
        game_vars.append(label.object_velocity_x)
        game_vars.append(label.object_velocity_y)
        game_vars.append(label.object_velocity_z)
        game_vars.append(label.object_angle)
        game_vars.append(label.height)
    
    while len(game_vars) < number*8+2:
        game_vars.append(0)
    while len(obj_ids) < number:
        obj_ids.append(0)
    
    depth_map = state.depth_buffer
    small_map = state.automap_buffer
    normalized_depth = image_process(depth_map / depth_map.max() * 255).astype(np.uint8)
    normalized_small_map = image_process(small_map / small_map.max() * 255).astype(np.uint8)
    return game_vars,obj_ids, normalized_depth, normalized_small_map

# 初始化 DoomGame
game = vzd.DoomGame()
print(vzd.scenarios_path)
game.load_config(vzd.scenarios_path + "/deadly_corridor.cfg")

# 启用深度图
game.set_available_game_variables([
    vzd.GameVariable.HEALTH,        # 血量
    vzd.GameVariable.AMMO2, 
    vzd.GameVariable.KILLCOUNT
])
game.set_screen_format(vzd.ScreenFormat.GRAY8)  # 设置屏幕格式为灰度
game.set_depth_buffer_enabled(True)  # 启用深度缓冲区
game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)

game.set_window_visible(False)
game.init()

agent = ppo.PPO(input_dim=10*8+2, output_dim=7)


step=0
while True:
    game.new_episode()  # 重新开始游戏
    
    state_list = []
    obj_ids_list = []
    image1_list = []
    image2_list = []
    previous_kill_count = 0
    previous_health = 100
    first_step = True
    
    while not game.is_episode_finished():
        
        #print(game.get_ticrate())
        # 获取深度图
        #if step%7==0:
        
        if len(state_list) == agent.warmup_steps:
            
            with torch.no_grad():
                if first_step:
                    torch_state_list = torch.tensor(state_list, dtype=torch.float32).unsqueeze(0)
                    torch_obj_ids_list = torch.tensor(obj_ids_list, dtype=torch.int64).unsqueeze(0)
                    torch_image1_list = torch.tensor(image1_list, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                    torch_image2_list = torch.tensor(image2_list, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                    
            
            
                action,action_log_prob,value = agent.choose_act(torch_state_list, torch_obj_ids_list, torch_image1_list, torch_image2_list)

            
            action_list = np.zeros(7)
            action_list[action] = 1
            reward = game.make_action(action_list, 7)/2
            current_kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
            #print((current_kill_count - previous_kill_count) * 80)
            #print((current_health - previous_health) * 2)
            reward += (current_kill_count - previous_kill_count) * 50
            reward += (current_health - previous_health) * 2
            previous_kill_count = current_kill_count
            previous_health = current_health
            reward = reward/100
            
            done = game.is_episode_finished()
            
            if done:
                with torch.no_grad():
                    agent.store(
                        torch_state_list.squeeze(0), 
                        torch_obj_ids_list.squeeze(0), 
                        torch_image1_list.squeeze(0), 
                        torch_image2_list.squeeze(0), 
                        action,
                        reward,
                        torch_state_list.squeeze(0),
                        torch_obj_ids_list.squeeze(0),
                        torch_image1_list.squeeze(0),
                        torch_image2_list.squeeze(0),
                        done,
                        action_log_prob,
                        value
                        )
                    step+=1
                if step%320==0:
                    agent.train()
                    step=0
                break
            
            state_list.pop(0)
            obj_ids_list.pop(0)
            image1_list.pop(0)
            image2_list.pop(0)

            new_game_vars, new_obj_ids, new_normalized_depth, new_normalized_small_map = get_state(game)
            state_list.append(new_game_vars)
            obj_ids_list.append(new_obj_ids)
            image1_list.append(new_normalized_depth)
            image2_list.append(new_normalized_small_map)
            next_torch_state_list = torch.tensor(state_list, dtype=torch.float32).unsqueeze(0)
            next_torch_image1_list = torch.tensor(image1_list, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            next_torch_image2_list = torch.tensor(image2_list, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            next_torch_obj_ids_list = torch.tensor(obj_ids_list, dtype=torch.int64).unsqueeze(0)
            if not first_step:
                with torch.no_grad():
                    agent.store(
                        torch_state_list.squeeze(0), 
                        torch_obj_ids_list.squeeze(0), 
                        torch_image1_list.squeeze(0), 
                        torch_image2_list.squeeze(0), 
                        action,
                        reward,
                        next_torch_state_list.squeeze(0),
                        next_torch_obj_ids_list.squeeze(0),
                        next_torch_image1_list.squeeze(0),
                        next_torch_image2_list.squeeze(0),
                        done,
                        action_log_prob,
                        value
                        )
                    step+=1
                if step%320==0:
                    agent.train()
                    step=0
                    break
                
            else:
                first_step = False
            
            torch_state_list = next_torch_state_list
            torch_obj_ids_list = next_torch_obj_ids_list
            torch_image1_list = next_torch_image1_list
            torch_image2_list = next_torch_image2_list
            #print(action)
            

            
        else:
            game_vars, obj_ids, normalized_depth, normalized_small_map = get_state(game)
            state_list.append(game_vars)
            obj_ids_list.append(obj_ids)
            image1_list.append(normalized_depth)
            image2_list.append(normalized_small_map)
                
        
        
        

    print("Episode finished! Restarting...")

cv2.destroyAllWindows()
game.close()
