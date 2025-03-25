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
    state = game.get_state()
    game_vars = state.game_variables
    depth_map = state.depth_buffer
    small_map = state.automap_buffer
    normalized_depth = image_process(depth_map / depth_map.max() * 255).astype(np.uint8)
    normalized_small_map = image_process(small_map / small_map.max() * 255).astype(np.uint8)
    return game_vars, normalized_depth, normalized_small_map

# 初始化 DoomGame
game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/deadly_corridor.cfg")

# 启用深度图
game.set_available_game_variables([
    vzd.GameVariable.HEALTH,        # 血量
    vzd.GameVariable.AMMO2,         # 子弹
])
game.set_screen_format(vzd.ScreenFormat.GRAY8)  # 设置屏幕格式为灰度
game.set_depth_buffer_enabled(True)  # 启用深度缓冲区
game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)
game.init()

agent = ppo.PPO(input_dim=2, output_dim=7)
while True:
    game.new_episode()  # 重新开始游戏
    
    state_list = []
    image1_list = []
    image2_list = []
    first_step = True
    while not game.is_episode_finished():
        
        game_vars, normalized_depth, normalized_small_map = get_state(game)
        game.make_action([0,0,0,0,0,1,0])
        cv2.imshow("depth", normalized_depth)
        cv2.imshow("small_map", normalized_small_map)
        cv2.waitKey(1)
