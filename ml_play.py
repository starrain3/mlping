"""
The template of the main script of the machine learning process
"""
import pickle
from os import path
import statistics 
import numpy as np
from mlgame.communication import ml as comm



def ml_loop(self):
    """
    The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    tmp = [100, 400]
    ball_served = False
    filename = path.join(path.dirname(__file__), 'save', 'clf_KMeans_BallAndDirection1.pickle')
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    # s = [93, 93]

    def get_direction(ball_x, ball_y, ball_pre_x, ball_pre_y):
        VectorX = ball_x - ball_pre_x
        VectorY = ball_y - ball_pre_y
        if (VectorX >= 0 and VectorY >= 0):
            return 0
        elif (VectorX > 0 and VectorY < 0):
            return 1
        elif (VectorX < 0 and VectorY > 0):
            return 2
        elif (VectorX < 0 and VectorY < 0):
            return 3

    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()
    

    # 3. Start an endless loop.
    while True:  #(frame_ary, Balls, BlockerPos, P1PlatformPos, commands_ary, direction, vx, vy, des)目前學習對象 BALLXY BLOCKERX  DIRECT VX VY
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.recv_from_game()
        ball_xy = []
        ball_xy.append(scene_info['ball'])
        feature = []
        feature.append(scene_info['ball'][0])
        feature.append(scene_info['ball'][1])
        feature.append(scene_info['blocker'][0])
        feature.append(get_direction(ball_xy[0][0],ball_xy[0][1],tmp[0],tmp[1]))
        vectorx = scene_info['ball'][0] - tmp[0]
        vectory = scene_info['ball'][1] - tmp[1]
        feature.append(vectorx)
        feature.append(vectory)
        # s = [feature[0], feature[1]]
        feature = np.array(feature)
        feature = feature.reshape((-1,6))
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info['status'] == 'GAME_1P_WIN' or scene_info['status'] == 'GAME_2P_WIN' or scene_info['status'] == 'GAME_DRAW':
            # Do some stuff if needed
            ball_served = False

            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            y = clf.predict(feature)
            x = scene_info['platform_1P'][0]
            if(y>470):
                y = 480
            if (y-x)>20:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                print('RIGHT')
            
            elif (y-x)<20:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
                print('LEFT')
            else:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
                print('NONE')
            # if y == 0:
            #     comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            #     print('NONE')
            # elif y == 1:
            #     comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
            #     print('LEFT')
            # elif y == 2:
            #     comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            #     print('RIGHT')
        tmp = scene_info['ball']
