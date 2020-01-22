import json, os
import numpy as np
import matplotlib.pyplot as plt

from utils import incr_ratserize, to_stroke_list

canvas = plt.figure(frameon=False, figsize=(2.25, 2.25))

def custom_to_npz(human):
    human_x, human_y, human_p = human['clickX'], human['clickY'], human['clickDrag']
    human_xy = np.array([human_x, human_y], dtype=np.float32).T
    human_xy -= human_xy[0,:] # center the data
    human_xy[1:,:] -= human_xy[:-1,:]
    human_p = 1. - np.array(human_p, dtype=np.float32)
    # some manipulation to make it compatible with QuickDraw
    human_p = np.array([*human_p, 1.])[1:][:,np.newaxis]
    human = np.hstack((human_xy[1:,:], human_p[1:,:]))
    human = human[1:,:] #.astype(np.int64)

    return human

L = os.listdir('./resources/humanstudy')

for i_sample, file in enumerate(L):
    with open(os.path.join('./resources/humanstudy', file), 'r') as f:
        Q = json.load(f)

    human = Q['human_data']
    ai = Q['ai_data']

    human = to_stroke_list(custom_to_npz(human))
    # ai = to_stroke_list(custom_to_npz(ai))

    human_r = incr_ratserize(human, canvas, coarse=6)
    # ai_r = incr_ratserize(ai, canvas)

    os.mkdir(f'resources/sample{i_sample}')
    # os.mkdir(f'resources/sample{i_sample}/human')
    # os.mkdir(f'resources/sample{i_sample}/ai')
    
    for i, b in enumerate(human_r):
        plt.imsave(os.path.join(f'resources/sample{i_sample}', f'{i}.png'), b.squeeze(), cmap='gray')
    # for i, b in enumerate(ai_r):
    #     plt.imsave(os.path.join(f'resources/sample{i_sample}/ai', f'{i}.png'), b.squeeze())