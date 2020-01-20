import json
import numpy as np
import matplotlib.pyplot as plt

from utils import rasterize, to_stroke_list

canvas = plt.figure(frameon=False, figsize=(2.25, 2.25))

with open('1579287746.275986.txt', 'r') as f:
    Q = json.load(f)

human = Q['human_data']
ai = Q['ai_data']

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

human = to_stroke_list(custom_to_npz(human))
ai = to_stroke_list(custom_to_npz(ai))

# plt.close(canvas)

human_r = rasterize(human, canvas)
ai_r = rasterize(ai, canvas)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(human_r, cmap='gray')
ax[1].imshow(ai_r, cmap='gray')

plt.show()
plt.close()