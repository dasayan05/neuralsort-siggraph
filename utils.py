import torch
import torch, numpy as np
import matplotlib.pyplot as plt

def rasterize(stroke_list, fig, max_val=255):
    for stroke in stroke_list:
        stroke = stroke[:,:2].astype(np.int64)
        plt.plot(stroke[:,0], stroke[:,1])
    plt.gca().invert_yaxis(); plt.axis('off')
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    plt.gca().cla()
    X = X[...,:3] / float(max_val)
    X = X.mean(2)
    X[X == 1.] = 0.; X[X > 0.] = 1.
    return X.astype(np.float32)

def accept_fstrokes(s, f):
    if len(s) != f:
        return False, None
    else:
        return True, s

def accept_ltefstrokes(s, f):
    if len(s) > f:
        return False, None
    else:
        return True, s

def accept_withinfg_strokes(s, f, g):
    if (len(s) < f) or (len(s) > g):
        return False, None
    else:
        return True, s

def prerender_stroke(stroke_list, fig):
    R = []
    for stroke in stroke_list:
        stroke = [stroke,]
        R.append( torch.tensor(rasterize(stroke, fig)).unsqueeze(0) )
    return torch.stack(R, 0)