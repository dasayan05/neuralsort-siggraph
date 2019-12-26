import torch
import torch, numpy as np
import matplotlib.pyplot as plt

def rasterize(stroke_list, fig, xlim=[0,255], ylim=[0,255]):
    for stroke in stroke_list:
        stroke = stroke[:,:2].astype(np.int64)
        plt.plot(stroke[:,0], stroke[:,1])
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.gca().invert_yaxis(); plt.axis('off')
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    plt.gca().cla()
    X = X[...,:3] / 255.
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

def prerender_stroke(stroke_list, fig, xlim=[0,255], ylim=[0,255]):
    R = []
    for stroke in stroke_list:
        stroke = [stroke,]
        R.append( torch.tensor(rasterize(stroke, fig, xlim, ylim)).unsqueeze(0) )
    # breakpoint()
    return torch.stack(R, 0)

def incr_ratserize(stroke_list, fig, xlim=[0,255], ylim=[0,255], coarse=2):
    R = []
    incomplete_sketch = []
    for stroke in stroke_list:
        incomplete_sketch.append( np.empty((0, 3)) )
        try:
            for pixels in np.array_split(stroke, stroke.shape[0] // coarse, 0):
                incomplete_sketch[-1] = np.vstack((incomplete_sketch[-1], pixels))
                R.append( torch.tensor(rasterize(incomplete_sketch, fig, xlim, ylim)).unsqueeze(0) )
        except ValueError as verr:
            incomplete_sketch[-1] = stroke
    
    return torch.stack(R, 0)

def permuter(L, t):
    return [L[i] for i in t]