import torch
import torch, numpy as np
import matplotlib.pyplot as plt

def to_stroke_list(sketch):
    ## sketch: an `.npz` style sketch from QuickDraw
    sketch = np.vstack((np.array([0, 0, 0]), sketch))
    sketch[:,:2] = np.cumsum(sketch[:,:2], axis=0)

    # range normalization
    xmin, xmax = sketch[:,0].min(), sketch[:,0].max()
    ymin, ymax = sketch[:,1].min(), sketch[:,1].max()

    sketch[:,0] = ((sketch[:,0] - xmin) / float(xmax - xmin)) * 255.
    sketch[:,1] = ((sketch[:,1] - ymin) / float(ymax - ymin)) * 255.
    sketch = sketch.astype(np.int64)

    stroke_list = np.split(sketch[:,:2], np.where(sketch[:,2])[0] + 1, axis=0)[:-1]
    return stroke_list

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

def prerender_group(groups, fig, xlim=[0,255], ylim=[0,255]):
    R = []
    for stroke in groups:
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

def listofindex(l):
    l = l.split(',') # must be comma separated string
    return [int(q) for q in l]

def subset(l, inds):
    ll = []
    for i in inds:
        ll.append(l[i])
    return ll

def stroke_grouping(stroke_list, num_groups=5):
    n_strokes = len(stroke_list)
    if n_strokes <= num_groups:
        n_stroke_per_group = 1
    else:
        if n_strokes % num_groups == 0:
            n_stroke_per_group = n_strokes // num_groups
        else:
            n_stroke_per_group = (n_strokes // num_groups) + 1
    groups = []
    i = 0
    for g in range(num_groups):
        if stroke_list[i:i+n_stroke_per_group].__len__() != 0:
            groups.append( stroke_list[i:i+n_stroke_per_group] )
            i += n_stroke_per_group
    if i <= n_strokes - 1:
        groups[-1].extend(stroke_list[i:])
    return groups

binary_xor = lambda a, b: (a.type(torch.uint8) ^ b.type(torch.uint8)).type(torch.float32)

def render_perm(rG, perm):
    R = [rG[perm[0]],]
    for p in perm[1:]:
        R.append(binary_xor(rG[p], R[-1]))
    R = torch.stack(R, 0)
    return R