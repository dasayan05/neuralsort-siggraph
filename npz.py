import numpy as np
from scipy.special import softmax

class NPZWriter(object):
    def __init__(self, filename):
        super().__init__()

        # Track parameters
        self.filename = filename
        # The whole structure
        self.Q_train, self.Q_test, self.Q_valid = [], [], []

    def add(self, stroke_list):
        sketch = np.concatenate(stroke_list, 0).astype(np.int16)
        # sketch = np.vstack((np.array([0, 0, 0]), sketch))
        sketch[:,:2] = sketch[:,:2] - sketch[0,:2]
        sketch[1:,:2] -= sketch[:-1,:2]
        sketch = sketch[1:,:]
        
        u = np.random.rand()

        if u >= 0. and u < 0.9:
            self.Q_train.append(sketch)
        elif u >= 0.9 and u < 0.95:
            self.Q_test.append(sketch)
        else:
            self.Q_valid.append(sketch)
        
    def flush(self):
        Q_train = np.array(self.Q_train, dtype=np.object)
        Q_test = np.array(self.Q_test, dtype=np.object)
        Q_valid = np.array(self.Q_valid, dtype=np.object)

        with open(self.filename, 'wb') as f:
            np.savez(f, train=Q_train, test=Q_test, valid=Q_valid)

class MetricWriter(object):
    def __init__(self, filename):
        super().__init__()

        # Track parameters
        self.filename = filename
        # The whole structure
        self.R, self.O, self.P, self.G = [], [], [], []

    def add(self, r, o, p, g):
        r_x = np.linspace(0., 1., num=r.shape[0]); r_x = r_x[:,np.newaxis]; r_y = r
        o_x = np.linspace(0., 1., num=o.shape[0]); o_x = o_x[:,np.newaxis]; o_y = o
        p_x = np.linspace(0., 1., num=p.shape[0]); p_x = p_x[:,np.newaxis]; p_y = p
        g_x = np.linspace(0., 1., num=g.shape[0]); g_x = g_x[:,np.newaxis]; g_y = g

        r = np.hstack((r_x, r_y))
        o = np.hstack((o_x, o_y))
        p = np.hstack((p_x, p_y))
        g = np.hstack((g_x, g_y))

        self.R.append(r)
        self.O.append(o)
        self.P.append(p)
        self.G.append(g)
    
    def flush(self):
        R = np.array(self.R, dtype=np.object)
        O = np.array(self.O, dtype=np.object)
        P = np.array(self.P, dtype=np.object)
        G = np.array(self.G, dtype=np.object)

        with open(self.filename, 'wb') as f:
            np.savez(f, rand=R, orig=O, pred=P, gred=G)