import numpy as np

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

        if u >= 0. and u < 0.8:
            self.Q_train.append(sketch)
        elif u >= 0.8 and u < 0.9:
            self.Q_test.append(sketch)
        else:
            self.Q_valid.append(sketch)
        
    def flush(self):
        Q_train = np.array(self.Q_train, dtype=np.object)
        Q_test = np.array(self.Q_test, dtype=np.object)
        Q_valid = np.array(self.Q_valid, dtype=np.object)

        with open(self.filename, 'wb') as f:
            np.savez(f, train=Q_train, test=Q_test, valid=Q_valid)