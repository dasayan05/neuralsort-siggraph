import os, random, pdb
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils import unpack_drawing, struct

class QuickDraw(Dataset):
    def __init__(self, root, categories=[], max_samples=80000, normalize_xy=True, dtype=np.float32, verbose=False,
            *, cache=None # not to be used
        ):
        super().__init__()

        # Track the parameters
        if os.path.exists(root):
            self.root = root
        
        if len(categories) == 0:
            self.categories = os.listdir(self.root)
        else:
            self.categories = [cat + '.bin' for cat in categories]
        
        self.normalize_xy = normalize_xy
        self.dtype = dtype
        self.verbose = verbose
        self.max_samples = max_samples

        # The cached data
        if cache != None:
            self.cache = cache
        else:
            self.cache = []
            for cat_idx, category in enumerate(self.categories):
                bin_file_path = os.path.join(self.root, category)
                n_samples = 0
                with open(bin_file_path, 'rb') as file:
                    while True:
                        try:
                            drawing = unpack_drawing(file)
                            self.cache.append((drawing['image'], cat_idx))
                            n_samples += 1
                            if n_samples >= max_samples:
                                break
                        except struct.error:
                            break
                if self.verbose:
                    print('[Info] {} sketches read from {}'.format(n_samples, bin_file_path))

            random.shuffle(self.cache)

        if self.verbose:
            print('[Info] Dataset with {} samples created'.format(len(self)))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i_sketch):
        qd_sketch, c_id = self.cache[i_sketch]
        n_strokes = len(qd_sketch) # sketch contains 'n_strokes' strokes
        sketch = np.empty((0, 3), dtype=self.dtype) # 2 for X-Y pair, 1 for pen state P

        for i_stroke in range(n_strokes):
            stroke = np.array(qd_sketch[i_stroke], dtype=self.dtype).T
            if self.normalize_xy:
                norm_factor = np.sqrt((stroke**2).sum(1)).max()
                stroke = stroke / (norm_factor + np.finfo(self.dtype).eps)
            # The pen states. Only the one at the end of stroke has 1, rest 0
            p = np.zeros((stroke.shape[0], 1), dtype=self.dtype); p[-1, 0] = 1.
            
            # stack up strokes to make sketch
            sketch = np.vstack((sketch, np.hstack((stroke, p))))

        return sketch, c_id

    def collate(batch):
        lengths = torch.tensor([x.shape[0] for (x, _) in batch])
        padded_seq_inp = pad_sequence([torch.tensor(x) for (x, _) in batch])
        labels = torch.tensor([c for (_, c) in batch])
        return pack_padded_sequence(padded_seq_inp, lengths, enforce_sorted=False), labels

    def get_dataloader(self, batch_size, shuffle = True, pin_memory = True):
        return DataLoader(self, batch_size=batch_size, collate_fn=QuickDraw.collate, shuffle=shuffle, pin_memory=pin_memory)

    def split(self, proportion=0.8):
        train_samples = int(len(self) * proportion)
        qd_test = QuickDraw(self.root, self.categories, self.max_samples, self.normalize_xy, self.dtype, self.verbose,
            cache=self.cache[train_samples:])
        self.cache = self.cache[:train_samples]

        if self.verbose:
            print('[Info] Dataset with {} samples created'.format(len(self)))
        
        return self, qd_test

if __name__ == '__main__':
    import sys

    qd = QuickDraw(sys.argv[1], categories=['airplane', 'bus'], max_samples=1000, verbose=True)
    qdl = qd.get_dataloader(4)
    for X, Y in qdl:
        print(Y); break