import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNSketchEncoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, dtype=torch.float32, bidirectional=True, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden = n_input, n_hidden
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout

        self.cell = nn.GRU(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional, dropout=self.dropout)

    def forward(self, x):
        # Initial hidden state
        self.h_initial = torch.zeros(self.n_layer * self.bidirectional, x.batch_sizes.max(), self.n_hidden, dtype=self.dtype)
        if torch.cuda.is_available():
            self.h_initial = self.h_initial.cuda()

        # breakpoint()
        _, h_final = self.cell(x, self.h_initial)
        h_final = h_final.view(self.n_layer, self.bidirectional, -1, self.n_hidden)
        return torch.cat((h_final[-1, 0, :], h_final[-1, 1, :]), 1)

class RNNSketchClassifier(nn.Module):
    def __init__(self, n_input, n_embedding, n_layer, n_classes, dtype=torch.float32, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_embedding = n_embedding
        self.n_classes = n_classes
        self.dtype = dtype
        self.dropout = dropout

        self.sketchenc = RNNSketchEncoder(n_input, self.n_embedding, n_layer, dtype=self.dtype, dropout=self.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_embedding * self.sketchenc.bidirectional, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes)
        )

    def forward(self, x):
        x = self.sketchenc(x)
        x = self.classifier(x)
        return x

class Embedder(object):
    def __init__(self, encoder, sketch, device):
        super().__init__()

        # Track parameters
        self.encoder = encoder
        self.sketch = sketch
        self.device = device
        self.encoder = encoder.to(self.device)
        self.n_strokes = len(sketch)

        # breakpoint()
        self.full_sketch = torch.cat(self.sketch, dim=0)
        self.sketch.append(self.full_sketch)
        self.lengths = torch.tensor([q.shape[0] for q in self.sketch], device=self.device)
        self.sketch = nn.utils.rnn.pad_sequence(self.sketch)
        self.packed_strokes = nn.utils.rnn.pack_padded_sequence(self.sketch, self.lengths, enforce_sorted=False)

        # get embeddings for all
        # self.encoder.eval()
        self.latent = self.encoder(self.packed_strokes)

        self.emb_strokes, self.emb_full = self.latent[:-1, :], self.latent[-1, :].unsqueeze(0)

        # Augmented stroke embeddings (concat'ed with full sketch embedding)
        self.aug_emb_strokes = torch.cat((self.emb_strokes, self.emb_full.repeat(self.emb_strokes.shape[0], 1)), dim=1)

    def get_aug_embeddings(self):
        return self.aug_emb_strokes

class ScoreFunction(nn.Module):
    def __init__(self, n_strokes, n_aug_emb):
        super().__init__()

        # Track parameters
        self.n_strokes = n_strokes
        self.n_aug_emb = n_aug_emb

        # weights
        self.l1 = nn.Linear(self.n_aug_emb, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.sigmoid(self.l3(x))

if __name__ == '__main__':
    import sys
    sketchclf = RNNSketchClassifier(3, 256, 3, n_classes=2)
    
    from quickdraw.quickdraw import QuickDraw
    qd = QuickDraw(sys.argv[1], categories=['airplane', 'bus'], max_sketches_each_cat=10, verbose=True, mode=QuickDraw.STROKESET)
    qdl = qd.get_dataloader(4)
    for B in qdl:
        for sketch, c in B:
            sketch = Embedder(sketchclf.sketchenc, sketch)