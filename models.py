import torch
import torch.nn as nn
import pdb

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

if __name__ == '__main__':
    sketchclf = RNNSketchClassifier(3, 256, 3, n_classes=2)
    print(sketchclf)