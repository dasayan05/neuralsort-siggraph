import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchANet(torch.nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        # Track parameters
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(1, 64, (15, 15), stride=3)
        self.conv2 = torch.nn.Conv2d(64, 128, (5, 5), stride=1)
        self.conv3 = torch.nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 512, (7, 7), stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(512, 512, (1, 1), stride=1, padding=0)

        self.linear = torch.nn.Linear(512, self.num_classes)

    def forward(self, x, feature=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (3, 3), stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (3, 3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, (3, 3), stride=2)
        x = F.dropout(F.relu(self.conv6(x)))
        x = F.dropout(F.relu(self.conv7(x)))
        x = x.view(-1, 512)
        
        if feature:
            return x
        else:
            return self.linear(x)

class Embedder(object):
    def __init__(self, encoder, sketch, device):
        super().__init__()

        # Track parameters
        self.encoder = encoder
        self.sketch = sketch
        self.device = device
        self.encoder = encoder.to(self.device)
        self.n_strokes = sketch.shape[0]

    def sandwitch(self, perm=None):
        if perm is None:
            perm = torch.eye(self.n_strokes).to(self.device)
        
        combined = torch.einsum('ab,bijk->aijk', perm, self.sketch)
        return torch.clamp(combined.sum(0), 0., 1.).unsqueeze(0)

    def get_aug_embeddings(self, perm=None):
        stroke_emb = self.encoder(self.sketch, feature=True)
        sketch_emb = self.encoder(self.sandwitch(), feature=True)
        
        return torch.cat((stroke_emb, sketch_emb.repeat(self.n_strokes, 1)), 1)

class ScoreFunction(nn.Module):
    def __init__(self, n_aug_emb):
        super().__init__()

        # Track parameters
        self.n_aug_emb = n_aug_emb

        # weights
        self.l1 = nn.Linear(self.n_aug_emb, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.sigmoid(self.l3(x))