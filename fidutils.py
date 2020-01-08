import torch, os
import matplotlib.pyplot as plt
from utils import rasterize, to_stroke_list
from models import SketchANet

TRAINED_MODEL = './c0123456789'

# DON'T change this. The size is important
canvas = plt.figure(frameon=False, figsize=(2.25, 2.25))

# Instance of the model
sketchanet = SketchANet(num_classes=10)
if os.path.exists(TRAINED_MODEL):
    sketchanet.load_state_dict(torch.load(TRAINED_MODEL))
else:
    raise FileNotFoundError(TRAINED_MODEL + ' not found')

if torch.cuda.is_available():
    sketchanet = sketchanet.cuda()

def FID_helper_embedding(L):
    embs = []
    for sketch, _ in L:
        raster = rasterize(to_stroke_list(sketch), canvas)
        raster = torch.tensor(raster).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            raster = raster.cuda()

        embedding = sketchanet(raster, feature=True)
        embs.append(embedding)

    embs = torch.cat(embs, 0)
    return embs.detach().cpu().numpy()