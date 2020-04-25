import os, itertools
import torch, numpy as np
import matplotlib.pyplot as plt

from npz import NPZWriter
from models import SketchANet
from quickdraw.quickdraw import QuickDraw
from utils import stroke_grouping, prerender_group, render_perm

all_classes = [ 'book', 'cat', 'chandelier', 'computer', 'cruise ship', 'face', 'flower', 'pineapple', 'sun',
                'bicycle', 'binoculars', 'birthday cake', 'guitar', 'windmill', 'piano', 'calculator', 'cow',
                'truck', 'butterfly', 'mosquito' ]

def get_perm_sketch(groups, perm):
    groups = [groups[p] for p in perm] # permuted groups
    stroke_list = []
    for g in groups:
        stroke_list.extend(g)
    return stroke_list

def main( args ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chosen_classes = [all_classes[args.category],] # just one class at a time

    qd = QuickDraw(args.root, categories=chosen_classes, max_sketches_each_cat=args.max_sketches_each_cat, verbose=True, npz=True,
        normalize_xy=False, mode=QuickDraw.STROKESET)
    qdltrain = qd.get_dataloader(1)

    model = SketchANet(len(all_classes))
    model = model.to(device)
    embmodel_path = os.path.join(args.base, args.embmodel)
    if os.path.exists(embmodel_path):
        model.load_state_dict(torch.load(embmodel_path))
    else:
        raise FileNotFoundError('args.embmodel not found')
    model.eval()

    fig = plt.figure(frameon=False, figsize=(2.25, 2.25)) # the canvas

    npzwriter = NPZWriter(os.path.join(args.base, all_classes[args.category] + '_exh.npz'))

    for i, B in enumerate(qdltrain):
        S, _ = B[0] # just one sample
        grouped_S = stroke_grouping(S, num_groups=args.group)
        rendered_G = prerender_group(grouped_S, fig).to(device)

        max_efe, best_perm = 0., None

        N = rendered_G.shape[0] # mostly 'args.group, sometimes 'args.group - 1'
        for ip, perm in enumerate(itertools.permutations(list(range(N)))):
            permed_G = render_perm(rendered_G, perm)
            with torch.no_grad():
                preds = torch.softmax(model(permed_G), 1)
            efe = preds[:,args.category].detach().cpu().numpy().mean() # Early recognition efficacy
            if efe > max_efe:
                max_efe = efe; best_perm = perm
        
        perm_stroke_list = get_perm_sketch(grouped_S, best_perm)
        npzwriter.add(perm_stroke_list)
        
        if i % args.save_interval == 0:
            npzwriter.flush()

    npzwriter.flush()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='Base path')
    parser.add_argument('--root', type=str, required=True, help='Root folder of QuickDraw (.npz)')
    parser.add_argument('-c', '--category', type=int, choices=list(range(len(all_classes))), help='Index of the category ?')
    parser.add_argument('--max_sketches_each_cat', '-n', type=int, required=False, default=50000, help='Max no. of sketches each category')
    parser.add_argument('-g', '--group', type=int, required=False, default=5, help='Number of groups')
    parser.add_argument('--embmodel', type=str, required=True, help='Embedding model (pre-trained) file')
    parser.add_argument('--save_interval', type=int, required=False, default=100, help='Save the .npz after this many samples')
    args = parser.parse_args()
    
    main( args )