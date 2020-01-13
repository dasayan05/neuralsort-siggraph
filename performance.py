import torch
import os, pickle, numpy as np
import matplotlib.pyplot as plt

from utils import rasterize, incr_ratserize, permuter
from models import SketchANet
from npz import MetricWriter

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

    # Need some hacks because the generated .pickle is not compatible with original .npz
    sketch[-1, -1] = 1 # Hacky stuff, DONT use elsewhere
    stroke_list = np.split(sketch, np.where(sketch[:,2])[0] + 1, axis=0)[:-2] # Also hacky
    return stroke_list

def main( args ):
    with open(args.reordered, 'rb') as f:
        reordered = pickle.load(f)
    with open(args.original, 'rb') as f:
        original = pickle.load(f)

    sketchclf = SketchANet(args.num_classes)
    if os.path.exists(args.classifier):
        sketchclf.load_state_dict(torch.load(args.classifier))
    else:
        raise FileNotFoundError('args.classifier not found')
    if torch.cuda.is_available():
        sketchclf = sketchclf.cuda()
    sketchclf.eval()

    canvas = plt.figure(frameon=False, figsize=(2.25, 2.25))
    metricwriter = MetricWriter(os.path.basename(args.classifier) + f's{args.cid}' + '_p.npz')

    total, correct = 0, 0

    with torch.no_grad():
        for i, ((orig_sk, _), (perm_sk, _)) in enumerate(zip(original, reordered)):
            try:
                print(f'Sample processed: {i}')
                orig_stroke_list, perm_stroke_list = to_stroke_list(orig_sk), to_stroke_list(perm_sk)
                # just for the sake of API; rand not needed
                rand_stroke_list = permuter(orig_stroke_list, np.random.permutation(len(orig_stroke_list)).tolist())

                rand_incr_rasters = incr_ratserize(rand_stroke_list, canvas)
                orig_incr_rasters = incr_ratserize(orig_stroke_list, canvas)
                perm_incr_rasters = incr_ratserize(perm_stroke_list, canvas)
                
                if torch.cuda.is_available():
                    rand_incr_rasters = rand_incr_rasters.cuda()
                    orig_incr_rasters = orig_incr_rasters.cuda()
                    perm_incr_rasters = perm_incr_rasters.cuda()

                rand = torch.softmax(sketchclf(rand_incr_rasters), 1)
                orig = torch.softmax(sketchclf(orig_incr_rasters), 1)
                pred = torch.softmax(sketchclf(perm_incr_rasters), 1)
                metricwriter.add(rand[:,args.cid].unsqueeze(1).cpu().numpy(),
                                 orig[:,args.cid].unsqueeze(1).cpu().numpy(),
                                 pred[:,args.cid].unsqueeze(1).cpu().numpy())

                if i % 50 == 0:
                    metricwriter.flush()

                orig = (orig.argmax(1) == args.cid).nonzero()
                pred = (pred.argmax(1) == args.cid).nonzero()

                total += 1
                if orig.numel() == 0:
                    if pred.numel() > 0:
                        correct += 1
                    else:
                        total -= 1
                else:
                    if pred.numel() > 0:
                        if pred[0] < orig[0]:
                            correct += 1

            except RuntimeError:
                continue
            except KeyboardInterrupt:
                break

        efficiency = float(correct) / total
        print('[Efficiency] {}/{} == {}'.format(correct, total, efficiency))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--reordered', type=str, required=True, help='.pickle for reordered')
    parser.add_argument('--original', type=str, required=True, help='.pickle for original')
    parser.add_argument('-m', '--classifier', type=str, required=True, help='saved model of classifier')
    parser.add_argument('-c', '--num_classes', type=int, required=False, default=10, help='number of classes')
    parser.add_argument('--cid', type=int, required=True, help='index of the class wrt to the given classifier')
    args = parser.parse_args()

    main( args )