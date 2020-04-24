import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_image_data_format('channels_first')

import torch, os, numpy as np
import matplotlib.pyplot as plt
from torch.utils import tensorboard as tb

from tf_models import SketchANet
from quickdraw.quickdraw import QuickDraw
from utils import rasterize, accept_withinfg_strokes
from utils import listofindex, subset

def main( args ):
    all_classes = [ 'book', 'cat', 'chandelier', 'computer', 'cruise ship', 'face', 'flower', 'pineapple', 'sun',
                    'bicycle', 'binoculars', 'birthday cake', 'guitar', 'windmill', 'piano', 'calculator', 'cow',
                    'truck', 'butterfly', 'mosquito' ]
    
    chosen_classes = subset(all_classes, args.clf_classes)
    qd = QuickDraw(args.root, categories=chosen_classes, max_sketches_each_cat=args.max_sketches_each_cat, verbose=True,
        normalize_xy=False, mode=QuickDraw.STROKESET, filter_func=lambda s: accept_withinfg_strokes(s, args.min_strokes, args.max_strokes), npz=args.npz)
    # qdl = qd.get_dataloader(args.batch_size)
    qdtrain, qdtest = qd.split(0.8)
    qdltrain, qdltest = qdtrain.get_dataloader(args.batch_size), qdtest.get_dataloader(args.batch_size)

    model = SketchANet(len(chosen_classes))
    # sched = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.8)

    # Tensorboard stuff
    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))

    fig = plt.figure(frameon=False, figsize=(2.25, 2.25))

    count = 0
    best_acc = 0
    for e in range(args.epochs):
        for i, B in enumerate(qdltrain):
            C = [c for _, c in B]
            B = [rasterize(s, fig)[np.newaxis, ...] for s, _ in B]

            X = np.stack(B, 0)
            Y = keras.utils.to_categorical(C, len(chosen_classes))

            H = model.fit(X, Y, batch_size=args.batch_size, epochs=1, verbose=0)
            loss = H.history['loss'][0]

            if i % args.print_interval == 0:
                print(f'[Training] {i}/{e}/{args.epochs} -> Loss: {loss}')
                writer.add_scalar('train-loss', loss.item(), count)

            count += 1

        accuracy = 0.0
        for i, B in enumerate(qdltest):
            C = [c for _, c in B]
            B = [rasterize(s, fig)[np.newaxis, ...] for s, _ in B]

            X = np.stack(B, 0)
            Y = keras.utils.to_categorical(C, len(chosen_classes))

            _, acc = model.evaluate(X, Y, batch_size=args.batch_size, verbose=0)
            accuracy = ((accuracy * i) + acc) / (i + 1)
        
        # sched.step() # invoke LR scheduler

        accuracy *= 100.0
        print(f'[Testing] -/{e}/{args.epochs} -> Accuracy: {accuracy} %')
        writer.add_scalar('test-accuracy', accuracy/100., e)
        if accuracy > best_acc:
            model.save(os.path.join(args.base, f'e{e}_' + args.modelname + '.h5'), save_format='h5', include_optimizer=False)
            best_acc = accuracy

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='base path')
    parser.add_argument('--root', type=str, required=True, help='root of quickdraw')
    parser.add_argument('--npz', action='store_true', help='use .npz files (if not, .bin files)')
    parser.add_argument('--max_sketches_each_cat', '-n', type=int, required=False, default=15000, help='Max no. of sketches each category')
    parser.add_argument('-c', '--clf_classes', type=listofindex, required=True, help='List of class indecies')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=100, help='No. of epochs')
    parser.add_argument('-f', '--max_strokes', type=int, required=False, default=10, help='max no. of strokes')
    parser.add_argument('-g', '--min_strokes', type=int, required=False, default=7, help='min no. of strokes')
    parser.add_argument('-m', '--modelname', type=str, required=True, help='name of model')
    parser.add_argument('--tag', type=str, required=True, help='a tag for recognizing model in TB')
    parser.add_argument('-i', '--print_interval', type=int, required=False, default=10, help='Print loss after this many iterations')
    args = parser.parse_args()

    main( args )