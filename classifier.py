import torch, os, numpy as np
import matplotlib.pyplot as plt
from torch.utils import tensorboard as tb

from models import SketchANet
from quickdraw.quickdraw import QuickDraw
from utils import rasterize, accept_fstrokes

def main( args ):
    chosen_classes = [ 'cat', 'chair', 'face', 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe' ]
    chosen_classes = chosen_classes[:args.n_classes]
    qd = QuickDraw(args.root, categories=chosen_classes, max_sketches_each_cat=35000 // len(chosen_classes), verbose=True,
        normalize_xy=False, mode=QuickDraw.STROKESET, filter_func=lambda s: accept_fstrokes(s, args.n_strokes))
    # qdl = qd.get_dataloader(args.batch_size)
    qdtrain, qdtest = qd.split(0.8)
    qdltrain, qdltest = qdtrain.get_dataloader(args.batch_size), qdtest.get_dataloader(args.batch_size)

    model = SketchANet(len(chosen_classes))
    if torch.cuda.is_available():
        model = model.cuda()
    
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    # Tensorboard stuff
    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))

    fig = plt.figure(frameon=False, figsize=(2.25, 2.25))

    count = 0
    best_acc = 0
    for e in range(args.epochs):
        for i, B in enumerate(qdltrain):
            C = [c for _, c in B]
            B = [torch.tensor(rasterize(s, fig)).unsqueeze(0) for s, _ in B]

            X = torch.stack(B, 0)
            Y = torch.tensor(C)

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            optim.zero_grad()

            output = model(X)
            loss = crit(output, Y)
            
            if i % args.print_interval == 0:
                print(f'[Training] {i}/{e}/{args.epochs} -> Loss: {loss.item()}')
                writer.add_scalar('train-loss', loss.item(), count)
            
            loss.backward()
            optim.step()

            count += 1

        correct, total = 0, 0
        for i, B in enumerate(qdltest):
            C = [c for _, c in B]
            B = [torch.tensor(rasterize(s, fig)).unsqueeze(0) for s, _ in B]

            X = torch.stack(B, 0)
            Y = torch.tensor(C)

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            output = model(X)
            _, predicted = torch.max(output, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
        
        accuracy = (correct / total) * 100
        print(f'[Testing] -/{e}/{args.epochs} -> Accuracy: {accuracy} %')
        writer.add_scalar('test-accuracy', accuracy/100., e)
        if accuracy > best_acc:
            torch.save(model.state_dict(), os.path.join(args.base, args.modelname))
            best_acc = accuracy

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='base path')
    parser.add_argument('--root', type=str, required=True, help='root of quickdraw')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('-c', '--n_classes', type=int, required=False, default=10, help='Number of classes for the classification task')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=100, help='No. of epochs')
    parser.add_argument('-f', '--n_strokes', type=int, required=False, default=9, help='how many strokes')
    parser.add_argument('-m', '--modelname', type=str, required=False, help='name of model')
    parser.add_argument('--tag', type=str, required=True, help='a tag for recognizing model in TB')
    parser.add_argument('-i', '--print_interval', type=int, required=False, default=10, help='Print loss after this many iterations')
    args = parser.parse_args()

    if not hasattr(args, 'modelname'):
        args.modelname = args.tag

    main( args )