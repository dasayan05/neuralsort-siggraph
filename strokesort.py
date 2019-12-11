import torch, os, random, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils import tensorboard as tb

from quickdraw.quickdraw import QuickDraw
from models import Embedder, ScoreFunction
from sketchanet import SketchANet
from train_sketchanet import rasterize

def prerender_stroke(stroke_list, fig):
    R = []
    for stroke in stroke_list:
        stroke = [stroke,]
        R.append( torch.tensor(rasterize(stroke, fig)).unsqueeze(0) )
    return torch.stack(R, 0)

def accept_fstrokes(s, f):
    if len(s) != f:
        return False, None
    else:
        return True, s

def stochastic_neural_sort(s, tau):
    ''' The core NeuralSort algorithm '''
    
    def deterministic_neural_sort(s, tau):
        device = s.device # Detect the device type of the score 's'
        
        n = s.size()[1]
        one = torch.ones((n, 1), device=device)
        A_s = torch.abs(s - s.permute(0, 2, 1))
        B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (n + 1 - 2 * (torch.arange(n, dtype=s.dtype, device=device) + 1))
        C = torch.matmul(s, scaling.unsqueeze(0))
        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / tau)
        
        return P_hat

    def sample_gumbel(samples_shape, device, dtype=torch.float32, eps = 1e-10):
        U = torch.rand(samples_shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + eps) + eps)
    
    batch_size, n, _ = s.size()
    log_s_perturb = torch.log(s) + sample_gumbel([batch_size, n, 1], s.device, s.dtype)
    log_s_perturb = log_s_perturb.view(batch_size, n, 1)
    P_hat = deterministic_neural_sort(log_s_perturb, tau)
    P_hat = P_hat.view(batch_size, n, n)
    
    return P_hat

def main( args ):
    chosen_classes = [ 'cat', 'chair', 'face'] #, 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe' ]
    qd = QuickDraw(args.root, categories=chosen_classes, max_sketches_each_cat=10000, verbose=True, normalize_xy=False,
        mode=QuickDraw.STROKESET, filter_func=lambda s: accept_fstrokes(s, args.n_strokes))
    qdtrain, qdtest = qd.split(0.98)
    qdltrain = qdtrain.get_dataloader(args.batch_size)
    qdltest = qdtest.get_dataloader(1)

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs'))
    
    sketchclf = SketchANet(len(chosen_classes))
    if os.path.exists(os.path.join(args.base, args.embmodel)):
        sketchclf.load_state_dict(torch.load(os.path.join(args.base, args.embmodel)))
    else:
        raise 'args.embmodel not found'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # score function
    score = ScoreFunction(args.n_strokes, args.embdim + args.embdim)
    score = score.to(device)
    sketchclf = sketchclf.to(device)
    sketchclf.eval()

    # loss function
    xentropy = torch.nn.CrossEntropyLoss()

    # optimizer
    optim = torch.optim.Adam(score.parameters(), lr=args.lr)
    fig = plt.figure(frameon=False, figsize=(2.25, 2.25))

    count = 0
    stemfig, stemax = plt.subplots(1, 2)

    for e in range(args.epochs):
        score.train()
        for iteration, B in enumerate(qdltrain):
            all_preds, all_labels = [], []
            for stroke_list, label in B:
                random.shuffle(stroke_list) # randomize the stroke order
                raster_strokes = prerender_stroke(stroke_list, fig)
                if torch.cuda.is_available():
                    raster_strokes = raster_strokes.cuda()

                embedder = Embedder(sketchclf, raster_strokes, device=device)
                aug = embedder.get_aug_embeddings()

                scores = score(aug)
                
                p_relaxed = stochastic_neural_sort(scores.unsqueeze(0), 1 / (1 + e**0.5))
                p_discrete = torch.zeros((1, args.n_strokes, args.n_strokes), dtype=torch.float32, device=device)
                p_discrete[torch.arange(1, device=device).view(-1, 1).repeat(1, args.n_strokes),
                       torch.arange(args.n_strokes, device=device).view(1, -1).repeat(1, 1),
                       torch.argmax(p_relaxed, dim=-1)] = 1
                
                # permutation matrix
                p = p_relaxed + p_discrete.detach() - p_relaxed.detach() # ST Gradient Estimator
                p = p.squeeze()

                perms = []
                for i in range(1, args.n_strokes + 1):
                    p_ = p[:i]
                    perms.append( embedder.sandwitch(perm=p_) )

                all_perms = torch.cat(perms, 0)
                preds = sketchclf(all_perms, feature=False) # as a classifier

                all_labels.append( torch.tensor(label, device=device).repeat(args.n_strokes) )
                all_preds.append(preds)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0).flatten()
            
            loss = xentropy(all_preds, all_labels)

            if iteration % args.interval == 0:
                print(f'[Training] [{iteration}/{e}/{args.epochs}] -> Loss: {loss}')
                writer.add_scalar('Train loss', loss.item(), count)
                count += 1
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        torch.save(score.state_dict(), os.path.join(args.base, 'model.pth'))

        # Testing
        score.eval()
        with torch.no_grad():
            for i_batch, B in enumerate(qdltest):
                i_sample = i_batch

                stroke_list, label = B[0]
                random.shuffle(stroke_list)

                raster_strokes = prerender_stroke(stroke_list, fig)
                if torch.cuda.is_available():
                    raster_strokes = raster_strokes.cuda()

                embedder = Embedder(sketchclf, raster_strokes, device=device)

                # classification score for randomized permutations
                p = torch.eye(args.n_strokes, device=device)
                perms = []
                for i in range(1, args.n_strokes + 1):
                    p_ = p[:i]
                    perms.append( embedder.sandwitch(perm=p_) )
                all_perms = torch.cat(perms, 0)
                preds = sketchclf(all_perms, feature=False)
                preds = torch.softmax(preds, 1)
                cls_score = preds[:,label].squeeze().cpu().numpy()
                stemax[0].stem(cls_score, use_line_collection=True)

                aug = embedder.get_aug_embeddings()
                scores = score(aug)
                
                p_relaxed = stochastic_neural_sort(scores.unsqueeze(0), 1 / (1 + e**0.5))
                p_discrete = torch.zeros((1, args.n_strokes, args.n_strokes), dtype=torch.float32, device=device)
                p_discrete[torch.arange(1, device=device).view(-1, 1).repeat(1, args.n_strokes),
                       torch.arange(args.n_strokes, device=device).view(1, -1).repeat(1, 1),
                       torch.argmax(p_relaxed, dim=-1)] = 1
                
                # permutation matrix
                p = p_relaxed + p_discrete.detach() - p_relaxed.detach() # ST Gradient Estimator
                p = p.squeeze()

                perms = []
                for i in range(1, args.n_strokes + 1):
                    p_ = p[:i]
                    perms.append( embedder.sandwitch(perm=p_) )

                all_perms = torch.cat(perms, 0)
                preds = sketchclf(all_perms, feature=False) # as a classifier
                preds = torch.softmax(preds, 1)
                cls_score = preds[:,label].squeeze().cpu().numpy()
                stemax[1].stem(cls_score, use_line_collection=True)

                plt.savefig( os.path.join(args.base, 'logs', str(i_sample)+'.png') )
                stemax[0].clear()
                stemax[1].clear()
                if i_sample > 10:
                    break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='base path')
    parser.add_argument('--root', type=str, required=True, help='QuickDraw folder path (containing .bin files)')
    parser.add_argument('--embmodel', type=str, required=True, help='Embedding model (pre-trained) file')
    parser.add_argument('--embdim', type=int, required=False, default=512, help='latent dim in the embedding model')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='Logging interval')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=10, help='no. of epochs')
    parser.add_argument('-f', '--n_strokes', type=int, required=False, default=9, help='pick up fixed no. of strokes')
    args = parser.parse_args()

    main( args )
    