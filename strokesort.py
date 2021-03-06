import torch, os, random, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils import tensorboard as tb

from quickdraw.quickdraw import QuickDraw
from models import Embedder, ScoreFunction, SketchANet
from utils import rasterize, incr_ratserize, prerender_stroke, accept_withinfg_strokes, permuter
from utils import listofindex, subset
from npz import NPZWriter, MetricWriter

def analyse(embedder, perm, savefile, device, n_strokes):
    # create visualizations of the model prediction

    p_eye = torch.eye(n_strokes, device=device) # for input-order
    
    figtest, axtest = plt.subplots(n_strokes, 4)
    figtest.set_figheight(10)
    figtest.set_figwidth(10)

    for q, p in enumerate([p_eye, perm]): # 'perm' is the permutation from the model
        perms = []
        for i in range(1, n_strokes + 1):
            p_ = p[:i]
            perms.append( embedder.sandwitch(perm=p_) )
        all_perms = torch.cat(perms, 0)
        preds = embedder.encoder(all_perms, feature=False)
        preds = torch.softmax(preds, 1)

        for i in range(n_strokes):
            img = all_perms[i,...].squeeze().cpu().numpy()
            pred = preds[i,...].squeeze().cpu().numpy()
            axtest[i,0 if q==0 else 2].imshow(img)
            axtest[i,0 if q==0 else 2].axis('off')
            axtest[i,1 if q==0 else 3].stem(pred, use_line_collection=True)
            axtest[i,1 if q==0 else 3].axis('off')

    axtest[0,0].set_title('Original Order')
    axtest[0,1].set_title('Classif. score')
    axtest[0,2].set_title('Model output')
    axtest[0,3].set_title('Classif. score')

    figtest.savefig(savefile)
    plt.close(figtest)

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

def stochastic_neural_sort(s, tau):
    ''' The core NeuralSort algorithm '''

    def sample_gumbel(samples_shape, device, dtype=torch.float32, eps = 1e-10):
        U = torch.rand(samples_shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + eps) + eps)
    
    batch_size, n, _ = s.size()
    log_s_perturb = torch.log(s) + sample_gumbel([batch_size, n, 1], s.device, s.dtype)
    log_s_perturb = log_s_perturb.view(batch_size, n, 1)
    P_hat = deterministic_neural_sort(log_s_perturb, tau)
    P_hat = P_hat.view(batch_size, n, n)
    
    return P_hat

def greedy(stroke_list, classifier, label, fig, device):
    perm_stroke_list = []
    perm_stroke_idx_list = []
    for _ in range(len(stroke_list)):
        best_score = 0.0
        best_stroke = None
        best_stroke_idx = None
        for i_stroke, stroke in enumerate(stroke_list):
            if i_stroke not in perm_stroke_idx_list:
                R = torch.tensor(rasterize([*perm_stroke_list, stroke], fig), device=device).unsqueeze(0).unsqueeze(0)
                acc = torch.softmax(classifier(R).squeeze(), 0)
                if acc[label] > best_score:
                    best_score = acc[label]
                    best_stroke = stroke
                    best_stroke_idx = i_stroke
        perm_stroke_list.append(best_stroke)
        perm_stroke_idx_list.append(best_stroke_idx)
    return perm_stroke_list


def main( args ):
    all_classes = [ 'book', 'cat', 'chandelier', 'computer', 'cruise ship', 'face', 'flower', 'pineapple', 'sun',
                    'bicycle', 'binoculars', 'birthday cake', 'guitar', 'windmill', 'piano', 'calculator', 'cow',
                    'truck', 'butterfly', 'mosquito' ]
    
    clf_classes = subset(all_classes, args.clf_classes)
    sort_classes = subset(all_classes, args.sort_classes)
    label_map = {}
    for si, s in enumerate(sort_classes):
        label_map[si] = clf_classes.index(s)

    qd = QuickDraw(args.root, categories=sort_classes, npz=args.npz,
        max_sketches_each_cat=args.max_sketches_each_cat, verbose=True, normalize_xy=False,
        mode=QuickDraw.STROKESET, filter_func=lambda s: accept_withinfg_strokes(s, args.min_strokes, args.max_strokes))
    
    # qdtrain, qdtest = qd.split(0.98)
    qdltrain = qd.get_dataloader(args.batch_size)
    qdltest = qd.get_dataloader(1)

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))
    
    sketchclf = SketchANet(len(clf_classes))
    if os.path.exists(os.path.join(args.base, args.embmodel)):
        sketchclf.load_state_dict(torch.load(os.path.join(args.base, args.embmodel)))
    else:
        raise FileNotFoundError('args.embmodel not found')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # score function
    score = ScoreFunction(args.embdim + args.embdim)
    score = score.to(device)
    score_model_path = os.path.join(args.base, args.modelname)
    if os.path.exists(score_model_path):
        score.load_state_dict(torch.load(score_model_path))
        skip_train = True
        print('model found; skipping training')
    else:
        skip_train = False
    
    sketchclf = sketchclf.to(device)
    sketchclf.eval() # just as a guiding signal

    # loss function
    xentropy = torch.nn.CrossEntropyLoss()

    # optimizer
    optim = torch.optim.Adam(score.parameters(), lr=args.lr)
    # sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.75)
    canvas = plt.figure(frameon=False, figsize=(2.25, 2.25))

    count = 0

    for e in range(args.epochs):
        
        ####### The NPZ Writer
        if args.producenpz:
            npzwriter = NPZWriter(os.path.join(args.base, f'e{e}_' + args.npzfile))
        ####### The Metric Writer
        if args.metric:
            metricwriter = MetricWriter(os.path.join(args.base, f'e{e}_' + args.metricfile))
        
        score.train()
        for iteration, B in enumerate(qdltrain):
            if skip_train:
                break
            try:
                with torch.autograd.detect_anomaly():
                    all_preds, all_labels = [], []
                    for stroke_list, label in B:
                        random.shuffle(stroke_list) # randomize the stroke order
                        label = label_map[label] # label mapping

                        # separate stroke-count for separate samples;
                        # this is no longer provided by user
                        n_strokes = len(stroke_list)

                        raster_strokes = prerender_stroke(stroke_list, canvas)
                        if torch.cuda.is_available():
                            raster_strokes = raster_strokes.cuda()

                        embedder = Embedder(sketchclf, raster_strokes, device=device)
                        aug = embedder.get_aug_embeddings()

                        scores = score(aug)
                        
                        p_relaxed = stochastic_neural_sort(scores.unsqueeze(0), 1 / (1 + e**0.5))
                        p_discrete = torch.zeros((1, n_strokes, n_strokes), dtype=torch.float32, device=device)
                        p_discrete[torch.arange(1, device=device).view(-1, 1).repeat(1, n_strokes),
                               torch.arange(n_strokes, device=device).view(1, -1).repeat(1, 1),
                               torch.argmax(p_relaxed, dim=-1)] = 1
                        
                        # permutation matrix
                        p = p_relaxed + p_discrete.detach() - p_relaxed.detach() # ST Gradient Estimator
                        p = p.squeeze()

                        perms = []
                        for i in range(1, n_strokes + 1):
                            p_ = p[:i]
                            perms.append( embedder.sandwitch(perm=p_) )

                        all_perms = torch.cat(perms, 0)
                        preds = sketchclf(all_perms, feature=False) # as a classifier

                        all_labels.append( torch.tensor(label, device=device).repeat(n_strokes) )
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
            except:
                continue

        if not skip_train:
            torch.save(score.state_dict(), score_model_path)
            print('[Saved] {}'.format(args.modelname))
        else:
            print('[Saved] {}'.format('Skiped trained; saving not required'))

        # Evaluation time
        score.eval()
        with torch.no_grad():
            total, correct = 0, 0

            for i_batch, B in enumerate(qdltest):
                i_sample = i_batch

                stroke_list, label = B[0] # Just one sample in batch
                label = label_map[label] # label mapping

                # random.shuffle(stroke_list)

                # separate stroke-count for separate samples;
                # this is no longer provided by user
                n_strokes = len(stroke_list)

                raster_strokes = prerender_stroke(stroke_list, canvas)
                if torch.cuda.is_available():
                    raster_strokes = raster_strokes.cuda()

                embedder = Embedder(sketchclf, raster_strokes, device=device)
                
                aug = embedder.get_aug_embeddings()
                scores = score(aug)
                
                p_relaxed = stochastic_neural_sort(scores.unsqueeze(0), 1 / (1 + e**0.5))
                p_discrete = torch.zeros((1, n_strokes, n_strokes), dtype=torch.float32, device=device)
                p_discrete[torch.arange(1, device=device).view(-1, 1).repeat(1, n_strokes),
                       torch.arange(n_strokes, device=device).view(1, -1).repeat(1, 1),
                       torch.argmax(p_relaxed, dim=-1)] = 1
                
                # permutation matrix
                p = p_relaxed + p_discrete.detach() - p_relaxed.detach() # ST Gradient Estimator
                p = p.squeeze()

                perm_stroke_list = permuter(stroke_list, p.argmax(1))

                if (i_sample < args.n_viz) and args.viz:
                    savefile = os.path.join(args.base, 'logs', args.modelname + '_' + str(i_sample) + '.png')
                    analyse(embedder, p, savefile, device, n_strokes)

                # prepare for writing
                if args.producenpz:
                    npzwriter.add(perm_stroke_list)
                    if i_sample % 500 == 0:
                        npzwriter.flush()

                if args.metric and (i_sample < args.n_metric):
                    rand_stroke_list = permuter(stroke_list, np.random.permutation(n_strokes).tolist())
                    orig_stroke_list = stroke_list
                    gred_stroke_list = greedy(stroke_list, sketchclf, label, canvas, device)

                    rand_incr_rasters = incr_ratserize(rand_stroke_list, canvas)
                    orig_incr_rasters = incr_ratserize(orig_stroke_list, canvas)
                    perm_incr_rasters = incr_ratserize(perm_stroke_list, canvas)
                    gred_incr_rasters = incr_ratserize(gred_stroke_list, canvas)

                    if torch.cuda.is_available():
                        rand_incr_rasters = rand_incr_rasters.cuda()
                        orig_incr_rasters = orig_incr_rasters.cuda()
                        perm_incr_rasters = perm_incr_rasters.cuda()
                        gred_incr_rasters = gred_incr_rasters.cuda()

                    rand = torch.softmax(sketchclf(rand_incr_rasters), 1)
                    orig = torch.softmax(sketchclf(orig_incr_rasters), 1)
                    pred = torch.softmax(sketchclf(perm_incr_rasters), 1)
                    gred = torch.softmax(sketchclf(gred_incr_rasters), 1)
                    metricwriter.add(rand[:,label].unsqueeze(1).cpu().numpy(),
                                     orig[:,label].unsqueeze(1).cpu().numpy(),
                                     pred[:,label].unsqueeze(1).cpu().numpy(),
                                     gred[:,label].unsqueeze(1).cpu().numpy())
                    print(f'{i_sample}/{args.n_metric} metric written')

                    if i_sample % 50 == 0:
                        metricwriter.flush()

                    orig = (orig.argmax(1) == label).nonzero()
                    pred = (pred.argmax(1) == label).nonzero()

                    total += 1
                    if orig.numel() == 0:
                        if pred.numel() > 0:
                            correct += 1
                        else:
                            total -= 1
                    else:
                        if pred.numel() > 0:
                            if pred[0] <= orig[0]:
                                correct += 1
    
            # print efficiency
            if args.metric:
                efficiency = float(correct) / total
                print('[Efficiency] {}/{} == {}'.format(correct, total, efficiency))
                writer.add_scalar("Efficiency", efficiency, global_step=e)
                metricwriter.flush()

        if args.producenpz:
            npzwriter.flush()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=False, default='.', help='base path')
    parser.add_argument('--root', type=str, required=True, help='QuickDraw folder path (containing .bin files)')
    parser.add_argument('--npz', action='store_true', help='use .npz files (if not, .bin files)')
    parser.add_argument('--max_sketches_each_cat', '-n', type=int, required=False, default=15000, help='Max no. of sketches each category')
    parser.add_argument('--embmodel', type=str, required=True, help='Embedding model (pre-trained) file')
    parser.add_argument('--embdim', type=int, required=False, default=512, help='latent dim in the embedding model')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('-i', '--interval', type=int, required=False, default=10, help='Logging interval')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=10, help='no. of epochs')
    parser.add_argument('-f', '--max_strokes', type=int, required=False, default=10, help='max no. of strokes')
    parser.add_argument('-g', '--min_strokes', type=int, required=False, default=7, help='min no. of strokes')
    parser.add_argument('-c', '--clf_classes', type=listofindex, required=True, help='List of class indecies in the classifier')
    parser.add_argument('-s', '--sort_classes', type=listofindex, required=True, help='List of class indecies in the neuralsort')
    parser.add_argument('-m', '--modelname', type=str, required=True, help='name of the model')
    parser.add_argument('--tag', type=str, required=True, help='a tag for recognizing model in TB')
    parser.add_argument('--viz', action='store_true', help='want visualizations?')
    parser.add_argument('--n_viz', '-z', type=int, required=False, default=25, help='How many samples to visualize')
    parser.add_argument('--n_metric', type=int, required=False, default=1000, help='How many samples to use for metric calc')
    parser.add_argument('--producenpz', action='store_true', help='want to produce .npz file ?')
    parser.add_argument('--npzfile', type=str, required=False, default='output.npz', help='NPZ file name')
    parser.add_argument('--metric', action='store_true', help='compute metric (early recog.) ?')
    parser.add_argument('--metricfile', type=str, required=False, default='metric.npz', )
    args = parser.parse_args()

    main( args )
    