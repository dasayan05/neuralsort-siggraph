import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from quickdraw.quickdraw import QuickDraw
from models import RNNSketchClassifier, Embedder, ScoreFunction

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
    chosen_classes = [ 'cat', 'chair', 'face', 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe' ]
    qd = QuickDraw(args.root, categories=chosen_classes, max_sketches_each_cat=15000, verbose=True,
        mode=QuickDraw.STROKESET, filter_func=lambda s: accept_fstrokes(s, args.n_strokes))
    qdl = qd.get_dataloader(args.batch_size)
    
    sketchclf = RNNSketchClassifier(3, args.embdim // 2, args.emblayer, n_classes=len(chosen_classes))
    with open(args.embmodel, 'rb') as f:
        sketchclf.load_state_dict(torch.load(f))
    # sketchclf.eval() # freeze this module

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # score function
    score = ScoreFunction(args.n_strokes, args.embdim + args.embdim)
    score = score.to(device)
    sketchclf = sketchclf.to(device)

    # loss function
    xentropy = torch.nn.CrossEntropyLoss()

    # optimizer
    optim = torch.optim.Adam(score.parameters(), lr=args.lr)

    for e in range(args.epochs):
        for iteration, B in enumerate(qdl):
            all_preds, all_labels = [], []
            for stroke_list, label in B:
                
                stroke_list = [torch.tensor(stroke, device=device) for stroke in stroke_list]
                stroke_lens = torch.tensor([stroke.shape[0] for stroke in stroke_list], device=device)
                n_strokes = len(stroke_lens)

                padded_strokes = pad_sequence(stroke_list, batch_first=True)
                
                embedder = Embedder(sketchclf.sketchenc, stroke_list, device=device)
                aug = embedder.get_aug_embeddings()

                scores = score(aug)
                
                p_relaxed = stochastic_neural_sort(scores.unsqueeze(0), 1 / (1 + e**0.5))
                p_discrete = torch.zeros((1, n_strokes, n_strokes), dtype=torch.float32, device=device)
                p_discrete[torch.arange(1, device=device).view(-1, 1).repeat(1, n_strokes),
                       torch.arange(n_strokes, device=device).view(1, -1).repeat(1, 1),
                       torch.argmax(p_relaxed, dim=-1)] = 1
                
                # permutation matrix
                p = p_relaxed + p_discrete.detach() - p_relaxed.detach() # ST Gradient Estimator

                reord_strokes = torch.einsum('ab,bij->aij', p.squeeze(), padded_strokes)
                reord_lens = torch.matmul(p.squeeze(), stroke_lens.float())

                preds = []
                for i in range(1, n_strokes + 1):
                    first_i_strokes = reord_strokes[:i,...]
                    first_i_lens = reord_lens[:i]
                    first_i_sketch = []
                    total_len = torch.zeros((1,), device=device)
                    for stroke_j, len_j in zip(first_i_strokes, first_i_lens):
                        first_i_sketch.append( stroke_j[:int(len_j), ...] )
                        total_len += int(len_j)
                    partial_sketch = torch.cat(first_i_sketch, dim=0)
                    partial_sketch = pack_padded_sequence(partial_sketch.unsqueeze(0), total_len, batch_first=True, enforce_sorted=False)

                    p_ = sketchclf(partial_sketch)
                    preds.append(p_)

                preds = torch.cat(preds, dim=0)
                all_labels.append( torch.tensor(label, device=device).repeat(n_strokes) )
                all_preds.append(preds)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0).flatten()
            
            loss = xentropy(all_preds, all_labels)

            if iteration % args.interval == 0:
                print(f'[Training] [{iteration}/{e}/{args.epochs}] -> Loss: {loss}')
            
            optim.zero_grad()
            loss.backward()
            optim.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='QuickDraw folder path (containing .bin files)')
    parser.add_argument('--embmodel', type=str, required=True, help='Embedding model (pre-trained) file')
    parser.add_argument('--emblayer', type=int, required=False, default=3, help='Layers in the embedding model')
    parser.add_argument('--embdim', type=int, required=False, default=512, help='latent dim in the embedding model')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='Logging interval')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=10, help='no. of epochs')
    parser.add_argument('-f', '--n_strokes', type=int, required=False, default=9, help='pick up fixed no. of strokes')
    args = parser.parse_args()

    main( args )
    