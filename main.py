import torch.nn as nn
import pdb, torch, numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, TensorDataset, DataLoader

def gen_dataset(*class_mu, sample_per_class = 100, noise_var_scale = 0.1, dtype=torch.float32):
    images = np.vstack([i.flatten() + (np.random.randn(sample_per_class, class_mu[0].size) * noise_var_scale)
                        for i in class_mu])
    labels = np.hstack([np.ones(sample_per_class) * i for i in range(len(class_mu))])
    
    images, labels = torch.tensor(images, dtype=dtype), torch.tensor(labels, dtype=torch.int64)
    return TensorDataset(images, labels)

class Classifier(object):
    def __init__(self, *, input_size=None, num_classes=None, classifier=None, cuda=True):

        if classifier == None:
            # if classifier not provided, build a simple Linear layer ..
            assert input_size != None, 'If classifier not provided, then provide input and output sizes'
            assert num_classes != None, 'If classifier not provided, then provide input and output sizes'
            self.input_size = input_size   # Track the parameters
            self.num_classes = num_classes # Track the parameters

            self.classifier = nn.Sequential(nn.Linear(self.input_size, self.num_classes))
        else:
            # .. if classifier provided, use that :)
            self.classifier = classifier

        # Track the parameters
        self.cuda = cuda

        if self.cuda:
            self.classifier = self.classifier.cuda()

    def train(self, train_dl, lr=4e-2, epochs=100):
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        for e in range(epochs):
            for i, (X, Y) in enumerate(train_dl):
                if self.cuda:
                    X, Y = X.cuda(), Y.cuda()
                
                y_ = self.classifier(X)
                loss = loss_func(y_, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        
    def evaluate(self, test_dl):
        correct, total = 0, 0
        for i, (X, Y) in enumerate(test_dl):
            if self.cuda:
                X, Y = X.cuda(), Y.cuda()
            
            y_ = self.classifier(X)
            _, pred_labels = torch.max(y_, 1)
            correct += sum(pred_labels == Y)
            total += pred_labels.shape[0]
        
        print('Evaluation accuracy: {}'.format(correct / total * 100.0))

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
    
    # creating three base classes for the toy dataset
    class1_mu = np.array([[1., 1., 1.],
                          [1., 0., 1.],
                          [1., 1., 1.]])

    class2_mu = np.array([[1., 0., 1.],
                          [0., 1., 0.],
                          [1., 0., 1.]])

    class3_mu = np.array([[0., 1., 0.],
                          [1., 1., 1.],
                          [0., 1., 0.]])
    n_samples = 100
    split_prop = 0.6
    seq_len = class1_mu.size
    n_classes = 3
    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    dtype = torch.float32

    dataset = gen_dataset(class1_mu, class2_mu, class3_mu, sample_per_class=n_samples)
    n_train_samples = int(len(dataset) * split_prop)
    n_test_samples = len(dataset) - n_train_samples
    train_dataset, test_dataset = random_split(dataset, [n_train_samples, n_test_samples])

    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, pin_memory=True)

    classifier = Classifier(input_size=seq_len, num_classes=3, cuda=cuda)
    classifier.train(train_dl)
    classifier.evaluate(test_dl)

    score = nn.Sequential(
        nn.Linear(seq_len + seq_len, 1), # One sequence has only one element (rest zero),
        nn.Sigmoid()                     # the other sequence is the full sequence (context)
    )
    if cuda:
        score = score.cuda()
    
    optim = torch.optim.Adam(score.parameters(), lr=args.lr)

    mask_matrix = torch.tensor(np.tril(np.ones((seq_len, seq_len))), dtype=dtype, device=device)
    mask_tensor = torch.zeros((seq_len, seq_len, seq_len), dtype=dtype, device=device)
    for i in range(seq_len):
        mask_tensor[i,i,i] = 1

    whole_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    loss_func = nn.CrossEntropyLoss()
    loss_record = []

    for e in range(args.epochs):
        for i, (X, Y) in enumerate(whole_dl):
            if cuda:
                X, Y = X.cuda(), Y.cuda()
            
            batch_size = X.shape[0]
            # pdb.set_trace()
            x = torch.einsum('nd,dij->nij', X, mask_tensor)
            # x = torch.tensor(x.astype(np.float32))
            # pdb.set_trace()

            y = Y.unsqueeze(1).repeat(1, seq_len).view(-1)

            h = torch.cat([x, x.sum(dim=1, keepdim=True).repeat(1, seq_len, 1)], dim=-1)
            s = score(h)
            # pdb.set_trace()

            p_relaxed = stochastic_neural_sort(s, 1 / (1 + e**0.5))

            p_discrete = torch.zeros((batch_size, seq_len, seq_len), dtype=dtype, device=device)
            # pdb.set_trace()
            p_discrete[torch.arange(batch_size, device=device).view(-1, 1).repeat(1, seq_len),
                       torch.arange(seq_len, device=device).view(1, -1).repeat(batch_size, 1),
                       torch.argmax(p_relaxed, dim=-1)] = 1

            # permutation matrix
            p = p_relaxed + p_discrete.detach() - p_relaxed.detach() # ST Gradient Estimator
            
            final_mask = torch.matmul(mask_matrix.view(1, seq_len, seq_len), p)
            masked_input = torch.matmul(final_mask, x)

            out = classifier.classifier(masked_input)

            loss = loss_func(out.view(-1, n_classes), y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if e % args.interval == 0:
                print(f'[Training] [{i}/{e}/{args.epochs}] -> Loss: {loss}')
                loss_record.append(loss.item())

    plt.plot(loss_record)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, required=False, default=10000, help='no of epochs to train')
    parser.add_argument('--lr', type=float, required=False, default=1e-3, help='learning rate')
    parser.add_argument('--cuda', action='store_true', help='Want GPU support ?')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='logging interval')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=32, help='batch size')

    args = parser.parse_args()

    main( args )


# In[15]:


# # Testing
# x = images[np.arange(1,300,2)]
# x = np.einsum('nd,dij->nij', x, mask_tensor)
# x = torch.tensor(x.astype(np.float32))

# y = torch.tensor(labels[rnd_idx].astype(np.int64)).unsqueeze(1).repeat(1,9).view(-1)

# h = torch.cat([x, x.sum(dim=1, keepdim=True).repeat(1,9,1)], dim=-1)
# s = score_func(h)


# # In[16]:


# s = s.detach().cpu().numpy()

# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# ax1.matshow(base1.reshape(3,3))
# ax2.matshow(s_[0].reshape(3,3))
# ax3.matshow(s[0].reshape(3,3))


# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# ax1.matshow(base2.reshape(3,3))
# ax2.matshow(s_[50].reshape(3,3))
# ax3.matshow(s[50].reshape(3,3))

# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# ax1.matshow(base3.reshape(3,3))
# ax2.matshow(s_[100].reshape(3,3))
# ax3.matshow(s[100].reshape(3,3))

# In[ ]:




