import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, interp1d
from sklearn.metrics import auc

INTERP = 'cubic'

def main( args ):
    fig = plt.figure()
    with open(args.metricfile, 'rb') as f:
        Q = np.load(f, allow_pickle=True, encoding='bytes')
        if args.greedy:
            T = Q['rand'], Q['orig'], Q['pred'], Q['gred']
        else:
            T = Q['rand'], Q['orig'], Q['pred'], [None]*len(Q['orig'])
        
        c = np.linspace(0., 1., num = 100)
        
        if args.greedy:
            r_ac, o_ac, p_ac, g_ac = np.zeros_like(c), np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)
        else:
            r_ac, o_ac, p_ac = np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)

        for i, (r, o, p, g) in enumerate(zip(*T)):
            try:
                r_spl = interp1d(r[:,0], r[:,1], kind=INTERP)
                r_ac = (i * r_ac + r_spl(c)) / (i + 1)
            except:
                pass

            try:
                o_spl = interp1d(o[:,0], o[:,1], kind=INTERP)
                o_ac = (i * o_ac + o_spl(c)) / (i + 1)
            except:
                pass
            
            try:
                p_spl = interp1d(p[:,0], p[:,1], kind=INTERP)
                p_ac = (i * p_ac + p_spl(c)) / (i + 1)
            except:
                pass

            if args.greedy:
                try:
                    g_spl = interp1d(g[:,0], g[:,1], kind=INTERP)
                    g_ac = (i * g_ac + g_spl(c)) / (i + 1)
                except:
                    pass
        
        if not args.ignorerandom:
            plt.plot(c, r_ac, color='r')
            print('Random order AUC:', auc(c, r_ac))
        plt.plot(c, o_ac, color='g')
        print('Original order AUC:', auc(c, o_ac))
        plt.plot(c, p_ac, color='b')
        print('NeuralSort order AUC:', auc(c, p_ac))

        if args.greedy:
            plt.plot(c, g_ac, color='m')
            print('Greedy order AUC:', auc(c, g_ac))
        
        if not args.ignorerandom:
            if args.greedy:
                plt.legend(['random order', 'human order', 'model order', 'greedy order'])
            else:
                plt.legend(['random order', 'human order', 'model order'])
        else:
            if args.greedy:
                plt.legend(['human order', 'model order', 'greedy order'])
            else:
                plt.legend(['human order', 'model order'])
        
        plt.xlabel('sketch completion percentage')
        plt.ylabel('class recognition accuracy')
        if args.category != '':
            plt.title(f'class: {args.category}')
        plt.savefig(args.outfile)
    plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metricfile', type=str, required=True, help='The metric file')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='Output graph file')
    parser.add_argument('-c', '--category', type=str, required=False, default='', help='name of category')
    parser.add_argument('--ignorerandom', action='store_true', help='Ignore the random curve')
    parser.add_argument('--greedy', action='store_true', help='Greed strategy included')
    args = parser.parse_args()

    main( args )