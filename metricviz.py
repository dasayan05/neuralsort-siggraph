import numpy as np
import matplotlib.pyplot as plt

def main( args ):
    fig = plt.figure()
    with open(args.metricfile, 'rb') as f:
        Q = np.load(f, allow_pickle=True, encoding='bytes')
        R, O, P = Q['rand'], Q['orig'], Q['pred']
        for r, o, p in zip(R, O, P):
            plt.plot(r[:,0], r[:,1], color='r')
            plt.plot(o[:,0], o[:,1], color='g')
            plt.plot(p[:,0], p[:,1], color='b')
        plt.savefig(args.outfile)
    plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metricfile', type=str, required=True, help='The metric file')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='Output graph file')
    args = parser.parse_args()

    main( args )