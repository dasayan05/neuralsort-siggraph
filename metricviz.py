import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, interp1d

def main( args ):
    fig = plt.figure()
    with open(args.metricfile, 'rb') as f:
        Q = np.load(f, allow_pickle=True, encoding='bytes')
        R, O, P = Q['rand'], Q['orig'], Q['pred']
        c = np.linspace(0., 1., num = 100)
        r_ac, o_ac, p_ac = np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)
        for i, (r, o, p) in enumerate(zip(R, O, P)):
            r_spl = interp1d(r[:,0], r[:,1])
            o_spl = interp1d(o[:,0], o[:,1])
            p_spl = interp1d(p[:,0], p[:,1])
            
            r_ac = (i * r_ac + r_spl(c)) / (i + 1)
            o_ac = (i * o_ac + o_spl(c)) / (i + 1)
            p_ac = (i * p_ac + p_spl(c)) / (i + 1)
        
        plt.plot(r_ac, c, color='r')
        plt.plot(o_ac, c, color='g')
        plt.plot(p_ac, c, color='b')
        plt.savefig(args.outfile)
    plt.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metricfile', type=str, required=True, help='The metric file')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='Output graph file')
    args = parser.parse_args()

    main( args )