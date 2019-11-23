import torch

from quickdraw.quickdraw import QuickDraw
from models import RNNSketchClassifier

def main( args ):
    chosen_classes = [ 'cat', 'chair', 'face', 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe' ]
    qd = QuickDraw(args.root, categories=chosen_classes, max_samples=25000, verbose=True,
                    filter_func=lambda x: (True, x))
    qdstrain, qdstest = qd.split(0.8)
    qdltrain, qdltest = qdstrain.get_dataloader(args.batch_size), qdstest.get_dataloader(args.batch_size)

    model = RNNSketchClassifier(3, args.hidden, args.layers, n_classes=len(chosen_classes), dropout=args.dropout)
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    bceloss = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    
    for e in range(args.epochs):
        model.train()
        for i, (X, Y) in enumerate(qdltrain):
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            y = model(X)

            loss = bceloss(y, Y)

            if i % args.print_interval == 0:
                print(f'[Training] [{i}/{e}/{args.epochs}] -> Loss: {loss}')

            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for i, (X, Y) in enumerate(qdltest):
                if torch.cuda.is_available():
                    X, Y = X.cuda(), Y.cuda()

                y = model(X)

                total += Y.shape[0]
                correct += (torch.argmax(y, 1) == Y).sum()

        accuracy = (correct / float(total)) * 100.0
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), args.modelfile)
            best_accuracy = accuracy
        
        print(f'[Testing] [-/{e}/{args.epochs}] -> Accuracy: {accuracy}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='QuickDraw folder path (containing .bin files)')
    parser.add_argument('-b','--batch_size', type=int, required=False, default=32, help='batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-m', '--modelfile', type=str, required=False, default='sketchclf.pth', help='model file name')
    parser.add_argument('-d', '--hidden', type=int, required=False, default=256, help='no. of hidden neurons')
    parser.add_argument('-l', '--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=500, help='no of epochs')
    parser.add_argument('--dropout', type=float, required=False, default=0.5, help='dropout probability')
    parser.add_argument('-i', '--print_interval', type=int, required=False, default=100, help='loss printing interval')
    args = parser.parse_args()

    main( args )