import argparse
from engine_co_gcn import *
from models_double_gcn import *
from coco_double_gcn import *
from util import *
import os
import sys

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.001, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume2', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--inp_name', default='data/coco/coco_glo', type=str, metavar='N')
parser.add_argument('--warmup', default=5, type=int, metavar='N')
parser.add_argument('--arg_kl', default=1, type=float, metavar='N')
parser.add_argument('--lmda', default=0.5, type=float, metavar='N')
parser.add_argument('--os', default='0,1,2,3', type=str, metavar='N')
parser.add_argument('--cos_l', default=1, type=int, metavar='N')
parser.add_argument('--lr_decay', default=0.1, type=float, metavar='N')
parser.add_argument('--gamma_neg', default=4, type=float, metavar='N')
parser.add_argument('--gamma_pos', default=0, type=float, metavar='N')
parser.add_argument('--alpha', default=1, type=float, metavar='N')
parser.add_argument('--drop', default=0.1, type=float, metavar='N')
parser.add_argument('--t_pos', default=1, type=float, metavar='N')
parser.add_argument('--t_neg', default=0, type=float, metavar='N')
parser.add_argument('--missing', default=0.9, type=float, metavar='N')
parser.add_argument('--rate', default=0.4, type=float, metavar='N')

class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        logits = F.sigmoid(logits)
        #print(logits.mean(), logits.min(), logits.max())
        loss = - self.pos_weight * target * torch.log(logits.clamp(min=self.eps)) - \
               (1 - target) * torch.log((1 - logits).clamp(min=self.eps))
        #print(loss.mean(), loss.min(), loss.max())
        '''
                if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        '''
        return loss
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= (one_sided_w.detach())

        return -loss #/ x.shape[0]



def main_coco_missing():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = True
    console = 'record_missing_coco/lr-' + str(args.lr)
    if not os.path.exists(console):
        os.makedirs(console)
    sys.stdout = open(console + '/console.txt', 'w')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.os
    inp_name = args.inp_name
    warm_up = args.warmup
    if args.cos_l == 1:
        cos_learning = True
    else:
        cos_learning = False
    arg_kl = args.arg_kl
    lmda = args.lmda
    lr_decay = args.lr_decay

    train_dataset = COCO2014_miss(args.data, phase='train', inp_name=inp_name, per=args.missing)
    val_dataset = COCO2014(args.data, phase='val', inp_name=inp_name)

    num_classes = 80

    # load model
    vgcn = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/coco/coco_adj.pkl', in_channel=300)
    sgcn = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/coco/coco_adj.pkl', in_channel=300)
    criterion = BCELosswithLogits()
    # define optimizer
    optimizerv = torch.optim.Adam(vgcn.get_config_optim(args.lr, args.lrp),
                                 lr=args.lr,
                                 # momentum=args.momentum,
                                 weight_decay=args.weight_decay
                                 )
    emav = ExponentialMovingAverage(vgcn.parameters(), args.ema)
    optimizers = torch.optim.Adam(sgcn.get_config_optim(args.lr, args.lrp),
                                  lr=args.lr,
                                  # momentum=args.momentum,
                                  weight_decay=args.weight_decay
                                  )
    emas = ExponentialMovingAverage(sgcn.parameters(), args.ema)
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes}
    state['rate_neg'] = args.t_neg
    state['rate'] = args.t_pos
    state['t_pos'] = args.t_pos
    state['t_neg'] = args.t_neg
    state['resume2'] = args.resume2
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/voc2007/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['lr_cons'] = state['lr']
    state['warm_up'] = warm_up
    state['cos_learning'] = cos_learning
    state['arg_kl'] = arg_kl
    state['lmda'] = lmda
    state['train_loss'] = []
    state['train_mAP'] = []
    state['test_loss'] = []
    state['test_mAP'] = []
    state['console'] = console
    state['lr_decay'] = lr_decay
    state['alpha'] = args.alpha
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(sgcn, criterion, train_dataset, val_dataset, optimizers, emas, vgcn, optimizerv, emav)
    sys.stdout.close()

if __name__ == '__main__':
    main_coco_missing()
