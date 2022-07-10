import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
import math
import torch
import math
from typing import Iterable, Optional
import weakref
import copy
from torch.autograd import Variable
import torch

def calt(y, t):
    ys = np.argsort(y)
    ts = np.argsort(t)
    ys = torch.from_numpy(ys)
    ts = torch.from_numpy(ts)
    #print('y:\n{}\n ys:\n{}\n t:\n{}\n ts:\n{}'.format(y, ys, t, ts))
    ya = torch.zeros(y.shape[0], y.shape[1])
    ta = torch.zeros(t.shape[0], t.shape[1])
    #id1 = torch.zeros(y.shape[1])
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            ya[i][ys[i][j]] = j
            ta[i][ts[i][j]] = j
    #print(id1)
    #ya = ys[ta[0]]
    #for i in range(y.shape[0]):
        #ya[i] = ys[i][id1]
        #ta[i] = ts[i][id1]
    #ya = ya[ys]
    #ta = ta[ts]
    #print(ya)
    #print(ta)
    num = (ya - ta).abs().sum(dim=1) / y.shape[1]
    #print(num)
    return num


def fin(y, t, forget_rate, mul=1):
    a = (y - t).abs() * mul
    if mul == 1:
        a *= (1 - t)
    b = a.view(1, -1)
    num_forget = max(int(forget_rate * a.shape[0] * a.shape[1]), 1)
    #print('a:{} for:{} num_f:{}'.format(a.shape, forget_rate, num_forget))
    v, indic = torch.topk(b, num_forget)
    #print('b:{} v:{} indic:{} mul:{}'.format(b, v, indic, mul))
    return indic


def cal_loss(in1, in2, y, t, criterion, y2):
    l = criterion(y, t)
    a1 = in1.data.cpu().numpy() // y.shape[1]
    a2 = in1.data.cpu().numpy() % y.shape[1]
    #su = (t[a1, a2] == 0).data.sum()
    #l[a1, a2] *= (y2[a1, a2] - t[a1, a2]).abs()
    l[a1, a2] *= t[a1, a2] #去掉损失非常大的负标签，因为可能是missing label
    a1 = in2.data.cpu().numpy() // y.shape[1]
    a2 = in2.data.cpu().numpy() % y.shape[1]
    l[a1, a2] *= t[a1, a2] #去掉损失非常小的负标签，因为需要专注于正例
    #h = float(y < 0.1 and t < 0.1)
    #l *= float(torch.ones(l.shape[0], l.shape[1]) - float(y < 0.1 and t < 0.1)).cuda()
    #su += (t[a1, a2] == 0).data.sum()
    #if su == l.shape[0] * l.shape[1]:
        #su -= 1
    return l.mean() #/ (l.shape[0] * l.shape[1] - su)


def sel(y1, y2, t, forget_rate, forget_rate_neg, ind, criterion): #选标签
    #print('loss1:{} loss2:{}'.format(loss1, loss2))
    y1s = torch.sigmoid(y1)
    y2s = torch.sigmoid(y2)
    loss1 = cal_loss(fin(y2s, t, forget_rate).detach(), fin(t, y1s, forget_rate_neg, mul=-1).detach(), y1, t, criterion, y1s.detach())
    loss2 = cal_loss(fin(y1s, t, forget_rate).detach(), fin(t, y2s, forget_rate_neg, mul=-1).detach(), y2, t, criterion, y2s.detach())
    return loss1, loss2

def val(y1, y2, t, forget_rate, ind, criterion, t_pos, t_neg): #
    y2s = torch.sigmoid(y2)
    l = criterion(y1, t)
    mul1 = 1 - (1 - t) * (y2s < t_neg).float()
    mul2 = 1 - t * (y2s > t_pos).float()
    return (l * mul1.detach() * mul2.detach()).mean()

def both(y1, y2, t, forget_rate, ind, criterion, t_pos, t_neg):
    y1s = torch.sigmoid(y1)
    y2s = torch.sigmoid(y2)
    l1 = criterion(y1, t)
    l2 = criterion(y2, t)
    mul1 = (1 - t) * (y1s < t_neg).float()
    mul2 = t * (y1s > t_pos).float()
    mul3 = (1 - t) * (y2s < t_neg).float()
    mul4 = t * (y2s > t_pos).float()
    return (l1 * (1 - mul1.detach() * mul3.detach())).mean(), (l2 * (1 - mul1.detach() * mul3.detach())).mean()

def loss_coteaching(y_1, y_2, t, forget_rate, forget_rate_neg, ind,criterion, t_pos = 1, t_neg = 0):
    #return loss_jocor(y_1, y_2, t, forget_rate, ind, True, co_lambda=0.1)
    return sel(y_1, y_2, t, forget_rate, forget_rate_neg, ind, criterion)
    #return criterion(y_1, t).mean(), criterion(y_2, t).mean()
    #return both(y_1, y_2, t, forget_rate, ind, criterion, t_pos, t_neg)
    #return val(y_1, y_2, t, forget_rate, ind, criterion, t_pos, t_neg), val(y_2, y_1, t, forget_rate, ind, criterion, t_pos, t_neg)
    #print('')
    loss_1 = criterion(y_1, t).sum(dim=1)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    #print('loss1:{} ind1:{}'.format(loss_1.shape, ind_1_sorted.shape))
    #for i in range(loss_1.shape[1]):
        #print('i:',i, loss_1[0][i], ind_1_sorted[0][i])
    loss_1_sorted = loss_1[ind_1_sorted]
    #print('loss_1:{} ind_1:{} loss_1_sorted:{}'.format(loss_1, ind_1_sorted, loss_1_sorted))
    #print('loss1:{} ind1:{} loss1_sort:{}'.format(loss_1, ind_1_sorted, loss_1_sorted))
    loss_2 = criterion(y_2, t).sum(dim=1)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    if num_remember == 0:
        num_remember = ind_1_sorted.shape[0]
    ind_1_update = ind_1_sorted[:num_remember].cpu()
    ind_2_update = ind_2_sorted[:num_remember].cpu()
    #if len(ind_1_update) == 0:
        #ind_1_update = ind_1_sorted.cpu().numpy()
        #ind_2_update = ind_2_sorted.cpu().numpy()
        #num_remember = ind_1_update.shape[0]

    #pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]]) / float(num_remember)
    #pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]]) / float(num_remember)

    loss_1_update = criterion(y_1[ind_2_update.cuda()], t[ind_2_update.cuda()])
    loss_2_update = criterion(y_2[ind_1_update.cuda()], t[ind_1_update.cuda()])

    return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember#, pure_ratio_1, pure_ratio_2

def dif(y1, y2, t, thres = 0.75):
    di = torch.zeros(y1.shape[0])
    p1 = (torch.sigmoid(y1) > thres)
    p2 = (torch.sigmoid(y2) > thres)
    for i in range(p1.shape[0]):
        for j in range(p1.shape[1]):
            if p1[i][j] != p2[i][j] or p1[i][j] != t[i][j]:
                di[i] = 1
                break
    return di

def agree(y1, y2, t, thres = 0.1):
    di = torch.ones(y1.shape[0])
    p1 = (torch.sigmoid(y1) - t).abs().data.cpu().numpy()
    p2 = (torch.sigmoid(y2) - t).abs().data.cpu().numpy()
    for i in range(y1.shape[0]):
        if p1[i].max() < thres and p2[i].max() < thres:
            di[i] = 0
    return di

def mixup_data(x, y, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    #lam = np.random.beta(alpha, alpha)
    lam = alpha

    batch_size = x.size()[0]
    #lam = np.random.uniform(0.2, 0.8)
    #lam = Variable(torch.from_numpy(lam).view(-1,1).squeeze().cuda(), requires_grad=False)
    index = torch.randperm(batch_size).cuda()
    #print('lam:', lam)
    #print('x:', x)
    #for i in range(batch_size):
    mixed_x = lam * x + (1 - lam) * x[index, :]
    #mixed_y = lam * y + (1 - lam) * y[index, :]
    y_a, y_b = y, y[index]
    #return mixed_x, mixed_y, lam
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


    def __str__(self):
        return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples
        self.thres = 0.5

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        output = torch.sigmoid(output)
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count > 0:
            precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= self.thres else -1
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= self.thres)
            Nc[k] = np.sum(targets * (scores >= self.thres))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1
