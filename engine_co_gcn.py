import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from util import *
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
import math
import matplotlib.pyplot as plt
import types
import math
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps
tta_epoch = 300
num_gradual = 10
num_be = 0

sns.set()
tqdm.monitor_interval = 0
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []
        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

        self.test_acc = []
        self.test_cf1 = []
        self.test_of1 = []

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['meter_loss'].add(self.state['loss_batch'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True, ema=None):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            self.state['norm'] = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.ToTensor(),
                #normalize,
            ])
        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                #normalize,
            ])

        self.state['best_score'] = 0

    #def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None, ema = None):
    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None, ema=None, model2 = None, optimizer2 = None, ema2 = None):
        self.init_learning(model, criterion)
        self.state['print_dis'] = False
        self.rate_schedule = self.gen_forget_rate(self.state['rate'])
        self.rate_schedule_neg = self.gen_forget_rate(self.state['rate_neg'])
        #ema.register()
        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,drop_last=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,drop_last=True,
                                                 num_workers=self.state['workers'])
        #print('train:{} test:{}'.format(len(train_loader), len(val_loader)))
        #x = input()
        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                self.state['epoch'] = self.state['start_epoch']

                pretrained_dict = checkpoint['state_dict']
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model.state_dict()}
                model.load_state_dict(pretrained_dict)
                checkpoint2 = torch.load(self.state['resume2'])
                self.state['start_epoch'] = checkpoint2['epoch']
                self.state['best_score'] = checkpoint2['best_score']

                pretrained_dict2 = checkpoint2['state_dict']
                pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in model2.state_dict()}
                model2.load_state_dict(pretrained_dict2)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True


            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            model2 = torch.nn.DataParallel(model2, device_ids=self.state['device_ids']).cuda()


            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.state['epoch'] = 0
            self.state['batch_id'] = 0
            return

        # TODO define optimizer
        #scheduler = OneCycleLR(optimizer, max_lr=self.state['lr'], steps_per_epoch=len(train_loader), epochs=self.state['max_epochs'] - self.state['start_epoch'], pct_start=0.2)
        #scheduler = 1

        #print('learn3:')
        #for param_group in optimizer.param_groups:
           #print(param_group['lr'])
        self.state['print_dis'] = False

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            if epoch == 400:
                break
            self.state['batch_id'] = 0
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)
            if epoch == self.state['max_epochs'] - 1:
                break

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch, ema, model2 = model2, optimizer2=optimizer2, ema2=ema2)
            # evaluate on validation set
            if not self.test():
                continue

            prec1 = self.validate(val_loader, model, criterion, ema, 0)
            if self.state['epoch'] % 1 == 1:
                prec2 = self.validate(val_loader, model2, criterion, ema2, 1)
            else:
                prec2 = 0
            print('epoch:{} prec1:{} prec2:{}'.format(epoch, prec1, prec2))
            prec1 = max(prec1, prec2)
            self.test_acc.append(prec1)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))

        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch, ema, model2, optimizer2, ema2):

        # switch to train mode
        model.train()
        model2.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)
        #self.on_start_epoch(True, model2, criterion, data_loader, optimizer2)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')
        end = time.time()
        #numm = 1
        for i, (input, target) in enumerate(data_loader):

            #ind = indexes.cpu().numpy().transpose()
            ind = torch.zeros(self.state['batch_size'])
            for i in range(ind.shape[0]):
                ind[i] = i
                #numm += 1
            ind = ind.numpy()
            # measure data loading time
            #print('input_len:{} input0:{} input1:{} input2:{}:'.format(len(input), input[0], input[1], input[2]))
            #input0: batch*3*448*448  input1:name input2:inp
            #x = input()
            #print('before cut:', input[0][0][0])

            #cut = Cutout(1,112)
            #for i in range(input[0].shape[0]):
                #input[0][i] = cut(input[0][i])

            #xp = cut(input[0][0])
            #print('aft cut:', xp[0] == input[0][0][0])
            #xpp = input()
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)
            #self.state['target_cpu'] = self.state['target']
            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)


            self.on_forward(True, model, criterion, data_loader, optimizer, ema=ema, model2=model2, optimizer2=optimizer2, ema2=ema2, ind=ind)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)
            #scheduler.step()

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion, ema, semantic):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)
        model = model
        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        if self.state['print_dis']:
            self.state['total_neg'] = 0
            self.state['total_pos'] = 0
            self.state['sum_neg'] = 0
            self.state['sum_pos'] = 0

            self.state['max_pos'] = 0
            self.state['min_pos'] = 1
            self.state['max_neg'] = 0
            self.state['min_neg'] = 1
            self.state['dis_neg'] = torch.zeros(15)
            self.state['dis_pos'] = torch.zeros(15)
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(False, model, criterion, data_loader, ema=ema, semantic=semantic)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)
        if self.state['print_dis']:
            print('total_neg:{} avg_neg:{} total_pos:{} avg_pos:{}'.format(self.state['total_neg'], self.state['sum_neg'] / self.state['total_neg'], self.state['total_pos'], self.state['sum_pos'] / self.state['total_pos']))
            print('per_neg:{} per_pos:{}'.format(self.state['total_neg'] / (self.state['total_neg'] + self.state['total_pos']), self.state['total_pos'] / (self.state['total_neg'] + self.state['total_pos'])))
            print('max_neg:{} min_neg:{} max_pos:{} min_pos:{}'.format(self.state['max_neg'], self.state['min_neg'], self.state['max_pos'], self.state['min_pos']))
            print('dis_neg:{} dis_pos:{}'.format(self.state['dis_neg'], self.state['dis_pos']))
        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def warm_learning(self, optimizer):
        lr_list = []
        decay = (self.state['epoch'] + 1) / self.state['warm_up']
        decay_pre = (self.state['epoch']) / self.state['warm_up']
        #print('decay:', decay)
        #print('decay_pre:', decay_pre)

        for param_group in optimizer.param_groups:
            #print('lr:', param_group['lr'])
            param_group['lr'] = param_group['lr'] * decay
            if decay_pre > 0:
                param_group['lr'] = param_group['lr'] / decay_pre
            lr_list.append(param_group['lr'])
        #print('lr_list:', lr_list)
        return np.unique(lr_list)

    def cos_learning(self, optimizer):
        lr_list = []
        #decay = 0.5 * (1 + torch.cos(torch.tensor(self.state['epoch'] * 3.14159265359 / self.state['max_epochs'])))
        #decay_pre = 0.5 * (1 + torch.cos(torch.tensor((self.state['epoch'] - 1) * 3.14159265359 / self.state['max_epochs'])))
        #print('epoch:{} max:{}'.format(self.state['epoch'], self.state['max_epochs']))
        decay = 0.5 * (1 + math.cos(self.state['epoch'] * 3.14159265359 / self.state['max_epochs']))
        decay_pre = 0.5 * (
                    1 + math.cos((self.state['epoch'] - 1) * 3.14159265359 / self.state['max_epochs']))
        #print('decay:', decay)
        #print('decay_pre:', decay_pre)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            if decay_pre > 0:
                param_group['lr'] = param_group['lr'] / decay_pre
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if self.state['epoch'] < self.state['warm_up']:
            return self.warm_learning(optimizer)
        if self.state['cos_learning']:
            return self.cos_learning(optimizer)
        lr_list = []
        # decay = 0.5 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        #print('self:{} step:{} {}'.format(self.state['epoch'], self.state['epoch_step'], np.array(self.state['epoch_step'])))
        decay = self.state['lr_decay'] if self.state['epoch'] % self.state['epoch_step'][0] == 0 and self.state['epoch'] != 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)



class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        #print('ap per class:', 100 * self.state['ap_meter'].value())
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if not training:
            self.test_cf1.append(CF1)
            self.test_of1.append(OF1)
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                self.state['train_loss'].append(loss)
                self.state['train_mAP'].append(map)
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t'
                      'OR_3: {OR:.4f}\t'
                      'OF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\t'
                      'CR_3: {CR:.4f}\t'
                      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
                self.state['test_loss'].append(loss)
                self.state['test_mAP'].append(map)
        if self.state['epoch'] + 1 == self.state['max_epochs']:
            print(self.state['train_loss'])
            print(self.state['train_mAP'])
            print(self.state['test_loss'])
            print(self.state['test_mAP'])
            plt.figure('Line fig')
            print('epoch:',self.state['max_epochs'])

            x_list = np.zeros(self.state['max_epochs'] - 1)
            train_l = np.zeros(self.state['max_epochs'] - 1)
            test_l = np.zeros(self.state['max_epochs'] - 1)
            train_ap = np.zeros(self.state['max_epochs'] - 1)
            test_ap = np.zeros(self.state['max_epochs'] - 1)
            for i in range(self.state['max_epochs'] - 1):
                x_list[i] = i
                train_l[i] = self.state['train_loss'][i]
                train_ap[i] = self.state['train_mAP'][i]

            for i in range(self.state['max_epochs'] - 1):
                test_l[i] = self.state['test_loss'][i]
                test_ap[i] = self.state['test_mAP'][i]
            ax = plt.gca()
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.plot(x_list, train_l, color='r', linewidth=1, alpha=0.6)
            ax.plot(x_list, test_l, color='b', linewidth=1, alpha=0.6)
            # plt.legend()
            plt.savefig(self.state['console'] + '/loss.png', dpi=300)

            bx = plt.gca()
            bx.set_xlabel('epoch')
            bx.set_ylabel('mAP')
            bx.plot(x_list, train_ap, color='r', linewidth=1, alpha=0.6)
            bx.plot(x_list, test_ap, color='b', linewidth=1, alpha=0.6)
            # plt.legend()
            plt.savefig(self.state['console'] + '/mAP.png', dpi=300)
        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
        self.state['batch_id'] += 1
        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])
        del self.state['output'], self.state['target_gt']

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def mix(self):
        return False
        return self.state['epoch'] >= 5 and self.state['epoch'] < 20  and (self.state['epoch'] % 5 == 3 or self.state['epoch'] % 5 == 4)
    def test(self):
        if self.mix():
            return False
        return self.state['epoch'] % 5 == 0 or self.state['epoch'] >= 0
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True, ema=None, model2=None, optimizer2=None, ema2=None, ind=None, semantic=None):
        feature_var = torch.autograd.Variable(self.state['feature']).float().detach()
        target_var = torch.autograd.Variable(self.state['target']).float().detach()
        inp_var = torch.autograd.Variable(self.state['input'][1]).float().detach()  # one hot
        inp_var_word = torch.autograd.Variable(self.state['input'][1]).float().detach()  # one hot
        coss = torch.autograd.Variable(self.state['input'][2]).float().detach()  # one hot
        if not training:
            feature_var.volatile = True
            #target_var_cpu.volatile = True
            target_var.volatile = True
            inp_var.volatile = True
            inp_var_word.volatile = True
            coss.volatile = True
            if semantic:
                #print('inp:', inp_var_word.shape)
                self.state['output'] = model(feature_var, inp_var_word)
            else:
                self.state['output'] = model(feature_var, inp_var)
            self.state['loss'] = criterion(self.state['output'], target_var).mean()
            object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor']
            if self.state['print_dis']:
                l = torch.sigmoid(self.state['output'])
                for x in range(target_var.shape[0]):
                    prin = False
                    for y in range(target_var.shape[1]):
                        #print(target_var[x][y].data.cpu())

                        if target_var[x][y].data.cpu().numpy() < 0.1:
                            if l[x][y].data.cpu().numpy() > 0.75:
                                prin = True
                            self.state['total_neg'] += 1
                            self.state['sum_neg'] += l[x][y]
                            self.state['max_neg'] = max(self.state['max_neg'], l[x][y].data.cpu().numpy())
                            self.state['min_neg'] = min(self.state['min_neg'], l[x][y].data.cpu().numpy())
                            self.state['dis_neg'][int(l[x][y].data.cpu().numpy() * 10)] += 1
                        else:
                            if l[x][y].data.cpu().numpy() < 0.1:
                                prin = True
                            self.state['total_pos'] += 1
                            self.state['sum_pos'] += (1 - l[x][y])
                            self.state['max_pos'] = max(self.state['max_pos'], l[x][y].data.cpu().numpy())
                            self.state['min_pos'] = min(self.state['min_pos'], l[x][y].data.cpu().numpy())
                            self.state['dis_pos'][int(l[x][y].data.cpu().numpy() * 10)] += 1
                    if prin:

                        l_pre = []
                        l_tar = []
                        for y in range(target_var.shape[1]):
                            if target_var[x][y].data.cpu().numpy() > 0.9:
                                l_tar.append(object_categories[y])
                            if l[x][y].data.cpu().numpy() > 0.75:
                                l_pre.append(object_categories[y])
                        print('img:{} pre:{} tar:{}'.format(self.state['out'][x], l_pre, l_tar) )
        else:
            if self.mix():
            #if self.state['epoch'] % 5 == 0 and self.state['epoch'] >= 10:
                #print('feature_var:', feature_var.shape)
                inputs, targets_a, targets_b, lam = mixup_data(feature_var.cuda(), target_var, self.state['alpha'])
                #inputs, targets, lam = mixup_data(feature_var.cuda(), target_var, self.state['alpha'])
                # self.state['target_gt'] = (lam * targets_a + (1 - lam) * targets_b).clone().detach().data
                #self.state['target_gt'] = targets.data
                #inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
                #outputs = net(inputs)
                #loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                self.state['output'] = model(inputs, inp_var)
                self.state['output2'] = model2(inputs, inp_var_word)
                #self.state['output'], kl = model(feature_var, inp_var, inp_var_word, training, self.state['lmda'], coss)
                #self.state['loss'] = criterion(self.state['output'], targets)
                self.state['loss'] = mixup_criterion(criterion, self.state['output'], targets_a, targets_b, lam).sum(dim=1)
                self.state['loss2'] = mixup_criterion(criterion, self.state['output2'], targets_a, targets_b, lam).sum(dim=1)
            else:

                self.state['output'] = model(feature_var.clone().detach(), inp_var)
                self.state['output2'] = model2(feature_var.clone().detach(), inp_var_word)
                #self.state['output2'] = model2(feature_var, inp_var)
                self.state['loss'] = criterion(self.state['output'], target_var).sum(dim=1)
                self.state['loss2'] = criterion(self.state['output2'], target_var).sum(dim=1)
        init_epoch = 100
        if training:
            # visual = gcn_visual[:2048].permute(1, 0)
            # word = gcn_word[:2048].permute(1, 0)
            # print('visual')
            # visual.sum().backward()
            # print('visual:{} word:{}'.format(gcn_visual.shape, gcn_word.shape))
            # cos1, cos2 = get_cos(fea, visual), get_cos(fea, word)
            # print('cos1:{} {}'.format(cos1.min(), cos1.max()))
            # print('cos2:{} {}'.format(cos2.min(), cos2.max()))
            # print('cos1:', cos1)
            # print('cos2:', cos2)
            # cos1.sum().backward()
            # kl = (self_kl(cos2, cos1) / fea.size(0)) * 100
            # print('kl:', kl.requires_grad)
            # kl.backward()
            # print('ok')
            # print('kl:{} kl_grad:{}'.format(kl, kl.requires_grad))
            # kl = torch.autograd.Variable(torch.tensor(kl)).float()
            # print('output:{} feature:{} gcn_visual:{} gcn_word:{}'.format(self.state['output'].shape, feature.shape, gcn_visual.shape, gcn_word.shape))
            # print('visual0:', gcn_visual[0:2])
            # print('visual1:', gcn_visual[2048:2050])
            # kl = kl_divergence(nn.sigmoid())
            # print('loss:{} kl:{}'.format(self.state['loss'], kl))
            # self.state['loss'] += kl
            # print('loss2:{} grad2:{}'.format(self.state['loss'], self.state['loss'].requires_grad))
            # print('after sum loss:', self.state['loss'])

            #print('rate:', self.rate_schedule, self.rate_schedule.shape)
            #print(self.rate_schedule[self.state['epoch']])
            #print(self.rate_schedule[self.state['epoch'] + 1])
            #print('output:{} target:{}'.format(self.state['output'].shape, target_var.shape))
            #for i in range(self.rate_schedule.shape[0]):
                #print(self.rate_schedule[i])
            if self.mix() or self.state['epoch'] <= -1:
                loss_1 = self.state['loss'].mean()
                loss_2 = self.state['loss2'].mean()

            elif self.state['epoch'] < init_epoch:
                loss_1, loss_2 = loss_coteaching(self.state['output'], self.state['output2'], target_var.clone().detach(), self.rate_schedule[self.state['epoch']], self.rate_schedule_neg[self.state['epoch']], ind,criterion, self.state['t_pos'], self.state['t_neg'])
            else:
                loss_1, loss_2 = loss_coteaching_plus(self.state['output'], self.state['output2'], target_var, self.rate_schedule[self.state['epoch']], ind, self.state['epoch'] * self.state['batch_id'], criterion)
            #del target_var
            #print('pro:{} label:{}'.format(torch.sigmoid(self.state['output'])[0].view(1, -1), target_var[0].view(1, -1)))
            #print('loss1:{} loss2:{}'.format(self.state['loss'].mean(), self.state['loss2'].mean()))
            #print('pred2:{} tar:{}'.format(torch.sigmoid(self.state['output2']).mean(), target_var.mean()))
            #print('loss:{}'.format(self.state['loss'].mean()))
            #print('loss2:{}'.format(self.state['loss2'].mean()))
            #print('target1:{} inp:{}'.format(target_var.sum(), inp_var.sum()))

            #print('target2:', target_var.sum())
            # print('target2:{} inp:{}'.format(target_var.sum(), inp_var_word.sum()))
            optimizer2.zero_grad()
            # self.state['loss2'].backward()
            loss_2.backward()
            nn.utils.clip_grad_norm(model2.parameters(), max_norm=10.0)
            optimizer2.step()


            optimizer2.zero_grad()


            optimizer.zero_grad()
            #self.state['loss'].backward()
            loss_1.backward()

            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()

            optimizer.zero_grad()






            self.state['loss'] = self.state['loss2']
            self.state['output'] = self.state['output2']
        else:
            del target_var



    # define drop rate schedule
    def gen_forget_rate(self, forget_rate, fr_type='type_1'):
        #forget_rate = self.state['rate']

        # if fr_type == 'type_1':
        rate_schedule = np.ones(self.state['max_epochs']) * forget_rate
        # print('shape:', rate_schedule.shape)
        be = num_be
        rate_schedule[0:be] = 0
        rate_schedule[be:num_gradual + be] = np.linspace(0, forget_rate, num_gradual)
        #rate_schedule[0] = rate_schedule[1]
        print('rate:', rate_schedule)
        # print('rate:', rate_schedule)
        # if fr_type=='type_2':
        #    rate_schedule = np.ones(args.n_epoch)*forget_rate
        #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
        #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

        return torch.from_numpy(rate_schedule)



    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):



        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0
        '''
                if training:
            smooth = 0.001
            self.state['target_gt'] = self.state['target'].clone()
            # print('shape:', self.state['target'].shape)
            for i in range(self.state['target'].shape[0]):
                summ = self.state['target'][i][self.state['target'][i] == 1].sum()
                # print('state_i:', self.state['target'][i])
                # print('summ:', summ)
                self.state['target'][i][self.state['target'][i] == 1] = 1 - smooth
                self.state['target'][i][self.state['target'][i] == -1] = summ * smooth / (
                            self.state['target'].shape[1] - summ)
        else:
        '''

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]