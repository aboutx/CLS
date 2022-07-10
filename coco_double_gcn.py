import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *
import torchvision.models as models
#from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
import time
def _read_conds(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _write_conds(dat, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dat, f)
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def download_coco2014(root, phase):
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    data = os.path.join(root, 'data/')
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file,data)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.Popen('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        print('anno_file:{} anno:{} anno_list:{}'.format(len(annotations_file), len(annotations), len(anno_list)))
        print('images:', len(images), images[0])
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')
    #asdf = input()

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        #with open(inp_name, 'rb') as f:
            #self.inp = pickle.load(f)
        self.inp_name = inp_name


        self.inp = (0, 1, 2)

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        #print('img_list:', self.img_list)
        #sdf = input()
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target


class COCO2014_miss(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None, per=0.8):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        print(self.img_list[0], len(self.img_list))
        self.num_classes = len(self.cat2idx)
        #with open(inp_name, 'rb') as f:
            #self.inp = pickle.load(f)
        self.inp = 0
        self.inp_name = inp_name
        tmp = 0
        #with open('data/coco/coco_wordnet_0', 'rb') as f:
            #word_all = pickle.load(f)
        word_all = 0
        num_target = 0
        for i in range(len(self.img_list)):
            target = self.img_list[i]['labels']
            se = len(target)
            num_target += se
            if se == 0:
                print(i, target)
        num_target /= len(self.img_list)
        print('avg:', num_target)
        self.classes = COCO_CLASSES
        self.tar = []
        path = 'data/coco/missing' + str(per)
        if not os.path.exists(path):
            for i in range(len(self.img_list)):
                if i % 1000 == 0:
                    print(i)
                item = self.img_list[i]
                filename = item['file_name']
                labels = sorted(item['labels'])
                img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                target = np.zeros(self.num_classes, np.float32) - 1
                target[labels] = 1
                for j in range(len(labels)):
                    if target[labels[j]] == 1 and torch.rand(1)[0] < per:
                        target[labels[j]] = -1
                if target.sum() != -target.shape[0]:
                    self.tar.append((filename, target))
            _write_conds(self.tar, path)
        self.img_list = _read_conds(path)
        print(len(self.img_list))
        num_target = 0
        for i in range(len(self.img_list)):
            path, target = self.img_list[i]
            se = (target == 1).sum()
            num_target += se
            if se == 0:
                print(i, target)
        num_target /= len(self.img_list)
        print('avg label:', num_target)

        num = 0
        while False:
            item = self.img_list[num]
            print('name:', item['file_name'])
            COCO_CLASSES.sort()
            for id in item['labels']:
                print(COCO_CLASSES[id])
            num += 1
            xi = input()
        #self.inp = self.get_inp('data/coco/feature_coco_multi' + str(per))

        #with open('data/coco/google_new_coco', 'rb') as f:
            #self.inp = pickle.load(f)

        self.inp = (self.inp, tmp, word_all)

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]

        filename = item[0]
        target = item[1]
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return (img, filename, self.inp), target

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target

    '''
        def get_inp(self, path):
        if not os.path.exists(path):
            model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights="imagenet",
                    input_tensor=None,
                    input_shape=None,
                    pooling='avg',
                    classes=1000,
                )
            inp = torch.zeros(80, 2048)
            num = torch.zeros(80)
            t0 = time.time()
            for i in range(len(self.img_list)):
                if i % 1000 == 0:
                    #break
                    print('i:{} time:{}'.format(i, (time.time() - t0) / (i + 1)))
                filename, target = self.img_list[i]
                img_path = os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                fea = model.predict(img_data).squeeze()
                fea = torch.from_numpy(fea)
                #print('fea:', fea.shape)
                for j in range(target.shape[0]):
                    if target[j] == 1:
                        inp[j] += fea
                        num[j] += 1
            num *= 10
            for i in range(inp.shape[0]):
                inp[i] /= num[i]
            _write_conds(inp.numpy(), path)
        inp = _read_conds(path)
        inp /= 10
        #inp[inp > 2] = torch.rand(1)[0]
        print('visual feature:', inp.min(), inp.max(), inp.mean(), inp.std())
        return inp
    '''