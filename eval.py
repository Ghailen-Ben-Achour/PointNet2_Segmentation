"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene,S3DISDataset
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
classes = ["Terrain",
            "Building",
            "UrbanAsset",
            "Vegetation",
            "Car"]
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')#32
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='semseg', help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--data_path', default='/content/gdrive/My Drive/AI/Swiss/sampling/*', help='path to data')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b,n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    #visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 5
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = args.data_path

    TEST_DATASET_WHOLE_SCENE = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET_WHOLE_SCENE, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    log_string("The number of test data is: %d" %  len(TEST_DATASET_WHOLE_SCENE))
    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    checkpoint = torch.load(str(experiment_dir) + '/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        log_string('---- EVALUATION ----' )
        for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points, target = data
            print(points.shape,target.shape)
            pts=np.zeros((points.shape[1],4))
            points = points.data.numpy()
            pts[...,:3]=points[...,:3]
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_val = np.argmax(pred_val, 2)
            pts[:,3]=pred_val[0,:]
            np.save(os.path.join('./test',str(i)+'.npy'), pts)

if __name__ == '__main__':
    args = parse_args()
    main(args)
