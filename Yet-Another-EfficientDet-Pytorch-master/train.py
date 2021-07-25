# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import argparse
import datetime
import os
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string, postprocess
from efficientdet.utils import BBoxTransform, ClipBoxes, Anchors_Face_Only
from efficientdet.model import Regressor_Face_Only, Classifier_Face_Only
from efficientdet.model_original import Regressor



class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

threshold = 0.2
iou_threshold = 0.2
def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='Face', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=True,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default='weights/efficientdet-d0.pth',
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, compound_coef=0, num_classes=1, debug = False, **kwargs):
        super().__init__()
        ## kwargs는 적당한 Parameters를 입력하면, 그걸 알아서 적당한 인자에게 전달해주는 듯. 즉 변수명만 맞추면 될 듯.
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.compound_coef = compound_coef
        self.num_classes = num_classes
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales


        self.regressor = Regressor_Face_Only(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = Classifier_Face_Only(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors_Face_Only(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)


    def forward(self, imgs, annotations, obj_list=None):
        BiFPN_outputs, regression, classification, anchors = self.model(imgs)
        """if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss"""
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        # Postprocessing for extrating the body's region
        # out shape = [image_num, roi_num, 세부 항목]
        out = postprocess(imgs,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

        # Save the body's bounding box result
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []

        # [img, xmins, ymins, xmaxs, ymaxs]
        img_per_roi = []

        ############## Extract the bounding box coordinate of the human in the images with batch sizes ###
        # i = image
        for i in range(len(imgs)):
            x_mins.clear()
            y_mins.clear()
            x_maxs.clear()
            y_maxs.clear()
            # 'j' is the order of rois in the 'out array'
            for j in range(len(out[i]['rois'])):
                x_min, y_min, x_max, y_max = out[i]['rois'][j].astype(np.int)
                obj = obj_list[out[i]['class_ids'][j]]
                score = float(out[i]['scores'][j])
                x_mins.append(x_min)
                y_mins.append(y_min)
                x_maxs.append(x_max)
                y_maxs.append(y_max)
            img_per_roi.append([x_mins.copy(), y_mins.copy(), x_maxs.copy(), y_maxs.copy()]) # Body bounding box coordinates

        ## img_per_roi.shape = [img_num, 4(x_min...y_max), detected anchor num]

        #################### Should i use the affine function on here? ...

        ## Resample BiFPN weights by the body region
        resampled_BiFPN_outputs_list = []
        resampled_BiFPN_outputs_list_per_feature_maps_num = []
        resampled_BiFPN_outputs_list_per_img = []
        feature_map_minimum_size = [8, 7, 6, 5, 4]
        # BiFPN_outputs = [feature map number, Image number, channel number, height, width]
        ## i is the pyramid's number
        for i in range(len(BiFPN_outputs)): # For the resolution of Feature map /
            # [feature_maps, images, channels, height, width]
            # Resize the bounding box coordinates for adapting feature maps which is extracted by BiFPN
            # j is the batch size's index
            for j in range(len(imgs)):
                ## The bounding box's coordinate's resize process
                x_ratio = imgs[j].shape[2] / BiFPN_outputs[i].shape[3]
                y_ratio = imgs[j].shape[1] / BiFPN_outputs[i].shape[2]
                x_mins, y_mins, x_maxs, y_maxs = img_per_roi[j]
                x_mins_resized, x_maxs_resized = [int(x_min / x_ratio) for x_min in x_mins], [int(x_max / x_ratio) for x_max in x_maxs]
                y_mins_resized, y_maxs_resized = [int(y_min / y_ratio) for y_min in y_mins], [int(y_max / y_ratio) for y_max in y_maxs]
                # Re-proposal of feature map which is extracted by pyramid feature map by using predicted bounding boxes

                # k is the bpunding box's number in one image
                for k in range(len(x_mins_resized)):
                    if x_mins_resized[k] >= x_maxs_resized[k]:
                        # if the minimum x value of bounding box is bigger than feature map size
                        if x_mins_resized[k] >= feature_map_minimum_size[i]:
                            x_mins_resized[k] = feature_map_minimum_size[i] - 2
                            x_maxs_resized[k] = feature_map_minimum_size[i]
                        elif x_maxs_resized[k] <= 0:
                            x_mins_resized[k] = 0
                            x_maxs_resized[k] = 1
                        else:
                            x_maxs_resized[k] = x_mins_resized[k]
                            x_mins_resized[k] -= 1

                for k in range(len(x_maxs)): # 여기서 앵커 개수에 따라서 피처맵을 차등 분류한다.
                    # 여기서 최소 크기를 지켜주고, regression 및 classification 할 때 Conv로 차원읆 맞춰줘야할 듯 하다.
                    print(BiFPN_outputs[i][j].shape)
                    print(y_maxs_resized[k] - y_mins_resized[k])
                    print(x_maxs_resized[k] - x_mins_resized[k])
                    resampled_BiFPN_outputs_list.append(BiFPN_outputs[i][j][:,y_mins_resized[k]:y_maxs_resized[k],x_mins_resized[k]:x_maxs_resized[k]])
                    print("shape")
                    print(resampled_BiFPN_outputs_list[len(resampled_BiFPN_outputs_list)-1].shape)
                resampled_BiFPN_outputs_list_per_img.append(resampled_BiFPN_outputs_list.copy())
                resampled_BiFPN_outputs_list.clear()
            resampled_BiFPN_outputs_list_per_feature_maps_num.append(resampled_BiFPN_outputs_list_per_img.copy())
            resampled_BiFPN_outputs_list_per_img.clear()




        # Convert BiFPN outputs to tuple type
        resampled_BiFPN_outputs = tuple(resampled_BiFPN_outputs_list_per_feature_maps_num)

        # Feed forward to prediction layers
        regression = self.regressor(resampled_BiFPN_outputs)
        classification = self.classifier(resampled_BiFPN_outputs)


        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model.requires_grad_(False)
    model.eval()
    models = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        models = models.cuda()
        if params.num_gpus > 1:
            models = CustomDataParallel(models, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(models)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW([{'params':models.classifier.parameters()},{'params':models.regressor.parameters()}], opt.lr)
    else:
        optimizer = torch.optim.SGD(models.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    models.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = models(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(models, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                models.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = models(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(models, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                models.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(models, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
