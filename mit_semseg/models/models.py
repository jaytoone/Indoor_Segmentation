import torch
import torch.nn as nn
from . import resnet, resnext, mobilenet, hrnet
from .unet import UNet
from .pytorch_refinenet import RefineNet4Cascade
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

from PIL import Image
import cv2
from datetime import datetime
import numpy as np


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        # print('preds.shape :', preds.shape)
        # print('label.shape :', label.shape)
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


def denormalize(x):
  mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).type_as(x)
  std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).type_as(x)
  return x * std + mean

from torchvision import transforms
import numpy as np

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, unet, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.unet = unet
        self.crit = crit
        self.crit_binary = nn.BCEWithLogitsLoss()
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, object_index=0, segSize=None):

        # print(type(feed_dict))
        # print('feed dict[0] img_data shape:', feed_dict[0]['img_data'].shape)

        if type(feed_dict) is list:
          feed_dict = feed_dict[0]
          # print(feed_dict)
          # also, convert to torch.cuda.FloatTensor
          if torch.cuda.is_available():
            feed_dict['img_data'] = feed_dict['img_data'].cuda()
            feed_dict['seg_label'] = feed_dict['seg_label'].cuda()
          else:
            raise RunTimeError('Cannot convert torch.Floattensor into torch.cuda.FloatTensor')

        org_image = denormalize(feed_dict['img_data'])
        org_image_ = org_image.cpu().detach()
        np_org_image = [np.asarray(transforms.ToPILImage()(org_image_[i])) for i in range(len(org_image_))]
        np_org_image = np.array(np_org_image)
        # print('After image data from feed_dict')
        # print('torch.cuda.memory_allocated() :', torch.cuda.memory_allocated())
        # print('torch.cuda.memory_reserved() :', torch.cuda.memory_reserved())
        # print()

        if segSize is None:
          label_size = (feed_dict['seg_label'].shape[2], feed_dict['seg_label'].shape[1])
        else:
          label_size = (segSize[1], segSize[0])

        np_org_image = self.resize_batch(np_org_image, label_size)

        edge_batch = self.canny_edge(np_org_image)
        edge_batch = self.resize_batch(edge_batch, label_size)
        edge_batch = np.expand_dims(edge_batch, axis=3)
        # print('edge_batch.max() :', edge_batch.max())

        # training
        if segSize is None:

            if self.deep_sup_scale is not None: # use deep supervision technique
                # (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                # pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            #   New Input = Original Image + Binary Pred + Edge Map   #
            #   torch.Size([4, 3, 480, 704])
            # print("feed_dict['img_data'].shape :", feed_dict['img_data'].shape)
            # org_image = [denormalize(feed_dict['img_data'][i]).cpu().detach().numpy() for i in range(len(feed_dict['img_data']))]
            # org_image = np.array(org_image)
            # print('org_image shape :', org_image.shape)
            
            # print('org_image min max :', org_image[0].min(), org_image[0].max())
            # print('np_org_image min max :', np_org_image.min(), np_org_image.max())
            # print('np_org_image.shape :', np_org_image.shape)
            # print('np_org_image.dtype :', np_org_image.dtype) # uint8
            # plt.imshow(np_org_image)
            # plt.axis('off')
            # plt.savefig('./edge_tensor/%s.png' % int(datetime.now().timestamp()))

            # print('np_org_image.shape :', np_org_image.shape)
            # print("feed_dict['img_data'].shape :", feed_dict['img_data'].shape)
            # print("feed_dict['seg_label'].shape :", feed_dict['seg_label'].shape)

            # print('label shape :', feed_dict['seg_label'].shape)
            # print('self.resize_batch(np_org_image).shape :', np_org_image.shape)
            # print('self.resize_batch(edge_batch).shape :', edge_batch.shape)

            # print('pred :', pred)
            # print('pred.dtype :', pred.dtype)
            # print('pred.shape :', pred.shape)

            # _, pred_m = torch.max(pred, dim=1)
            # pred_m = pred_m.cpu().detach().numpy()
            # pred_m_ = np.uint8(pred_m==0) * 255
            # pred_ = pred_m_

            #                     Put Index of Object What you want to Change                 #
            #   Wall : 0 /  Floor, Flooring : 3 / Ceiling : 5

            pred_s = pred[:, [object_index], :, :].squeeze(1)
            pred_s = pred_s.cpu().detach().numpy()
            pred_scaled = (pred_s - pred_s.min()) / (pred_s.max() - pred_s.min()) * 255

            pred_ = pred_scaled

            # print('pred_m.shape :', pred_m.shape)
            # print('pred_m.min() :', pred_m.min())

            #     이거 취소 -> Max가 아니라 Binary Image로 도출해야하니까 Wall에 해당하는 Channel만 추출한다.    #            
            #   Tensor Slicing    #
            # pred_ = torch.sigmoid(pred[:, [0], :, :]).squeeze(1)
            # pred_ = pred_.cpu().detach().numpy()
            # pred__ = np.uint8(pred_ > 0.5) * 255
            # print('pred[:, [0], :, :].shape :', pred[:, [0], :, :].shape)
            # print('pred_.shape :', pred_.shape)
            # print('pred_.min() :', pred_.min())
            # print('pred_.max() :', pred_.max())
            # plt.show()


            # plt.subplot(151)
            # plt.imshow(np_org_image[0])
            
            # plt.subplot(152)
            # plt.imshow(np.squeeze(edge_batch[0], axis=2))            

            # plt.subplot(153)
            # plt.imshow(pred__[0])

            # plt.subplot(154)
            # plt.imshow(pred_m_[0])

            # print('np_org_image.dtype :', np_org_image.dtype)
            # print('pred_.dtype :', pred_.dtype)
            # print('edge_batch.dtype :', edge_batch.dtype)

            pred_ = np.expand_dims(pred_, axis=3)
            # print('np.expand_dims(pred_, axis=3).shape :', pred_.shape)
            # print('np.expand_dims(edge_batch, axis=3).shape :', edge_batch.shape)            

            # print('np_org_image.shape :', np_org_image.shape)
            # print('pred_.shape :', pred_.shape)
            # print('edge_batch.shape :', edge_batch.shape)

            new_input = np.concatenate((np_org_image, pred_, edge_batch), axis=3)
            # new_input = np.concatenate((np_org_image, pred_), axis=3)
            # print('new_input.shape :', new_input.shape)
            new_tensor = torch.from_numpy(np.transpose(new_input, (0, 3, 1, 2))).cuda().type(torch.float32).contiguous()
            # print('new tensor.shape:', new_tensor.shape)

            new_pred = self.unet(new_tensor)

            # print('new_pred.dtype :', new_pred.dtype)
            # print('new_pred.shape :', new_pred.shape)

            # _, new_pred_m = torch.max(new_pred, dim=1)
            # new_pred_m = new_pred_m.cpu().detach().numpy()
            # new_pred_m_ = np.uint8(new_pred_m==0) * 255

            # new_pred_ = torch.sigmoid(new_pred).squeeze(1)
            # print('new_pred_.max() :', new_pred_.max())
            # new_pred_ = np.uint8(new_pred_ > 0.5) * 255

            # plt.subplot(155)
            # plt.imshow(new_pred_m_[0])

            # print('new_pred.dtype :', new_pred.dtype)
            # print("feed_dict['seg_label'].dtype :", feed_dict['seg_label'].dtype)
            segment = feed_dict['seg_label'].cpu().detach().numpy()
            # print('segment.min() :', segment.min()) # = -1 but wall index = 0

            # binary_seg_label의 wall index = 0, else = 1로 설정한다. -> new_pred의 wall label 값을 '0'으로 설정하기 위함 #
            binary_seg_label = torch.from_numpy(np.where(segment == object_index, 0, 1)).cuda().type(torch.int64)
            # print('binary_seg_label.dtype :', binary_seg_label.dtype)
            # print('binary_seg_label.shape :', binary_seg_label.shape)

            segment = np.uint8(np.where(segment == object_index, 0, 1)) * 255


            new_pred_s = new_pred[:, [0], :, :].squeeze(1)
            new_pred_s = new_pred_s.cpu().detach().numpy()
            new_pred_scaled = (new_pred_s - new_pred_s.min()) / (new_pred_s.max() - new_pred_s.min()) * 255


            # plt.figure(figsize=(10, 10))
            # plt.subplot(121)
            # plt.imshow(pred_[0], 'gray')
            # plt.subplot(122)
            # plt.imshow(new_pred_scaled[0], 'gray')

            # plt.savefig('./tensor_img/%s.png' % int(datetime.now().timestamp()))
            # print('tensor_img saved')

            # feed_dict['seg_label'] = feed_dict['seg_label'].type(torch.float32)
            #   기존의 criterion인 NLLLoss를 사용할 경우 dimension을 맞춰줄 필요가 없지만
            #   지금 사용하는 binary_crit => BCEWithLogitsLoss()이기 때문에 dimension을 맞추어준다.
            loss = self.crit(pred, feed_dict['seg_label'])
            loss_binary = self.crit(new_pred, binary_seg_label)
            # print('loss_binary :', loss_binary)

            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                # loss_deepsup = self.crit(pred_deepsup, binary_seg_label)
                loss += loss_deepsup * self.deep_sup_scale

            loss += loss_binary

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            acc_binary = self.pixel_acc(new_pred, binary_seg_label)
            return loss, acc, acc_binary
            
        # inference
        else:


            # print('inference working')
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            
            # return pred 
            return pred[:, [object_index], :, :]


            # _, pred_ = torch.max(pred, dim=1)
            # pred_ = pred_.cpu().detach().numpy()
            # pred__ = np.uint8(pred_==object_index) * 255
            # pred_ = np.expand_dims(pred__, axis=3)

            pred_s = pred[:, [object_index], :, :].squeeze(1)
            pred_s = pred_s.cpu().detach().numpy()
            # pred_s = pred
            pred_scaled = (pred_s - pred_s.min()) / (pred_s.max() - pred_s.min()) * 255
            pred_ = np.expand_dims(pred_scaled, axis=3)

            # return pred_

            # print('np_org_image.shape :', np_org_image.shape)
            # print('pred_.shape :', pred_.shape)
            # print('edge_batch.shape :', edge_batch.shape)

            new_input = np.concatenate((np_org_image, pred_, edge_batch), axis=3)
            # new_input = np.concatenate((np_org_image, pred_), axis=3)

            # print('new_input.shape :', new_input.shape)
            new_tensor = torch.from_numpy(np.transpose(new_input, (0, 3, 1, 2))).cuda().type(torch.float32).contiguous()
            # print('new tensor.shape:', new_tensor.shape)

            new_pred = self.unet(new_tensor)
            # _, new_pred_m = torch.max(new_pred, dim=1)
            # new_pred_m = new_pred_m.cpu().detach().numpy()
            # new_pred_m_ = np.uint8(new_pred_m==0) * 255
            
            new_pred_s = new_pred[:, [0], :, :].squeeze(1)
            new_pred_s = new_pred_s.cpu().detach().numpy()
            new_pred_scaled = (new_pred_s - new_pred_s.min()) / (new_pred_s.max() - new_pred_s.min()) * 255
            new_pred_ = np.expand_dims(new_pred_scaled, axis=3)


            # del new_input
            # del new_pred_s
            # torch.cuda.empty_cache()
            # # print('del new_pred_s')

            # # new_pred_ = new_pred[:, [0], :, :].squeeze(1)
            # # new_pred_ = torch.sigmoid(new_pred_)
            # # # # print('new_pred_.max() :', new_pred_.max())
            # # # # print('new_pred_.shape :', new_pred_.shape)
            # # new_pred_ = new_pred_.cpu().detach().numpy()
            # # new_pred_scaled = (new_pred_ - new_pred_.min()) / (new_pred_.max() - new_pred_.min())
            # # # print('new_pred_scaled.max() :', new_pred_scaled.max())
            # # # print('new_pred_scaled.min() :', new_pred_scaled.min())
            # # new_pred_scaled = np.uint8(new_pred_scaled > 0.5) * 255

            # # plt.subplot(121)
            # # # plt.imshow(np_org_image[0])
            # # plt.imshow(pred_scaled[0])
            # # # plt.imshow(new_pred_m_[0])
            # # plt.subplot(122)
            # # # plt.imshow(new_pred_m[0])
            # # plt.imshow(new_pred_scaled[0])
            # # # plt.imshow(np.uint8(new_pred_scaled[0]))
            # # plt.savefig('./test_result/%s.png' % int(datetime.now().timestamp()))

            # return new_pred


        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            # model = Resnet(orig_resnet).to(device)
            # summary(model, (3, 256, 256))

    def resize_batch(self, image_batch, label_size):

      resize_batch = list()
      for i in range(len(image_batch)):
        
        np_image = image_batch[i]
        if len(image_batch.shape) == 4:
          resize_image = Image.fromarray(np_image).convert('RGB').resize(label_size, Image.NEAREST)
        else:
          # print('len(image_batch) :', len(image_batch))
          resize_image = Image.fromarray(np_image).resize(label_size, Image.NEAREST)
        # print('resize_image.size :', resize_image.size)

        resize_batch.append(np.array(resize_image))

      return np.array(resize_batch)

    

    def canny_edge(self, image_batch):

        edge_batch = list()
        for i in range(len(image_batch)):

          image = image_batch[i]
          # plt.subplot(121)
          # plt.imshow(image)
          # plt.axis('off')
          
          # image = image.astype('uint8')
          # print('image shape in image_batch :', image.shape)          
          hsv_ = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

          hsv_added = cv2.addWeighted(image, 0.7, hsv_, 0.3, 0)
          kernel_size, low_threshold, high_threshold = 5, 0, 150
          hsv_added = cv2.GaussianBlur(hsv_added, (kernel_size, kernel_size), 0)
          hsv_added = cv2.Canny(hsv_added, low_threshold, high_threshold)
          hsv_added = np.invert(hsv_added)

          # print('hsv_added.max() :', hsv_added.max())
          # print('hsv_added.dtype :', hsv_added.dtype) # uint8

          # plt.subplot(122)
          # plt.imshow(hsv_added)
          # # plt.axis('off')
          # plt.savefig('./edge_tensor/%s.png' % int(datetime.now().timestamp()))

          edge_batch.append(hsv_added)

        return np.array(edge_batch)


from torchsummary import summary

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            # model = Resnet(orig_resnet).to(device)
            # summary(model, (3, 256, 256))

        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            # pretrained = True if len(weights) == 0 else False
            # orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            # model = UPerNet(Resnet(orig_resnet)).to(device)
            # summary(model, (3, 256, 256))
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    @staticmethod
    def build_unet(n_channels=5, 
        n_classes=2, 
        bilinear=True,
        weights=''):

        unet = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        # unet = RefineNet4Cascade((n_channels, feature_size), num_classes=2)

        if len(weights) > 0:
            print('Loading weights for unet')
            unet.load_state_dict(
              torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return unet



def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

import matplotlib.pyplot as plt

class Resnet(nn.Module):
    def __init__(self, orig_resnet, fpn_dim=64):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.edge_conv = nn.Sequential(
            nn.Conv2d(fpn_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))   # 64
        # self.show_tensor(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);       # 1024    
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

    def show_tensor(self, x):

        show_x = x
        print('show_x shape :', show_x.shape)    
        # print(show_x.shape[1])    
        
        show_x = self.edge_conv(show_x)
        # show_x = nn.functional.sigmoid(show_x)
        # print('show_x shape :', show_x.shape)    
        image =  show_x.cpu().detach().numpy()[0].reshape(show_x.shape[2], show_x.shape[3])
        print('image shape :', image.shape)     
        # print(show_x[0])      

        plt.imshow(image)
        # plt.imshow(image, 'gray')
        # plt.show()
        plt.axis('off')
        plt.savefig('./encoder_tensor_image.png')



class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        # if self.use_softmax: # is True during inference
        if segSize is not None:
            x = nn.functional.interpolate(
              x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)

            torch.cuda.empty_cache()
            # print('After get x from C1')
            # print('torch.cuda.memory_allocated() :', torch.cuda.memory_allocated())
            # print('torch.cuda.memory_reserved() :', torch.cuda.memory_reserved())
            # print()
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(fpn_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        # print('fusion_list[1] shape :', fusion_list[1].shape)
        # show_x = nn.functional.interpolate(
        #   fusion_list[1], size=segSize, mode='bilinear', align_corners=False)
        # self.show_tensor(fusion_list[-1]) 
        
        fusion_out = torch.cat(fusion_list, 1)
        # print('fusion_out(concated) :', fusion_out.shape)
        x = self.conv_last(fusion_out)
        # print('x shape :', x.shape)

        # print('self.use_softmax :', self.use_softmax)
        # print('segSize :', segSize)

        # if self.use_softmax:  # is True during inference

        if segSize is not None:
          x = nn.functional.interpolate(
              x, size=segSize, mode='bilinear', align_corners=False)
          x = nn.functional.softmax(x, dim=1)
          return x

        x = nn.functional.log_softmax(x, dim=1)
        # print('x shape :', x.shape)
        # print()

        return x

    def show_tensor(self, x):

        show_x = x
        print('show_x shape :', show_x.shape)    
        # print(show_x.shape[1])    
        
        show_x = self.edge_conv(show_x)
        # show_x = nn.functional.sigmoid(show_x)
        # print('show_x shape :', show_x.shape)    
        image =  show_x.cpu().detach().numpy()[0].reshape(show_x.shape[2], show_x.shape[3])
        print('image shape :', image.shape)     
        # print(show_x[0])      

        plt.imshow(image)
        # plt.imshow(image, 'gray')
        # plt.show()
        plt.axis('off')
        plt.savefig('./decoder_tensor_image.png')


# class RefineNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#           super(UNet, self).__init__()
#           self.n_channels = n_channels
#           self.n_classes = n_classes
#           self.bilinear = bilinear

#           self.conv1x1 = nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#             )


#       def forward(self, x):
          
#           x = nn.functional.log_softmax(x, dim=1)
#           return x
