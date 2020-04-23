import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from util import *

def parse_cfg(cfg_file):
    """
        Parses configuration file
        Returns list of blocks, in the YOLO network
    """
    with open(cfg_file, 'r') as f:
        lines = file.read().split("\n")
        lines = [x for x in lines if len(x)>0]
        lines = [x for x in lines x[0]!='#']
        lines = [x.strip() for x in lines]
        f.close()
    
    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for i, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type'] == "convolutional":
            activation = x['activation']
            if "batch_normalize" in x.keys():
                batch_norm = int(x['batch_normalize'])
                bias = False
            else:
                batch_norm = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{}".format(i), conv)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("bn_norm_{}".format(i), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{}".format(i), activn)

        elif x['type'] == "upsample":
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(i), upsample)
        
        elif x['type'] == "route":
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            
            if start > 0:
                start = start - i
            if end > 0:
                end = end - i
            
            if end < 0:
                filters = output_filters[i + start] + output_filters[i + end]
            else:
                filters = output_filters[i+start]

        elif x["type"] == "yolo":
            mask = x["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add("Detection_{}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, gpu=True):
        modules = self.blocks[1:]
        outputs = {}
        for i, module in enumerate(modules):
            write = 0
            module_type = module["type"]
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                
                if len(layers) == 1:
                    x = outputs[i+layers[0]]
                
                else:
                    if layers[1]>0:
                        layers[1] = layers[1] - i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            
            elif module_type == "shortcut":
                prev = int(module["from"])
                x = outputs[i-1]+outputs[i+prev]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(self.net_infor["classes"])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes)
                
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            
            outputs[i] = x
        
        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count = 5)
        self.header =  torch.from_numpy(header)
        self.seen - self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.module_list[i+1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                if batch_normalize:
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    # Number of biases
                    num_biases = conb.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr+num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

def get_test_input(path):
    img = cv2.imread(path)
    img = ccv2.resize(img, (416, 416))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]/255.0
    img = torch.from_numpy(img).float()
    img = Variable(img)
    return img

model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
inp = get_test_input("./image1.png")
pred = model(inp)
print(pred)