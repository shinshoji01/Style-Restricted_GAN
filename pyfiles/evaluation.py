import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models

from prdc import compute_prdc

from util import *

############################### Evaluation Method ######################################

class vgg_model():
    def __init__(self, model):
        layers = []
        layers += list(model.features.children())
        layers += list(model.avgpool.children())
        self.feature_extractor = nn.Sequential(*layers)
        layers = []
        layers += list(model.classifier.children())[:6]
        self.fcs = nn.Sequential(*layers)
        self.model = model
        
    def get(self, x, output_type="score"):
        if output_type=="feature":
            with torch.no_grad():
                x = self.feature_extractor(x)
                x = torch.flatten(x, 1)
                x = self.fcs(x)
                out = x

        elif output_type =="score":
            with torch.no_grad():
                output = self.model(x)
                out = output
        return out

class GAN_evaluation():
    def __init__(self, feature_extractor="vgg-initialization", device="cpu", 
                 classes=tuple(range(4)), reference=tuple(range(4))):
        self.fe = feature_extractor
        
        if "vgg" in self.fe and "ImageNet" in self.fe:
            model = models.vgg19_bn(pretrained=True).to(device)
            model.eval()
            self.model = vgg_model(model)
            
        elif "vgg" in self.fe and "initialization" in self.fe:
            model = models.vgg19_bn(pretrained=False).to(device)
            model.apply(weights_init)
            model.eval()
            self.model = vgg_model(model)
            
        elif "vgg" in self.fe and "CelebA" in self.fe:
            model = models.vgg19_bn(pretrained=False).to(device)
            model.classifier[6] = nn.Linear(in_features=4096, out_features=len(classes))
            
            model_path = f"../data/parameters/B/facial_recognizer_vgg_lr5e-05_epoch126.pth"
            model_ = torch.load(model_path) 
            model.load_state_dict(model_)
            model.eval()
            model = model.to(device)
            self.model = vgg_model(model)
            
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self.device = device
        
    def preprocess(self, tensor):
        images = []
        for i in range(tensor.shape[0]):
            t = tensor[i:i+1,:,:,:]
            image = image_from_output(t)[0]
            image = self.transform(image).numpy()
            images.append(image)
        images = torch.Tensor(np.array(images))
            
        return images
    
    def get_feature(self, tensor, batch=32, get_attention=False, thres=0.5):
        num = tensor.shape[0]
        for itr in range(num//batch+int(bool(num-batch*(num//batch)))):
            data = tensor[itr*batch:(itr+1)*batch]
            data = data.to(self.device)
            feature = cuda2numpy(self.model.get(data, "feature").reshape(data.shape[0], -1))
                    
            if itr==0:
                features = feature
            else:
                features = np.concatenate([features, feature], axis=0)
        return features
    
    def get_prdc(self, true, pred, nearest_k=5, preprocess=True, thres=0.5, batch=32):
        self.run_preprocess = preprocess
        if preprocess:
            true = self.preprocess(true)
            pred = self.preprocess(pred)
        f1 = self.get_feature(true)
        f2 = self.get_feature(pred)
        if f1.shape[1]==0:
            return {"precision": None,  "recall": None,  "density": None,  "coverage": None,  }
        metrics = compute_prdc(real_features=f1,
                               fake_features=f2,
                               nearest_k=nearest_k)
        return metrics
    
def evaluation_init(fe_list, classes, metrics):
    GAN_eval = {}
    for fe in fe_list:
        GAN_eval[fe] = {}
        for source_label in classes:
            GAN_eval[fe][source_label] = {}
            for target_label in classes:
                GAN_eval[fe][source_label][target_label] = {}
                GAN_eval[fe][source_label][target_label] = {}
                for metric in metrics.keys():
                    GAN_eval[fe][source_label][target_label][metric] = []
    return GAN_eval