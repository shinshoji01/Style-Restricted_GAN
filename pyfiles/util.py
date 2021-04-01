import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import pickle
import torch.optim as optim

from prdc import compute_prdc

def cuda2numpy(x):
    return x.detach().to("cpu").numpy()

def cuda2cpu(x):
    return x.detach().to("cpu")

def pickle_save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data

def min_max(x, axis=None, mean0=False, get_param=False):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min+1e-8)
    if mean0 :
        result = result*2 - 1
    if get_param:
        return result, min, max
    return result

def standardize_torch(x):
    xmean = torch.mean(x, dim=(1,2,3), keepdim=True)
    xstd = torch.std(x, dim=(1,2,3), keepdim=True)
    new_x = (x-xmean)/xstd
    return new_x 

def image_from_output(output):
    image_list = []
    output = output.detach().to("cpu").numpy()
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list

def normalize_dataset(data, axis=1, device="cpu", mean0=True):
    if mean0:
        data = (data + 1)/2
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(2*(axis==0)+1,2*(axis==1)+1,2*(axis==2)+1,2*(axis==3)+1)
    std = std.view(2*(axis==0)+1,2*(axis==1)+1,2*(axis==2)+1,2*(axis==3)+1)
    data = (data-mean)/std
    return data

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('linear') != -1:        
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('batchnorm') != -1:     
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
        
def class_encode(label, device, ref_class, source_label=None):
    class_label = torch.tensor(ref_class, dtype=torch.float32)[label].view(-1, ref_class.shape[1]).to(device)
    return class_label

def load_classifier(net, classifier_path):
    model = torch.load(classifier_path)
    print(net.load_state_dict(model, strict=False))
    net.load_state_dict(model, strict=False)
    return net

def get_target(label, classes, to_tensor=False, to_cuda=False, whole=False, shuffle=True):
    try:
        label = label.to("cpu").detach().numpy()
    except AttributeError:
        pass
    if whole:
        target = np.tile(np.arange(len(classes)), (label.shape[0], 1))
    else:
        target = np.reshape(np.tile(np.arange(len(classes)), (label.shape[0], 1))[np.array(1-np.eye(len(classes))[label], dtype=bool)], (-1, len(classes)-1))
    if shuffle:
        for i in range(target.shape[0]):
            np.random.shuffle(target[i,:])
    if to_tensor:
        target = torch.Tensor(target)
    if to_cuda:
        target = torch.Tensor(target).to("cuda")
    return target

def get_random_dataset(dataset, num, random=True, random_seed=0):
    if random:
        index = np.random.choice(np.arange(len(dataset)), num, False)
    else:
        np.random.seed(random_seed)
        index = np.random.choice(np.arange(len(dataset)), num, False)

    for i in range(len(index)):
        data = dataset[i][0].unsqueeze(0)
        if i == 0:
            data_list = data
        else:
            data_list = torch.cat([data_list, data], dim=0)
    return data_list

class SRGAN_training():
    def __init__(self, net, opt, criterion, lbd, unrolled_k, device, ref_label,
                 batch_size=64, encoded_feature="latent", styleINdataset=True, ndim=8):
        self.G, self.D, self.E = net[0].to(device), net[1].to(device), net[2].to(device)
        self.optG, self.optD, self.optE = opt[0], opt[1], opt[2]
        self.scheG, self.scheD, self.scheE = None, None, None
        self.criterion, self.criterion_class = criterion
        self.lbd = lbd
        self.k = unrolled_k
        self.device = device
        self.ref_label = ref_label
        self.n_batch = batch_size
        self.encoded_feature = encoded_feature
        self.styleINdataset = styleINdataset
        self.ndim = ndim
        self.source_image = None
        self.target_image = None
        self.recon_image = None
        self.label = None
        self.c_rand = None
        self.enc_info = None
        self.target_cenc = None
        if lbd["hist"]>0:
            self.hi = histogram_imitation(device)
    
    def opt_sche_initialization(self, lr=[0.0001, 0.0001, 0.0001]):
        lr_G, lr_D, lr_E = lr
        if self.optG==None:
            self.optG = optim.Adam(self.G.parameters(), lr=lr_G, betas=(0.5, 0.999))
        self.scheG = optim.lr_scheduler.ExponentialLR(self.optG, gamma=0.95)
        if self.optD==None:
            self.optD = optim.Adam(self.D.parameters(), lr=lr_D, betas=(0.5, 0.999))
        self.scheD = optim.lr_scheduler.ExponentialLR(self.optD, gamma=0.95)
        if self.optE==None:
            self.optE = optim.Adam(self.E.parameters(), lr=lr_E, betas=(0.5, 0.999))
        self.scheE = optim.lr_scheduler.ExponentialLR(self.optE, gamma=0.95)
        return
        
    def G_transformation(self, target_label, source_image, encoder=False, ref_image=None):
        if encoder:
            latent, mu, logvar, class_output, attention = self.E(ref_image)
            info = [latent, mu, logvar, class_output, attention]
            if self.encoded_feature == "latent":
                latent_vector = latent
            elif self.encoded_feature == "mu":
                latent_vector = mu
                
        else:
            latent_vector = torch.randn(source_image.shape[0], self.ndim).to(self.device)
            info = latent_vector
            
        class_vector = class_encode(target_label, self.device, self.ref_label)
        class_vector = torch.cat([class_vector, latent_vector], 1)
        target_image = self.G(source_image, class_vector)
        
        return target_image, info
        
    def update_D(self):
        self.D.zero_grad()
        if self.styleINdataset:
            c, random = self.label["index"]
            self.target_image, [_,self.c_rand,_,_,_] = self.G_transformation(self.label["target"], self.source_image, True, self.source_image[c][random])
        else:
            self.target_image, self.c_rand = self.G_transformation(self.label["target"], self.source_image, False)
        
        errD = 0
        # real image
        output, output_class = self.D(self.source_image)
        errD_real = get_loss_D(output, 1., self.criterion, self.device)
        errD_class = get_domainloss_D(output_class, class_encode(self.label["source"], self.device, self.ref_label), self.criterion_class)
        errD += errD_real + errD_class*self.lbd["class"]

        # fake image
        output, output_class = self.D(self.target_image.detach())
        errD_fake = get_loss_D(output, 0., self.criterion, self.device)
        errD += errD_fake

        # gradient penalty
        if self.lbd["gp"] > 0:
            errD_gp = get_gradient_penalty(self.D, self.source_image, self.target_image.detach())
            errD += errD_gp * self.lbd["gp"]
            
        errD.backward()
        self.optD.step()
        return errD
    
    def update_GandE(self):
        self.G.zero_grad()
        self.E.zero_grad()

        errG = 0
        errE = 0
        errE_output = 0

        ## ordinary SingleGAN loss
        recon_image, source_enc_info = self.G_transformation(self.label["source"], self.target_image, True, self.source_image)
        output, output_class = self.D(self.target_image)
        errG_dis = get_loss_D(output, 1., self.criterion, self.device)
        errG_class = get_domainloss_D(output_class, class_encode(self.label["target"], self.device, self.ref_label), self.criterion_class)
        errG_cycle = torch.mean(torch.abs(self.source_image - recon_image))
        errG += errG_dis + errG_class*self.lbd["class"] + errG_cycle*self.lbd["cycle"]
        errE_output += errG_cycle * self.lbd["cycle"]
        
        ## multimodal transformation (KL): Conventional KL
        if self.lbd["KL"] > 0:
            _, mu, logvar, _, _ = source_enc_info
            errE_KL = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp_()) 
            errE += errE_KL*self.lbd["KL"]
            errE_output += errE_KL*self.lbd["KL"]
            
        ## Identity loss under source style condition
        if self.lbd["idt"] > 0:
            identity_image, _ = self.G_transformation(self.label["source"], self.source_image, True, self.source_image)
            errG_idt = torch.mean(torch.abs(self.source_image - identity_image))
            errG += errG_idt*self.lbd["idt"]
            errE_output += errG_idt*self.lbd["idt"]
            
        ## encoder attention loss
        if self.lbd["attention"] > 0:
            _, _, _, _, attention = source_enc_info
            errE_att = get_focus_loss(attention, lbd=1)
            errE += errE_att*self.lbd["attention"]
            errE_output += errE_att*self.lbd["attention"]
        
        ## encoder classification loss
        if self.lbd["class_enc"] > 0:
            _, _, _, output_class_enc, _ = source_enc_info
            errE_class_enc = get_domainloss_D([output_class_enc], class_encode(self.label["source"], self.device, self.ref_label), self.criterion_class)
            errE += errE_class_enc*self.lbd["class_enc"]
            errE_output += errE_class_enc*self.lbd["class_enc"]
            
        ## batch size KL
        if self.lbd["batch_KL"] > 0:
            _, mu, _, _, _ = source_enc_info
            var = torch.var(mu, dim=0)*self.n_batch/(self.n_batch-1)
            mean = torch.mean(mu, dim=0)
            errE_bKL = -0.5 * torch.sum(1 + torch.log(var) - mean**2 - var) 
            errE += errE_bKL*self.lbd["batch_KL"]
            errE_output += errE_bKL*self.lbd["batch_KL"]
            
        ## correlative loss
            if self.lbd["corr_enc"] > 0:
                errE_corr = corrcoef_loss(mu.T, self.device)
                errE += errE_corr*self.lbd["corr_enc"]
                errE_output += errE_corr*self.lbd["corr_enc"]
                
        ## histgram imitation loss
            if self.lbd["hist"] > 0:
                errE_hist = self.hi.loss(mu)
                errE += errE_hist*self.lbd["hist"]
                errE_output += errE_hist*self.lbd["hist"]
                
        ## Consistency Regularization
        if self.lbd["consis_reg"] > 0:
            augmented = get_augmented_image(self.source_image, augment)
            _, augmented_mu, _, _, _ = self.E(augmented)
            _, source_mu, _, _, _ = source_enc_info
            
            errE_consis_reg = torch.mean(torch.abs(augmented_mu - source_mu)*2)
            errE += errE_consis_reg*self.lbd["consis_reg"]
            errE_output += errE_consis_reg*self.lbd["consis_reg"]
        
        errG.backward(retain_graph=True)
        errE.backward(retain_graph=True)
        self.optG.step()
        self.optE.step()
        
        ########################### update exclusively G ###########################
        self.G.zero_grad()
        self.E.zero_grad()
        
        errG_ex = 0
        ## multimodal transformation (regression loss)
        _, target_cenc, _, _, _ = self.E(self.target_image)
        errG_reg = torch.mean(torch.abs(self.c_rand - target_cenc))
        errG_ex += errG_reg * self.lbd["reg"]
        
        ## multimodal transformation (regression loss for identity images)
        if self.lbd["idt_reg"]*self.lbd["idt"] > 0:
            errG_idt_reg = 0
            
            ## random condition
            idt_random_image, [_,source_c_rand,_,_,_] = self.G_transformation(self.label["source"], self.source_image, True, self.source_image)
            _, idt_cenc_rand, _, _, _ = self.E(idt_random_image)
            errG_idt_reg += torch.mean(torch.abs(source_c_rand - idt_cenc_rand))
            errG_ex += errG_idt_reg * self.lbd["idt_reg"] * (self.lbd["idt"]/self.lbd["cycle"])
            
        errG_ex.backward()
        self.optG.step()
        
        errG += errG_ex
        
        return [errG, errE_output]
    
    def UnrolledUpdate(self):
        for i in range(self.k):

            # update D
            errD = self.update_D()
            if i==0:
                paramD = self.D.state_dict()
                errorD = errD

        # update G
        errorG, errorE = self.update_GandE()

        self.D.load_state_dict(paramD)
        return [errorG, errorD, errorE]
        
    def train(self, source_image, label):
        self.source_image = source_image
        self.label = label
        error = self.UnrolledUpdate()
        return error



def get_output_and_plot(sg, dataset, index, class_info, random_sample_num=5, styleINdataset=False, device="cuda"):
    classes, label_discription = class_info
    data = dataset[index]
    fixed_source_image = data[0].view(1, 3, 128, 128).to(device)
    fixed_source_label = torch.tensor(data[1]).view(1,)
    
    # get target label
    fixed_target_label = torch.tensor(get_target(fixed_source_label, classes, whole=False, shuffle=False))
    target_label = fixed_target_label[:,0:1]
    
    # get target image under source condition
    image, _ = sg.G_transformation(target_label, fixed_source_image, True, fixed_source_image)
    fixed_target_image = cuda2cpu(image)
    
    # get target image by several latent codes
    if styleINdataset:
        ref_image = get_random_dataset(dataset, random_sample_num).to(device)
        image, _ = sg.G_transformation(target_label.repeat(1,random_sample_num), fixed_source_image.repeat(random_sample_num,1,1,1), True, ref_image)
    else:
        image, _ = sg.G_transformation(target_label.repeat(1,random_sample_num), fixed_source_image.repeat(random_sample_num,1,1,1), False)
    target_image_list = cuda2cpu(image)
    
    # get reconstructed image under source condition
    image, _ = sg.G_transformation(fixed_source_label,target_image_list[0:1].to(device), True, fixed_source_image)
    fixed_recon_image = cuda2cpu(image)

    # get identity image under source condition
    image, _ = sg.G_transformation(fixed_source_label, fixed_source_image, True, fixed_source_image)
    fixed_identity_image = cuda2cpu(image)
    
    # get transformed image under class condition
    image, _ = sg.G_transformation(fixed_target_label, fixed_source_image.repeat(len(classes)-1,1,1,1), False)
    trans_image_list = cuda2cpu(image)
    
    # get reconstructed image by several latent code
    if styleINdataset:
        ref_image = get_random_dataset(dataset, random_sample_num).to(device)
        image, _ = sg.G_transformation(fixed_source_label.repeat(random_sample_num), target_image_list[0:1].repeat(random_sample_num,1,1,1).to(device), True, ref_image)
    else:
        image, _ = sg.G_transformation(fixed_source_label.repeat(random_sample_num), target_image_list[0:1].repeat(random_sample_num,1,1,1).to(device), False)
    recon_image_list = cuda2cpu(image)
    
    # get idt image by several latent code
    if styleINdataset:
        ref_image = get_random_dataset(dataset, random_sample_num).to(device)
        image, _ = sg.G_transformation(fixed_source_label.repeat(random_sample_num), fixed_source_image.repeat(random_sample_num,1,1,1), True, ref_image)
    else:
        image, _ = sg.G_transformation(fixed_source_label.repeat(random_sample_num), fixed_source_image.repeat(random_sample_num,1,1,1), False)
    idt_image_list = cuda2cpu(image)
    
    length = random_sample_num + 1
    width = 4
    fig = plt.figure(figsize=(5*width, 5*length))

    index = 1
    ax = fig.add_subplot(length, width, index)
    ax.imshow(image_from_output(fixed_source_image)[0])
    ax.set_title("source")
    index = 2
    ax = fig.add_subplot(length, width, index)
    ax.imshow(image_from_output(fixed_target_image)[0])
    ax.set_title("target by source condition")
    index = 3
    ax = fig.add_subplot(length, width, index)
    ax.imshow(image_from_output(fixed_recon_image)[0])
    ax.set_title("recon by source condition")
    index = 4
    ax = fig.add_subplot(length, width, index)
    ax.imshow(image_from_output(fixed_identity_image)[0])
    ax.set_title("identity image by source condition")

    for i in range(len(classes)-1):
        index = 4*(i+1)+1
        ax = fig.add_subplot(length, width, index)
        ax.imshow(image_from_output(trans_image_list[i:i+1])[0])
        ax.set_title(label_discription[fixed_target_label[0][i]])

    for i in range(random_sample_num):
        index = 4*(i+1)+2
        ax = fig.add_subplot(length, width, index)
        ax.imshow(image_from_output(target_image_list[i:i+1])[0])
        ax.set_title("target by random latent")

    for i in range(random_sample_num):
        index = 4*(i+1)+3
        ax = fig.add_subplot(length, width, index)
        ax.imshow(image_from_output(recon_image_list[i:i+1])[0])
        ax.set_title("recon by random latent")

    for i in range(random_sample_num):
        index = 4*(i+1)+4
        ax = fig.add_subplot(length, width, index)
        ax.imshow(image_from_output(idt_image_list[i:i+1])[0])
        ax.set_title("idt by random latent")
        
    return fig

def dic_init(get_edge=False):
    data = {}
    data["source"] = []
    data["target"] = []
    data["recon"] = []
    label = {}
    label["source"] = []
    label["target"] = []
    return data, label

def get_samples(netG, netE, dataset, index, latent=None, classes=tuple(range(4)), ref_label=None, 
                       ndim=8, scale=1, image_type="pil", batch=32, device="cuda", conventional_E=False):
    """
    get samples of the outputs in multimodal-SingleGAN, which is diversified the output with the latent code
    
    Parameters
    ------------
    netG : PyTorch model
        the generator of the SingleGAN
        
    dataset : Dataset in PyTorch
        dataset which is used in the inference task
        
    index : int
        the index which indicates data location in the dataset
        
    target_label : ndarray
        the index which indicates data location in the dataset
        
    latent : None or ndarray, shape=(sample_num, latent_dim)
        latent indicates the "style" of generated data
        
    classes : tuple
        the class list(tuple)
        
    ref_label : ndarray
        continuous class label (relational label or moving label)
        
    label_type : 'target_only', 'concatenation', or 'substruction'
        the target label for transformation
        'target_only' -> only target label
        'concatenation' -> concatenation of the target label and the source label
        'substruction' -> substruction of the target label and the source label
        
    transformation_type : 'normal', 'edge', or 'edge_emphasis'
        the represenation which is used for the input of the E
        
    ndim : int 
        the dimension of the latent code
    
    Returns
    ----------
    data : dic
        dictionary of whole data
        
    label : dic
        dictionary of whole label
        
    """
    
    fixed_source_image = dataset[index][0].view(1, 3, 128, 128).to(device)
    fixed_source_label = torch.tensor([dataset[index][1]])
    
    data, label = dic_init(False)
    label["source"] = cuda2numpy(fixed_source_label)
    if image_type=="pil":
        data["source"] = image_from_output(fixed_source_image)[0]
    elif image_type=="tensor":
        data["source"] = cuda2cpu(fixed_source_image)[0]

    netG.eval()
    netE.eval()
    if type(latent)==list:
        latent_list = []
        for v in latent:
            latent_list.append(torch.tensor(v, dtype=torch.float32).to(device))
    else:
        latent = torch.tensor(latent, dtype=torch.float32).to(device)
        latent_list = [latent]*len(classes)
        
    num = latent_list[0].shape[0]
    label["latent"] = {}
    data["target"] = {}
    for target_label in classes:
        label["latent"][target_label] = []
        data["target"][target_label] = []
        for itr in range(num//batch+int(bool(num-batch*(num//batch)))):
            target_label = torch.tensor([target_label])
            class_vector = class_encode(target_label, device, ref_label)
            latent_ = latent_list[target_label][itr*batch:(itr+1)*batch, :]
            class_vector = torch.cat([class_vector.repeat(latent_.shape[0],1), latent_], 1)
            target_image = netG(fixed_source_image.repeat(latent_.shape[0],1,1,1), class_vector)
            if conventional_E:
                class_vector = class_encode(target_label, device, ref_label)
                _, mu, _ = netE(target_image, class_vector.repeat(latent_.shape[0],1))
            else:
                _, mu, _, _, _ = netE(target_image)
            target_label = target_label.numpy()[0]
            label["latent"][target_label].append(cuda2numpy(mu))
            if image_type=="pil":
                data["target"][target_label] += image_from_output(target_image)
            elif image_type=="tensor":
                target_image = cuda2numpy(target_image)
                if itr == 0:
                    data["target"][target_label] = target_image
                else:
                    data["target"][target_label] = np.concatenate([data["target"][target_label], target_image], axis=0)
                    
        if image_type=="tensor":
            data["target"][target_label] = torch.Tensor(np.array(data["target"][target_label]))
            
    if image_type=="tensor":
        data["source"] = torch.Tensor(data["source"]).unsqueeze(0)
    return data, label

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


##################################### Loss ###########################################

def get_loss_D(outputs, target, criterion, device="cuda"):
    loss = 0.0
    for output in outputs:
        targets = torch.full((output.shape), target, device=device)
        loss += criterion(output, targets)
    return loss / len(outputs)

def get_domainloss_D(outputs_class, true_label, criterion_class):
    loss = 0.0
    for output_class in outputs_class:
        loss += criterion_class(output_class, true_label)
    return loss / len(outputs_class)

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.view(-1,1).expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)
    return c

def corrcoef_loss(m, device):
    num = m.shape[0]
    coco = corrcoef(m)
    lossmatrix = torch.abs((coco - torch.eye(num, device=device)))
    return torch.sum(lossmatrix) / (num*(num-1))

def get_focus_loss(focus_map, eps=0.01, lbd=0.2):
    minimizing = torch.mean(focus_map**2)
    segmenting = 1/torch.mean((focus_map-0.5)**2+eps)
    return lbd*minimizing + (1-lbd)*segmenting

### https://github.com/yunjey/stargan/blob/master/solver.py ###
def gradient_penalty_loss(y, x, device):
    grad_output = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, retain_graph=True,
                               create_graph=True, only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def get_gradient_penalty(netD, real_image, target_image, device):
    alpha = torch.rand(real_image.size(0), 1, 1, 1).to(device)
    x_hat = (alpha * real_image + (1-alpha) * target_image).requires_grad_(True)
    outputs, _ = netD(x_hat)
    loss = 0.0
    for output in outputs:
        loss += gradient_penalty_loss(output, x_hat, device)
    return loss / len(outputs)

## https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3 ##
class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma, device):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers.requires_grad = True

    def forward(self, x):
        self.centers = self.centers.to(x.device)
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x
    
class histogram_imitation():
    def __init__(self, device, bins=50, range_max=10, sigma=0.2, target_num=100000):
        self.device = device
        self.gausshist = GaussianHistogram(bins=bins, min=-range_max, max=range_max, sigma=sigma, device=device)
        target = torch.randn(target_num,1)
        gausshist_value = self.gausshist(target[:,0])
        self.target = (gausshist_value / gausshist_value.sum() + 1e-8).to(device)
    
    def loss(self, x):
        hist_loss = 0
        for i in range(x.shape[1]):
            gausshist_value = self.gausshist(x[:,i])
            input = gausshist_value / gausshist_value.sum() + 1e-8
            hist_loss += F.kl_div(input.log(), self.target, None, None, "sum")
        return hist_loss
    
class ToPIL(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return image_from_output(torch.reshape(img, (1,img.shape[0],img.shape[1],img.shape[2])))[0]

    def __repr__(self):
        return self.__class__.__name__
    
class MinMax(object):
    def __init__(self, mean0=True):
        self.mean0 = mean0
        pass
    def __call__(self, img):
        return torch.Tensor(min_max(cuda2numpy(img), mean0=self.mean0))
    def __repr__(self):
        return self.__class__.__name__

    
augment = transforms.Compose([
    ToPIL(),
    transforms.RandomAffine(degrees=0, translate=(0.15,0.05)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    MinMax(True),
])

def get_augmented_image(data, transform):
    for i in range(data.shape[0]):
        x = data[i]
        image = transform(x).view(1,3,128,128)
        if i == 0:
            new = image
        else:
            new = torch.cat([new, image])
    return new