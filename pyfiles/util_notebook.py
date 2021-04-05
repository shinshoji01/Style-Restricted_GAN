import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import pandas as pd

from util import *

def get_adjustable_parameters(notebook_no=1):
    if notebook_no == 1:
        models = [
            ["conventionalKL", 1, 0],
            ["preposedKL", 1, 0],
            ["preposedKL", 5, 0.5],
        ]
        columns=["restriction_type", "unrolled_k", "idt_reg"]
        
    elif notebook_no == 2: 
        return None
    elif notebook_no == 3: 
        return None
    elif notebook_no == 5:
        return None
    
    return pd.DataFrame(np.array(models), columns=columns)

class SingleGAN_training():
    def __init__(self, net, criterion, lbd, unrolled_k, device, ref_label, n_batch, ndim, classes):
        self.G, self.D, self.E = net[0], net[1], net[2]
        self.optG, self.optD, self.optE = None, None, None
        self.scheG, self.scheD, self.scheE = None, None, None
        self.criterion = criterion
        self.lbd = lbd
        self.k = unrolled_k
        self.device = device
        self.ref_label = ref_label
        self.n_batch = n_batch
        self.ndim = ndim
        self.classes = classes
        self.source_image = None
        self.target_image = None
        self.label = None
        self.c_rand = None
        self.enc_info = None
        self.target_cenc = None
        if lbd["hist"]>0:
            self.hi = histogram_imitation(device)
    
    def opt_sche_initialization(self, lr=[0.0001, 0.0001, 0.0001]):
        lr_G, lr_D, lr_E = lr
        self.optG = optim.Adam(self.G.parameters(), lr=lr_G, betas=(0.5, 0.999))
        self.scheG = optim.lr_scheduler.ExponentialLR(self.optG, gamma=0.95)
        self.optD = []
        self.scheD = []
        for i in self.classes:
            self.optD.append(optim.Adam(self.D[i].parameters(), lr=lr_D, betas=(0.5, 0.999)))
            self.scheD.append(optim.lr_scheduler.ExponentialLR(self.optD[i], gamma=0.95))
        self.optE = optim.Adam(self.E.parameters(), lr=lr_E, betas=(0.5, 0.999))
        self.scheE = optim.lr_scheduler.ExponentialLR(self.optE, gamma=0.95)
        return
        
    def G_transformation(self, target_label, source_image, encoder=False, ref_image=None):
        if encoder:
            class_vector = class_encode(target_label, self.device, self.ref_label)
            latent, mu, logvar = self.E(ref_image, class_vector)
            info = [latent, mu, logvar]
            latent_vector = latent
        else:
            latent_vector = torch.randn(source_image.shape[0], self.ndim).to(self.device)
            info = latent_vector
            
        class_vector = class_encode(target_label, self.device, self.ref_label)
        class_vector = torch.cat([class_vector, latent_vector], 1)
        target_image = self.G(source_image, class_vector)
        
        return target_image, info
        
    def update_D(self):
        self.target_image, self.c_rand = self.G_transformation(self.label["target"], self.source_image, False)
        
        all_errD = 0
        for i in self.classes:
            self.D[i].zero_grad()
            errD = 0
            real_image = self.source_image[self.label["source"]==i]
            if real_image.shape[0]!=0:
                output = self.D[i](real_image)
                errD_real = get_loss_D(output, 1., self.criterion, self.device)
            else:
                errD_real = 0
            errD += errD_real
            
            fake_image = self.target_image[self.label["target"]==i].detach()
            if fake_image.shape[0]!=0:
                output = self.D[i](fake_image)
                errD_fake = get_loss_D(output, 0., self.criterion, self.device)
            else:
                errD_fake = 0
            errD += errD_fake
            errD.backward()
            self.optD[i].step()
            all_errD += errD/len(self.classes)
            
        return all_errD
    
    def update_GandE(self):
        self.G.zero_grad()
        self.E.zero_grad()

        errG = 0
        errE = 0
        errE_output = 0

        ## ordinary SingleGAN loss
        recon_image, source_enc_info = self.G_transformation(self.label["source"], self.target_image, True, self.source_image)
        for i in self.classes:
            fake_image = self.target_image[self.label["target"]==i]
            if fake_image.shape[0]!=0:
                output = self.D[i](fake_image)
                errG_dis = get_loss_D(output, 1., self.criterion, self.device)
            else:
                errG_dis = 0
            errG += errG_dis/len(self.classes)
        errG_cycle = torch.mean(torch.abs(self.source_image - recon_image))
        errG += errG_cycle*self.lbd["cycle"]
        errE_output += errG_cycle * self.lbd["cycle"]
        
        ## multimodal transformation (KL)
        _, mu, logvar = source_enc_info
        errE_KL = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp_()) 
        errE += errE_KL*self.lbd["KL"]
        errE_output += errE_KL*self.lbd["KL"]
        
        ## Identity loss under source style condition
        if self.lbd["idt"] > 0:
            identity_image, _ = self.G_transformation(self.label["source"], self.source_image, True, self.source_image)
            errG_idt = torch.mean(torch.abs(self.source_image - identity_image))
            errG += errG_idt*self.lbd["idt"]
            errE_output += errG_idt*self.lbd["idt"]
            
        ## batch size KL
        if self.lbd["batch_KL"] > 0:
            _, mu, _ = source_enc_info
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
            
        errG.backward(retain_graph=True)
        errE.backward(retain_graph=True)
        self.optG.step()
        self.optE.step()
        
        ## update exclusively G
        self.G.zero_grad()
        
        ## multimodal transformation (regression loss)
        errG_ex = 0
        
        class_vector = class_encode(self.label["target"], self.device, self.ref_label)
        _, target_cenc, _ = self.E(self.target_image, class_vector)
        errG_reg = torch.mean(torch.abs(self.c_rand - target_cenc))
        errG_ex += errG_reg * self.lbd["reg"]
        
        ## multimodal transformation (regression loss for identity images)
        if self.lbd["idt_reg"]*self.lbd["idt"] > 0:
            errG_idt_reg = 0
            
            ## random condition
            idt_random_image, source_c_rand = self.G_transformation(self.label["source"], self.source_image, False)
            class_vector = class_encode(self.label["source"], self.device, self.ref_label)
            _, idt_cenc_rand, _ = self.E(idt_random_image, class_vector)
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
                paramD = []
                for j in self.classes:
                    paramD.append(self.D[j].state_dict())
                errorD = errD

        # update G and E
        errorG, errorE = self.update_GandE()

        for j in self.classes:
            self.D[j].load_state_dict(paramD[j])
        return [errorG, errorD, errorE]
        
    def train(self, source_image, label):
        self.source_image = source_image
        self.label = label
        error = self.UnrolledUpdate()
        return error
    
class SingleGAN_training_singleD():
    def __init__(self, net, opt, criterion, lbd, unrolled_k, device, ref_label, ndim,
                 batch_size=64, encoded_feature="latent"):
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
            class_vector = class_encode(target_label, self.device, self.ref_label)
            latent, mu, logvar = self.E(ref_image, class_vector)
            info = [latent, mu, logvar]
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
            _, mu, logvar = source_enc_info
            errE_KL = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp_()) 
            errE += errE_KL*self.lbd["KL"]
            errE_output += errE_KL*self.lbd["KL"]
            
        ## Identity loss under source style condition
        if self.lbd["idt"] > 0:
            identity_image, _ = self.G_transformation(self.label["source"], self.source_image, True, self.source_image)
            errG_idt = torch.mean(torch.abs(self.source_image - identity_image))
            errG += errG_idt*self.lbd["idt"]
            errE_output += errG_idt*self.lbd["idt"]
            
        ## batch size KL
        if self.lbd["batch_KL"] > 0:
            _, mu, _ = source_enc_info
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
                
        errG.backward(retain_graph=True)
        errE.backward(retain_graph=True)
        self.optG.step()
        self.optE.step()
        
        ########################### update exclusively G ###########################
        self.G.zero_grad()
        self.E.zero_grad()
        
        errG_ex = 0
        ## multimodal transformation (regression loss)
        class_vector = class_encode(self.label["target"], self.device, self.ref_label)
        _, target_cenc, _ = self.E(self.target_image, class_vector)
        errG_reg = torch.mean(torch.abs(self.c_rand - target_cenc))
        errG_ex += errG_reg * self.lbd["reg"]
        
        ## multimodal transformation (regression loss for identity images)
        if self.lbd["idt_reg"]*self.lbd["idt"] > 0:
            errG_idt_reg = 0
            
            ## random condition
            idt_random_image, source_c_rand = self.G_transformation(self.label["source"], self.source_image, False)
            class_vector = class_encode(self.label["source"], self.device, self.ref_label)
            _, idt_cenc_rand, _ = self.E(idt_random_image, class_vector)
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