import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import pickle

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

# def standardize_torch(x):
#     xmean = torch.mean(x, dim=(1,2,3), keepdim=True)
#     xstd = torch.std(x, dim=(1,2,3), keepdim=True)
#     new_x = (x-xmean)/xstd
#     return new_x 

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

def plot_correlation_matrix(cm, save=False, save_dir="", save_name=""):
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(round(cm[i, j], 4)),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                fontsize=12)
    plt.tight_layout()
    if save:
        save_path = save_dir + save_name
        plt.savefig(fname=save_path, format="png", bbox_inches="tight")
    plt.show()
    return

############ https://www.kaggle.com/grfiv4/plot-a-confusion-matrix #############
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


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


## https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3 ##
class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
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
        self.gausshist = GaussianHistogram(bins=bins, min=-range_max, max=range_max, sigma=sigma)
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
    
