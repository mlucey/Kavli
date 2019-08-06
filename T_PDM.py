import math
import numpy as np
import string
from datetime import datetime
import os
from astropy.table import Table
import matplotlib.pyplot as plt;
import random


import matplotlib.axes as axes;
from matplotlib.patches import Ellipse
import seaborn as sns;
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

print(20*'=~')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
print(20*'=~')

def evaluate(tensors):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
      `namedtuple` or combinations thereof.

    Returns:
      ndarrays: Object with same structure as `tensors` except with `Tensor` or
        `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)

def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
  """Plots the mixture of Normal models to axis=ax comp=True plots all
  components of mixture model
  """
  # x = np.linspace(-10.5, 10.5, 250)
  x = np.linspace(-0.1, 1.1, 250)
  final = np.zeros_like(x)
  for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
    temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
    final = final + temp
    if comp:
      ax.plot(x, temp, label='Normal ' + str(i))
  ax.plot(x, final, label='Mixture of Normals ' + label)
  ax.legend(fontsize=13)
  return final

def GenData_lamost(fileIn = 'lamost_wise_gaia_PS1_2mass.fits'):
    filts = ['Jmag', 'Hmag', 'Kmag', 'phot_g_mean_mag', 'phot_bp_mean_mag',
              'phot_rp_mean_mag', 'gmag', 'rmag', 'imag',
              'zmag', 'ymag', 'W1mag', 'W2mag']  # train filters
    filts_target = ['DeltaP', 'Deltanu'] #
    # filts_target = ['Teff', 'log_g_'] #
    params = ['teff','logg']

    al = Table.read(fileIn)
    inds = np.where( ~(np.isnan(al['Jmag'])) & ~(np.isnan(al['Hmag'])) & ~(np.isnan(al['Kmag'])) & ~(np.isnan(al['phot_g_mean_mag'])) & ~(np.isnan(al['phot_rp_mean_mag'])) & ~(np.isnan(al['phot_bp_mean_mag']))& ~(np.isnan(al['gmag'])) & ~(np.isnan(al['rmag'])) & ~(np.isnan(al['imag'])) & ~(np.isnan(al['zmag'])) & ~(np.isnan(al['ymag'])) & ~(np.isnan(al['W1mag'])) & ~(np.isnan(al['W2mag'])) )[0]
    #inds = np.where( (al['Jmag']>-1000) & (al['Hmag']>-1000) & (al['Kmag']>-1000) & (al['phot_g_mean_mag']>-1000) & (al['phot_rp_mean_mag']>-1000) &

                     # (al['gmag']>-1000) & (al['rmag']>-1000) & (al['imag']>-1000) & (al['zmag']>-1000) & (al['ymag']>-1000) & (al['W1mag']>-1000) & (al['W2mag']>-1000))[0]
    print(len(al))
    print(inds)
    gps = al[inds]
    x_train_all_1 = np.array([gps[filts[0]], gps[filts[1]], gps[filts[2]], gps[filts[3]], gps[filts[4]], gps[filts[5]], gps[filts[6]], gps[filts[7]], gps[filts[8]], gps[filts[9]], gps[filts[10]], gps[filts[11]], gps[filts[12]]]).T

    #inds = np.where((gps['DeltaP']>-1000) & (gps['Deltanu']>-1000))[0]
    #inds = np.where((gps['DeltaP']>-1000) & (gps['Deltanu']>-1000)& ~(np.logical_and(gps['DeltaP']>=100, gps['DeltaP']<=250)))[0]
    #inds = np.where((gps['teff']>-1000) & (gps['logg']>-1000) )[0]
    inds = np.where(~(np.isnan(gps['teff'])) & ~(np.isnan(gps['logg'])) & (gps['logg']<3.5) )[0]

    al =  gps[inds]

    #
    # ## RC pristine
    # pristine_inds = np.where(al['Class'] != 'RC_Pristine')
    # al =  al[pristine_inds]

    np.random.seed(123)

    x_train_all = np.array([al[filts[0]], al[filts[1]], al[filts[2]], al[filts[3]], al[filts[4]], al[filts[5]], al[filts[6]], al[filts[7]], al[filts[8]], al[filts[9]], al[filts[10]], al[filts[11]], al[filts[12]]]).T
    y_train_all = np.array(al[params[0]]).T
    params = np.array([al[params[0]],al[params[1]]]).T
    #classy = np.zeros(len(al))
    #inds  = np.where((al['Class'] == 'RC') | (al['Class'] == 'RC_Pristine'))[0]
    #classy[inds] = 1
    #print(len(inds))
    #alset = set(range(len(al)))
    #cset = set(inds)
    #no_rc = np.array(list(alset-cset),dtype='int')[0:len(inds)+1]
    print(x_train_all)
    #x_rc = x_train_all[inds]
    #y_rc = y_train_all[inds]

    #x_train_all = np.concatenate([x_rc,x_train_all[no_rc]],axis=0)
    #y_train_all = np.concatenate([y_rc,y_train_all[no_rc]],axis=0)
    #classy = np.concatenate([classy[inds],classy[no_rc]],axis=None)



    xmax = np.max(x_train_all, axis = None)
    xmin = np.min(x_train_all, axis = None)
    print(xmax)
    print(xmin)
    x_train_rescaled = (x_train_all - xmin) / (xmax - xmin)
    print(x_train_rescaled)
    #import pdb ; pdb.set_trace()
    ymax = np.max(y_train_all, axis = 0)
    ymin = np.min(y_train_all, axis = 0)

    y_train_rescaled = (y_train_all - ymin) / (ymax - ymin)

    TrainshuffleOrder = np.arange(x_train_rescaled.shape[0])
    np.random.shuffle(TrainshuffleOrder)

    x_train_shuffled = x_train_rescaled[TrainshuffleOrder]
    y_train_shuffled = y_train_rescaled[TrainshuffleOrder]
    #classy_shuffled = classy[TrainshuffleOrder]
    params_shuffled = params[TrainshuffleOrder]

    X_train = x_train_shuffled[:num_train]  # color mag
    X_test = x_train_shuffled[num_train + 1: num_train + num_test]  # color mag


    y_train = y_train_shuffled[:num_train]  # spec z
    y_test = y_train_shuffled[num_train + 1: num_train + num_test]  # spec z
    #classy = classy_shuffled[:num_train]
    params = params_shuffled[:num_train]
    # return (X_train[:, 2:8], y_train, X_test[:, 2:8], y_test)
    # return (X_train[:, :5], y_train[:, 0], X_test[:, :5], y_test[:, 0])
    #print(len(np.where(classy==1)[0]))
    #print(len(classy))
    #print(float(len(np.where(classy==1)[0]))/float(len(classy)))

    return X_train, y_train, X_test, y_test, params, ymax, ymin, xmax, xmin

 # fileIn='lamost_rc_wise_gaia_PS1_2mass.fits'

 # X_train, y_train, X_test, y_test = GenData_skymapper(fileIn = 'apogee_rc_skymapper.fits')



def neural_network_mod():
    """
    loc, scale, logits = NN(x; theta)

    Args:
      X: Input Tensor containing input data for the MDN
    Returns:
      locs: The means of the normal distributions that our data is divided into.
      scales: The scales of the normal distributions that our data is divided
        into.
      logits: The probabilities of ou categorical distribution that decides
        which normal distribution our data points most probably belong to.
    """
    X = tf.placeholder(tf.float64,name='X',shape=(None,D))
    # 2 hidden layers with 15 hidden units
    net = tf.layers.dense(X, 32, activation=tf.nn.relu)
    net = tf.layers.dense(net, 16, activation=tf.nn.relu)
    net = tf.layers.dense(net, 8, activation=tf.nn.relu)
    locs = tf.layers.dense(net, K, activation=None)
    scales = tf.layers.dense(net, K, activation=tf.exp)
    logits = tf.layers.dense(net, K, activation=None)
    outdict= {'locs':locs, 'scales':scales, 'logits':logits}
    hub.add_signature(inputs=X,outputs=outdict)

    return locs, scales, logits


def mixture_model(X,Y,learning_rate=1e-3,decay_rate=.95,step=1000):
    dict = neural_network(tf.convert_to_tensor(X),as_dict=True)
    locs = dict['locs'] ; scales = dict['scales'] ; logits = dict['logits']
    cat = tfd.Categorical(logits=logits)
    components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                  in zip(tf.unstack(tf.transpose(locs)),
                         tf.unstack(tf.transpose(scales)))]

    y = tfd.Mixture(cat=cat, components=components)
    #define loss function
    log_likelihood = y.log_prob(Y)
    # log_likelihood = -tf.reduce_sum(log_likelihood/(1. + y_train)**2 )
    log_likelihood = -tf.reduce_sum(log_likelihood )
    global_step = tf.Variable(0, trainable=False)
    decayed_lr = tf.train.exponential_decay(learning_rate,
                                        global_step, step,
                                        decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_lr)
    train_op = optimizer.minimize(log_likelihood)
    evaluate(tf.global_variables_initializer())
    return log_likelihood, train_op, logits, locs, scales

def train(log_likelihood,train_op,n_epoch):
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    for i in range(n_epoch):
        _, loss_value = evaluate([train_op, log_likelihood])
        train_loss[i] = loss_value
    plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train Loss')
    return train_loss


def get_predictions(logits,locs,scales):
    pred_weights, pred_means, pred_std = evaluate([tf.nn.softmax(logits), locs, scales])
    return pred_weights, pred_means, pred_std

def plot_pdfs(pred_means,pred_weights,pred_std,num=10):
    obj = [random.randint(0,num_train-1) for x in range(num)]
    #obj = [93, 402, 120,789,231,4,985]
    print(obj)
    fig, axes = plt.subplots(nrows=num, ncols=1, sharex = True, figsize=(8, 7))
    allfs = []
    for i in range(len(obj)):
        fs = plot_normal_mix(pred_weights[obj][i], pred_means[obj][i],
                    pred_std[obj][i], axes[i], comp=False)
        allfs.append(fs)
        axes[i].axvline(x=y_train[obj][i], color='black', alpha=0.5)
        axes[i].text(0.3, 4.0, 'ID: ' +str(obj[i]), horizontalalignment='center',
        verticalalignment='center')

    plt.xlabel(r' rescaled[$z_{pred}]$', fontsize = 19)
    plt.show()

def plot_pred_mean(pred_means,pred_weights,pred_std,ymax,ymin,y_train,select='no'):
    y_pred = np.sum(pred_means*pred_weights, axis = 1)
    y_pred_std = np.sum(pred_std*pred_weights, axis = 1)

    plt.figure(22, figsize=(9,8))

    #ymax=1
    #ymin=0
    if select == 'yes':
        y_pred = y_pred[obj]
        y_train = y_train[obj]
        y_pred_std = y_pred_std[obj]

    # plt.scatter(y_test, y_pred, facecolors='k', s = 1)

    plt.errorbar( (ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)

    #switched
    #plt.errorbar(  (ymax - ymin)*(y_pred)+ymin, (ymax - ymin)*(y_train)+ymin, yerr= (ymax - ymin)*(y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)

    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*( y_train)+ymin, 'k')

    plt.ylabel(r'$z_{pred}$', fontsize = 19)
    plt.xlabel(r'$z_{true}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('weight x mean')
    plt.tight_layout()
    plt.show()

def plot_pred_peak(pred_means,pred_weights,pred_std,ymax,ymin,y_train,select='no'):
    def peak(weight,sigma):
        return weight/np.sqrt(2*np.pi*sigma**2)

    peak_max = np.argmax(peak(pred_weights,pred_std),axis=1)
    y_pred = np.array([pred_means[i,peak_max[i]] for i in range(len(y_train))])
    y_pred_std = np.array([pred_std[i,peak_max[i]] for i in range(len(y_train))])
    plt.figure(24, figsize=(9, 8))
    if select == 'yes':
        y_pred = y_pred[obj]
        y_train = y_train[obj]
        y_pred_std = y_pred_std[obj]
    # plt.scatter(y_test, y_pred, facecolors='k', s = 1)
    plt.errorbar((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(
      y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)
    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_test)+ymin, (ymax - ymin)*(y_test)+ymin, 'k')
    plt.ylabel(r'$z_{pred}$', fontsize = 19)
    plt.xlabel(r'$z_{true}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('highest peak')
    plt.tight_layout()
    plt.show()

def plot_pred_weight(pred_means,pred_weights,pred_std,ymax,ymin,y_train,select='no'):
    weight_max = np.argmax(pred_weights, axis = 1)  ## argmax or max???

    y_pred = np.array([pred_means[i,weight_max[i]] for i in range(len(y_train))])
    y_pred_std = np.array([pred_std[i,weight_max[i]] for i in range(len(y_train))])

    plt.figure(29, figsize=(9, 8))
    if select == 'yes':
        y_pred = y_pred[obj]
        y_train = y_train[obj]
        y_pred_std = y_pred_std[obj]

    # plt.scatter(y_test, y_pred, facecolors='k', s = 1)
    plt.errorbar((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(
      y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)

    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_test)+ymin, (ymax - ymin)*(y_test)+ymin, 'k')
    plt.ylabel(r'$z_{pred}$', fontsize = 19)
    plt.xlabel(r'$z_{true}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('highest weight')
    plt.tight_layout()
    plt.show()

def select_rc(pred_means,pred_weights,pred_std,ymax,ymin,y_train,cmethod='peak',cut=200):
    print(cmethod)
    rcs =np.zeros(len(y_train))
    if cmethod == 'peak':
        def peak(weight,sigma):
            return weight/np.sqrt(2*np.pi*sigma**2)

        peak_max = np.argmax(peak(pred_weights,pred_std),axis=1)
        y_pred = np.array([pred_means[i,peak_max[i]] for i in range(len(y_train))])
        y_pred_std = np.array([pred_std[i,peak_max[i]] for i in range(len(y_train))])
    if cmethod == 'weight':
        weight_max = np.argmax(pred_weights, axis = 1)  ## argmax or max???
        #print(weight_max.shape)
        y_pred = np.array([pred_means[i,weight_max[i]] for i in range(len(y_train))])
        y_pred_std = np.array([pred_std[i,weight_max[i]] for i in range(len(y_train))])
    y_pred = (ymax - ymin)*(y_pred)+ymin
    y_pred_std = (ymax - ymin)*(y_pred_std)
    y_train = (ymax - ymin)*(y_train)+ymin

    yes = np.where(y_pred>cut)[0]
    rcs[yes] =1
    #print(rcs.shape)
    return rcs

def contamination(pred_means,pred_weights,pred_std,ymax,ymin,y_train,cut=200):
    rcs = select_rc(pred_means,pred_weights,pred_std,ymax,ymin,y_train,cmethod='peak')
    trcs = np.zeros(len(y_train))
    y_train = (ymax - ymin)*(y_train)+ymin
    tyes = np.where(y_train>cut)[0]
    trcs[tyes] = 1
    false_positive_p = np.where((rcs==1) & (trcs==0))[0]
    true_positive_p = np.where((rcs==1) & (trcs==1))[0]
    positive_p = np.where(rcs==1)[0]
    tpositive = np.where(trcs==1)[0]
    rcs = select_rc(pred_means,pred_weights,pred_std,ymax,ymin,y_train,cmethod='weight')
    #print(rcs)
    false_positive_w = np.where((rcs==1) & (trcs==0))[0]
    #print(false_positive_w.shape)
    true_positive_w = np.where((rcs==1) & (trcs==1))[0]
    positive_w = np.where(rcs==1)[0]
    return float(len(false_positive_p))/float(len(positive_p)), float(len(false_positive_w))/float(len(positive_w)), float(len(true_positive_p))/float(len(tpositive)), float(len(true_positive_w))/float(len(tpositive))

def binning(pred_means,pred_weights,pred_std,ymax,ymin,y_train,params,cut=200,tbins=10,gbins=10):
    bin_teff = int((np.max(params[:,0])-np.min(params[:,0]))/tbins)
    bin_logg = int((np.max(params[:,1])-np.min(params[:,1]))/gbins)
    cont = np.zeros((tbins,gbins))
    pps = np.zeros((tbins,gbins))
    for i in range(tbins):
        for j in range(gbins):
            rcs = select_rc(pred_means,pred_weights,pred_std,ymax,ymin,y_train,cmethod='peak')
            trcs = np.zeros(len(y_train))
            y_train = (ymax - ymin)*(y_train)+ymin
            tyes = np.where(y_train>cut)[0]
            trcs[tyes] = 1
            false_positive_p = np.where((rcs==1) & (trcs==0) & (np.logical_and(np.min(params[:,0])+i*bin_teff<params[:,0],params[:,0]<np.min(params[:,0])+(i+1)*bin_teff)) & (np.logical_and(np.min(params[:,1])+j*bin_logg<params[:,1],params[:,1]<np.min(params[:,1])+(j+1)*bin_logg)))[0]
            true_positive_p = np.where((rcs==1) & (trcs==1) & (np.logical_and(np.min(params[:,0])+i*bin_teff<params[:,0],params[:,0]<np.min(params[:,0])+(i+1)*bin_teff)) & (np.logical_and(np.min(params[:,1])+j*bin_logg<params[:,1],params[:,1]<np.min(params[:,1])+(j+1)*bin_logg)))[0]
            positive = np.where((rcs==1) & (np.logical_and(np.min(params[:,0])+i*bin_teff<params[:,0],params[:,0]<np.min(params[:,0])+(i+1)*bin_teff)) & (np.logical_and(np.min(params[:,1])+j*bin_logg<params[:,1],params[:,1]<np.min(params[:,1])+(j+1)*bin_logg)))[0]
            cont[i,j] = float(len(false_positive_p))/float(len(positive))
            pps[i,j] = float(len(true_positive_p))/float(len(positive))
            tots[i,j] = len(positive)
    fig, ax = plt.subplots()
    plt.imshow(cont)
    plt.colorbar()
    #ax.set_yticks(range(len(colors)))
    ax.set_yticklabels(np.linspace(np.min(params[:,1]),np.max(params[:,1]),gbins))
    ax.set_xticklabels(np.linspace(np.min(params[:,0]),np.max(params[:,0]),tbins))
    plt.title('Contamination')
    plt.show()
    fig, ax = plt.subplots()
    plt.imshow(pps)
    plt.colorbar()
    #ax.set_yticks(range(len(colors)))
    ax.set_yticklabels(np.linspace(np.min(params[:,1]),np.max(params[:,1]),gbins))
    ax.set_xticklabels(np.linspace(np.min(params[:,0]),np.max(params[:,0]),tbins))
    plt.title('Sucessful IDs')
    return cont, pps

def per_stats(pred_means,pred_weights,pred_std,ymax,ymin,y_train):
    y_pred = np.sum(pred_means*pred_weights, axis = 1)
    y_pred_std = np.sum(pred_std*pred_weights, axis = 1)
    y_pred = (ymax - ymin)*(y_pred)+ymin
    y_pred_std = (ymax - ymin)*(y_pred_std)
    y_train = (ymax - ymin)*(y_train)+ymin
    diff = y_pred-y_train
    mean_diff = np.mean(diff)
    med_diff = np.median(diff)
    std_diff = np.std(diff)
    mean_sigma = np.mean(y_pred_std)
    med_sigma = np.median(y_pred_std)
    std_sigma = np.std(y_pred_std)
    return mean_diff, med_diff, std_diff, mean_sigma, med_sigma, std_sigma

def testing(save_mod,X_test):
    neural_network = hub.Module(save_mod)
    dict = neural_network(X_test,as_dict=True)
    locs = dict['locs'] ; scales = dict['scales'] ; logits = dict['logits']
    pred_weights, pred_means, pred_std = evaluate([tf.nn.softmax(logits), locs, scales])
    return pred_weights, pred_means, pred_std

def plot_cum_sigma(pred_weights,pred_std,ymax,ymin):
    #y_pred_std = np.sum(pred_std*pred_weights, axis = 1)

    weight_max = np.argmax(pred_weights, axis = 1)  ## argmax or max???
    y_pred_std = np.array([pred_std[i,weight_max[i]] for i in range(len(y_train))])
    y_pred_std = (ymax - ymin)*(y_pred_std)
    plt.figure(222)
    plt.hist(y_pred_std,100, density=True, histtype='step',
                           cumulative=True,color='k')
    plt.xlabel('Sigma')
    plt.show()

n_epochs = 100000 #1000 #20000 #20000
# N = 4000  # number of data points  -- replaced by num_trai
D = 13 #6  # number of features  (8 for DES, 6 for COSMOS)
K = 5 # number of mixture components

learning_rate = 1e-3
decay_rate= .8
step=100

num_train = 100000 #800000
num_test = 100000 #10000 #params.num_test # 32

save_mod = '/home/mrl2968/Desktop/Kavli/Tmodels/lr'+str(learning_rate)+'_dr'+str(decay_rate)+'_step'+str(step)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train)

############training

X_train, y_train, X_test, y_test, params, ymax, ymin, xmax, xmin = GenData_lamost(fileIn = 'lamost_wise_gaia_PS1_2mass.fits')
#import pdb ; pdb.set_trace()

net_spec = hub.create_module_spec(neural_network_mod)
neural_network = hub.Module(net_spec,name='neural_network',trainable=True)

log_likelihood, train_op, logits, locs, scales  = mixture_model(X_train,y_train,learning_rate=learning_rate,decay_rate=decay_rate)

train_loss = train(log_likelihood,train_op,n_epochs)
#save network
neural_network.export(save_mod,sess)

pred_weights, pred_means, pred_std = get_predictions(logits, locs, scales)
print(pred_means)

plot_pdfs(pred_means,pred_weights,pred_std)

plot_pred_mean(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

mean_diff, med_diff, std_diff, mean_sigma, med_sigma, std_sigma = per_stats(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

plot_cum_sigma(pred_weights,pred_std,ymax,ymin)

"""
#plot_pred_peak(pred_means,pred_weights,pred_std,ymax,ymin,y_train)
#plot_pred_weight(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

#contamp, contamw, pp, pw = contamination(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

#bin_contam, bin_pp, bin_tot = binning(pred_means,pred_weights,pred_std,ymax,ymin,y_train,params,cut=200,tbins=10,gbins=10)

######testing


test_weights, test_means, test_std = testing(save_mod,X_test)
plot_pdfs(test_means,test_weights,test_std)

plot_pred_mean(test_means,test_weights,test_std,ymax,ymin,y_train)

plot_cum_sigma(test_weights,test_std,ymax,ymin)

test_mean_diff, test_med_diff, test_std_diff, test_mean_sigma, test_med_sigma, test_std_sigma = per_stats(test_means,test_weights,test_std,ymax,ymin,y_test)
"""
