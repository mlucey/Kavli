import math
import numpy as np
import string
from datetime import datetime
import os
from astropy.table import Table


#import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt;
import random


import matplotlib.axes as axes;
from matplotlib.patches import Ellipse
#import seaborn as sns;
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


def rev_sigmoid(x):
    return 100/(1+100*np.exp(x))
    #return np.abs(x)/(1+np.abs(x))

def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
  """Plots the mixture of Normal models to axis=ax comp=True plots all
  components of mixture model1.0/(1+np.exp(x))
  """
  # x = np.linspace(-10.5, 10.5, 250)
  x = np.linspace(-0.1, 1.1, 250)
  final = np.zeros_like(x)
  for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
    temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
    final = final + temp
    if comp:
      ax.plot(x, temp, label='Normal ' + str(i))
  ax.plot(x, final, color='k', label='Mixture of Normals ' + label)
  #ax.legend(fontsize=13)
  return final

#def GenData_lamost(fileIn = 'lamost_rc_wise_gaia_PS1_2mass.fits'):
def GenData_lamost(fileIn = 'rgb_p.fits',copy=False):
    al = Table.read(fileIn)
    filts = ['j_m', 'h_m', 'ks_m', 'phot_g_mean_mag', 'phot_bp_mean_mag',
              'phot_rp_mean_mag', 'g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag',
              'z_mean_psf_mag', 'y_mean_psf_mag', 'w1mpro', 'w2mpro','parallax'] # train filters
    #filts_target = ['DeltaP', 'Deltanu'] #
    filts_target = ['deltap']#, 'Deltanu'] #
    # filts_target = ['Teff', 'log_g_'] #
    param = ['teff','log_g']
    #params = ['teff','logg']


    inds = np.where( ~(np.isnan(al[filts[0]])) & ~(np.isnan(al[filts[1]])) & ~(np.isnan(al[filts[2]])) & ~(np.isnan(al[filts[3]])) & ~(np.isnan(al[filts[4]])) & ~(np.isnan(al[filts[5]]))& ~(np.isnan(al[filts[6]])) & ~(np.isnan(al[filts[7]])) & ~(np.isnan(al[filts[8]])) & ~(np.isnan(al[filts[9]])) & ~(np.isnan(al[filts[10]])) & ~(np.isnan(al[filts[11]])) & ~(np.isnan(al[filts[12]])) & ~(np.isnan(al[filts[13]])) )[0]

    #inds = np.where( (al['Jmag']>-1000) & (al['Hmag']>-1000) & (al['Kmag']>-1000) & (al['phot_g_mean_mag']>-1000) & (al['phot_rp_mean_mag']>-1000) &
                #      (al['gmag']>-1000) & (al['rmag']>-1000) & (al['imag']>-1000) & (al['zmag']>-1000) & (al['ymag']>-1000) & (al['W1mag']>-1000) & (al['W2mag']>-1000) & (al['parallax']))[0]
    gps = al[inds]

    tinds = inds
    #inds = np.where((gps['DeltaP']>-1000) & (gps['Deltanu']>-1000))[0]
    inds = np.where((gps['deltap']>-1000))[0] #& (gps['Deltanu']>-1000))[0]
    #inds = np.where((gps['DeltaP']>-1000) & (gps['Deltanu']>-1000)& ~(np.logical_and(gps['DeltaP']>=100, gps['DeltaP']<=250)))[0]
    # inds = np.where((gps['Teff']>-1000) & (gps['log_g_']>-1000) )[0]

    al =  gps[inds]
    tinds =tinds[inds]

    #
    # ## RC pristine
    # pristine_inds = np.where(al['Class'] != 'RC_Pristine')
    # al =  al[pristine_inds]

    np.random.seed(123)

    x_train_all = np.array([al[filts[0]], al[filts[1]], al[filts[2]], al[filts[3]], al[filts[4]], al[filts[5]], al[filts[6]], al[filts[7]], al[filts[8]], al[filts[9]], al[filts[10]], al[filts[11]], al[filts[12]],al[filts[13]]]).T
    y_train_all = np.array(al[filts_target[0]]).T
    params = np.array([al[param[0]],al[param[1]]]).T
    ids = np.array(al['id']).T
    classy = np.zeros(len(al))
    inds  = np.where((al['class'] == 'RC         ') | (al['class'] == 'RC_Pristine'))[0]
    #inds  = np.where((al['Class'] == 'R') )[0]
    classy[inds] = 1
    print(len(inds))
    alset = set(range(len(al)))
    cset = set(inds)
    no_rc = np.array(list(alset-cset),dtype='int')[0:len(inds)+1]

    x_rc = x_train_all[inds]
    y_rc = y_train_all[inds]

    #x_train_all = np.concatenate([x_rc,x_train_all[no_rc]],axis=0)
    #y_train_all = np.concatenate([y_rc,y_train_all[no_rc]],axis=0)
    #classy = np.concatenate([classy[inds],classy[no_rc]],axis=None)

    xmax_a = np.max(x_train_all[:,13])
    xmin_a = np.min(x_train_all[:,13])
    x_train_rescaled = np.zeros((len(x_train_all),len(x_train_all[0])))
    x_train_rescaled[:,13] = (x_train_all[:,13]-xmin_a)/(xmax_a-xmin_a)
    print(xmax_a)
    print(xmin_a)
    xmax = np.max(x_train_all[:,:13], axis = None)
    xmin = np.min(x_train_all[:,:13], axis = None)
    x_train_rescaled[:,:13] = (x_train_all[:,:13] - xmin) / (xmax - xmin)

    ymax = np.max(y_train_all, axis = 0)
    ymin = np.min(y_train_all, axis = 0)

    y_train_rescaled = (y_train_all - ymin) / (ymax - ymin)


  # color mag

    if copy:
        t2 = Table.read('Tables/test_rc_nu.fits')
        x_test_all = np.array([t2[filts[0]], t2[filts[1]], t2[filts[2]], t2[filts[3]], t2[filts[4]], t2[filts[5]], t2[filts[6]], t2[filts[7]], t2[filts[8]], t2[filts[9]], t2[filts[10]], t2[filts[11]], t2[filts[12]],t2[filts[13]]]).T
        y_test_all = np.array(t2[param[0]]).T
        X_test = np.zeros((len(x_test_all),len(x_test_all[0])))
        X_test[:,13] = (x_test_all[:,13]-xmin_a)/(xmax_a-xmin_a)
        X_test[:,:13] = (x_test_all[:,:13] - xmin) / (xmax - xmin)
        y_test= (y_test_all - ymin) / (ymax - ymin)
        test_ids = t2['id']
        ismem = np.in1d(ids,test_ids,invert=True)
        test_tinds = np.where(ismem==False)[0]
        train_inds = np.where(ismem==True)[0]
        al_train = al[train_inds]
        train_inds = np.where(al_train['s_n']>75)[0][:num_train]
        X_train = x_train_rescaled[train_inds]
        y_train = y_train_rescaled[train_inds]
        train_tinds= train_inds
        params = params[train_tinds]
    else:
        TrainshuffleOrder = np.arange(x_train_rescaled.shape[0])
        np.random.shuffle(TrainshuffleOrder)

        x_train_shuffled = x_train_rescaled[TrainshuffleOrder]
        y_train_shuffled = y_train_rescaled[TrainshuffleOrder]
        classy_shuffled = classy[TrainshuffleOrder]
        params_shuffled = params[TrainshuffleOrder]
        tinds = tinds[TrainshuffleOrder]
        X_train = x_train_shuffled[:num_train]  # color mag
        X_test = x_train_shuffled[num_train + 1: num_train + num_test]

        y_train = y_train_shuffled[:num_train]  # spec z
        y_test = y_train_shuffled[num_train + 1: num_train + num_test]  # spec z
        classy = classy_shuffled[:num_train]
        params = params_shuffled[:num_train]
        train_tinds = tinds[:num_train]
        test_tinds = tinds[num_train+1:num_train+num_test]
        # return (X_train[:, 2:8], y_train, X_test[:, 2:8], y_test)
        # return (X_train[:, :5], y_train[:, 0], X_test[:, :5], y_test[:, 0])
        print(len(np.where(classy==1)[0]))
        print(len(classy))
        print(float(len(np.where(classy==1)[0]))/float(len(classy)))

    return X_train, y_train, X_test, y_test, classy, params, ymax, ymin, xmax, xmin, xmax_a, xmin_a, train_tinds, test_tinds, ids

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


def mixture_model(X,Y,learning_rate=1e-3,decay_rate=.95,step=1000,train=True):
    if train:
        dict = neural_network(tf.convert_to_tensor(X),as_dict=True)
    else:
        dict = neural_network_t(tf.convert_to_tensor(X),as_dict=True)
    locs = dict['locs'] ; scales = dict['scales'] ; logits = dict['logits']
    cat = tfd.Categorical(logits=logits)
    components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                  in zip(tf.unstack(tf.transpose(locs)),
                         tf.unstack(tf.transpose(scales)))]

    y = tfd.Mixture(cat=cat, components=components)
    #define loss function
    log_likelihood = y.log_prob(Y)
    #log_likelihood = -tf.reduce_sum(log_likelihood/(1. + Y)**2 )
    y_mean = np.median(Y)
    #log_likelihood = -tf.reduce_sum(log_likelihood*rev_sigmoid(Y))
    log_likelihood = -tf.reduce_sum(log_likelihood)
    #log_likelihood = -tf.reduce_sum(log_likelihood*(y_mean-Y)**4 )
    if train:
        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.train.exponential_decay(learning_rate,
                                        global_step, step,
                                        decay_rate)
        optimizer = tf.train.AdamOptimizer(decayed_lr)
        train_op = optimizer.minimize(log_likelihood)
        evaluate(tf.global_variables_initializer())
        return log_likelihood, train_op, logits, locs, scales
    else:
        evaluate(tf.global_variables_initializer())
        return log_likelihood, logits, locs, scales


def train(log_likelihood,train_op,n_epoch):
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    for i in range(n_epoch):
        _, loss_value = evaluate([train_op, log_likelihood])
        train_loss[i] = loss_value
    plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train Loss')
    #plt.show()
    plt.savefig('../Plots/loss_function.pdf')
    return train_loss


def get_predictions(logits,locs,scales):
    pred_weights, pred_means, pred_std = evaluate([tf.nn.softmax(logits), locs, scales])
    return pred_weights, pred_means, pred_std

def plot_pdfs(pred_means,pred_weights,pred_std,num=10,train=True):
    if train:
        obj = [random.randint(0,num_train-1) for x in range(num)]
    else:
        obj = [random.randint(0,num_test-1) for x in range(num)]
    #obj = [93, 402, 120,789,231,4,985]
    print(obj)
    fig, axes = plt.subplots(nrows=num, ncols=1, sharex = True, figsize=(8, 7))
    allfs = []
    for i in range(len(obj)):
        fs = plot_normal_mix(pred_weights[obj][i], pred_means[obj][i],
                    pred_std[obj][i], axes[i], comp=False)
        allfs.append(fs)
        axes[i].axvline(x=y_train[obj][i], color='black', alpha=0.5)
    plt.xlabel('Normalized Period Spacing')
    plt.tight_layout()
    if train:
        plt.savefig('Plots/train_pdfs_p.pdf')
    else:
        plt.savefig('Plots/test_pdfs_p.pdf')
        #axes[i].text(0.3, 4.0, 'ID: ' +str(obj[i]), horizontalalignment='center',
        #verticalalignment='center')

def plot_pred_mean(pred_means,pred_weights,pred_std,ymax,ymin,y_train,select='no',train=True):
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
    #plt.errorbar((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(
      #y_pred_std), fmt='o',color='k', ecolor='grey', ms = 2, alpha = 0.1)
    plt.scatter((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin,color='k', s = 1)
    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*( y_train)+ymin, 'k')
    plt.ylabel(r'$\Delta P_{phot}$', fontsize = 19)
    plt.xlabel(r'$\Delta P_{spec}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('weight x mean')
    plt.tight_layout()
    if train:
        plt.savefig('Plots/train_scatter_p.pdf')
    else:
        plt.savefig('Plots/test_scatter_p.pdf')
    #plt.show()
    #plt.savefig('../Plots/pred_mean.pdf')

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
    #plt.errorbar((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(
      #y_pred_std), fmt='o',color='k', ecolor='grey', ms = 2, alpha = 0.1)
    plt.scatter((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin,color='k', s = 1)
    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_test)+ymin, (ymax - ymin)*(y_test)+ymin, 'k')
    plt.ylabel(r'$\Delta P_{phot}$', fontsize = 19)
    plt.xlabel(r'$\Delta P_{spec}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('highest peak')
    plt.tight_layout()
    plt.show()
    #plt.savefig('../Plots/pred_peak.pdf')


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
    #plt.errorbar((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(
      #y_pred_std), fmt='o',color='k', ecolor='grey', ms = 2, alpha = 0.1)
    plt.scatter((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin,  color='k', s = 1)

    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_test)+ymin, (ymax - ymin)*(y_test)+ymin, 'k')
    plt.ylabel(r'$\Delta P_{phot}$', fontsize = 19)
    plt.xlabel(r'$\Delta P_{spec}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('highest weight')
    plt.tight_layout()
    plt.show()
    #plt.savefig('../Plots/pred_weight.pdf')

def select_rc(pred_means,pred_weights,pred_std,ymax,ymin,cut,cmethod='peak'):
    print(cmethod)
    rcs =np.zeros(len(pred_means))
    if cmethod == 'peak':
        def peak(weight,sigma):
            return weight/np.sqrt(2*np.pi*sigma**2)

        peak_max = np.argmax(peak(pred_weights,pred_std),axis=1)
        y_pred = np.array([pred_means[i,peak_max[i]] for i in range(len(pred_means))])
        y_pred_std = np.array([pred_std[i,peak_max[i]] for i in range(len(pred_means))])
    if cmethod == 'weight':
        weight_max = np.argmax(pred_weights, axis = 1)  ## argmax or max???
        #print(weight_max.shape)
        y_pred = np.array([pred_means[i,weight_max[i]] for i in range(len(pred_means))])
        y_pred_std = np.array([pred_std[i,weight_max[i]] for i in range(len(pred_means))])
    if cmethod == 'mean':
        y_pred = np.sum(pred_means*pred_weights, axis = 1)
        y_pred_std = np.sum(pred_std*pred_weights, axis = 1)
    y_pred = (ymax - ymin)*(y_pred)+ymin
    #print(y_pred.shape)
    y_pred_std = (ymax - ymin)*(y_pred_std)
    #y_train = (ymax - ymin)*(y_train)+ymin

    yes = np.where(y_pred>cut)[0]
    rcs[yes] =1
    #print(rcs.shape)
    return rcs

def contamination(pred_means,pred_weights,pred_std,ymax,ymin,classy):
    rcs = select_rc(pred_means,pred_weights,pred_std,ymax,ymin,cut,cmethod='peak')
    trcs = classy

    false_positive_p = np.where((rcs==1) & (trcs==0))[0]
    true_positive_p = np.where((rcs==1) & (trcs==1))[0]
    positive_p = np.where(rcs==1)[0]
    tpositive = np.where(trcs==1)[0]
    rcs = select_rc(pred_means,pred_weights,pred_std,ymax,ymin,cut,cmethod='weight')
    #print(rcs)
    false_positive_w = np.where((rcs==1) & (trcs==0))[0]
    #print(false_positive_w.shape)
    true_positive_w = np.where((rcs==1) & (trcs==1))[0]
    positive_w = np.where(rcs==1)[0]
    rcs = select_rc(pred_means,pred_weights,pred_std,ymax,ymin,cut,cmethod='mean')
    #print(rcs)
    false_positive_m = np.where((rcs==1) & (trcs==0))[0]
    #print(false_positive_w.shape)
    true_positive_m = np.where((rcs==1) & (trcs==1))[0]
    positive_m = np.where(rcs==1)[0]
    return float(len(false_positive_p))/float(len(positive_p)), float(len(false_positive_w))/float(len(positive_w)), float(len(false_positive_m))/float(len(positive_m)), float(len(true_positive_p))/float(len(tpositive)), float(len(true_positive_w))/float(len(tpositive)), float(len(true_positive_m))/float(len(tpositive))

def binning(pred_means,pred_weights,pred_std,ymax,ymin,y_train,params,cut,tbins=10,gbins=10):
    bin_teff = int((np.max(params[:,0])-np.min(params[:,0]))/tbins)
    bin_logg = np.round((np.max(params[:,1])-np.min(params[:,1]))/float(gbins),decimals=2)
    cont = np.zeros((tbins,gbins))
    pps = np.zeros((tbins,gbins))
    tots= np.zeros((tbins,gbins))
    rcs = select_rc(pred_means,pred_weights,pred_std,ymax,ymin,cut,cmethod='peak',cut=cut)
    trcs = np.zeros(len(y_train))
    y_train = (ymax - ymin)*(y_train)+ymin
    tyes = np.where(y_train>cut)[0]
    trcs[tyes] = 1
    print(trcs)
    for i in range(tbins):
        for j in range(gbins):

            false_positive_p = np.where((rcs==1) & (trcs==0) & (np.logical_and(np.min(params[:,0])+i*bin_teff<params[:,0],params[:,0]<np.min(params[:,0])+(i+1)*bin_teff)) & (np.logical_and(np.min(params[:,1])+j*bin_logg<params[:,1],params[:,1]<np.min(params[:,1])+(j+1)*bin_logg)))[0]
            true_positive_p = np.where((rcs==1) & (trcs==1) & (np.logical_and(np.min(params[:,0])+i*bin_teff<params[:,0],params[:,0]<np.min(params[:,0])+(i+1)*bin_teff)) & (np.logical_and(np.min(params[:,1])+j*bin_logg<params[:,1],params[:,1]<np.min(params[:,1])+(j+1)*bin_logg)))[0]
            positive = np.where((rcs==1) & (np.logical_and(np.min(params[:,0])+i*bin_teff<params[:,0],params[:,0]<np.min(params[:,0])+(i+1)*bin_teff)) & (np.logical_and(np.min(params[:,1])+j*bin_logg<params[:,1],params[:,1]<np.min(params[:,1])+(j+1)*bin_logg)))[0]
            tpositive = np.where((trcs==1) & (np.logical_and(np.min(params[:,0])+i*bin_teff<params[:,0],params[:,0]<np.min(params[:,0])+(i+1)*bin_teff)) & (np.logical_and(np.min(params[:,1])+j*bin_logg<params[:,1],params[:,1]<np.min(params[:,1])+(j+1)*bin_logg)))[0]
            if (len(positive) == 0) | (len(tpositive)==0):
                cont[i,j] = 0
                pps[i,j] = 0
                tots[i,j] = 0
            else:
                cont[i,j] = float(len(false_positive_p))/float(len(positive))
                pps[i,j] = float(len(true_positive_p))/float(len(tpositive))
                tots[i,j] = len(positive)
    fig, ax = plt.subplots()
    plt.imshow(cont)
    plt.colorbar()
    ax.set_yticks(range(gbins))
    #ax.set_xticks(range(int(tbins/2)))
    ax.set_yticklabels(np.round(np.linspace(np.min(params[:,1]),np.max(params[:,1]),gbins),decimals=2))
    ax.set_xticklabels(np.round(np.linspace(np.min(params[:,0]),np.max(params[:,0]),tbins/2+1),decimals=2))
    plt.title('Contamination')
    #plt.show()
    plt.savefig('Plots/bins_fp.pdf')
    fig, ax = plt.subplots()
    plt.imshow(pps)
    plt.colorbar()
    ax.set_yticks(range(gbins))
    #ax.set_xticks(range(int(tbins/2)))
    ax.set_yticklabels(np.round(np.linspace(np.min(params[:,1]),np.max(params[:,1]),gbins),decimals=2))
    ax.set_xticklabels(np.round(np.linspace(np.min(params[:,0]),np.max(params[:,0]),tbins/2+1),decimals=2))
    plt.title('Sucessful IDs')
    #plt.show()
    plt.savefig('Plots/bins_tp.pdf')
    fig, ax = plt.subplots()
    plt.imshow(tots)
    plt.colorbar()
    ax.set_yticks(range(gbins))
    #ax.set_xticks(range(int(tbins/2)))
    ax.set_yticklabels(np.round(np.linspace(np.min(params[:,1]),np.max(params[:,1]),gbins),decimals=2))
    ax.set_xticklabels(np.round(np.linspace(np.min(params[:,0]),np.max(params[:,0]),tbins/2+1),decimals=2))
    plt.title('Total IDS')
    #plt.show()
    plt.savefig('Plots/bins_total.pdf')
    return cont, pps, tots

def testing(test_x,test_y):

    log_likelihood,  logits, locs, scales = mixture_model(test_x,test_y,train=False)
    #_, loss_value = evaluate([train_op, log_likelihood])
    pred_weights, pred_means, pred_std = get_predictions(logits,locs,scales)
    return pred_weights, pred_means, pred_std

n_epochs = 1000000 #1000 #20000 #20000

# N = 4000  # number of data points  -- replaced by num_trai
D = 14 #6  # number of features  (8 for DES, 6 for COSMOS)
K = 2 # number of mixture components

learning_rate = 1e-3
decay_rate= 1e-6
step=1000

cut = 225
num_train = 30000 #800000
num_test = 100000 #10000 #params.num_test # 32

save_mod = 'PModels/lr'+str(learning_rate)+'_dr'+str(decay_rate)+'_step'+str(step)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train)


#X_train, y_train, X_test, y_test, classy, params, ymax, ymin, xmax, xmin = GenData_lamost(fileIn = '../Data/lamost_rc_wise_gaia_PS1_2mass.fits')

X_train, y_train, X_test, y_test, classy, params, ymax, ymin, xmax, xmin, xmax_a, xmin_a, train_tinds, test_tinds, ids = GenData_lamost(fileIn = 'lamost_rc_qual.fits',copy=True)

#import pdb ; pdb.set_trace()
net_spec = hub.create_module_spec(neural_network_mod)
neural_network = hub.Module(net_spec,name='neural_network',trainable=True)

log_likelihood, train_op, logits, locs, scales  = mixture_model(X_train,y_train,learning_rate=learning_rate,decay_rate=decay_rate)

train_loss = train(log_likelihood,train_op,n_epochs)
#save network
#neural_network.export(save_mod,sess)

pred_weights, pred_means, pred_std = get_predictions(logits, locs, scales)
#print(pred_means)

plot_pdfs(pred_means,pred_weights,pred_std)

plot_pred_mean(pred_means,pred_weights,pred_std,ymax,ymin,y_train)
#plot_pred_peak(pred_means,pred_weights,pred_std,ymax,ymin,y_train)
#plot_pred_weight(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

contamp, contamw, contamm, pp, pw, pm = contamination(pred_means,pred_weights,pred_std,ymax,ymin,classy)

#bin_contam, bin_pp, bin_tot = binning(pred_means,pred_weights,pred_std,ymax,ymin,y_train,params,cut=200,tbins=10,gbins=10)


#load saved network
#neural_network_t = hub.Module(save_mod)

######testing
"""
test_weights, test_means, test_std = testing(X_test,y_test)
plot_pdfs(test_means,test_weights,test_std,train=False)

plot_pred_mean(test_means,test_weights,test_std,ymax,ymin,y_test)

#test_mean_diff, test_med_diff, test_std_diff, test_mean_sigma, test_med_sigma, test_std_sigma = per_stats(test_means,test_weights,test_std,ymax,ymin,y_test)
"""
#def load_data(filein='lamost_rc_wise_gaia_PS1_2mass.fits',y_exist=True):
def load_data(filein='../../all_drimmel.fits',y_exist=True):
    filts = ['Jmag', 'Hmag', 'Kmag', 'phot_g_mean_mag', 'phot_bp_mean_mag',
              'phot_rp_mean_mag', 'gmag', 'rmag', 'imag',
              'zmag', 'ymag', 'W1mag', 'W2mag','parallax']  # train filters#
    #params = ['Teff','log_g_']
    params = ['Teff','log_g_']
    goal = 'DeltaP'
    al = Table.read(filein)
    inds = np.where( ~(np.isnan(al['Jmag'])) & ~(np.isnan(al['Hmag'])) & ~(np.isnan(al['Kmag'])) & ~(np.isnan(al['phot_g_mean_mag'])) & ~(np.isnan(al['phot_rp_mean_mag'])) & ~(np.isnan(al['phot_bp_mean_mag']))& ~(np.isnan(al['gmag'])) & ~(np.isnan(al['rmag'])) & ~(np.isnan(al['imag'])) & ~(np.isnan(al['zmag'])) & ~(np.isnan(al['ymag'])) & ~(np.isnan(al['W1mag'])) & ~(np.isnan(al['W2mag'])) & ~(np.isnan(al['parallax'])) )[0]
    al = al[inds]

    x_train_all = np.array([al[filts[0]], al[filts[1]], al[filts[2]], al[filts[3]], al[filts[4]], al[filts[5]], al[filts[6]], al[filts[7]], al[filts[8]], al[filts[9]], al[filts[10]], al[filts[11]], al[filts[12]],al[filts[13]]]).T

    x_train_rescaled = np.zeros((len(x_train_all),len(x_train_all[0])))
    x_train_rescaled[:,13] = (x_train_all[:,13]-xmin_a)/(xmax_a-xmin_a)

    x_train_rescaled[:,:13] = (x_train_all[:,:13] - xmin) / (xmax - xmin)
    if y_exist:
        y_train_all = np.array(al[goal]).T
        y_train_rescaled = (y_train_all - ymin) / (ymax - ymin)
        classy = np.zeros(len(al))
        #inds  = np.where((al['Class'] == 'RC         ') | (al['Class'] == 'RC_Pristine'))[0]
        inds = np.where(al['Class']=='R')[0]
        classy[inds] = 1
        return x_train_rescaled, y_train_rescaled, classy
    else:
        return x_train_rescaled

def save_inf(pred_means,pred_weights,pred_std,ids,filein='tables/test_rc_nu.fits',test=True):
    y_pred = np.sum(pred_means*pred_weights, axis = 1)
    y_pred_std = np.sum(pred_std*pred_weights, axis = 1)
    y_pred = (ymax - ymin)*(y_pred)+ymin
    y_pred_std = (ymax - ymin)*(y_pred_std)
    if test:
        al = Table.read(filein)
        al['Deltap_phot'] = y_pred
        al['Deltap_phot_error'] = y_pred_std
        al.write('Tables/test_rc_nu_p.fits',overwrite=True)
    else:
        al = Table.read(filein)
        inds = np.where( ~(np.isnan(al['Jmag'])) & ~(np.isnan(al['Hmag'])) & ~(np.isnan(al['Kmag'])) & ~(np.isnan(al['phot_g_mean_mag'])) & ~(np.isnan(al['phot_rp_mean_mag'])) & ~(np.isnan(al['phot_bp_mean_mag']))& ~(np.isnan(al['gmag'])) & ~(np.isnan(al['rmag'])) & ~(np.isnan(al['imag'])) & ~(np.isnan(al['zmag'])) & ~(np.isnan(al['ymag'])) & ~(np.isnan(al['W1mag'])) & ~(np.isnan(al['W2mag'])) )[0]
        pred_in = np.full(len(al),np.nan)
        pred_std_in = np.full(len(al),np.nan)
        pred_in[inds] = y_pred
        pred_std_in[inds] = y_pred_std

        al['Teff_phot'] = pred_in
        al['e_Teff_phot'] = pred_std_in
        al.write(filein[:-5]+'_phot.fits',overwrite=True)

#testing
test_weights, test_means, test_std = testing(X_test,y_test)
plot_pdfs(test_means,test_weights,test_std,train=False)

plot_pred_mean(test_means,test_weights,test_std,ymax,ymin,y_test,train=False)

save_inf(test_means,test_weights,test_std,ids,test=True)

## determing parameters of rc catalog stars
"""
rc_x, rc_y, rc_classy = load_data()
rc_weights, rc_means, rc_std = testing(rc_x,rc_y)
plot_pdfs(rc_means,rc_weights,rc_std,train=False)
contamp, contamw, contamm, pp, pw, pm = contamination(rc_means,rc_weights,rc_std,ymax,ymin,rc_classy)
"""
#plot_pred_peak(rc_means,rc_weights,rc_std,ymax,ymin,rc_y)
#plot_pred_mean(rc_means,rc_weights,rc_std,ymax,ymin,rc_y)
"""
#Creating Catalog
rc_x, rc_y, rc_classy = load_data(y_exist=False)
rc_weights, rc_means, rc_std = testing(rc_x,rc_y)
plot_pdfs(rc_means,rc_weights,rc_std,train=False)


#bin_contam, bin_pp, bin_tot = binning(rc_means,rc_weights,rc_std,ymax,ymin,rc_y,params,cut=200,tbins=10,gbins=10)

#rc_mean_diff, rc_med_diff, rc_std_diff, rc_mean_sigma, rc_med_sigma, rc_std_sigma = per_stats(rc_means,rc_weights,rc_std,ymax,ymin,rc_y)

save_inf(rc_means,rc_weights,rc_std)
"""
