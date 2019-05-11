import numpy as np
import tensorflow as tf
import tensorlayer as tl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import *
from utilities import *
from data import *
from networks import *

setGPU('0')
log = Logger('log/sentiment-msda', clear=True)
cls = CLS(5000, 2, bottle_neck_dim = 500).cuda()
discriminator = LargeDiscriminator(5000).cuda()
scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=3000)
optimizer_cls = OptimWithSheduler(optim.Adam(cls.parameters(), weight_decay = 5e-4, lr = 5e-5),
                                  scheduler)
optimizer_discriminator = OptimWithSheduler(optim.Adam(discriminator.parameters(), weight_decay = 5e-4, lr = 5e-5),
                                  scheduler)


source_train = np.load('sentimenthx.npy').astype('float32')[19731:27677]/84
target_train = np.load('sentimenthx.npy').astype('float32')[0:6464]/84
source_label0 = (np.load('sentiment_label.npy')[19731:27677] + 1) / 2
source_label = np.zeros((7946,2)).astype('float32')
for i in range(7946):
    source_label[i] = one_hot(2, source_label0[i])
target_label0 = (np.load('sentiment_label.npy')[0:6464] + 1) / 2
target_label = np.zeros((6464, 2)).astype('float32')
for i in range(6464):
    target_label[i] = one_hot(2, target_label0[i])
# =====================train
k=0
while k < 5:
    mini_batches_source = get_mini_batches(source_train, source_label, 64)
    mini_batches_target = get_mini_batches(target_train, target_label, 64)
    for (i, ((im_source, label_source,), (im_target, label_target,))) in enumerate(
            zip(mini_batches_source, mini_batches_target)):
        
        # =========================generate transferable examples
        label_source_0 = Variable(torch.from_numpy(label_source)).cuda()
        feature_fooling = Variable(torch.from_numpy(im_target).cuda(),requires_grad = True)
        feature_fooling_c = Variable(torch.from_numpy(im_source).cuda(),requires_grad = True)
        feature_fooling_0 = feature_fooling.detach()
        
        for i in range(2):
            scores = discriminator(feature_fooling)
            loss = BCELossForMultiClassification(torch.ones_like(scores) , 1 - scores) - 0.1 * torch.sum((feature_fooling - feature_fooling_0) * (feature_fooling - feature_fooling_0))
            loss.backward()
            g = feature_fooling.grad
            feature_fooling = feature_fooling + 0.001 * g 
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling = Variable(feature_fooling.data.cpu().cuda(),requires_grad = True)
        
        feature_fooling_c1 = feature_fooling_c.detach()
        for xs in range(2):
            scorec = discriminator.forward(feature_fooling_c)
            losss = BCELossForMultiClassification(torch.ones_like(scorec) ,  scorec) - 0.1 * torch.sum((feature_fooling_c - feature_fooling_c1) * (feature_fooling_c - feature_fooling_c1))
            losss.backward()
            gss = feature_fooling_c.grad
            feature_fooling_c = feature_fooling_c +  0.001 * gss
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling_c = Variable(feature_fooling_c.data.cpu().cuda(),requires_grad = True)
        
        for xss in range(3):
            _,_,_,scorec = cls.forward(feature_fooling_c)
            loss = CrossEntropyLoss(label_source_0, scorec) - 0.1 * torch.sum((feature_fooling_c - feature_fooling_c1) * (feature_fooling_c - feature_fooling_c1))
            loss.backward()
            gs = feature_fooling_c.grad
            feature_fooling_c = feature_fooling_c +  0.001 * gs
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling_c = Variable(feature_fooling_c.data.cpu().cuda(),requires_grad = True)
            
        # =========================forward pass
        feature_source = Variable(torch.from_numpy(im_source)).cuda()
        label_source = Variable(torch.from_numpy(label_source)).cuda()
        feature_target = Variable(torch.from_numpy(im_target)).cuda()
        label_target = Variable(torch.from_numpy(label_target)).cuda()
        
        _, _, __, predict_prob_source = cls.forward(feature_source)
        _, _, __, predict_prob_target = cls.forward(feature_target)
        _, _, __, predict_prob_fooling = cls.forward(feature_fooling)
        _, _, __, predict_prob_fooling_c = cls.forward(feature_fooling_c)

        domain_prob_source = discriminator.forward(feature_source)
        domain_prob_target = discriminator.forward(feature_target)
        domain_prob_fooling = discriminator.forward(feature_fooling)
        domain_prob_fooling_c = discriminator.forward(feature_fooling_c)
        
        dloss = BCELossForMultiClassification(torch.ones_like(domain_prob_source), domain_prob_source)
        dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_target), 1 - domain_prob_target)
        dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_fooling_c), domain_prob_fooling_c.detach())
        dloss += BCELossForMultiClassification(torch.ones_like(domain_prob_fooling), 1 - domain_prob_fooling.detach())
        
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        ce_extra_c = CrossEntropyLoss(label_source, predict_prob_fooling_c)
        

        dis = torch.sum((predict_prob_fooling - predict_prob_target) * (predict_prob_fooling - predict_prob_target))
        

        with OptimizerManager([optimizer_cls , optimizer_discriminator]):
            loss = ce + ce_extra_c + dis + dloss
            loss.backward()
                        
        log.step += 1
    
        if log.step % 10 == 1:
            counter = AccuracyCounter()
            counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
            acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype = np.float32))).cuda()
            track_scalars(log, ['ce', 'acc_train', 'dis','ce_extra_c','dloss'], globals())

        if log.step % 100 == 0:
            clear_output()
    k += 1       

# ======================test
with TrainingModeManager([cls], train = False) as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
    for (i, (im, label)) in enumerate(mini_batches_target):
        fs = Variable(torch.from_numpy(im), volatile = True).cuda()
        label = Variable(torch.from_numpy(label), volatile = True).cuda()
        __, fs, _, predict_prob = cls.forward(fs)
        predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
        label = np.argmax(label, axis = -1).reshape(-1, 1)
        predict_index = np.argmax(predict_prob, axis = -1).reshape(-1, 1)
        accumulator.updateData(globals())
        if i % 10 == 0:
            print(i)

for x in accumulator.keys():
    globals()[x] = accumulator[x]
print('acc')
print(float(np.sum(label.flatten() == predict_index.flatten()) )/ label.flatten().shape[0])


