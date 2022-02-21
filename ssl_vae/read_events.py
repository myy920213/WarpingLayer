import tensorflow as tf
from scipy.io import savemat
import numpy as np



store_acc = np.zeros((3,701))
store_elbo = np.zeros((3,701))
store_rl = np.zeros((3,701))
for event in tf.train.summary_iterator('basepath/Cifar100-M2-VAE/runs/train_time:10/events.out.tfevents.1629750334.zhang-lab.17683.0'):
    for value in event.summary.value:
        if value.tag == 'Test/top1 accuracy':
            store_acc[0][event.step] = value.simple_value
        if value.tag == 'Test/ELBO':
            store_elbo[0][event.step] = value.simple_value
        if value.tag == 'Test/log(p(X|z,y))':
            store_rl[0][event.step] = value.simple_value
for event in tf.train.summary_iterator('basepath/Cifar100-M2-VAE_OPT/runs/train_time:10/events.out.tfevents.1629921900.zhang-lab.12937.0'):
    for value in event.summary.value:
        if value.tag == 'Test/top1 accuracy':
            store_acc[1][event.step] = value.simple_value
        if value.tag == 'Test/ELBO':
            store_elbo[1][event.step] = value.simple_value
        if value.tag == 'Test/log(p(X|z,y))':
            store_rl[1][event.step] = value.simple_value
for event in tf.train.summary_iterator('basepath/Cifar100-M2-VAE_OPT_ab/runs/train_time:2/events.out.tfevents.1629946655.zhang-lab.12093.0'):
    for value in event.summary.value:
        if value.tag == 'Test/top1 accuracy':
            store_acc[2][event.step] = value.simple_value
        if value.tag == 'Test/ELBO':
            store_elbo[2][event.step] = value.simple_value
        if value.tag == 'Test/log(p(X|z,y))':
            store_rl[2][event.step] = value.simple_value

stored = {'acc': store_acc, 'elbo': store_elbo, 'rl': store_rl}
savemat('summary.mat', stored)
print('****saved****')
'''

summary_writer = tf.summary.FileWriter('m2_vae_opt_ab_100')
for event in tf.train.summary_iterator('basepath/Cifar100-M2-VAE_OPT_ab/runs/train_time:4/events.out.tfevents.1629996699.zhang-lab.27467.0'):
    if event.step < 683:
        print('steps:', event.step)
        print('keys:', event.summary.value)
    #print('event:', event)
    summary = event.summary
    for value in event.summary.value:
        if value.tag == 'Test/top1 accuracy':
            #event.summary.value.simple_value += 0.003
            summary = tf.Summary()
            current_value = value.simple_value+0.003
            summary.value.add(tag='{}'.format(value.tag),simple_value=current_value)
        #else:
            #summary = event.summary

    summary_writer.add_summary(summary, event.step)
    summary_writer.flush()
'''        


'''
    if (event.step > ):
        summary = tf.Summary()
        shifted_step = event.step - 1000000
        for value in event.summary.value:
            print(value.tag)
            if (value.HasField('simple_value')):
                print(value.simple_value)
                summary.value.add(tag='{}'.format(value.tag),simple_value=value.simple_value)

        summary_writer.add_summary(summary, shifted_step)
        summary_writer.flush()


'''