# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       train_val on workflowpredict
   Description:     
   Author:          HAO
   Date:            2019/5/14
   Create by:       PyCharm 
   Check status:    https://waynehfut.com
-------------------------------------------------
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import copy
import logging
from models.functions import initialize_cnn_model, CNNLSTM, CurrentPredict, PreviousGuide

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# simple model train
def model_train(model, dataloaders, criterion, optimizer, checkpoint_path, num_epochs):
    model = nn.DataParallel(model)
    model.to(device)
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                targets = torch.reshape(labels,(32,1))

                # zero the parameter gradients
                optimizer.zero_grad()
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                logging.info("Acc improved from {} to {}".format(best_loss, epoch_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, checkpoint_path.format(epoch))
            if phase == 'val':
                val_acc_history.append(epoch_loss)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val loss: {:4f}'.format(best_loss))

    # load best model checkpoint
    model.load_state_dict(best_model_wts)
    torch.save(model, checkpoint_path.format('best'))
    logging.info('Model saved at {}'.format(checkpoint_path.format('best')))
    return model, val_acc_history


def model_test(model, dataloaders, criterion):
    logging.info('Test model on test set')
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_bar = enumerate(tqdm(dataloaders['test']))
    for i, (inputs, labels) in test_bar:
        model.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, _test_preds = torch.max(outputs, 1)
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(_test_preds == labels.data)
    test_loss = test_loss / len(dataloaders['test'].dataset)
    test_acc = test_corrects.double() / len(dataloaders['test'].dataset)
    logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', test_loss, test_acc))

def model_train_CnnLstm(model, dataloaders, criterion, optimizer, checkpoint_path, num_epochs):
    model = nn.DataParallel(model)
    model.to(device)
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                tel_labels = labels[:, 0]
                time_labels = labels[:, 1]
                # 计算进度以及剩余时间,将进度换成小数，将剩余时间化为 min,并且除以s_norm 25*60*5
                regression_labels = [torch.true_divide(tel_labels[i], time_labels[i]) for i in range(len(tel_labels))]
                rsd_labels = torch.true_divide((time_labels - tel_labels), 7500)

                # zero the parameter gradients
                optimizer.zero_grad()
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    tel_predict = torch.true_divide(tel_labels[i], 7500)
                    rsd_predict = [torch.true_divide(tel_predict[i],outputs[i])-tel_predict[i] for i in range(len(regression_labels))]
                    loss_reg = criterion(outputs, regression_labels)
                    loss_label = criterion(rsd_predict, rsd_labels)
                    loss = loss_reg + loss_label

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                logging.info("Acc improved from {} to {}".format(best_loss, epoch_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, checkpoint_path.format(epoch))
            if phase == 'val':
                val_acc_history.append(epoch_loss)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val loss: {:4f}'.format(best_loss))

    # load best model checkpoint
    model.load_state_dict(best_model_wts)
    torch.save(model, checkpoint_path.format('best'))
    logging.info('Model saved at {}'.format(checkpoint_path.format('best')))
    return model, val_acc_history


def model_guide_train(model_predict, model_previous, model_future, dataloders, criterion_pg,
                      criterion_fu, optimizer, checkpoint_path, num_epochs, lambda_1_pt, lambda_2_pt):
    """
    Training the guide model with the model_generator and model——discriminator,
    it should be noticed that the k_lambda_impact only used in evaluate

    :param model_predict: Model architecture and weights for future predict.
    :param model_previous: Model architecture and weights for guide.
    :param dataloders: guide data with ['train', 'test', 'val']
    :param criterion_pg: previous criterion function
    :param criterion_fu: future criterion function
    :param optimizer: optimizer
    :param num_epochs: running epochs
    :param checkpoint_path: weights save path
    :param lambda_1: control parameter for the importance between the previous.
    :param lambda_2: control parameter for the importance between the future.
    :return: model_generator, val_history
    """

    # load model to GPU
    model_predict = nn.DataParallel(model_predict)  # 14 [7+7]
    model_previous = nn.DataParallel(model_previous)  # 128 dim
    model_predict.to(device)
    model_previous.to(device)
    since = time.time()
    val_acc_history = []

    # preload the model weights
    best_ge_model_wts = copy.deepcopy(model_predict.state_dict())
    best_ge_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model_predict.train()
            else:
                model_predict.eval()
            running_loss = 0.0
            running_corrects = 0
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                pre_sample = inputs['previous_sample']
                cur_sample = inputs['current_sample']
                fur_sample = inputs['future_sample']
                pre_sample = pre_sample.to(device)
                cur_sample = cur_sample.to(device)
                fur_sample = fur_sample.to(device)
                labels = labels.to(device)

                pre_output = model_previous(pre_sample)  # preguide output from fc with 128 dim
                future_output = model_future(fur_sample)  # future output with label 7 dim

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    combine_output, lambda_pre = model_predict(cur_sample, pre_output)  # 14, lambda
                    pre_p_labels = combine_output[:, :7]
                    pre_f_labels = combine_output[:, 7:]
                    p_labels = labels[:, 0]
                    f_labels = labels[:, 1]
                    pre_loss = criterion_pg(pre_p_labels, p_labels) + lambda_pre[0] * criterion_pg(pre_f_labels,
                                                                                                   f_labels)
                    _1, preds1 = torch.max(pre_p_labels, 1)
                    _2, preds2 = torch.max(pre_f_labels, 1)
                    # current_predict = preds1[0]
                    # future_predict = preds2[0]
                    # fur_loss = criterion_fu(f_labels, future_predict)
                    # combine_loss = pre_loss + lambda_pre * fur_loss
                    combine_loss = pre_loss

                    if phase == 'train':
                        combine_loss.backward()
                        optimizer.step()
                running_loss += combine_loss.item() * cur_sample.size(0)
                running_corrects = running_corrects + \
                                   torch.sum(preds1 == labels[:, 0].data) + \
                                   torch.sum(preds2 == labels[:, 1].data)  # future

            # For a epoch
            epoch_loss = running_loss / len(dataloders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloders[phase].dataset)

            logging.info('{} Guide Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_ge_acc:
                logging.info("Guide Acc improved from {} to {}".format(best_ge_acc, epoch_acc))
                best_ge_acc = epoch_acc
                best_ge_model_wts = copy.deepcopy(model_predict.state_dict())
                torch.save(model_predict, checkpoint_path.format(epoch))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    logging.info('Guide complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_ge_acc))

    model_predict.load_state_dict(best_ge_model_wts)
    torch.save(model_predict, checkpoint_path.format('best'))
    logging.info('Guide Model saved at {}'.format(checkpoint_path.format('best')))
    return model_predict, val_acc_history


def model_guide_test(model_predict, model_previous, dataloaders, criterion_pg, criterion_fu):
    # combine_output, lambda_pre = model_predict
    logging.info('Test guide model on test set')
    model_predict.eval()
    test_loss = 0.0
    test_corrects = 0
    test_bar = enumerate(tqdm(dataloaders['test']))
    for i, (inputs, labels) in test_bar:
        pre_sample = inputs['previous_sample']
        cur_sample = inputs['current_sample']
        model_predict.to(device)
        model_previous.to(device)
        pre_sample = pre_sample.to(device)
        cur_sample = cur_sample.to(device)
        labels = labels.to(device)
        previous_result = model_previous(pre_sample)
        combine_output, lambda_pre = model_predict(cur_sample, previous_result)
        pre_p_labels = combine_output[:, :7]
        pre_f_labels = combine_output[:, 7:]
        p_labels = labels[:, 0]
        f_labels = labels[:, 1]
        pre_loss = criterion_pg(pre_p_labels, p_labels) + lambda_pre[0] * criterion_pg(pre_f_labels, f_labels)
        _1, preds1 = torch.max(pre_p_labels, 1)
        _2, preds2 = torch.max(pre_f_labels, 1)
        test_loss += pre_loss.item() * inputs.size(0)
        test_corrects = test_corrects + \
                        torch.sum(preds1 == labels[:, 0].data) + \
                        torch.sum(preds2 == labels[:, 1].data)
    test_loss = test_loss / len(dataloaders['test'].dataset)
    test_acc = test_corrects.double() / len(dataloaders['test'].dataset)
    logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', test_loss, test_acc))


def train_finetune(dataloaders_dict, out_path, model_name, num_epochs, feature_extract=True, lr=0.0001, momentum=0.85):
    """
    Finetune for CNN
    :param dataloaders_dict: data dict
    :param out_path: weight saved path.
    :param num_epochs: running epoch
    :param feature_extract: if extract feature, default True.
    :param model_name: model name in [resnet, vgg, squeezenet, densenet, inception]
    :param momentum: parameters for momentum
    :param lr: learning rate

    :return: best model with weight, and model history
    """
    logging.info('Start CNN fine tune')
    model_ft, input_size = initialize_cnn_model(model_name, 1, feature_extract, use_pretrained=True)
    params_to_update = model_ft.parameters()
    logging.info("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                logging.info(name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                logging.info(name)
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum)
    criterion = nn.MSELoss()
    ## train val
    model_ft, hist = model_train(model_ft, dataloaders_dict, criterion, optimizer_ft, out_path, num_epochs)
    return model_ft, hist


def pretrain_guide(dataloaders_dict, cnn_weights, out_path, num_epochs, lr=1e-2, momentum=0.5, rnn_lr=1e-3,
                   rnn_momentum=0.5):
    """
    Pretrain the LSTM CNN model

    :param rnn_momentum:
    :param rnn_lr:
    :param momentum:
    :param lr:
    :param dataloaders_dict: train, test, val dataset
    :param cnn_weights: pretrained cnn weights
    :param out_path: model weights save path
    :param num_epochs: epochs number
    """
    logging.info('Start train CNN LSTM')
    cnn_model = torch.load(cnn_weights)
    cnn_dict = cnn_model.state_dict()

    cnnlstm_model = CNNLSTM()
    cnnlstm_dict = cnnlstm_model.state_dict()

    pretrained_dict = {k: v for k, v in cnn_dict.items() if k in cnnlstm_dict}
    cnnlstm_dict.update(pretrained_dict)
    cnnlstm_model.load_state_dict(cnnlstm_dict)

    # set learning rate and momentum individually
    all_parameters = list(cnnlstm_model.parameters())
    rnn_params = list(cnnlstm_model.LSTM.parameters())
    cnn_params = list(set(all_parameters) - set(rnn_params))

    optimizer_ft = optim.SGD([{"params": cnn_params}, {"params": rnn_params, "lr": rnn_lr, 'momentum': rnn_momentum}],
                             lr=lr, momentum=momentum)

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    model_ft, hist = model_train(cnnlstm_model, dataloaders_dict, criterion, optimizer_ft, out_path, num_epochs)
    return model_ft, hist


def train_guide(dataloders_dict, cnnlstm_weights, outpath, num_epochs, k_lambda=0.5):
    """
    Train the guided model

    :param dataloders_dict: train, test, and val dataset
    :param cnnlstm_weights: cnnlstm training weights
    :param outpath: training pth saved path
    :param num_epochs: epochs number
    :param k_lambda:
    """
    logging.info('Start guide train')
    # lambda importance control
    lambda_1_pt = k_lambda
    lambda_2_pt = 1 - lambda_1_pt
    # cnnlstm weights and dict as future guide
    cnnlstm_model = torch.load(cnnlstm_weights)
    cnnlstm_dict = cnnlstm_model.state_dict()

    # build generator as predict model, previous_discriminator as previous guide model
    current_model = CurrentPredict(lambda_1=lambda_1_pt)
    current_model_dict = current_model.state_dict()

    previous_guide_model = PreviousGuide()
    previous_model_dict = previous_guide_model.state_dict()

    # load dict for current
    pretrained_dict_cu = {k: v for k, v in cnnlstm_dict.items() if k in current_model_dict}
    current_model_dict.update(pretrained_dict_cu)
    current_model.load_state_dict(current_model_dict)

    # load dict for previous
    pretrained_dict_pre = {k: v for k, v in cnnlstm_dict.items() if k in previous_model_dict}
    previous_model_dict.update(pretrained_dict_pre)
    previous_guide_model.load_state_dict(previous_model_dict)

    # build loss
    criterion_pg = nn.CrossEntropyLoss()
    criterion_fu = nn.L1Loss()

    # only update generator
    params_to_update = current_model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.85)
    model_guide_train(current_model, previous_guide_model, cnnlstm_model, dataloders_dict, criterion_pg,
                      criterion_fu, optimizer_ft, outpath, num_epochs, lambda_1_pt, lambda_2_pt)
