import os
import time, datetime
import cv2, copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from torch.optim import lr_scheduler

# Custom libraries:
from models.TDFA import TDFA
from models.STSTNet import STSTNet
from getdata import *
from ISDA_im import EstimatorCV, ISDALoss
from config import args
import resultAnalysis
from last_test import finalltest
from preprocess import deletefiles

# Relevant data preservation files
modelpath = r'./result/modelparameters.txt'
fdme2path = r'./result/modelresult.txt'
alldatapath = r'./result/dataresult.txt'

losok = 0

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def datasplit(fold, paths):
    """Divide the training set and test set samples"""
    f = open(os.path.join(paths, 'comp3C_label.txt'))
    alltrains = []
    imgdat = f.readlines()
    for i in range(len(imgdat)):
        if imgdat[i][0:imgdat[i].find('/')] != fold:
            alltrains.append(imgdat[i])
        pass
    f.close()
    train_size = int(len(alltrains) * 0.8)
    val_size = len(alltrains) - train_size
    outtrainsets = []
    outvalsets = []
    trainsets, valsets = torch.utils.data.random_split(alltrains, [train_size, val_size])

    for i in trainsets.indices:
        outtrainsets.append(trainsets.dataset[i])
        pass
    for j in valsets.indices:
        outvalsets.append(valsets.dataset[j])
        pass
    outtrainsets = alltrains

    return outtrainsets, outvalsets


device = torch.device(args.gpunum if torch.cuda.is_available() else 'cpu')
class_num = 3

# each fold of training and validation
def trainFold(fold, record_path, tu0=1.0, tu1=1.0, tu2=1.0, trafo_oth=None):
    alldatafile = trafo_oth[2]
    fdme2 = trafo_oth[3]
    record_file = record_path + '/training_process.txt'
    savemodel = r'./result/modelpth'
    if not os.path.exists(savemodel):
        os.makedirs(savemodel)
        pass
    trainlosspath = r'./result'

    global best_prec1
    best_prec1 = 0

    global losok


    global f1s, accs, uf1s, uars
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # unify or separately transform inputs
    # 3 classification of CDE
    if args.chooseNormal == 3:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [113.08480479, 113.45608963, 113.17704464]],
                                         std=[x / 255.0 for x in [16.41538289, 17.02655186, 17.19560204]])
        pass

    # whether use data augment on the image
    if args.is_augment == True:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((144, 144)),
            transforms.CenterCrop((120, 120)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            normalize,
        ])
        # print("Enter transform_train")
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        pass
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    datapaths = r'../comp3C5C_dataset/'
    traind, vald = datasplit(fold, datapaths)

    # whether use data split by random_split
    Experimental_Protocol = "LOSO"
    if Experimental_Protocol == "LOSO":
        if args.is_split == True:
            trainset = GetData_split('../comp3C5C_dataset/' + fold + '/train/', 'trn_label.txt', transform_train, 4)
            # testset = GetData_split('../comp3C5C_dataset/' + fold + '/test', 'comp3C_label.txt', transform_test, 4)
            pass
        else:
            trainset = GetData_newrawloso2(traind, fold, transform_train, 4, foldid=2)
            testset = GetData_newrawloso2(datapaths, fold, transform_test, 4, foldid=0)
            pass
    # LOVO experimental protocol
    if Experimental_Protocol == "LOVO":
        if args.is_split == True:
            trainset = GetData_split('../comp3C5C_dataset/' + fold + '/train/', 'trn_label.txt', transform_train, 4)
            valset = GetData_split('../comp3C5C_dataset/' + fold + '/train', 'val_label.txt', transform_train, 4)
            testset = GetData_split('../comp3C5C_dataset/' + fold + '/test', 'comp3C_label.txt', transform_test, 4)
        else:
            dataset = GetData_raw('../comp3C5C_dataset/' + fold + '/train/', transform_train, 4)
            testset = GetData_raw('../comp3C5C_dataset/' + fold + '/test/', transform_test, 4)
        pass
    # get the train,val,test set
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.trainbatch_size, shuffle=args.train_shuffle,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.testbatch_size, shuffle=False, **kwargs)

    # create model，size=10
    if args.ispretrained:
        print(f'****model.initdata= {trafo_oth[0]}')
        print(f'***fc.initdata= {trafo_oth[1]}')
        model = torch.load(trafo_oth[0], map_location='cuda:0')
        fc = torch.load(trafo_oth[1], map_location='cuda:0')
        pass
    else:
        if args.model == 'TDFA':
            model = TDFA(args.size)
            # feature_num=128,class_num=3
        fc = Full_layer(int(model.feature_num), class_num)
        pass
    # weights manually initialized
    # if args.isXavier:
    #     model.apply(weights_init)
    #     fc.apply(weights_init)
    #     pass
    # Whether to save the current initialization model parameters
    if args.issavewbinit:
        saveinitmodel = '../initmodel'
        # If this folder saveinitmodel does not exist, create it first.
        if not os.path.exists(saveinitmodel):
            os.mkdir(saveinitmodel)
            pass
        torch.save(model, os.path.join(saveinitmodel, 'tdfaNmodel_' + fold + '_init' + '.pth'))
        torch.save(fc, os.path.join(saveinitmodel, 'fcmodel_' + fold + '_init' + '.pth'))
        pass

    # define loss function (criterion) and optimizer
    # Information on the weights assigned to each category
    u1 = trafo_oth[5][0]
    u2 = trafo_oth[5][1]
    u3 = trafo_oth[5][2]
    balencW = [u1, u2, u3]
    isda_criterion = ISDALoss(int(model.feature_num), class_num, balencW).to(device)
    ce_criterion = nn.CrossEntropyLoss().to(device)
    # Optimizer SGD
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 {'params': fc.parameters()}],
                                lr=args.initial_learning_rate,
                                momentum=args.momentum,
                                nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam([{'params': model.parameters()},
    #                               {'params': fc.parameters()}],
    #                              lr=args.initial_learning_rate)
    if args.lr_strategy == 0:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    if args.lr_strategy == 1:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer)
    if args.lr_strategy == 2:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if fold == 'sub01':
        files = open(modelpath, 'w+', encoding='utf-8')
        print('【0】datasets：', file=files)
        print(f'total length={len(trainset) + len(testset)}', file=files)
        print(f'trainset length={len(trainset)}', file=files)
        print(f'testset length={len(testset)}', file=files)

        print("\n【1】model structure：", file=files)
        print(model, file=files)
        print("fc structure：", file=files)
        print(fc, file=files)

        print('Number of final features: {}'.format(
            int(model.feature_num))
        )
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])
            + sum([p.data.nelement() for p in fc.parameters()])
        ))
        print('model parameter number: {}Million'.format(
            (sum([p.data.nelement() for p in model.parameters()])
             + sum([p.data.nelement() for p in fc.parameters()])) / 1000000
        ), file=files)

        print("\n【2】optimizer：", file=files)
        print(f'optimizer.lr=={optimizer.param_groups[0]["lr"]}', file=files)
        files.close()
        pass
    model = model.to(device)
    fc = fc.to(device)
    best_conf = []
    best_test_prec = 0
    trainloss = []
    epochx = []
    allaccc = []
    uf1S = []
    uarS = []

    stopflag = -1
    lrall = []
    testresult = []
    valuar = []
    valuf1 = []
    valaccs = []
    testuar = []
    testuf1 = []
    testaccs = []
    valloss = []

    losok += 1
    best_uf1 = 0
    best_uar = 0
    best_valloss = 0
    bestmetrics = []
    for epoch in range(args.Epoch):
        # Adjust the value of the learning rate according to the number of iterations
        print("Epoch [(current/total)=={}/{}]".format(epoch + 1, args.Epoch))
        adjust_learning_rate(optimizer, epoch + 1)
        # scheduler.step()

        print(f"current epoch= {epoch},lr={optimizer.param_groups[0]['lr']}")
        lrall.append(optimizer.param_groups[0]['lr'])
        global EPOCH
        EPOCH = epoch
        # train for one epoch

        features, targets, epochloss, acc0, umetrics = train(train_loader, model, fc, isda_criterion, optimizer, epoch,
                                                             fold, wi=[tu0, tu1, tu2], train_P=trafo_oth[4])
        test_prec, test_conf, testacc0, testumetrics = test(test_loader, model, fc, ce_criterion, epoch,
                                                            record_file)

        # remember best prec and save bestmodel
        is_best = testacc0 >= best_prec1
        best_prec1 = max(testacc0, best_prec1)
        best_uf1 = max(testumetrics[0][0], best_uf1)
        best_uar = max(testumetrics[0][1], best_uar)
        best_valloss = min(testumetrics[0][4], best_valloss)

        trainloss.append(epochloss)
        epochx.append(epoch)

        allaccc.append(acc0)
        uf1S.append(umetrics[0][0])
        uarS.append(umetrics[0][1])

        valaccs.append(testacc0)
        valuf1.append(testumetrics[0][0])
        valuar.append(testumetrics[0][1])
        valloss.append(float(testumetrics[0][4]))
        # Save the best model
        if is_best == True:
            nums = 0
            testaccs.append(testacc0)
            testuf1.append(testumetrics[0][0])
            testuar.append(testumetrics[0][1])
            bestmetrics.append(
                [float(f'{testacc0:.6f}'), float(f'{testumetrics[0][0]:.6f}'), float(f'{testumetrics[0][1]:.6f}'),
                 testumetrics[0][4]])

            testresult.append(
                f"{fold}_epoch{epoch}: testACC={testacc0:.6f},  testUF1={testumetrics[0][0]:.6f},  testUAR={testumetrics[0][1]:.6f} ; testLoss={testumetrics[0][4]:.6f}")
            for i in range(len(testumetrics[0][3])):
                if testumetrics[0][3][i] == testumetrics[0][2][i]:
                    nums += 1
                    pass
                pass
            testresult.append(f'y_true={testumetrics[0][3]}')
            testresult.append(f'y_pred={testumetrics[0][2]},right_num={nums}')
            torch.save(model, os.path.join(savemodel, 'tdfaNmodel_' + fold + "epoch" + str(epoch) + '.pth'))
            torch.save(fc, os.path.join(savemodel, 'fcmodel_' + fold + "epoch" + str(epoch) + '.pth'))
            pass

        pass
    trainloss2 = []
    trainacc = []
    trainuf1 = []
    trainuar = []
    savelrall = []
    trainloss2.extend([round(i, 4) for i in trainloss])
    trainacc.extend([round(i, 4) for i in allaccc])
    trainuf1.extend([round(i, 4) for i in uf1S])
    trainuar.extend([round(i, 4) for i in uarS])
    savelrall.extend([round(i, 8) for i in lrall])

    valacc2 = []
    valuf12 = []
    valuar2 = []
    valacc2.extend([round(i, 4) for i in valaccs])
    valuf12.extend([round(i, 4) for i in valuf1])
    valuar2.extend([round(i, 4) for i in valuar])

    testacc2 = []
    testuf12 = []
    testuar2 = []
    testacc2.extend([round(i, 4) for i in testaccs])
    testuf12.extend([round(i, 4) for i in testuf1])
    testuar2.extend([round(i, 4) for i in testuar])

    print(f"LOSO={losok},  EPOCH={args.Epoch}, testsample：【{fold}】", file=fdme2)
    print("aver_trainloss[last10]=", np.mean(trainloss2[-10:]), file=fdme2)
    print("aver_trainacc[last10]=", np.mean(trainacc[-10:]), file=fdme2)
    print('\n', "aver_testacc[last10]=", np.mean(valacc2[-10:]), file=fdme2)
    print(f"******Test result :", file=fdme2)
    for i in testresult:
        print(i, file=fdme2)
        pass
    # ******Best Model Filtering*****************************************************start
    allmodellist = np.array(bestmetrics)
    maxacc = np.max(allmodellist[:, 0])
    bestacclist = []
    for i in range(len(bestmetrics)):
        if maxacc == bestmetrics[i][0]:
            bestacclist.append(bestmetrics[i])
            pass
        pass
    bestacclists = np.array(bestacclist)
    maxuf1 = np.max(bestacclists[:, 1])
    maxuar = np.max(bestacclists[:, 2])
    minloss = np.min(bestacclists[:, 3])

    bestacclist0 = []
    for k in range(0, len(testresult), 3):
        topacc = float(testresult[k][testresult[k].find('=', testresult[k].find(':')) + 1:
                                     testresult[k].find(',')])
        if maxacc == topacc:
            bestacclist0.append(testresult[k])
            pass
        pass

    bestacclist = bestacclist0
    list2 = []
    list3 = []
    for k in range(len(bestacclist)):
        topuf1 = float(bestacclist[k][bestacclist[k].find('testUF1') + len('testUF1='):
                                      bestacclist[k].find(',', bestacclist[k].find('testUF1')) - 1])
        topuar = float(bestacclist[k][bestacclist[k].find('testUAR') + len('testUAR='):
                                      bestacclist[k].find(';') - 1])
        toploss = float(bestacclist[k][bestacclist[k].find('=', bestacclist[k].find(';')) + 1:])
        if topuf1 == maxuf1 and topuar == maxuar and toploss == minloss:
            list3.append(bestacclist[k])
            pass
        if topuf1 == maxuf1 and topuar == maxuar and toploss != minloss:
            # 这里将取loss最小值
            list2.append(bestacclist[k])
            pass
        pass
    if len(list3) != 0:
        toptestresult = list3[0]
        pass
    if len(list3) == 0 and len(list2) != 0:
        temploss = []
        for k in range(len(list2)):
            toploss = float(list2[k][list2[k].find('=', list2[k].find(';')) + 1:])
            temploss.append(toploss)
            pass
        tempminloss = np.min(np.array(temploss))
        for k in range(len(list2)):
            toploss = float(list2[k][list2[k].find('=', list2[k].find(';')) + 1:])
            if tempminloss == toploss:
                toptestresult = list2[k]
                pass
            pass
        pass
    if len(list3) == 0 and len(list2) == 0:
        for k in range(len(bestacclist)):
            toploss = float(bestacclist[k][bestacclist[k].find('=', bestacclist[k].find(';')) + 1:])
            if toploss == minloss:
                toptestresult = bestacclist[k]
                pass
            pass
        pass

    # Determination of the best model and its metrics
    print(f"******Top Test results:", file=fdme2)
    print(toptestresult, file=fdme2)
    print(70 * '+', file=fdme2)

    trainsavemodel = os.path.join(trainlosspath, 'trainresult').replace('\\', '/')
    testsavemodel = os.path.join(trainlosspath, 'testresult').replace('\\', '/')
    if not os.path.exists(trainsavemodel):
        os.makedirs(trainsavemodel)
    if not os.path.exists(testsavemodel):
        os.makedirs(testsavemodel)

    # Output experimental results for each model
    print(f'【LOSO testsub={fold}】', file=alldatafile)
    print(f'【epochx】=  {epochx}', file=alldatafile)
    print(f'【trainLoss】=  {trainloss2}', file=alldatafile)
    print(f'【trainAcc】=  {trainacc}', file=alldatafile)

    print(f'\n【testAcc】=  {valacc2}', file=alldatafile)
    print(f'【testLoss】=  {valloss}', file=alldatafile)
    print(f'【LR】=  {savelrall}', file=alldatafile)
    print(80 * '+', file=alldatafile)

    deletemodels(bestmodel=toptestresult, folds=fold)
    return toptestresult


def deletemodels(bestmodel, folds):
    temps = r'./result/modelpth'
    allmodels = os.listdir(temps)
    for i in range(len(allmodels)):
        if folds == allmodels[i][allmodels[i].find('_') + 1:allmodels[i].find('epoch')]:
            if bestmodel[bestmodel.find('epoch'):bestmodel.find(':')] != allmodels[i][
                                                                         allmodels[i].find('epoch'): allmodels[
                                                                             i].find('.pth')]:
                deltpath = os.path.join(temps, allmodels[i])
                os.remove(deltpath)
                pass
            pass
        pass
    pass


def train(train_loader, model, fc, isdacriterion, optimizer, epoch, fold, wi=[0.333, 0.333, 0.333], train_P=None):
    """Train for one epoch on the training set"""
    # batch_time = Initializemetrics()

    losses = Initializemetrics()
    print(f'train_P={train_P}')
    print(f'wi={wi}')
    ratio = train_P * (epoch / (args.Epoch))
    # switch to train mode
    model.train()
    fc.train()
    epochy_trues = []
    epochy_preds = []
    umetric = []
    balancelabels = []
    allmu = []

    # the dataloader of four inputs u,v,os,maguv
    tq = tqdm(train_loader, leave=True, ncols=100)
    batchs = 0
    iteration = 0
    old_features = 0
    old_targets = 0

    for i, (x1, x2, x3, x4, target) in enumerate(tq):
        iteration += 1
        torch.set_printoptions(precision=8, sci_mode=False)
        tq.set_description(f"test_subs={fold}")
        balancelabels.extend(target.data.tolist())
        target = target.to(device)
        batchs += 1
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)

        input_var1 = torch.autograd.Variable(x2)
        input_var2 = torch.autograd.Variable(x3)
        input_var3 = torch.autograd.Variable(x1)
        input_var4 = torch.autograd.Variable(x4)
        target_var = torch.autograd.Variable(target)

        mus = adjust_Balance_W(epoch, balancelabels)
        allmu.append(mus)

        # compute output

        loss, output, features, otheris = isdacriterion(model, fc, input_var1, input_var2, input_var3, input_var4,
                                                        target_var,
                                                        ratio, wi,
                                                        others=[fold, epoch, iteration, old_features, old_targets])

        if iteration == 1:
            all_features = features.cpu().detach().numpy()
            all_targets = target.cpu().detach().numpy()
        else:
            all_features = np.concatenate((all_features, features.cpu().detach().numpy()), axis=0)
            all_targets = np.concatenate((all_targets, target.cpu().detach().numpy()), axis=0)
            pass
        old_features = all_features
        old_targets = all_targets

        # measure accuracy and record loss

        _, pred = output.topk(1, 1, True, True)
        epochy_trues.extend(target.data.cpu().numpy().tolist())
        epochy_preds.extend(pred.view(1, -1).cpu().numpy().tolist()[0])

        print(
            f"\nepoch={epoch}/{args.Epoch},batch_i/allbatch={batchs}/{len(train_loader)},batch_loss={loss.data.item()}")
        losses.update(loss.data.item(), x2.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pass

    f1s, accs, uf1s, uars = modelmetric(epochy_preds, epochy_trues)
    umetric.append([uf1s, uars, epochy_preds, epochy_trues])

    return all_features, all_targets, losses.ave, accs, umetric


def test(test_loader, model, fc, criterion, epoch, record_file):
    """Perform test on the test set"""
    losses = Initializemetrics()
    top1 = Initializemetrics()

    train_batches_num = len(test_loader)

    # switch to evaluate mode
    model.eval()
    fc.eval()
    testepochy_trues = []
    testepochy_preds = []
    testumetric = []

    # end = time.time()
    conf_matrix = torch.zeros(3, 3)
    for i, (x, x2, x3, x4, target) in enumerate(test_loader):
        target = target.to(device)
        x = x.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)

        input_var1 = torch.autograd.Variable(x2)
        input_var2 = torch.autograd.Variable(x3)
        input_var3 = torch.autograd.Variable(x)
        input_var4 = torch.autograd.Variable(x4)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            features = model(input_var1, input_var2, input_var3, input_var4)
            if i == 0:
                all_features = features.cpu().detach().numpy()
                all_targets = target.cpu().detach().numpy()
            else:
                all_features = np.concatenate((all_features, features.cpu().detach().numpy()), axis=0)
                all_targets = np.concatenate((all_targets, target.cpu().detach().numpy()), axis=0)
            output = fc(features)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]

        losses.update(loss.data.item(), x2.size(0))
        top1.update(prec1.item(), x2.size(0))

        _, pred = output.topk(1, 1, True, True)
        testepochy_trues.extend(target.data.cpu().numpy().tolist())
        testepochy_preds.extend(pred.view(1, -1).cpu().numpy().tolist()[0])

        pass
    f1s, testacc, testuf1s, testuars = modelmetric(testepochy_preds, testepochy_trues)
    testumetric.append([testuf1s, testuars, testepochy_preds, testepochy_trues, float(f'{losses.ave:.6f}')])
    return top1.ave, conf_matrix.numpy(), testacc, testumetric


# Perform initialization methods
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0)
        pass
    if isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        pass
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        pass
    pass


# Construct classifiers of different sizes based on input parameters feature_num, class_num
class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        # feature_num=128,class_num=3
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


class Initializemetrics(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0
        pass


    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count
        pass


def adjust_Balance_W(epoch, label):
    classnum = []
    balanceratio = []
    mu = []
    epsilon = 0.002
    classnum.extend([label.count(i) for i in range(3)])
    balanceratio.extend([j / sum(classnum) + epsilon for j in classnum])
    mu.append(1.0 / (1 + balanceratio[0] / balanceratio[1] + balanceratio[0] / balanceratio[2]))
    mu.append(mu[0] * balanceratio[0] / balanceratio[1])
    mu.append(mu[0] * balanceratio[0] / balanceratio[2])

    return mu


def adjust_learning_rate(optimizer, epoch):
    """Keep only the best results and delete the rest of the cases"""
    if args.changelr == 12:
        if epoch < 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01
                pass
            pass
        if 20 <= epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
                pass
            pass
        pass
    pass

# Calculate F1,UF1,UAR,Acc, confusion matrix
def modelmetric(output, target):
    y_true = target
    y_pred = output
    f1 = resultAnalysis.GENf1_score(y_true, y_pred)
    acc = resultAnalysis.GENaccuracy_score(y_true, y_pred)
    uf1, uar = resultAnalysis.recognition_evaluation(y_true, y_pred)

    return f1, acc, uf1, uar


def accuracy(output, target, topk=(1,)):
    """Calculate model accuracy metrics"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def computefinalresult(finalresults):
    # Save all test result data for LOSO Best Model
    # (ACC,UF1,UAR,Loss)
    mid = 3
    alltestmetris = np.zeros((len(finalresults), mid))
    allvalmetris = np.zeros((len(finalresults), 4))
    for k in range(len(finalresults)):
        testresus = simpletop(finalresults[k])

        alltestmetris[k][0] = testresus[0]
        alltestmetris[k][1] = testresus[1]
        alltestmetris[k][2] = testresus[2]
        pass
    return allvalmetris, alltestmetris


def simpletop(topfinalresult):
    tacci = topfinalresult.find('testACC=')
    tuf1i = topfinalresult.find('testUF1=')
    tuari = topfinalresult.find('testUAR=')
    finalacc = topfinalresult[tacci + len('testACC='):tacci + len('testACC=') + 6]
    finaluf1 = topfinalresult[tuf1i + len('testUF1='):tuf1i + len('testUF1=') + 6]
    finaluar = topfinalresult[tuari + len('testUAR='):tuari + len('testUAR=') + 6]
    testresus = [float(finalacc.strip()), float(finaluf1.strip()), float(finaluar.strip())]
    return testresus


# **************************************************************************************************

# Function entry
def mains(inu0, inu1, inu2, classW, othersP=None):
    alldatafile = open(alldatapath, 'a+')
    fdme2 = open(fdme2path, 'a+')
    finaltopresults = []


    startime = f"{datetime.datetime.now():%Y.%m.%d.%H.%M.%S}"
    startt = time.time()
    labelpath = r'../comp3C5C_dataset/comp3C_label.txt'
    with open(labelpath, 'r+') as f:
        files = f.readlines()
        pass
    folds = []
    for name in files:
        if name[0:name.find('/')] in folds:
            continue
        folds.append(name[0:name.find('/')])
        pass
    i = 0
    # LOSO
    initpath = r'../pretrained/comp3init1'
    modelsluist = os.listdir(initpath)
    for fold in folds:
        record_path = args.save_path + '/' + fold
        if fold != 'total' and not os.path.exists(record_path):
            print('*************' + fold + ' starts training!*****************')
            if args.ispretrained:
                for k in range(len(modelsluist)):
                    if modelsluist[k][0:5] == 'tdfaN':
                        if fold == modelsluist[k][
                                   modelsluist[k].find('_') + 1:modelsluist[k].find('_', modelsluist[k].find('_') + 1)]:
                            tdfaN = os.path.join(initpath, modelsluist[k])
                            pass
                        pass
                    if modelsluist[k][0:3] == 'fcm':
                        if fold == modelsluist[k][
                                   modelsluist[k].find('_') + 1:modelsluist[k].find('_', modelsluist[k].find('_') + 1)]:
                            fcmodel = os.path.join(initpath, modelsluist[k])
                            pass
                        pass
                    pass
                pass
            else:
                tdfaN = None
                fcmodel = None
                pass

            # where fold refers to the testset samples
            # ******************Breakpoint training problems:
            # if args.savecheckpoint:
            #     if fold == args.breaksubxx:
            #         print(f'Breakpoints stop training ...... when fold={fold}')
            #         break
            #         pass
            #     pass
            # if args.loadcheckpoint:
            #     if fold in config.beforsubs:
            #         print(f'Skip the trained samples!!! Now fold={fold}')
            #         continue
            #         pass
            #     pass

            i += 1
            topfinalresult = trainFold(fold, record_path, tu0=inu0, tu1=inu1, tu2=inu2,
                                       trafo_oth=[tdfaN, fcmodel, alldatafile, fdme2, othersP, classW])

            finaltopresults.append(topfinalresult)
            pass
        pass
    allvalmetris, alltestmetris = computefinalresult(finaltopresults)

    # Select the best models to save in
    print(f'\n////////////////////【bestStart, the best model for all LOSOs is as follows:】///////////////////////////', file=fdme2)
    for i in finaltopresults:
        print(i, file=fdme2)
        pass
    print(f'////////////////////【bestEnd, the best model for all LOSOs is as above】///////////////////////////', file=fdme2)

    print(f'####【MAX and Aver results for all of the above LOSOs:】####', file=fdme2)
    print(
        f'【Max】testing：ACC={np.max(alltestmetris, axis=0)[0]:.4f}, UF1={np.max(alltestmetris, axis=0)[1]:.4f}, UAR={np.max(alltestmetris, axis=0)[2]:.4f}',
        file=fdme2)
    print(
        f'【Aver】testing：ACC={np.mean(alltestmetris, axis=0)[0]:.4f}, UF1={np.mean(alltestmetris, axis=0)[1]:.4f}, UAR={np.mean(alltestmetris, axis=0)[2]:.4f}',
        file=fdme2)
    print(f'### u0={inu0}, u1={inu1}, u2={inu2},isdaW={othersP}，classW={classW}', file=fdme2)
    print(70 * '+', file=fdme2)

    endtime = f"{datetime.datetime.now():%Y.%m.%d.%H.%M.%S}"
    endt = time.time()
    print(
        f'training:\n{startime}---{endtime}\ntraining time:\n{endt - startt:.4f}(s) or {(endt - startt) / 60:.4f}(min) or {(endt - startt) / 3600:.4f}(h)',
        file=fdme2)
    fdme2.close()
    alldatafile.close()

    print("The model training is Done!!! ")
    print("The model training is Done!!! \n")

    print("*************Finally, the model starts testing!!! ")
    finalltest()
    # deletefiles(specialpath='./result')
    print("*************Finally, the model tests Done!!! ")

    pass


# ***************************************************

if __name__ == '__main__':


    logpath = r'./result'
    # If the folder doesn't exist, create it.
    if not os.path.exists(logpath):
        os.mkdir(logpath)
        pass
    # print(torch.cuda.is_available())
    deletefiles(specialpath=r'./result', flags=1)
    fdme3pathn = r'./result/gridSearchResult.txt'
    preci = 1000
    c0 = 1.0
    c1 = 1.0
    c2 = 1.0
    lossW = [c0, c1, c2]
    mains(0.333, 0.333, 0.333, classW=lossW, othersP=0.5)

    pass
