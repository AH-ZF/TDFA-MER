'''
This py module focuses on testing the best performance of the model
'''

from torchvision import transforms
import os
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import args
from getdata import GetData_newrawloso2
from models.TDFA import TDFA
import resultAnalysis

device = torch.device(args.gpunum if torch.cuda.is_available() else 'cpu')

#
if args.chooseNormal == 3:
    # 3 classification of CDE
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [113.08480479, 113.45608963, 113.17704464]],
                                     std=[x / 255.0 for x in [16.41538289, 17.02655186, 17.19560204]])
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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
kwargs = {'num_workers': 0, 'pin_memory': True}

class_num = 3



# Initialization related metrics
class Initializemetrics(object):
    """Computes the metrics and current value"""

    def __init__(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count



def modelmetric(output, target):
    y_true = target
    y_pred = output
    # Calculate F1,UF1,UAR,Acc, confusion matrix
    f1 = resultAnalysis.GENf1_score(y_true, y_pred)
    selff1 = resultAnalysis.self_f1(y_true, y_pred)
    acc = resultAnalysis.GENaccuracy_score(y_true, y_pred)
    uf1, uar = resultAnalysis.recognition_evaluation(y_true, y_pred)
    paaccs, paf1_score, pauf1, pauar = resultAnalysis.paper_metric(y_true, y_pred)
    papermet = [paaccs, paf1_score, pauf1, pauar]

    return f1, selff1, acc, uf1, uar, papermet


def test(test_loader, model, fc, criterion):
    """Perform test on the testset"""

    losses = Initializemetrics()

    # Entering Test Mode
    model.eval()
    fc.eval()
    testepochy_trues = []
    testepochy_preds = []
    losssub = []


    tq = tqdm(test_loader, leave=True, ncols=100)
    for i, (x, x2, x3, x4, target) in enumerate(tq):
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
        losssub.append((loss.data.item()) * (x2.size(0)))
        losses.update(loss.data.item(), x2.size(0))

        _, pred = output.topk(1, 1, True, True)
        testepochy_trues.extend(target.data.cpu().numpy().tolist())
        testepochy_preds.extend(pred.view(1, -1).cpu().numpy().tolist()[0])
        pass

    return testepochy_preds, testepochy_trues, losssub,all_features,all_targets


def main(tdfaNmodels, fcmodels, fold):
    testset = GetData_newrawloso2('../comp3C5C_dataset/', fold, transform_test, 4, foldid=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, **kwargs)
    # [ ] Load model parameters
    model = torch.load(tdfaNmodels, map_location='cuda:0').to(device)
    fc = torch.load(fcmodels, map_location='cuda:0').to(device)

    ce_criterion = nn.CrossEntropyLoss().to(device)

    testepochy_preds, testepochy_trues, losssubs,rec_features,rec_targets = test(test_loader, model, fc, ce_criterion)

    return testepochy_preds, testepochy_trues, losssubs,rec_features,rec_targets


fdme2pathn = r'./result/modelresult.txt'
fdme3pathn = r'./result/gridSearchResult.txt'
modelres = r'./result/modelpth'


def newmetric():
    fdmenew = open(fdme2pathn, 'r+', encoding='utf-8')
    lines = fdmenew.readlines()
    bestmodel = []
    tdfaNmodellist = []
    fcmodellist = []
    testsub = []
    modelname = os.listdir(modelres)
    for i in range(len(lines)):
        if lines[i].find('bestStart') != -1:
            stindx = i
            pass
        if lines[i].find('bestEnd') != -1:
            endindx = i
            pass
        pass
    bestmodel = lines[stindx + 1:endindx]
    for j in bestmodel:
        if len(j) < 2:
            continue
        for k in modelname:
            if k[0:5] == 'tdfaN':
                if j[0:j.find('_')] == k[k.find('_') + 1:k.find('epoch')] and \
                        j[j.find('epoch'):j.find(':')] == k[
                                                          k.find('epoch'):k.find(
                                                              '.')]:
                    tdfaNmodellist.append(os.path.join(modelres, k).replace('\\', '/'))
                    testsub.append(j[0:j.find('_')])
                    pass
                pass
            if k[0:2] == 'fc':
                if j[0:j.find('_')] == k[k.find('_') + 1:k.find('epoch')] and \
                        j[j.find('epoch'):j.find(':')] == k[
                                                          k.find(
                                                              'epoch'):k.find(
                                                              '.')]:
                    fcmodellist.append(os.path.join(modelres, k).replace('\\', '/'))
                    pass
                pass
            pass
        pass

    return tdfaNmodellist, fcmodellist, testsub


# Remove redundant model files pth
def deletefiles(tdfaNmodellists, fcmodellists):
    path = r'./result/modelpth'
    files = os.listdir(path)
    finalfiles = []
    for i in range(len(tdfaNmodellists)):
        finalfiles.append(tdfaNmodellists[i][tdfaNmodellists[i].find('tdfaN'):])
        pass
    for i in range(len(fcmodellists)):
        finalfiles.append(fcmodellists[i][fcmodellists[i].find('fcmodel'):])
        pass
    for fil in files:
        if fil not in finalfiles:
            deltpath = os.path.join(path, fil)
            os.remove(deltpath)
            pass
        pass
    pass


def testresult(tdfaNmodellists, fcmodellists, testsubs, dataname):
    y_preds = []
    y_true = []
    allloss = []
    allnums = []
    singlesub = []
    singlenum = []
    samps = 0
    for fold in testsubs:
        samps += 1
        print(f"test subjects={fold}")
        for i in tdfaNmodellists:
            if i[i.find('_') + 1:i.find('epo')] == fold:
                tdfaNmodel = i
            pass
        for i in fcmodellists:
            if i[i.find('_') + 1:i.find('epo')] == fold:
                fcmodel = i
            pass
        print(f'tdfaN_best model={tdfaNmodel}')
        print(f'fc_best model={fcmodel}')

        testsuby_preds, testsub_trues, mlosssubs,rec_fea,rec_tar = main(tdfaNmodel, fcmodel, fold)

        if samps == 1:
            fin_features = rec_fea
            fin_targets = rec_tar
        else:
            fin_features = np.concatenate((fin_features, rec_fea), axis=0)
            fin_targets = np.concatenate((fin_targets, rec_tar), axis=0)


        singlesub.append([fold + ': ', testsub_trues, testsuby_preds])
        nums = 0
        for i in range(len(testsub_trues)):
            if testsub_trues[i] == testsuby_preds[i]:
                nums += 1
            pass
        print(f'{fold}: Number of samples predicted correctly ={nums}')
        allnums.append(nums)

        print(f'y_preds={testsuby_preds}')
        print(f'y_true={testsub_trues}')
        y_preds.extend(testsuby_preds)
        singlenum.append(len(testsub_trues))
        y_true.extend(testsub_trues)
        allloss.extend(mlosssubs)
        pass
    print(f"End of test, calculate metrics, sample size={len(y_preds)}")
    print(f"End of Test, Calculate Metrics, Total Number of Correct Samples Predicted={sum(allnums)}")
    print(f"End of Test, Calculate Metrics, Number of correct samples predicted={allnums}")

    f1s, selff1s, testacc, testuf1s, testuars, papermets = modelmetric(y_preds, y_true)
    finaloss = sum(allloss) / len(y_preds)
    girdf = open(fdme3pathn, 'a+', encoding='utf-8')
    print(f'***Finally Performance :{dataname}', file=girdf)
    print(
        f'Allrightnum={sum(allnums)}/{len(y_preds)}, Acc={testacc:.6f},F1={papermets[1]:.6f}, UF1={testuf1s:.6f}, UAR={testuars:.6f}, Loss={finaloss:.6f}',
        file=girdf)
    print(f"Number of correct samples predicted={allnums}", file=girdf)
    girdf.close()

    with open(fdme2pathn, 'a+', encoding='utf-8') as testf:
        print(72 * '+', file=testf)
        print(f'********Finally Performance :{dataname}', file=testf)
        print(
            f'Allrightnum={sum(allnums)}/{len(y_preds)}, Acc={testacc:.6f},F1={papermets[1]:.6f}, UF1={testuf1s:.6f}, UAR={testuars:.6f}, Loss={finaloss:.6f}',
            file=testf)
        print(f"Number of correct samples predicted={allnums}", file=testf)
        print(f"Theoretical sample={singlenum}", file=testf)
        pass
    resultAnalysis.GENconfusion_matrix(y_true, y_preds, savepath='./result/', index=0,
                                       imgnname='AAtestCMatrix' + dataname)

    pass


def finalltest():

    # Extract the trained model
    tdfaNmodellists, fcmodellists, testsubs = newmetric()

    lengths = len(testsubs)
    cassubs = []
    smicsubs = []
    sammsubs = []
    for j in range(lengths):
        # Calculate casme only
        if testsubs[j][0:3] == 'sub':
            cassubs.append(testsubs[j])
            pass
        # Calculate smic only
        if testsubs[j][0:3] != 'sub' and testsubs[j][0] == 's':
            smicsubs.append(testsubs[j])
            pass
        # Calculate samm only
        if testsubs[j][0:3] != 'sub' and testsubs[j][0] != 's':
            sammsubs.append(testsubs[j])
            pass
        pass

    testresult(tdfaNmodellists, fcmodellists, testsubs=sammsubs, dataname='SAMM')
    testresult(tdfaNmodellists, fcmodellists, testsubs=cassubs, dataname='CASME II')
    testresult(tdfaNmodellists, fcmodellists, testsubs=smicsubs, dataname='SMIC')
    testresult(tdfaNmodellists, fcmodellists, testsubs=testsubs, dataname='Comp3C')



if __name__ == '__main__':
    print('Overall testing starts ......')
    finalltest()
    print('The overall test is over!!! ')
    print('The overall test is over!!! ')
    print('The overall test is over!!! ')
    pass
