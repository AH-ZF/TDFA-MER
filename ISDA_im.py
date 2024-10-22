import numpy as np
import torch
import torch.nn as nn
from config import args

device = torch.device(args.gpunum if torch.cuda.is_available() else 'cpu')


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        # shape 3*128*128
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(device)
        self.Ave = torch.zeros(class_num, feature_num).to(device)
        self.Amount = torch.zeros(class_num).to(device)

    def update_CV(self, features, labels):
        # N,C,A  batch_size,class_num,feature_num
        # N=10,C=3,A=128
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(device)
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1
        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        # N*C*A, Feature Difference
        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)
        # permute  bmm  b*n*m b*m*p ->b*n*p
        # C*A*N C*N*A-> C*A*A
        '''
        var_temp refers to the covariance matrix under this batch
        '''
        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))
        ap = onehot.sum(0)
        '''
        onehot into the sum after the number of labels of each class for statistical summation
        sum_weight_CV for dimension expansion, for 3 * 128 * 128, the result is the code mjt
        sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A) is equivalent to summing the total number of samples in the previous t steps
        Similarly sum_weight_AV, and weight_AV, just with different dimensions
        '''
        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)
        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0
        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0
        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )
        # Constructing the covariance matrix
        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)


# Constructs are weighted mean features
def activew(allfeatures, labels, clasnum, newfeature):
    classfeas = torch.zeros((clasnum, allfeatures.size(1))).to(device)

    oldfeature = newfeature[0]
    oldlabel = newfeature[1]
    currentfeature = newfeature[2]
    currentlabel = newfeature[3]
    currClassfeas = torch.zeros((clasnum, currentfeature.size(1))).to(device)
    cnum = currentfeature.size(0)
    if newfeature[4] == 1:
        for i in range(clasnum):
            currClassfeas[i] = torch.mean(currentfeature[currentlabel == i], dim=0)
            pass
        classfeas = currClassfeas
        pass
    else:
        onum = oldfeature.size(0)
        oldClassfeas = torch.zeros((clasnum, oldfeature.size(1))).to(device)
        for i in range(clasnum):
            oldClassfeas[i] = torch.mean(oldfeature[oldlabel == i], dim=0)
            pass
        for i in range(clasnum):
            classfeas[i] = (onum * oldClassfeas[i] + cnum * currClassfeas[i]) / (onum + cnum)
            pass
        pass

    return classfeas


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num, u):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.flagsubs = ' '
        self.flagepochs = -1
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        pass

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)
        # bmm is to perform 3D matrix multiplication.
        # ratio = 1.0
        # The sigma2 here is the equation 17 in the paper, sigma2=10*3*3
        sigma2 = ratio * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).to(device)
                            .expand(N, C, C)).sum(2).view(N, C)

        aug_result = y + sigma2
        deltay = sigma2

        return aug_result, deltay

    def forward(self, model, fc, x1, x2, x3, x4, target_x, ratio, balanceW=0, others=None):
        target_classy = torch.tensor((0, 1, 2), device=device)
        features = model(x1, x2, x3, x4)
        # The y here is the output
        y = fc(features)
        if others[2] == 1:
            all_features = features.cpu().detach().numpy()
            all_targets = target_x.cpu().detach().numpy()
            newfeature = [others[3], others[4], features,
                          target_x, others[2]]
            pass
        else:
            all_features = np.concatenate((others[3], features.cpu().detach().numpy()), axis=0)
            all_targets = np.concatenate((others[4], target_x.cpu().detach().numpy()), axis=0)
            newfeature = [torch.from_numpy(others[3]).to(device), torch.from_numpy(others[4]).to(device), features,
                          target_x, others[2]]

            pass
        # Get weighted mean feature classification results
        classfeates = activew(torch.from_numpy(all_features).to(device), torch.from_numpy(all_targets).to(device),
                              self.class_num, newfeature)
        classy = fc(classfeates)

        self.estimator.update_CV(features.detach(), target_x)
        isda_aug_y, deltays = self.isda_aug(fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio)

        loss = self.cross_entropy(isda_aug_y, target_x)
        lossy0 = self.cross_entropy(y, target_x)
        otheris = [isda_aug_y, deltays, lossy0]
        # The other two loss functions
        lossy = self.cross_entropy(y, target_x)
        meany = self.cross_entropy(classy, target_classy)

        # Comprehensive loss function
        loss = 0.333 * (loss + lossy + meany)

        return loss, y, features, otheris
