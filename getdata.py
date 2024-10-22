# get data and data augmentation

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
from config import args
from sklearn.model_selection import StratifiedKFold

'''
get data and then split data randomly
'''


class GetData_raw(Dataset):
    # separately transform to four inputs
    def __init__(self, path1, transform, count, is_reshape=True):
        super(GetData_raw, self).__init__()
        self.path = path1
        self.transform = transform
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []
        print(os.path.join(self.path, 'label.txt'))
        self.dataset.extend(open(os.path.join(self.path, 'label.txt')).readlines())

    def __getitem__(self, index):
        str1 = self.dataset[index].strip()
        imgdata = []
        for i in range(self.count):
            imgpath = os.path.join(self.path, str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape == True:
                im = cv2.resize(im, (120, 120))
            imgdata.append(self.transform(im))
        label = int(str1.split(',')[self.count])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)



if args.chooseNormal == 3:
    # 3 classification of CDE
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [113.08480479, 113.45608963, 113.17704464]],
                                     std=[x / 255.0 for x in [16.41538289, 17.02655186, 17.19560204]])
    pass

# Enhanced training set only
class GetData_newrawloso2(Dataset):
    # separately transform to four inputs
    def __init__(self, path1, fold, transform, count, foldid, is_reshape=False):
        super(GetData_newrawloso2, self).__init__()
        self.path = path1
        self.paths = r'../comp3C5C_dataset/'
        self.transform = transform
        self.transformnew = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.flags = foldid
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []

        # Training set data
        if self.flags == 2:
            self.dataset.extend(self.path)
            if args.is_augment:
                self.dataset.extend(self.path)
                pass
            pass
        # Test set data
        if self.flags == 0:
            f = open(os.path.join(self.path, 'comp3C_label.txt'))
            imgdat = f.readlines()
            for i in range(len(imgdat)):
                if imgdat[i][0:imgdat[i].find('/')] == fold:
                    self.dataset.append(imgdat[i])
                pass
            f.close()
            pass

    def __getitem__(self, index):
        # print(f"*******index={index}")
        str1 = self.dataset[index].strip()
        imgdata = []

        for i in range(self.count):
            imgpath = os.path.join(self.paths, str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape:
                im = cv2.resize(im, (120, 120))
                pass
            # imgdata.append(self.transform(im))
            if len(self.dataset) > 500:
                if index < len(self.dataset) / 2:
                    imgdata.append(self.transform(im))
                else:
                    imgdata.append(self.transformnew(im))
                    pass
            else:
                imgdata.append(self.transformnew(im))

            pass
        label = int(str1.split(',')[self.count])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)


class GetData_newrawloso(Dataset):
    # separately transform to four inputs
    def __init__(self, path1, fold, transform, count, foldid, is_reshape=True):
        super(GetData_newrawloso, self).__init__()
        self.path = path1
        self.transform = transform
        self.transformnew = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.flags = foldid
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []

        # Training set data
        if self.flags == 1:
            f = open(os.path.join(self.path, 'label.txt'))
            imgdat = f.readlines()
            for i in range(len(imgdat)):
                if imgdat[i][0:3] != fold:

                    self.dataset.append(imgdat[i])

                pass

            f.close()
            pass
        # Test set data
        if self.flags == 0:

            f = open(os.path.join(self.path, 'label.txt'))
            imgdat = f.readlines()
            for i in range(len(imgdat)):
                if imgdat[i][0:3] == fold:
                    self.dataset.append(imgdat[i])
                pass
            f.close()
            pass

    def __getitem__(self, index):
        str1 = self.dataset[index].strip()
        imgdata = []

        for i in range(self.count):
            imgpath = os.path.join(self.path, str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape == True:
                im = cv2.resize(im, (120, 120))
                pass
            if index % 2 == 0:
                imgdata.append(self.transform(im))
            else:
                imgdata.append(self.transformnew(im))
                pass
            pass
        label = int(str1.split(',')[self.count])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)


class GetData_rawloso(Dataset):
    # separately transform to four inputs
    def __init__(self, path1, fold, transform, count, foldid, is_reshape=True):
        super(GetData_rawloso, self).__init__()
        self.path = path1
        self.transform = transform
        self.flags = foldid
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []

        # Training set data
        if self.flags == 1:
            f = open(os.path.join(self.path, 'label.txt'))
            imgdat = f.readlines()
            for i in range(len(imgdat)):
                if imgdat[i][0:5] != fold:
                    self.dataset.append(imgdat[i])
                pass
            f.close()
            pass
        # Test set data
        if self.flags == 0:
            f = open(os.path.join(self.path, 'label.txt'))
            imgdat = f.readlines()
            for i in range(len(imgdat)):
                if imgdat[i][0:5] == fold:
                    self.dataset.append(imgdat[i])
                pass
            f.close()
            pass

    def __getitem__(self, index):
        str1 = self.dataset[index].strip()
        imgdata = []

        for i in range(self.count):
            imgpath = os.path.join(self.path, str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape == True:
                im = cv2.resize(im, (120, 120))
            imgdata.append(self.transform(im))
            pass

        label = int(str1.split(',')[self.count])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)


'''
get data by spliting data
'''


class GetData_split(Dataset):
    def __init__(self, path1, txt_name, transform, count, is_reshape=True):
        super(GetData_split, self).__init__()
        self.path = path1
        self.transform = transform
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []
        self.dataset.extend(open(os.path.join(self.path, txt_name)).readlines())

    def __getitem__(self, index):
        str1 = self.dataset[index].strip()
        imgdata = []
        for i in range(self.count):
            imgpath = os.path.join(self.path, str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape == True:
                im = cv2.resize(im, (120, 120))
            imgdata.append(self.transform(im))
        label = int(str1.split(',')[self.count])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)
