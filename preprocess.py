from typing import Union
import numpy as np
import pandas as pd
import cv2, os, xlrd, time
import copy, shutil, dlib
from tqdm import tqdm


def center_crop(img: np.array, crop_size: Union[tuple, int]) -> np.array:
    """Returns center cropped image

    Parameters
    ----------
    img : [type]
        Image to do center crop
    crop_size : Union[tuple, int]
        Crop size of the image

    Returns
    -------
    np.array
        Image after being center crop
    """
    width, height = img.shape[1], img.shape[0]

    # Height and width of the image
    mid_x, mid_y = int(width / 2), int(height / 2)

    if isinstance(crop_size, tuple):
        crop_width, crop_hight = int(crop_size[0] / 2), int(crop_size[1] / 2)
    else:
        crop_width, crop_hight = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[mid_y - crop_hight:mid_y + crop_hight, mid_x - crop_width:mid_x + crop_width]

    return crop_img


def TVL1_optical_flow(prev_frame: np.array, next_frame: np.array):
    """Compute the TV-L1 optical flow and normalized the result"""
    # Transform the image from BGR to Gray
    # if prev_frame.shape()
    # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    # next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

    # Create TV-L1 optical flow
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(scaleStep=0.5)
    flow = optical_flow.calc(prev_frame, next_frame, None)

    return flow


def TVL1_magnitude(flow: np.array):
    """Compute the magnitude of the frame"""
    flow = copy.deepcopy(flow)
    mag = np.sqrt(np.sum(flow ** 2, axis=-1))
    mag = normalized_channel(mag)

    return mag


def normalized_channel(frame):
    min_value = np.amin(frame, axis=(0, 1), keepdims=True)
    max_value = np.amax(frame, axis=(0, 1), keepdims=True)
    frame = (frame - min_value) / (max_value - min_value + 1e-8) * 255
    frame = np.minimum(frame, 255)
    frame = np.maximum(frame, 0)
    return frame.astype("uint8")


def optical_strain(flow: np.array) -> np.array:
    """Compute the optical strain for the given u, v
    Refer to: https://github.com/mariaoliverparera/mod-opticalStrain/blob/master/get_contours.py

    Parameters
    ----------
    flow : np.array
        Normalized horizontal and vertical optical flow fields

    Returns
    -------
    np.array
        Return the optical strain magnitude for u, v
    """
    flow = copy.deepcopy(normalized(flow))
    u = flow[..., 0]
    v = flow[..., 1]

    # Compute the gradient
    u_x = u - np.roll(u, 1, axis=1)
    u_y = u - np.roll(u, 1, axis=0)
    v_x = v - np.roll(v, 1, axis=1)
    v_y = v - np.roll(v, 1, axis=0)

    e_xy = 0.5 * (u_y + v_x)

    e_mag = np.sqrt(u_x ** 2 + 2 * (e_xy ** 2) + v_y ** 2)
    e_mag = normalized_channel(e_mag)

    return e_mag


def gray_frame(frame):
    """Transform the frame into Gray and normalized the result"""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return frame


def compute_features(onset_frame: np.array, apex_frame: np.array):
    # Compute flow
    flow = TVL1_optical_flow(prev_frame=onset_frame, next_frame=apex_frame)
    # Compute magnitude and strain
    flow_mag = TVL1_magnitude(flow)
    strain_mag = optical_strain(flow)

    # flow = np.minimum(normalized_(flow), 255)
    # flow = np.maximum(flow, 0)
    # flow = flow.astype("uint8")
    flow = normalized_channel(flow)

    # Use next_frame to make gray frame
    gray = gray_frame(apex_frame)
    # Four parameters: optical flow, optical flow amplitude, optical strain, gray scale image
    return flow, flow_mag, strain_mag, gray


def normalized(frame: np.array,
               g_min: float = -128,
               g_max: float = 128,
               lambda_: int = 16):
    # Do the normalization
    if len(frame.shape) > 2:
        f_min = np.amin(frame, axis=(0, 1))
        f_max = np.amax(frame, axis=(0, 1))
    else:
        f_min = np.min(frame)
        f_max = np.max(frame)

    frame = lambda_ * (frame - f_min) * (g_max - g_min) / (f_max - f_min + 1e-8) + g_min

    return frame


def normalized_(frame: np.array, scaling: int = 16, shifting: int = 128):
    return frame * scaling + shifting


def save_flow(video_flows, flow_path):
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, 'u', "{:06d}.png".format(i)),
                    flow[:, :, 0])  # Saving the optical flow in the x-direction
        cv2.imwrite(os.path.join(flow_path, 'v', "{:06d}.png".format(i)),
                    flow[:, :, 1])  # Saving the optical flow in the y-direction



def inputdata(path):
    filename = os.listdir(path)
    for objname in filename:
        objfile = os.path.join(path, objname)
        objv = os.listdir(objfile)
        for objfigure in objv:
            subobjv = os.path.join(objfile, objfigure)
            figures = os.listdir(subobjv)
            figuress = sorted(figures, key=lambda x: int(x[3:-4]))
            for imgname in figuress:
                imagename = os.path.join(subobjv, imgname)
                frame = cv2.imread(imagename, 1)
                grayimage = gray_frame(frame)

                print(" ")

    pass


def main(origin_path, newpath, listexcel):
    '''

    :param origin_path:
    :param newpath:
    :param listexcel:
    :return:
    '''
    for obj in listexcel:
        if str(obj[2]).find('.') > 0 and str(obj[3]).find('.') > 0:
            if str(obj[2]) != str(obj[3]):
                oripath = os.path.join(origin_path, 'sub' + obj[0], obj[1])
                objv = os.listdir(oripath)
                savepath = os.path.join(newpath, 'sub' + obj[0], obj[1])
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                for imgname in objv:
                    srcpath = os.path.join(oripath, imgname)
                    if int(imgname[3:-4]) == int(obj[2]):
                        shutil.copy2(srcpath, savepath)
                    if int(imgname[3:-4]) == int(obj[3]):
                        shutil.copy2(srcpath, savepath)

    pass


def oneapex(origin_path, excel_path, newpath):
    workbook = xlrd.open_workbook(excel_path)
    # Read the first sheet as an index:
    Data_sheet = workbook.sheet_by_index(0)
    rowNum = Data_sheet.nrows
    colNum = Data_sheet.ncols
    print(rowNum)
    print(colNum)
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    listexcel = []
    for i in range(1, rowNum):
        rows = Data_sheet.row_values(i)
        tmplist = []
        tmplist.append(rows[0])
        tmplist.append(rows[1])
        if rows[3] is not None:
            tmplist.append(rows[3])
        if rows[4] is not None:
            tmplist.append(rows[4])
        listexcel.append(tmplist)
        pass
    print(listexcel)

    main(origin_path, newpath, listexcel)
    pass


# Perform face detection cropping:
def facecut(image1, image2):
    face_path_68 = r"./shape_predictor_68_face_landmarks.dat"
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_path_68)
    # The face is detected, the face position rectangular box is extracted and the output is the rectangular box coordinates
    faces = detector(gray, 0)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 1)
    landmarks = predictor(gray, face)
    print(landmarks.num_parts)
    col = []
    row = []
    for n in range(0, landmarks.num_parts):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        col.append(landmarks.part(n).x)
        row.append(landmarks.part(n).y)
        text = str(n)
        # cv2.putText(image1, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1)
        # cv2.circle(image1, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)
        pass
    minx = min(col)
    maxx = max(col)
    miny = min(row)
    maxy = max(row)
    return image1[miny - 20:maxy, minx:maxx], image2[miny - 20:maxy, minx:maxx]


# Save optical flow images
def saveflow(u, v, savpath):
    cv2.imwrite(os.path.join(savpath, 'u.jpg'), u)
    cv2.imwrite(os.path.join(savpath, 'v.jpg'), v)

    pass


def osComputer(flow):
    flow = copy.deepcopy(normalized(flow))
    u = flow[..., 0]
    v = flow[..., 1]

    # Compute the gradient
    u_x = u - np.roll(u, 1, axis=1)
    u_y = u - np.roll(u, 1, axis=0)
    v_x = v - np.roll(v, 1, axis=1)
    v_y = v - np.roll(v, 1, axis=0)

    e_xy = 0.5 * (u_y + v_x)

    e_mag = np.sqrt(u_x ** 2 + 2 * (e_xy ** 2) + v_y ** 2)
    e_mag = normalized_channel(e_mag)
    return e_mag


def ofMagnitude(tempflowu, tempflowv, savepath):
    # flow = copy.deepcopy(flow)
    tempflowu = np.array(tempflowu)
    tempflowv = np.array(tempflowv)
    magu = np.round(np.sqrt(tempflowu ** 2 + tempflowv ** 2))

    cv2.imwrite(os.path.join(savepath, 'maguv.jpg'), magu)

    pass


def imagecut_resize():
    objs = os.listdir(oneapexpath)
    for objname in objs:
        objpath = os.path.join(oneapexpath, objname)
        objvideo = os.listdir(objpath)
        for objv in objvideo:
            objpath1 = os.path.join(objpath, objv)
            objvideo1 = os.listdir(objpath1)
            img = sorted(objvideo1, key=lambda x: int(x[3:-4]))
            image1 = cv2.imread(os.path.join(objpath1, img[0]))
            image2 = cv2.imread(os.path.join(objpath1, img[1]))
            img1, img2 = facecut(image1, image2)

            img1 = cv2.resize(img1, (120, 120))
            img2 = cv2.resize(img2, (120, 120))
            tempflowu = np.zeros_like(img1, dtype=float)
            tempflowv = np.zeros_like(img1, dtype=float)
            for i in range(img1.shape[2]):
                # Calculate the optical flow for each channel and save the dimension of flow as (120*120*2)
                flow = TVL1_optical_flow(img1[:, :, i], img2[:, :, i])
                # tempflowu[..., i] = flow[..., 0]
                # tempflowv[..., i] = flow[..., 1]
                maxu = max(flow[..., 0].flatten())
                minu = min(flow[..., 0].flatten())
                maxv = max(flow[..., 1].flatten())
                minv = min(flow[..., 1].flatten())
                tempflowu[..., i] = np.round(255 * (flow[..., 0] - minu) / (maxu - minu))
                tempflowv[..., i] = np.round(255 * (flow[..., 1] - minv) / (maxv - minv))
                pass
            savepath = os.path.join(ospath, objname, objv)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            saveflow(tempflowu, tempflowv, savepath)
            # Save three-channel optical flow magnitude images
            ofMagnitude(tempflowu, tempflowv, savepath)
    pass


def facedetect(images):
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(r"./data/haarcascades/haarcascade_frontalface_alt_tree.xml")
    faceRects = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    color = (0, 255, 0)
    if len(faceRects):
        for faceRect in faceRects:
            # x, y denote the coordinates; w, h denote the rectangle width and height
            x, y, w, h = faceRect
            print(faceRect)
            cv2.rectangle(images, (x, y), (x + h, y + w), color, 2)
            pass
        pass
    cv2.imshow("image", images)
    cv2.waitKey(0)
    pass


def ofcompute(oneapexpath, ospath):
    objs = os.listdir(oneapexpath)
    excel_path = r'./Dataset/CASME2_coding.xlsx'
    workbook = xlrd.open_workbook(excel_path)
    Data_sheet = workbook.sheet_by_index(0)
    rowNum = Data_sheet.nrows
    colNum = Data_sheet.ncols
    # tq = tqdm(objs, leave=True, ncols=100)
    k = 0
    counts0 = 0
    counts1 = 0
    counts2 = 0

    for objname in objs:
        objpath = os.path.join(oneapexpath, objname)
        objvideo = os.listdir(objpath)
        tq = tqdm(objvideo, leave=False, ncols=100)
        k += 1
        for objv in tq:
            tq.set_description("Subject_num [{}/{}]".format(k, len(objs)))
            objpath1 = os.path.join(objpath, objv)
            objvideo1 = os.listdir(objpath1)
            # Sorting, ensuring the relationship between the start and peak frames, cropping based on the first frame
            img = sorted(objvideo1, key=lambda x: int(x[3:-4]))
            image1 = cv2.imread(os.path.join(objpath1, img[0]))
            image2 = cv2.imread(os.path.join(objpath1, img[1]))
            # img1, img2 = facecut(image1, image2)
            # img1 = cv2.resize(img1, (120, 120))
            # img2 = cv2.resize(img2, (120, 120))
            img1 = image1
            img2 = image2
            # Save the u, v, os of the three channels
            tempflowu = np.zeros_like(img1, dtype=float)
            tempflowv = np.zeros_like(img1, dtype=float)
            tempflowos = np.zeros_like(img1, dtype=float)
            for i in range(img1.shape[2]):
                flow = TVL1_optical_flow(img1[:, :, i], img2[:, :, i])
                # tempflowu[..., i] = flow[..., 0]
                # tempflowv[..., i] = flow[..., 1]
                # Calculate optical strain
                flowOS = osComputer(flow)
                tempflowos[..., i] = flowOS
                maxu = max(flow[..., 0].flatten())
                minu = min(flow[..., 0].flatten())
                maxv = max(flow[..., 1].flatten())
                minv = min(flow[..., 1].flatten())
                tempflowu[..., i] = np.round(255 * (flow[..., 0] - minu) / (maxu - minu))
                tempflowv[..., i] = np.round(255 * (flow[..., 1] - minv) / (maxv - minv))
                pass
            savepath = os.path.join(ospath, objname, objv)
            resultpath = os.path.join(ospath, 'result.txt')
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            saveflow(tempflowu, tempflowv, savepath)
            # Save three-channel optical strain images
            cv2.imwrite(os.path.join(savepath, 'os.jpg'), tempflowos)

            f = open(resultpath, 'a+', encoding='utf-8')
            for i in range(rowNum):
                rows = Data_sheet.row_values(i)
                label = -1
                if 'sub' + rows[0] == objname and rows[1] == objv:
                    # Divided into 3 classifications
                    if rows[8] in ['disgust', 'repression']:
                        label = 0  # negative
                        counts0 += 1
                    if rows[8] == 'happiness':
                        label = 1  # positive
                        counts1 += 1
                    if rows[8] == 'surprise':
                        label = 2  # surprise
                        counts2 += 1
                    pass
                if label != -1:
                    fathpath = objname + '/' + objv + '/'
                    strings = fathpath + 'u.jpg' + ',' + \
                              fathpath + 'v.jpg' + ',' + \
                              fathpath + 'os.jpg' + ',' + \
                              fathpath + 'maguv.jpg' + ',' + \
                              str(label)
                    f.write(strings)
                    f.write('/n')
                    pass
                pass
            f.close()
            # Save three-channel optical flow magnitude images
            ofMagnitude(tempflowu, tempflowv, savepath)
            pass
        pass
    print(f"label0 ={counts0}")
    print(f"label1 ={counts1}")
    print(f"label2 ={counts2}")


pass



# Delete files in the specific directory
def deletefiles(specialpath, flags=0):
    file = os.listdir(specialpath)
    for i in range(len(file)):
        currfiles = os.path.join(specialpath, file[i])
        if os.path.isfile(currfiles):
            os.remove(currfiles)
            pass
        if os.path.isdir(currfiles):
            subfile = os.listdir(currfiles)
            for j in range(len(subfile)):
                subcurrfiles = os.path.join(currfiles, subfile[j])
                if os.path.isfile(subcurrfiles):
                    os.remove(subcurrfiles)
                    pass
                pass
            pass
        pass
    pass



if __name__ == '__main__':
    print("Opencv Version=", cv2.__version__)

    # origin_path = r'./Dataset/CASMEII/CASME2-RAW'
    # excel_path = r'./Dataset/CASME2_coding.xlsx'
    # oneapexpath = r'./Dataset/CASMEII/CASME2-one-apex'
    #
    # ospath = r'./Dataset/CASMEII/CASME2-of'
    ### Extract the start and peak frames of the sequence image and save them under the path oneapexpath.
    # The main purpose is to extract the start and apex frames.
    # oneapex(origin_path, excel_path, oneapexpath)
    ### 【】Face cropping, calculate the optical flow of the start and peak frames, save it under ospath 120*120*3
    # Mainly to calculate OF, OS, Magof
    # ofcompute(oneapexpath, ospath)

    print(' ')
