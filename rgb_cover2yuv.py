import argparse
import glob
import random

import cv2
import numpy
import numpy as np
import io

save_sub = '.png'
save_sign = '_420'


def populate_train_list(orig_images_path, file_sub, valid_rate, bool_random):
    # /.../my_directory/(orig_images_path)*.bmp(file_sub)
    # 1. 給資料夾 讀取裡面所有檔名
    image_list_orig = glob.glob(orig_images_path + '*' + file_sub)
    tmp_img_name = []
    for image in image_list_orig:
        image = image.split("/")[-1]
        tmp_img_name.append(image)

    # 2. 檔名切成兩份: train & valid
    train_keys = []
    val_keys = []
    len_img = len(tmp_img_name)
    train_rate = 10 - valid_rate
    for i in range(len_img):
        if i < len_img * train_rate / 10:
            train_keys.append(tmp_img_name[i])
        else:
            val_keys.append(tmp_img_name[i])

    # 3.補上路徑
    train_list = []
    val_list = []
    for train_image in train_keys:
        train_list.append(orig_images_path + train_image)
    for valid_image in val_keys:
        val_list.append(orig_images_path + valid_image)

    # 4.亂序
    if bool_random:
        random.shuffle(train_list)
        random.shuffle(val_list)

    return train_list, val_list


def save_path(orig_images_path, save_dict, save_index):
    image = orig_images_path.split("/")[-1]
    image = image.split(".")[0]

    full_path = save_dict + image + save_sign + save_index + save_sub
    # full_path = save_dict + image + save_sub
    return full_path


def config_path(now, index):
    if index == 2:
        return True
    else:
        if now == index:
            return True
    return False



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. config para
    ###############################################################################
    # (1) load_train_data
    # ---------------------------------------------------- #
    parser.add_argument('--Path_LoadImages', type=str,
                        default="test_images/")
    parser.add_argument('--SaveSubName', type=str,
                        default=".png")
    parser.add_argument('--Valid_rate', type=int, default=0)
    parser.add_argument('--random', type=bool, default=False)
    # ---------------------------------------------------- #

    # (2) save_output
    # ---------------------------------------------------- #
    parser.add_argument('--Path_Save420Images', type=str,
                        default="output_420_images/")
    parser.add_argument('--Path_Save444Images', type=str,
                        default="output_444_images/")
    # index: 0(only444), 1:(only420), 2:(both)
    parser.add_argument('--Save_index', type=int, default=0)
    # ---------------------------------------------------- #

    # (3) test_config
    parser.add_argument('--show', type=bool, default=False)

    conf = parser.parse_args()
    save_index = conf.Save_index
    ###############################################################################

    # 2. loading file name (willy 20210408)
    ###############################################################################
    train_list, valid_list = populate_train_list(conf.Path_LoadImages, conf.SaveSubName,
                                                 conf.Valid_rate, conf.random)
    ###############################################################################

    # 3. main code (willy 20210408)
    ###############################################################################
    for file_path in train_list:
        # (1) Building the input:
        # --------------------------------------- #
        img = cv2.imread(file_path)
        # Convert BGR to YCrCb (YCrCb apply YCrCb JPEG
        # (or YCC), "full range",
        # where Y range is [0, 255], and U, V range is
        # [0, 255] (this is the default JPEG format color space format).
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if config_path(0, save_index):
            cv2.imwrite(save_path(file_path, conf.Path_Save444Images, '_yuv'), yuv)

        if config_path(1, save_index):
            y, v, u = cv2.split(yuv)
            # --------------------------------------- #

            # 2. (important) 利用 opencv.resize 的補差 做到 420 的效果
            # --------------------------------------- #
            # a. 雙線性補差：uv四格平均
            u = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2))
            v = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2))

            # b. 鄰近補差：相同值擴充4格
            u = cv2.resize(u, (u.shape[1] * 2, u.shape[0] * 2),
                           interpolation=cv2.INTER_NEAREST)
            v = cv2.resize(v, (v.shape[1] * 2, v.shape[0] * 2),
                           interpolation=cv2.INTER_NEAREST)
            # --------------------------------------- #

            # 3. merge yuv
            # --------------------------------------- #
            # Stack planes to 3D matrix (use Y,V,U ordering)
            # yuv = cv2.merge((y, u, v))
            if y.shape != u.shape:
                for dim in range(len(y.shape)):
                    if y.shape[dim] != u.shape[dim]:
                        index_start = u.shape[dim]
                        index_end = y.shape[dim]
                        y = numpy.delete(y, numpy.s_[index_start:index_end], axis=dim)

            yuv_420 = cv2.merge([y, u, v])
            if conf.show:
                cv2.imshow("yuv incorrect colors", yuv_420)
                cv2.waitKey()
                cv2.destroyWindow('yuv incorrect colors')
            cv2.imwrite(save_path(file_path, conf.Path_Save420Images, '_yuv'), yuv_420)
        # --------------------------------------- #

        '''
        # 4. Convert to rgbScreenshot from 2021-04-08 15-36-57
        # --------------------------------------- #
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        if conf.show:
            cv2.imshow("bgr incorrect colors", bgr)
            cv2.waitKey()
            cv2.destroyWindow('bgr incorrect colors')
        cv2.imwrite(save_path(file_path, conf.Path_SaveImages,  '_bgr'), bgr)

        # --------------------------------------- #
        '''

    '''
    filename = '0801'
    # filename = 'Blue0'
    # filename = 'random0'
    # filename = 'Red_line0'

    file_type = '.png'

    img = cv2.imread(filename + file_type)
    # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # y, u, v = cv2.split(yuv)

    # Convert BGR to YCrCb (YCrCb apply YCrCb JPEG
    # (or YCC), "full range",
    # where Y range is [0, 255], and U, V range is
    # [0, 255] (this is the default JPEG format color space format).
    yvu = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, v, u = cv2.split(yvu)
    # --------------------------------------- #

    # test : split resize
    # b, g, r = cv2.split(img)

    # 2. (important) 利用 opencv.resize 的補差 做到 420 的效果
    # ----------------------------------------
    # a. 雙線性補差：uv四格平均
    u = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2))
    v = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2))

    # b. 鄰近補差：相同值擴充4格
    u = cv2.resize(u, (u.shape[1] * 2, u.shape[0] * 2),
                   interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(v, (v.shape[1] * 2, v.shape[0] * 2),
                   interpolation=cv2.INTER_NEAREST)
    # ----------------------------------------

    # 3.
    yuv = cv2.merge((y, u, v))  # Stack planes to 3D matrix (use Y,V,U ordering)
    cv2.imshow("yuv incorrect colors", yuv)
    cv2.waitKey()
    cv2.destroyWindow('yuv incorrect colors')

    yuv_data = cv2.merge([y, u, v])
    bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR)

    cv2.imshow("bgr incorrect colors", bgr)
    cv2.waitKey()
    cv2.destroyWindow('bgr incorrect colors')

    file_compress = '_420'
    file_savetype = '.bmp'
    cv2.imwrite(filename + file_compress + file_savetype, bgr)
    '''

    '''
    # Read YUV420 (I420 planar format) and convert to BGR
    # Open In-memory bytes streams (instead of using fifo)
    f = io.BytesIO()

    # Write Y, U and V to the "streams".
    # add remarks from willy : to byte data
    f.write(y.tobytes())
    f.write(u.tobytes())
    f.write(v.tobytes())

    # add remarks from willy : find the stream start
    f.seek(0)
    ###############################################################################

    y0 = f.read(y.size)
    u0 = f.read(y.size)
    v0 = f.read(y.size)
 
    # How to How should I be placing the u and v channel information in all_yuv_data?
    # -------------------------------------------------------------------------------

    # Read YUV420 (I420 planar format) and convert to BGR
    ###############################################################################
    data = f.read(y.size * 3 // 2)  # Read one frame (number of bytes is width*height*1.5).

    # Reshape data to numpy array with height*1.5 rows
    yuv_data = np.frombuffer(data, np.uint8).reshape(y.shape[0] * 3 // 2, y.shape[1])

    # Convert YUV to BGR
    bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420);

    # How to How should I be placing the u and v channel information in all_yuv_data?
    # -------------------------------------------------------------------------------
    # Example: place the channels one after the other (for a single frame)
    f.seek(0)
    y0 = f.read(y.size)
    u0 = f.read(y.size // 4)
    v0 = f.read(y.size // 4)
    yuv_data = y0 + u0 + v0
    yuv_data = np.frombuffer(yuv_data, np.uint8).reshape(y.shape[0] * 3 // 2, y.shape[1])
    bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
    ###############################################################################

    # Display result:
    cv2.imshow("bgr incorrect colors", bgr)
    cv2.waitKey()

    ###############################################################################
    f.seek(0)
    y = np.frombuffer(f.read(y.size), dtype=np.uint8).reshape(
        (y.shape[0], y.shape[1]))  # Read Y color channel and reshape to height x width numpy array
    u = np.frombuffer(f.read(y.size // 4), dtype=np.uint8).reshape(
        (y.shape[0] // 2, y.shape[1] // 2))  # Read U color channel and reshape to height x width numpy array
    v = np.frombuffer(f.read(y.size // 4), dtype=np.uint8).reshape(
        (y.shape[0] // 2, y.shape[1] // 2))  # Read V color channel and reshape to height x width numpy array

    # Resize u and v color channels to be the same size as y
    u = cv2.resize(u, (y.shape[1], y.shape[0]))
    v = cv2.resize(v, (y.shape[1], y.shape[0]))
    yvu = cv2.merge((y, v, u))  # Stack planes to 3D matrix (use Y,V,U ordering)
    '''

    # --------------------------------------------------------------------------- #
    ###############################################################################

    # save data
