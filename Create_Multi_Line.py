import os
import random
import numpy as np
import cv2
from random import randrange
import argparse
import datetime
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--min_range', type=int, default=4)
    parser.add_argument('--numWidth', type=int, default=16)
    parser.add_argument('--numHeight', type=int, default=16)
    parser.add_argument('--background', type=str, default='black')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--sub_folder', type=str, default='chessboard/')

    parser.add_argument('--line_color', type=str, default='random')
    parser.add_argument('--num_images', type=int, default=2)
    parser.add_argument('--num_start', type=int, default=0)

    parser.add_argument('--thickness', type=int, default=1)

    config = parser.parse_args()
    width = config.width
    height = config.height
    min_range = config.min_range

    numWidth = config.numWidth
    if numWidth > width / min_range:
        numWidth = width / min_range

    numHeight = config.numHeight
    if numHeight > height / min_range:
        numHeight = height / min_range

    num_images = config.num_images
    num_start = config.num_start

    # define background
    # ----------------------------------------------------- #
    BackGroundName = config.background
    color_background = [0, 0, 0]
    if BackGroundName == 'white':
        color_background = [255, 255, 255]
    elif BackGroundName == 'index':
        color_background = [255, 128, 0]
    # ----------------------------------------------------- #

    # define color
    # ----------------------------------------------------- #
    line_color = config.line_color
    color = [0, 0, 0]
    FileName = 'black_line'

    if line_color == 'red':
        color = [0, 0, 255]
        FileName = 'red_line'
    elif line_color == 'green':
        color = [0, 255, 0]
        FileName = 'green_line'
    elif line_color == 'blue':
        color = [255, 0, 0]
        FileName = 'blue_line'
    elif line_color == 'index':
        color = [255, 128, 0]
        FileName = 'index_line'
    elif line_color == 'random':
        color = [randrange(256), randrange(256), randrange(256)]
        if BackGroundName == 'mix':
            while color == [255, 255, 255] or color == [0, 0, 0]:
                color = [randrange(256), randrange(256), randrange(256)]
        else:
            while color == color_background:
                color = [randrange(256), randrange(256), randrange(256)]
        FileName = 'random_line'
    # ----------------------------------------------------- #

    # TODO: INIT IMAGE SAVE PATH
    # system path check ...
    # make directory
    # clear file in directory
    # ----------------------------------------------------- #
    save_path = config.save_path
    sub_folder = config.sub_folder

    # TODO: clear file in directory
    '''
    import os, shutil
    folder = '/path/to/folder'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    '''

    # TODO: check data in path
    '''    
    tmp_img_name = []
    image_list_orig = glob.glob(orig_images_path + "*.bmp")
    for image in image_list_orig:
        image = image.split("/")[-1]
        tmp_img_name.append(image)
    '''
    # ----------------------------------------------------- #

    folder_name = save_path + sub_folder
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    now = datetime.now()
    now = datetime.strftime(now, '%Y-%m-%d %H:%M:%S')
    folder_name_temp = folder_name + now + '/'
    if os.path.isdir(folder_name_temp):
        now = datetime.now()
        now = datetime.strftime(now, '%Y-%m-%d %H:%M:%S')
        folder_name_temp = folder_name + now + '/'
    folder_name = folder_name_temp
    os.mkdir(folder_name)
    folder_name = folder_name + FileName + '_' + BackGroundName + '_'

    for i in range(num_start, num_images):
        pt_width = []
        pt_height = []

        if line_color == 'random':
            color = [randrange(256), randrange(256), randrange(256)]
            while color == color_background:
                color = [randrange(256), randrange(256), randrange(256)]

        # define line point
        # ----------------------------------------------------- #
        for indexW in range(numWidth):
            randW = random.randint(1, width)

            while randW in pt_width:
                randW = random.randint(1, width)

            # TODO: 20210111... how to do reset for loop?  while
            bTrying = True
            while bTrying:
                bTrying = False
                for w in pt_width:
                    if abs(w - randW) < min_range:
                        randW = random.randint(1, width)
                        bTrying = True
                        break

            pt_width.append(randW)

        for indexH in range(numHeight):
            randH = random.randint(1, height)

            while randH in pt_height:
                randH = random.randint(1, height)

            bTrying = True
            while bTrying:
                bTrying = False
                for h in pt_height:
                    if abs(h - randH) < min_range:
                        randH = random.randint(1, height)
                        bTrying = True
                        break

            pt_height.append(randH)
        # ----------------------------------------------------- #

        img = np.zeros((height, width, 3), np.uint8)

        for h in range(height):
            for w in range(width):
                if h in pt_height or w in pt_width:
                    img[h, w] = color
                else:
                    if BackGroundName == 'mix':
                        if i % 2 == 0:
                            img[h, w] = [255, 255, 255]
                    else:
                        img[h, w] = color_background
        # save our image as a "jpg" image
        # for all path
        cv2.imwrite(folder_name + str(i).zfill(4) + ".png", img)
