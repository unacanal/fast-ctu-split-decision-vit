import os
import cv2
import numpy as np
from glob import glob
import shutil
import csv

fps120_sequence = ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'Lips', 'ReadySetGo', 'ShakeNDry', 'YachtRide'] # 8
# fps50_sequence = ['CityAlley', 'FlowerFocus', 'FlowerKids', 'FlowerPan', 'RaceNight', 'RiverBank', 'SunBath', 'Twilight'] # 8

def get_CU_VVC(sourcedir, csvpath, cupath, block_size, number_of_frames):
    csv_path_list = glob(csvpath + '/*.csv')
    print(len(csv_path_list))
    for csv_path in csv_path_list:
        csvfile_name = csv_path.split(os.path.sep)[8]
        if csvfile_name.split('_')[0] in fps120_sequence:
            fps = 120
        else:
            fps = 50
        sequence_source_path = sourcedir + '/' + \
                               csvfile_name.split('_')[0] + '_' + \
                               csvfile_name.split('_')[1] + '_' + \
                               csvfile_name.split('_')[2] + '_' + \
                               str(fps) + 'fps_420_10bit_YUV.yuv'
        w = int(csvfile_name.split('_')[2].split('x')[0])
        h = int(csvfile_name.split('_')[2].split('x')[1])

        y_all = get_Luma(sequence_source_path, number_of_frames, w, h)

        # load csv file
        f = open(csv_path)
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            poc = int(row[0])
            x = int(row[1])
            y = int(row[2])

            curr_frame = y_all[poc]

            # padded_curr_frame = np.zeros(shape=(h + 2 * block_size, w + 2 * block_size, 1), dtype=np.uint8)
            #
            # # padding
            # # 1) left above
            # padded_curr_frame[0:0 + block_size, 0:0 + block_size] = curr_frame[0, 0]
            # # 2) above
            # for i in range(block_size):
            #     padded_curr_frame[i, block_size:block_size + w] = curr_frame[0, 0:0 + w]
            # # 3) right above
            # padded_curr_frame[0:0 + block_size, w + block_size:w + block_size + block_size] = curr_frame[0, w - 1]
            # # 4) left bottom
            # padded_curr_frame[h + block_size:h + block_size + block_size, 0:0 + block_size] = curr_frame[h - 1, 0]
            # # 5) bottom
            # for i in range(block_size):
            #     padded_curr_frame[h + block_size + i, block_size:block_size + w] = curr_frame[h - 1, 0:0 + w]
            # # 6) right bottom
            # padded_curr_frame[h + block_size:h + block_size + block_size, w + block_size:w + block_size + block_size] = curr_frame[h - 1, w - 1]
            # # 7) left
            # for i in range(block_size):
            #     padded_curr_frame[block_size:block_size + h, i] = curr_frame[0:0 + h, 0]
            # # 8) right
            # for i in range(block_size):
            #     padded_curr_frame[block_size:block_size + h, block_size + w + i] = curr_frame[0:0 + h, w - 1]
            # # 9) center
            # padded_curr_frame[block_size:block_size + h, block_size:block_size + w] = curr_frame

            # curr_cu = curr_frame[block_size + y:block_size + y + block_size, block_size + x:block_size + x + block_size]

            curr_cu = curr_frame[y:y + block_size, x:x + block_size]

            save_cu_path = os.path.join(cupath, csvfile_name.split('.')[0])

            if os.path.isdir(save_cu_path) == False:
                os.makedirs(save_cu_path)

            block = row[0] + '_' + row[1] + 'x' + row[2] + '_' + row[5] # POC_X_Y_SPLIT
            cv2.imwrite(save_cu_path + '/' + block + '.png', curr_cu)
            print("Save---", save_cu_path + '/' + block)
    print('Finish getting CU_VVC')

def get_Luma(sequence_source_path, number_of_frames, w, h):
    y_all = []
    for i in range(number_of_frames):
        file_stream = open(sequence_source_path, 'rb')

        y = np.fromfile(file_stream, np.uint16, w * h).reshape((h, w))
        u = np.fromfile(file_stream, np.uint16, w * h // 4).reshape((h // 2, w // 2))
        v = np.fromfile(file_stream, np.uint16, w * h // 4).reshape((h // 2, w // 2))

        y_uint8 = np.empty((h, w, 1), dtype=np.uint8)
        cv2.normalize(y, y_uint8, 0., 255., cv2.NORM_MINMAX, cv2.CV_8U)
        #cv2.imwrite('y.png', y_uint8)

        y_all.append(y_uint8)

    file_stream.close()
    print('Finish Luma extraction')

    return y_all

if __name__ == "__main__":
    block_size = 128 # 32x32, 64x64, 128x128
    number_of_frames = 100
    # sourcedir = '/home/ubuntu/Dataset/Sequence/UVG'
    # inputpath = '/home/ubuntu/Dataset/UVG_VVC_BI/train/' + str(block_size) + 'x' + str(block_size) + '/Input'
    # gtpath = '/home/ubuntu/Dataset/UVG_VVC_BI/train/' + str(block_size) + 'x' + str(block_size) + '/GT'
    sourcedir = '/home/ubuntu/PycharmProjects/vit-pytorch/UVG/yuv'
    csvpath = '/home/ubuntu/PycharmProjects/vit-pytorch/UVG/csv/' + str(block_size) + 'x' + str(block_size)
    csvpath = '/home/ubuntu/PycharmProjects/vit-pytorch/UVG/csv/temp'
    cupath = '/home/ubuntu/PycharmProjects/vit-pytorch/FastQTMTDataset/train/' + str(block_size) + 'x' + str(block_size)

    get_CU_VVC(sourcedir, csvpath, cupath, block_size, number_of_frames)