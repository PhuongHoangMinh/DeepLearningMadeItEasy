import numpy as np
import scipy.misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import Utils
w = 224
h = 224

#data_path = '/home/kien/PycharmProjects/data/'
data_path = '../../data/'
img_data_path = data_path+'Flickr8k_Dataset/Flicker8k_Dataset/'
text_data_path = data_path+'Flickr8k_text/'

# onlyfiles = [f for f in listdir(img_data_path) if isfile(join(img_data_path, f))]
train_text_path = text_data_path+'Flickr_8k.trainImages.txt'
val_text_path = text_data_path+'Flickr_8k.devImages.txt'
test_text_path = text_data_path+'Flickr_8k.testImages.txt'

# Get train data
f = open(train_text_path)
lines = f.readlines()
f.close()

num_train = len(lines)
train_data = np.zeros((num_train, h, w, 3), dtype=np.float32)
print("Loading images into train_data....")
for i in range(len(lines)):
    I = mpimg.imread(img_data_path + lines[0][0:-1])
    train_data[i,:,:,:] = scipy.misc.imresize(I, [h, w], 'bicubic')
    if i % 100 == 0:
        print (i)

# Get val data
f = open(val_text_path)
lines = f.readlines()
f.close()

num_val = len(lines)
val_data = np.zeros((num_val, h, w, 3), dtype=np.float32)
print("Loading images into val_data....")
for i in range(len(lines)):
    I = mpimg.imread(img_data_path + lines[0][0:-1])
    val_data[i,:,:,:] = scipy.misc.imresize(I, [h, w], 'bicubic')
    if i % 100 == 0:
        print (i)

# Get test data
f = open(test_text_path)
lines = f.readlines()
f.close()

num_test = len(lines)
test_data = np.zeros((num_test, h, w, 3), dtype=np.float32)
print("Loading images into test_data....")
for i in range(len(lines)):
    I = mpimg.imread(img_data_path + lines[0][0:-1])
    test_data[i,:,:,:] = scipy.misc.imresize(I, [h, w], 'bicubic')
    if i % 100 == 0:
        print (i)

# Utils.SaveList([train_data, val_data, test_data], data_path+'Flickr8k_processed/Flickr8k_224_imgdata.dat')
image_dict = {}
image_dict['train_data'] = train_data
image_dict['val_data'] = val_data
image_dict['test_data'] = test_data

Utils.SaveH5(image_dict , data_path+'Flickr8k_processed/Flickr8k_224_imgdata.dat')
