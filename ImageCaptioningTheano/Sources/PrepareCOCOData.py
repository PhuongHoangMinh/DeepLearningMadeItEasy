import glob
import numpy as np
import scipy.misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import Utils
import pdb
def VGG_preprocess(data):
    VGG_MEAN = np.asarray([103.939, 116.779, 123.68])
    VGG_MEAN = np.reshape(VGG_MEAN, (1, 3, 1, 1))
    return np.asarray(data-VGG_MEAN, dtype=np.float32)

def create_train_data():
    w = 224
    h = 224
    max_sample_perh5 = 8000
    h5_counter = 0

    path_name = '../../data/mscoco/'
    file_type = '*.jpg'
    train_file_list = sorted(glob.glob(path_name + 'train2014/' + file_type))
    val_file_list   = sorted(glob.glob(path_name + 'val2014/' + file_type))
    test_file_list  = sorted(glob.glob(path_name + 'test2014/' + file_type))
    pdb.set_trace() 
    
    num_train = len(train_file_list)
    print("There are %d train img in total" % num_train)
    train_data = np.zeros((max_sample_perh5, h, w, 3), dtype = np.float32)
    print('loading train images into auxiliary file')
    for i in range(num_train):
    	I = mpimg.imread( train_file_list[i])
    
    	train_sample = np.zeros((I.shape[0], I.shape[1], 3), dtype = np.float32)
    	if len(I.shape) == 2:
    		# print(I.shape)
    		train_sample[:,:,0] = I
    		train_sample[:,:,1] = I
    	        train_sample[:,:,2] = I
    	else:
            train_sample = I
    
    	train_data[i%max_sample_perh5, :, :, :] = scipy.misc.imresize(train_sample, [h, w], 'bicubic')
    	if i%1000 == 0:
    		print(i)
        if ((i+1) % max_sample_perh5 == 0 or (i+1) == num_train):
            # process train_data using VGG

            # Change RGB to BGR
            train_data = train_data[:,:,:,[2,1,0]]

            # Transpose train_data into the shape of (8000, 3, 224, 224)
            train_data = np.transpose(train_data,(0, 3,1,2,))
            train_data = VGG_preprocess(train_data)

            image_dict = {}
            image_dict['train_X'] = train_data
            # image_dict['val_data'] = val_data
            Utils.SaveH5(image_dict , (path_name+'MSCOCO_processed/MSCOCO_224_imgdata_train_%d.h5') % (i/max_sample_perh5))
          
            if (num_train - (i+1) < max_sample_perh5):
                sample_size = num_train - (i+1)
            else:
                sample_size = max_sample_perh5
            train_data = np.zeros((sample_size, h, w, 3), dtype = np.float32)

def create_val_data():
    w = 224
    h = 224
    max_sample_perh5 = 8000
    h5_counter = 0

    path_name = '../../data/mscoco/'
    file_type = '*.jpg'
    train_file_list = sorted(glob.glob(path_name + 'train2014/' + file_type))
    val_file_list   = sorted(glob.glob(path_name + 'val2014/' + file_type))
    test_file_list  = sorted(glob.glob(path_name + 'test2014/' + file_type))
    pdb.set_trace() 
    
    num_val = len(val_file_list)
    print("There are %d val img in total" % num_val)
    val_data = np.zeros((max_sample_perh5, h, w, 3), dtype = np.float32)
    print('loading val images into auxiliary file')
    for i in range(num_val):
    	I = mpimg.imread(val_file_list[i])
    
    	val_sample = np.zeros((I.shape[0], I.shape[1], 3), dtype = np.float32)
    	if len(I.shape) == 2:
    		# print(I.shape)
    		val_sample[:,:,0] = I
    		val_sample[:,:,1] = I
    	        val_sample[:,:,2] = I
    	else:
            val_sample = I
    
    	val_data[i%max_sample_perh5, :, :, :] = scipy.misc.imresize(val_sample, [h, w], 'bicubic')
    	if i%1000 == 0:
    		print(i)
        if ((i+1) % max_sample_perh5 == 0 or (i+1) == num_val):
            # process val_data using VGG

            # Change RGB to BGR
            val_data = val_data[:,:,:,[2,1,0]]

            # Transpose val_data into the shape of (8000, 3, 224, 224)
            val_data = np.transpose(val_data,(0, 3,1,2,))
            val_data = VGG_preprocess(val_data)

            image_dict = {}
            image_dict['val_X'] = val_data
            # image_dict['val_data'] = val_data
            Utils.SaveH5(image_dict , (path_name+'MSCOCO_processed/MSCOCO_224_imgdata_val_%d.h5') % (i/max_sample_perh5))
          
            if (num_val - (i+1) < max_sample_perh5):
                sample_size = num_val - (i+1)
            else:
                sample_size = max_sample_perh5
            val_data = np.zeros((sample_size, h, w, 3), dtype = np.float32)


if __name__ == "__main__" :
    #create_train_data()
    create_val_data() 
