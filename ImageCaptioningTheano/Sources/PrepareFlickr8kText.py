import numpy as np
import scipy.misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import Utils
import h5py

def pad_caption(caption, max_len):
    while (len(caption) != (max_len)):
        caption += ['<PAD>']
    return caption

def find_max_len():
    # Find max len of flickr8k captions
    # Get captioning data
    print ("Fidning max len of the flickr 8k caption data...")
    data_path = '../../data/'	
    #data_path = '/home/kien/PycharmProjects/data/'
    text_data_path = data_path + 'Flickr8k_text/'
    f = open(text_data_path + 'Flickr8k.token.txt')
    captions = f.readlines()
    f.close()
    all_len = np.zeros(len(captions), dtype=np.int32)
    counter = 0
    for c in captions:
        caption_split = c.split('#')
        caption_str = caption_split[1][2:-1]
        caption_words = caption_str.split()
        all_len[counter] = len(caption_words)
        print(all_len[counter])
        counter += 1

    return max(all_len)


def process_text(max_len):
    w = 224
    h = 224
    # data_path = '/home/kien/PycharmProjects/data/'
    data_path = '../../data/'
    text_data_path = data_path+'Flickr8k_text/'
    img_data_path = data_path + 'Flickr8k_Dataset/Flicker8k_Dataset/'

    # Get train data
    f = open(text_data_path + 'Flickr_8k.trainImages.txt')
    train_file_names = f.read().splitlines()
    num_train = len(train_file_names)
    f.close()

    # Get val data
    f = open(text_data_path + 'Flickr_8k.devImages.txt')
    val_file_names = f.read().splitlines()
    num_val = len(val_file_names)
    f.close()

    # Get test data
    f = open(text_data_path + 'Flickr_8k.testImages.txt')
    test_file_names = f.read().splitlines()
    num_test = len(test_file_names)
    f.close()

    # Allocate image data
    train_data = np.zeros((num_train, h, w, 3), dtype=np.float32)
    val_data = np.zeros((num_val, h, w, 3), dtype=np.float32)
    test_data = np.zeros((num_test, h, w, 3), dtype=np.float32)

    # Get captioning data
    f = open(text_data_path + 'Flickr8k.token.txt')
    captions = f.readlines()
    f.close()
    train_caption_list = []
    val_caption_list = []
    test_caption_list = []

    vocab = []
    train_i = 0
    val_i = 0
    test_i = 0
    for i in range(0,len(captions),5):
        c = captions[i]
        caption_split = c.split('#')
        file_name = caption_split[0]
        caption_number = int(caption_split[1][0])

        caption_str = caption_split[1][2:-1].lower()
        caption = caption_str.plit()
        caption = pad_caption(['<START>'] + caption + ['<END>'], max_len)

        if (file_name in train_file_names):
            train_caption_list += [caption]
            I = mpimg.imread(img_data_path + file_name)
            train_data[train_i, :, :, :] = scipy.misc.imresize(I, [h, w], 'bicubic')
            train_i += 1
            bp = 1

        elif (file_name in val_file_names):
            val_caption_list += [caption]
            I = mpimg.imread(img_data_path + file_name)
            val_data[val_i, :, :, :] = scipy.misc.imresize(I, [h, w], 'bicubic')
            val_i += 1
            bp = 1

        elif (file_name in test_file_names):
            test_caption_list += [caption]
            I = mpimg.imread(img_data_path + file_name)
            test_data[test_i, :, :, :] = scipy.misc.imresize(I, [h, w], 'bicubic')
            test_i += 1
            bp = 1

        else:
            print('Caption doesn\'t belong to any image!\n')

        if (i%500 == 0):
            print (i/5)

        for v in caption:
            if (not(v in vocab)):
                vocab += [v]

        bp = 1

    save_data_path = data_path + '/Flickr8k_processed/'

    Utils.SaveList([train_data, val_data, test_data],
                   save_data_path + 'Flickr8k_224_imgdata.dat')
    Utils.SaveList([train_caption_list, val_caption_list, test_caption_list],
                   save_data_path + 'Flickr8k_caption_string.dat')
    Utils.SaveList([vocab],
                   save_data_path + 'Flickr8k_vocab.dat')

def create_train_label(max_len, n_words=None):
    # data_path = '/home/kien/PycharmProjects/data/Flickr8k_processed/'
    data_path = '../../data/Flickr8k_processed/'
    # data = Utils.LoadList(data_path + 'Flickr8k_imgdata.dat')
    # train_data = data[0]
    # val_data = data[1]
    # test_data = data[2]

    captions = Utils.LoadList(data_path + 'Flickr8k_caption_string.dat')
    train_captions = captions[0]
    val_captions = captions[1]
    test_captions = captions[2]

    vocab = Utils.LoadList(data_path + 'Flickr8k_vocab.dat')
    vocab = vocab[0]
    if (n_words == None):
        n_words = len(vocab)

    vocab[n_words-1:] = [] #Last word must be unknown
    vocab += ['<UNK>']

    # Better solutions?
    train_label = np.zeros((len(train_captions), max_len, n_words), dtype=np.uint32)
    val_label = np.zeros((len(val_captions), max_len, n_words), dtype=np.uint32)
    test_label = np.zeros((len(test_captions), max_len, n_words), dtype=np.uint32)

    print('Creating label for training captions...\n')
    for i in range(len(train_captions)):
        caption = train_captions[i]
        for j in range(len(caption)): # Should be max_len
            if (not (caption[j] in vocab)):
                word_index = n_words-1
                train_captions[i][j] = vocab[-1]
            else:
                word_index = vocab.index(caption[j])

            train_label[i, j, word_index] = 1

        if (i % 10 == 0):
            print(i)

    print('Creating label for validating captions...\n')
    for i in range(len(val_captions)):
        caption = val_captions[i]
        for j in range(len(caption)): # Should be max_len
            if (not (caption[j] in vocab)):
                word_index = n_words-1
                val_captions[i][j] = vocab[-1]
            else:
                word_index = vocab.index(caption[j])
            val_label[i, j, word_index] = 1

        if (i % 10 == 0):
            print(i)

    print('Creating label for testing captions...\n')
    for i in range(len(test_captions)):
        caption = test_captions[i]
        for j in range(len(caption)): # Should be max_len
            if (not (caption[j] in vocab)):
                word_index = n_words-1
                test_captions[i][j] = vocab[-1]
            else:
                word_index = vocab.index(caption[j])
            test_label[i, j, word_index] = 1

        if (i % 10 == 0):
            print(i)

    print('Saving captions...\n')
    Utils.SaveList([train_label, val_label, test_label],
                   data_path + ('Flickr8k_label_%d.dat' % n_words))
    
    #with h5py.File(data_path + 'Flickr8k_caption_string') as hf:

    Utils.SaveList([train_captions, val_captions, test_captions],
                   data_path + 'Flickr8k_caption_string_%d.dat' % n_words)

    Utils.SaveList([vocab],
                   data_path + 'Flickr8k_vocab_%d.dat' % n_words)

if __name__ =='__main__':
    # max_len = find_max_len() # It's actually 38
    max_len = 40
    process_text(max_len)

    #create_train_label(max_len, 2000)
