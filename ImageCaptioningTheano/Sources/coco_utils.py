import os, json
import numpy as np
import h5py

def load_co_co(base_dir='../../data/MSCOCO/coco_captioning',
			   max_train = None,
			   pca_features = None):
	'''
	des: loading MSCOCO data from preprocessed files including json(vocabulary), image_features as well as compresses image features from fc7, captions
	return: 
		data = {} - a dictionary with (key : string) and ( value : np.ndarray) 
	'''
	data = {}

	#captions data from h5 file
	caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
	with h5py.File(caption_file, 'r') as f:
		for k, v in f.items():
			data[k] = np.asarray(v)

	#train image data from h5 file
	if pca_features:
		train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
	else:
		train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
	with h5py.File(train_feat_file, 'r') as f:
		data['train_features'] = np.asarray(f['features'])

	#val image data from h5 file
	if pca_features:
		val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
	else:
		val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
	with h5py.File(val_feat_file, 'r') as f:
		data['val_features'] = np.asarray(f['features'])


	#vocab json data from json file
	dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
	with open(dict_file, 'r') as f:
		dict_data = json.load(f)
		for k, v in dict_data.items():
			data[k] = v


	#training image urls to load images on the fly
	train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
	with open(train_url_file, 'r') as f:
		train_urls = np.asarray([line.strip() for line in f])
	data['train_urls'] = train_urls

	#val image urls to load images on the fly
	val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
	with open(val_url_file, 'r') as f:
		val_urls = np.asarray([line.strip() for line in f])
	data['val_urls'] = val_urls


	#subsample the training data but in random ways
	if max_train is not None:
		num_train = data['train_captions'].shape[0]
		mask = np.random.randint(num_train, size = max_train)
		data['train_captions'] = data['train_captions'][mask]
		data['train_image_idx'] = data['train_image_idx'][mask]

	return data


def decode_captions(captions, idx_to_word):
	'''
	des: decode captions of a list of images
	params: captions:    - 2d arrays N*D where N - number of images, D is number of words in a caption
			idx_to_word: - dictionary of indexes and words
	return : a list of string captions
	'''
	singleton = False
  	if captions.ndim == 1:
            singleton = True
            captions = captions[None]
	
	decoded = []
	N,T = captions.shape

	for i in range(N):
		words = []
		for t in range(T):
			word = append(idx_to_word[captions[i,t]])
			if word != '<NULL>':
				words.append(word)
			if word == '<END>':
				break
			decoded.append(''.join(words))
	if singleton:
        	decoded = decoded[0]

	return decoded

def sample_coco_minibatch(data, batch_size = 100, split = 'train'):
	'''
	sample mini batch from data for testing or training purposes
	return : captions, features, urls
	'''
	split_size = data['%s_captions'%split].shape[0]
	mask 	   = np.random.choice(split_size, batch_size)
	captions   = data['%s_captions'%split][mask]
	image_idx  = data['%s_image_idx'%split][mask]
	features   = data['%s_features'%split][mask]
	urls       = data['%s_urls'%split][mask]

	return captions, features, urls



