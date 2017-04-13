# DeepLearningMadeItEasy
This repository consists of two parts. The first part is visualization of Convolution Neural Networks (CNNs) implemented in Python and Numpy. 
The second on is an implementation of Image Captioning papers from scratch on Theano.

Visualizing deep CNNs includes class visualization [1], feature inversion [2], and activation map visualization [2] [3] using gradient descent, gradient ascent and several regularization techniques
to generate images. The project utilized TinyImageNet and pretrained models from CNN for visual recognition course work from Stanford [6].

Image captioning is a commonly known as a difficult task in Computer Vision in which a machine algorithms should reason the content 
of images and generate captions describing the content in natural languages. The ideas was first presented in [4] [5] by combining 
CNNs and a variant of Recurrent Neural Networks (RNN) to learn a joint embedding sapce of the visual 
and textural features. 

In our implementation, we utilized the VGG-16 and LSTM to learn captions of images from MSCOCO dataset for ShowTell architecture. 
In addition,an attention based model proposed by Kelvin Xu et al. was also implemented. Spatial attention maps are generated as Figure below.

##References
[1]Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR Workshop 2014.

[2]Yosinski et al, "Understanding Neural Networks Through Deep Visualization", ICML 2015 Deep Learning Workshop

[3]Aravindh Mahendran, Andrea Vedaldi, "Understanding Deep Image Representations by Inverting them", CVPR 2015

[4]Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan. "Show and Tell: A Neural Image Caption Generator", CVPR, 2014

[5]Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. 
"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", CVPR, 2015

[6]http://cs231n.stanford.edu/

