# cCNN-Image-Classifier
Configurable Convolutional Neural Networks for Image Classification.

The programs in this package preprocess images using CV2, and build image classifiers and then classify images using Convolutional Neural Network (CNN). The CNN model can be conveniently specified using a "json" file. 

Datasets
Selected classes of images obtained from the links provided by ImageNet. For each class, several hundreds of images are needed for training the CNN model. Each class should use a separate folder. The image files must be in "jpg" format.

Configuration json file
Contains the paths of image folders for training the CNN model, and configurations of the CNN model.

Programs
All programs should be executed using command lines with arguments. To learn how to feed arguments, just execute the programs without any argument.

Image downloading and preprocessing
1) batch_image_download.py.
A "url" file should be first obtained from the ImageNet website. The "url" file is in the format of "groupid_imageid url". This program downloads the images for a group id according to the "url" file.
2) filter_images.py.
This program filters the downloaded images according to image sizes and channels.

Build the CNN model using the wrapper script "runCNN.sh".

Predit classes on new images using the wrapper script "predCNN.sh". 

Note: dataset.py, layer_function.py, train_cnn.py and predict_cnn.py are original python scripts which are called by the wrapper scripts.
