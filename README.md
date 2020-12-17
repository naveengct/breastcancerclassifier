# breastcancerclassifier
The idea is to develop a web application that predicts breast
cancer types (malignant or benign) upon uploading tumor images.Similarly used a traditional
approach developing my own CNN architecture and got 80% accuracy.In my second approach
is used transfer learning(ResNet and MobileNet) which is a pretrained model with frozen
weights and used data augmentation(flip,crop and adjust the color of the image) and showered
with 90% accuracy.Third approach is batch normalization then changed the optimizer ( Adam)
and added regularization and dropout.Then converted it into .h5 model and deployed in a web
application.
