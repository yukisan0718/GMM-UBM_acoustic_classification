GMM-UBM acoustic classification
====

## Overview
Implementation of an acoustic classification algorithm using a Gaussian mixture model, so-called GMM-UBM [1]. The GMM-UBM is known as a baseline model for many applications in the sound processing field.

The "UBM_training.py" has been implemented to train the UBM. For training the UBM, you have to collect audio dataset recorded by various speakers (in various environments) as many as possible.

The "MAP_adaption.py" has been implemented to adapt the UBM model into a specific class, based on the maximum posteriori estimation. The adapted model would be a binary detector for each class.


## Requirement
soundfile 0.10.3

matplotlib 3.1.0

joblib 0.14.1

numpy 1.18.1

scipy 1.4.1

scikit-learn 0.21.2


## Dataset preparation
An example of application is speaker recognition [2], or acoustic scenes and events classification [3]. In any case, please create a folder for each class in the "audio_data" directory, and put audio files (.wav format) in it.


## References
[1] D. A. Reynolds, T. F. Quatieri, and R. B. Dunn: 'Speaker Verification Using Adapted Gaussian Mixture Models', Digital Signal Processing, Vol.10, pp.19–41, (2000)

[2] A. Nagrani, J. S. Chung, and A. Zisserman: 'Voxceleb: A Largescale Speaker Identification Dataset', in Proceedings of Interspeech, (2017), [Online], Available: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

[3] D. Stowell, D. Giannoulis, E. Benetos, M. Lagrange, and M. D. Plumbley: 'Detection and Classification of Acoustic Scenes and Events', IEEE Transactions on Multimedia, Vol.17, No.10, pp.1733–1746, (2015)