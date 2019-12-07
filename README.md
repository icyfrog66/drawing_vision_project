# cvproject
CSE 455 Final Project


How to use:

`python3 getdata.py`

This command downloads data from the quickdraw dataset into the data folder.
After that, each data/class subdirectory includes all the images of the specific class.

`python3 network1.py`

This command will train on the neural network (currently the network only does a linear mapping).

https://docs.google.com/document/d/13X_dw9HP5LGxE2Sd8XTahElW6A-7wqLPSoxUviOpIz4/edit?usp=sharing

Document comparing the performance of pretrained Pytorch models (i.e. using network2.py). Resnet has the best val accuracy (= 0.828750) of all the models and it also does not take too much time to train (70 minutes).

https://drive.google.com/file/d/1UlezlRfi6RYVFhBhNi-OYo7hAYyqNDKQ/view?usp=sharing

Poster for presentation.

printfeatures.py should print out the last layer of a custom trained network (only around 70% test accuracy for now, but 2nd to last layer of features have been converted to an np array).
