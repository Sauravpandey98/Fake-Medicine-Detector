# Fake-Medicine-Detector
A fake medicine detector based on deep learning. It can detect fake or counterfeit medicines by comparing the images of packets of original medicine and medicine under test(MUT).

# Description
This detector is a part of Safemed project.Description of the trained model,dataset used and its application can be found in [Safemed](https://github.com/deepak2310gupta/SafeMed) repository.


# Requirements

Python version 3.2+ is preferred.

Following Python modules are required:
* Numpy
* Tflite



# How to Use
Just download the model and python file.And run the python file by giving paths of trained model,image of original medicine packet and image of medicine under test(MUT) packet as arguments.

An example of images of original and fake medicine packet is also provided in this repository.User can also use these images to test the model.


Example:

$git clone https://github.com/Sauravpandey98/Fake-Medicine-Detector.git

$cd /path to cloned folder

$ python3 Fake_medicine_detection.py /path to model /path to original medicine's packet image /path to MUT's packet image
