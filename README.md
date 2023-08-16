Group: 62
Team Member:
    Teja Bale (200050020)
    Krishna Kamal (200050142)
    Yash Mailapalli (200050160)
Project : Classify a remote sensing image using a kernel PCA based classifier.

To run this code:
Requirements:
    pandas
    numpy
    matplotlib.pyplot
    sklearn
    scikiplot
    tkinter

To install:
    pip3 install pandas
    pip3 install numpy
    pip3 install matplotlib
    pip3 install -U scikit-learn
    pip3 install -q scikit-plot
    sudo apt-get install python3-tk

The Datasets for the project are csv files and we can select this dataset file for this project from them in GUI


In the GUI:
    In the PCA kernal you have to select the kernal you want to use for the KernalPCA
    Components is the to dimesion of the data after PCA
    In SVM kernal choose kernal for SVM
    choose slack parameter for SVM
    In the test_size choose the fraction of testing data size
    finally select the dataset for project

After the Succesful run Accuracy and Images with only training data and image with classified points along with training data

In the ouput folder some outputs are given for the below values

Dataset = Indian pines
pca_kernal = 'poly'
svm_kernal = 'rbf'
n_components = 40
slack_parameter = 100
datasetcsv_path = 'Indianpines_Datasetcsv'
testsize = 0.20

To run the program high GPU required and it will take some time to train and it depends on the machine and Even it will take some time using GPU

