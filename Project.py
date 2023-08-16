import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score , classification_report,ConfusionMatrixDisplay
import scikitplot as skplt
import tkinter as tk
from tkinter import filedialog






def gui():
    pca_kernal = var1.get()
    svm_kernal = var3.get()
    n_components = var2.get()
    slack_parameter = var4.get()
    datasetcsv_path = var.get()
    testsize = var5.get()
    df = pd.read_csv(datasetcsv_path)

    df.head()

    X = df.iloc[:, :-1]
    Y = df.iloc[:,-1]
    X = np.array(X)
    Y = np.array(Y)

    if "PaviaU" in datasetcsv_path:
        size_tuple = (610,340)
    elif "Indianpines" in datasetcsv_path:
        size_tuple = (145,145)

    names = ['Ground','Alfalfa','Corn-notill','Corn-mintill',	'Corn','Grass-pasture','Grass-trees',
            'Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill',
            'Soybean-clean','Wheat','Woods','Buildings Grass Trees Drives','Stone Steel Towers']

    #ploting the original dataset
    plt.figure(figsize=(8, 6))
    plt.imshow(np.reshape(Y , size_tuple ), cmap='nipy_spectral')
    plt.colorbar()
    plt.axis('off')
    plt.title('Ground Truth')
    plt.savefig('ground_truth.png')
    plt.show()


    eigen_solver = 'arpack'
    

    PCA = KernelPCA(n_components = n_components, kernel=pca_kernal, eigen_solver = eigen_solver)
    X_pca = PCA.fit_transform(X)
    X_pca.shape

    X_train, X_test, Y_train, Y_test, indices_train, indices_test  = train_test_split(X_pca, Y, range(X.shape[0]),test_size = testsize, random_state = 11, stratify=Y)

    fig = plt.figure(figsize = (20, 10))

    for i in range(1,9):
        fig.add_subplot(2,4, i)
        plt.imshow(X[:,i].reshape(size_tuple), cmap='nipy_spectral')
        plt.axis('off')
        plt.title(f'Band - {i}')

    plt.savefig('Initial_Bands.png')

    fig = plt.figure(figsize = (20, 10))

    for i in range(1,9):
        fig.add_subplot(2,4, i)
        plt.imshow(X_pca[:,i].reshape(size_tuple), cmap='nipy_spectral')
        plt.axis('off')
        plt.title(f'Band - {i}')

    plt.savefig('PCA_Bands.png')

    SVM = SVC(C=slack_parameter, kernel=svm_kernal ,cache_size = 10*1024)
    SVM.fit(X_train, Y_train)

    Y_pred = SVM.predict(X_test)
    print(f'Accuracy: {accuracy_score(Y_test, Y_pred)* 100}%')

    print(classification_report(Y_test, Y_pred , target_names = names))

    skplt.metrics.plot_confusion_matrix(Y_test, Y_pred, figsize=(12,12));

    withtest = [0]*X.shape[0]
    withouttest = [0]*X.shape[0]
    for i in range(len(indices_train)):
        withtest[indices_train[i]] = Y[indices_train[i]]
        withouttest[indices_train[i]] = Y[indices_train[i]]

    for i in range(len(indices_test)):
        withtest[indices_test[i]] = Y_pred[i]
        withouttest[indices_train[i]] = 0

    plt.figure(figsize=(8, 6))
    plt.imshow(np.array(withouttest).reshape(size_tuple), cmap='nipy_spectral')
    plt.colorbar()
    plt.axis('off')
    plt.title('Only training data')
    plt.savefig('Only training data.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(np.array(withtest).reshape(size_tuple), cmap='nipy_spectral')
    plt.colorbar()
    plt.axis('off')
    plt.title('Test Data after Classification')
    plt.savefig('Classification_map.png')
    plt.show()

    root.destroy()

    



def choose_file():
    file_path = filedialog.askopenfilename()
    var.set(file_path)


root = tk.Tk()
root.title("GUI")


var = tk.StringVar(value="")
menu = tk.OptionMenu(root, var, "")
menu.pack()

button = tk.Button(root, text="Choose File", command=choose_file)
button.pack()


var1 = tk.StringVar(value="pca kernel")
options = ["rbf", "linear", "poly"]

name_label1 = tk.Label(root, text="Enter no.of components:")

var2 = tk.IntVar(value=0)
entry1 = tk.Entry(root, textvariable=var2)

var3 = tk.StringVar(value="svm kernel")
options = ["rbf", "linear", "poly"]

name_label2 = tk.Label(root, text="Enter slack parameter:")

var4 = tk.IntVar(value=0)
entry2 = tk.Entry(root, textvariable=var4)

name_label3 = tk.Label(root, text="Enter fraction for test data:")
var5 = tk.DoubleVar(value=0)
entry3 = tk.Entry(root, textvariable=var5)

menu1 = tk.OptionMenu(root, var1, *options)

menu2 = tk.OptionMenu(root, var3, *options)


menu1.pack()
name_label1.pack()
entry1.pack()
menu2.pack()
name_label2.pack()
entry2.pack()
name_label3.pack()
entry3.pack()

button = tk.Button(root, text=" Run ", command=gui)
button.pack()

label = tk.Label(root, text="")
label.pack()

root.mainloop()