#!/usr/bin/env python
# coding: utf-8

# 
# # Traffic Sign Recognition with 4GB dataset
# 
# This is ready to use preprocessed data for Traffic Signs saved into the nine pickle files.
# 
# **Original Datasets**
# 
# * train.pickle
# 
# * valid.pickle
# 
# * test.pickle
# 
# Code with detailed description on how datasets were preprocessed is in `datasets_preparing.py`
# 
# Before preprocessing training dataset was equalized making examples in the classes equal as it is shown on the figure below. Histogram of 43 classes for training dataset with their number of examples for Traffic Signs Classification before and after equalization by adding transformed images (brightness and rotation) from original dataset. After equalization, training dataset has increased up to 86989 examples.
# 
# 
# 
# 
# **Preprocessed Datasets**
# 
# * data0.pickle - Shuffling
# 
# * data1.pickle - Shuffling, /255.0 Normalization
# 
# * data2.pickle - Shuffling, /255.0 + Mean Normalization
# 
# * data3.pickle - Shuffling, /255.0 + Mean + STD Normalization
# 
# * data4.pickle - Grayscale, Shuffling
# 
# * data5.pickle - Grayscale, Shuffling, Local Histogram Equalization
# 
# * data6.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 Normalization
# 
# * data7.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean Normalization
# 
# * data8.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean + STD Normalization
# 
# 
# **Shapes of data0 - data3 are as following (RGB):**
# 
# - x_train: (86989, 3, 32, 32)
# - y_train: (86989,)
# - x_validation: (4410, 3, 32, 32)
# - y_validation: (4410,)
# - x_test: (12630, 3, 32, 32)
# - y_test: (12630,)
# 
# **Shapes of data4 - data8 are as following (Gray):**
# - x_train: (86989, 1, 32, 32)
# - y_train: (86989,)
# - x_validation: (4410, 1, 32, 32)
# - y_validation: (4410,)
# - x_test: (12630, 1, 32, 32)
# - y_test: (12630,)
# 
# 
# `mean_image` and `standard_deviation` were calculated from `train` dataset and applied to `validation` and `testing` datasets for appropriate datasets. When using user's image for classification, it has to be preprocessed firstly in the same way and in the same order according to the chosen dataset among nine.
# 
# # Imports

# In[2]:


import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from fastai.vision.all import *
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from fastai.vision.augment import RandTransform
import random
from sklearn.metrics import classification_report

sys.path.append('../Data/')


# # Downloading/Processing Data 

# In[49]:


def load_pickle(file='data0.pickle'):
    with open(f'./Data/_4GB/{file}', 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # latin1 for dictionary
    return data


def save_images_to_disk(images, labels, base_path):
    """
    Saves images stored as NumPy arrays to disk. Handles both grayscale and RGB images.

    Args:
    - images (list or array-like): List or array-like object containing the image data.
    - labels (list or array-like): List or array-like object containing the labels.
    - base_path (str): Base path where the images will be saved.
    """
    for i, (image, label) in enumerate(zip(images, labels)):
        # Check if image is grayscale (1, height, width) or RGB (3, height, width)
        if image.shape[0] == 1:
            # For grayscale, convert shape (1, height, width) to (height, width)
            image = image.squeeze(0)
            mode = 'L'  # Grayscale
        else:
            # For RGB, convert shape (3, height, width) to (height, width, 3)
            image = image.transpose(1, 2, 0)
            mode = 'RGB'

        # Convert to PIL image with correct mode
        img = Image.fromarray(image.astype('uint8'), mode)

        # Create label directory if it doesn't exist
        label_path = os.path.join(base_path, str(label))
        os.makedirs(label_path, exist_ok=True)

        # Save the image
        img.save(os.path.join(label_path, f'image_{i}.png'))


# In[13]:


for i in range(9):
    data = load_pickle(f'data{i}.pickle')
    save_images_to_disk(data['x_train'], data['y_train'], f'Data/_4GB/data{i}images/data{i}train')
    save_images_to_disk(data['x_validation'], data['y_validation'], f'Data/_4GB/data{i}images/data{i}validation')
    save_images_to_disk(data['x_test'], data['y_test'], f'Data/_4GB/data{i}images/data{i}test')
    print(f"Saved data{i}.pickle images to disk")


# # Fine-Tuning
# 
# ## Step 1: Load Data and find initial learning rate

# In[80]:


class SelectiveRotation(RandTransform):
    """
    Applies rotation to images except for those with specified labels.
    """
    def __init__(self, labels_to_avoid, degrees=10, **kwargs):
        super().__init__(**kwargs)
        self.labels_to_avoid = labels_to_avoid
        self.degrees = degrees

    def encodes(self, x: PILImage):
        # Extract label from file path, assuming label is the name of the parent directory
        label_name = x.parent.name
        if label_name not in self.labels_to_avoid:
            # Apply rotation
            degree = random.uniform(-self.degrees, self.degrees)
            x = x.rotate(degree)
        return x
    

def get_data_loaders_and_model(data_id=0, model=models.resnet50):
    data = load_pickle(f'data{data_id}.pickle')

    def get_label_from_path(item: Path):
        label = item.parent.name
        # Debug output
       # print(f"Reading path: {item}, Label extracted: {label}")
        return label

    path = Path(f'Data/_4GB/data{data_id}images')

    all_labels = sorted(set([p.parent.name for p in get_image_files(path)]))
    print("All labels found:", all_labels)  # Debug output

    reg_signs = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=all_labels)),
        get_items=get_image_files,
        splitter=GrandparentSplitter(
            train_name=f'data{data_id}train', valid_name=f'data{data_id}validation'),
        get_y=get_label_from_path,
        item_tfms=Resize(224),
    )

    labels_to_avoid = [str(label) for label in [38, 39, 36, 37]]

    aug_signs = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=all_labels)),
        get_items=get_image_files,
        splitter=GrandparentSplitter(
            train_name=f'data{data_id}train', valid_name=f'data{data_id}validation'),
        get_y=get_label_from_path,
        item_tfms=Resize(224),
        batch_tfms=[*aug_transforms(mult=1.0, do_flip=True, flip_vert=False,
                                    max_warp=0.0), SelectiveRotation(labels_to_avoid, degrees=10)]
    )

    reg_dls = reg_signs.dataloaders(path)
    aug_dls = aug_signs.dataloaders(path)

    test_dir = f'data{data_id}test'
    test_dl = reg_dls.test_dl(get_image_files(path/test_dir), with_labels=True)

    color_channels = 3 if data_id < 4 else 1

    learn = vision_learner(reg_dls, model, loss_func=CrossEntropyLossFlat(),
                           metrics=accuracy, n_in=color_channels, n_out=len(data['labels']), pretrained=True)

    return reg_dls, aug_dls, test_dl, data['labels'], learn


data_id = 0 # (0-8) pickle files to pull data from
model = models.resnet34
#model = models.resnet50
reg_dls, aug_dls, test_dl, labels, learn = get_data_loaders_and_model(data_id=data_id, model=model)

learn.lr_find()


# In[79]:


def print_data_samples(dl, name):
    print(f"Samples from {name}:")
    for xb, yb in dl:
        # Assuming xb is a batch of images and yb is a batch of labels
        for i, (x, y) in enumerate(zip(xb, yb)):
            img_path = dl.items[i]  # Get the path
            # Convert label index to actual label using the labels list if applicable
            label = dl.vocab[y] if hasattr(dl, 'vocab') else y
            print(f"Path: {img_path}, Label: {label}")
            if i >= 5:  # Print only first 5 samples to avoid too much output
                break
        break


print_data_samples(reg_dls.train, "Regular Training DataLoader")
print_data_samples(aug_dls.train, "Augmented Training DataLoader")
print_data_samples(test_dl, "Test DataLoader")


# ### Step 2: Phase 1 Training - train frozen network on unaugmented data

# In[4]:


# Train for 1-2 epochs with non-augmented data
suggested_lr = 0.0020892962347716093
# Layers are already frozen
learn.fit_one_cycle(3, lr_max=suggested_lr)  # 1-2 epochs


# # Step 3: Phase 1 Training - train frozen network on augmented data

# In[5]:


learn.dls = aug_dls

# Train for 2-3 more epochs with augmented data
learn.fit_one_cycle(2, lr_max=suggested_lr)


# # Step 4: Phase 2 Training - Unfreeze the network and find the new learning rate

# In[6]:


# Unfreeze the model
learn.unfreeze()
learn.lr_find()


# # Step 5: Phase 2 Training - Set differential learning rates and train the whole network. 
# 
# * Differential Learning rates sets higher learning rates for the last layers and lower learning rates for first layers

# In[7]:


# Set differential learning rates
# use suggested_lr for the high
high = 1.3182567499825382e-06
low = 10**-6.5
learn.fit_one_cycle(1, lr_max=slice(low, high))


# # Save Model

# In[8]:


learn.save('data0ResNet34')


# # Load Model
# 
# ### data0ResNet50
# 
# Using data0.pickle and the pre-trained ResNet50 model. 
# 
# 1. The initial learning rate was found
# 
# 2. Using the suggested learning rate, the model was trained for an additional 3 epochs on the data0 (non-augmented) data. 
# 
# 3. Then the was unfrozen and the suggested learning rate was found again.
# 
# 4. Using this suggested learning rate and another smaller learning rate (ie: $1/10$) differential learning rates were used to train the unfrozen model for one epoch.
# 
# 5. After this stage, the test accuracy was very high, but the validation accuracy was very low. I decided to introduce augmented data at this stage and train the model for an additional two epochs with the most recently suggested learning rate. 
# 
# 6. The final model was saved and the test accuracy was calculated:
# 
# Validation loss: 0.156088188290596
# Validation accuracy: 0.9716553092002869
# Test loss: 0.1470213681459427
# Test accuracy: 0.9702296257019043
# 
# 
# **NOTES**
# 
# The model seems to be mixing up ('Keep right', 'Keep left') and ('Go straight or right', 'Go straight or left'). Perhaps this is due to rotation in the data augmentation phase. 
# 
# * We could keep images with these labels from being rotated to possibly avoid the model's confusion

# In[7]:


model = models.resnet34
#model = models.resnet50

data_id = 0
_, _, test_dl, labels, learn = get_data_loaders_and_model(data_id=data_id, model=model)
learn = learn.load('data0ResNet34') 
#learn = learn.load('data0ResNet50')


# # Loss/Accuracy

# In[38]:


val_loss, val_acc = learn.validate()
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_acc}')

# Compute the test set loss and accuracy
test_loss, test_acc = learn.validate(dl=test_dl)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')


# In[16]:


from fastai.vision.all import *
from PIL import Image
import matplotlib.pyplot as plt


def plot_prediction(image_path, learner):

    labels = pd.read_csv('./Data/_4GB/label_names.csv')

    # Load the image
    img = PILImage.create(image_path)
    img = img.resize((224, 224))

    # Predict the label
    pred, pred_idx, probs = learner.predict(img)
    print(f"pred_idx=  {pred_idx}")
    print(f"probs= {probs}")
    # Extract the true label from the file path
    true = image_path.parent.name

    predicted_label = labels['SignName'].iloc[int(pred)]
    true_label = labels['SignName'].iloc[int(true)]
    print(f"True = {true}, Pred = {pred}")

    # Plot the image with predicted and true labels
    plt.imshow(img)
    plt.title(
        f'Predicted: {predicted_label}, True: {true_label}\nProbability: {probs[pred_idx]:.4f}')
    plt.axis('off')
    plt.show()

# Image path (assuming the image is stored in one of the test set directories)
image_path = Path('./Data/_4GB/data0images/data0test/0/image_143.png')
plot_prediction(image_path, learn)


# In[36]:


data = load_pickle()
labels = data['labels']


# In[37]:


labels


# In[45]:


from fastai.vision.all import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_prediction(image_path, labels, pred):
    

    # Load the image
    img = PILImage.create(image_path)
    # Ensure the image is resized as per model's training
    img = img.resize((224, 224))

    # Extract the true label from the file path
    true = image_path.parent.name

    # Mapping indices to actual label names
    predicted_label = labels['SignName'].iloc[int(pred)]
    true_label = labels['SignName'].iloc[int(true)]

    # Plot the image with predicted and true labels
    plt.imshow(img)
    plt.title(
        f'Predicted: {predicted_label}, True: {true_label}')
    plt.axis('off')
    plt.show()

labels = pd.read_csv('./Data/_4GB/label_names.csv')
# Get predictions and actual labels
preds, y_true = learn.get_preds(dl=test_dl)
pred_indices = preds.argmax(dim=1)

# Get image paths from the test dataloader directly
test_items = test_dl.items

# Filter and display only correct predictions
count = 0
for (pred, true, img_path) in zip(pred_indices, y_true, test_items):
    if pred == true and count < 10:
       plot_prediction(img_path, labels, pred)
       count += 1


# In[46]:


# Extract a few batches from the test data loader and print paths and labels
for x, y in test_dl:
    for xi, yi in zip(x, y):
        print(f"Label: {yi}, Path: {test_dl.dataset.items[yi]}")
    break  # Just check the first batch for a quick inspection


# In[41]:


count_correct/total


# In[20]:


preds, y_true = learn.get_preds(dl=test_dl)


# In[21]:


len(preds), len(y_true)


# In[31]:


data = load_pickle()
labels = data['labels']


# Convert probabilities to predicted class indices
pred_indices = preds.argmax(dim=1)

# Loop through the predictions and true labels
for pred_idx, true_idx in zip(pred_indices, y_true):
    if pred_idx == true_idx:
        # Fetch the label using the index
        label = labels[true_idx.item()]  # Convert tensor to integer if needed
        print(f"Successfully predicted label: {label}")


# # Confusion Matrix

# In[11]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12, 12))


# # Top Losses

# In[35]:


labels[38], labels[39], labels[36], labels[37]


# In[12]:


interp.plot_top_losses(5, nrows=1)


# # Classification Report

# In[14]:


from sklearn.metrics import classification_report
data = load_pickle(f'data0.pickle')
class_names = data['labels']
preds, y_true = learn.get_preds(dl=test_dl)
y_pred = np.argmax(preds, axis=1)
report = classification_report(y_true, y_pred, target_names=class_names)

print(report)


# # ROC/AUC

# In[15]:


def plot_average_roc(y_true, y_pred_probs, classes, n_classes):
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))

    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'.format(
                 roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.2f})'.format(
                 roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

def plot_multi_class_roc_auc(y_true, y_pred, classes):
    """
    Plot ROC curve and calculate AUC for each class in a multi-class classification problem.

    Args:
    - y_true (array-like): True binary labels or binary label indicators (n_samples, n_classes).
    - y_pred (array-like): Target scores, can either be probability estimates of the positive class,
                           confidence values, or non-thresholded measure of decisions (n_samples, n_classes).
    - classes (list): List containing the labels of the classes.
    """
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class and micro-averaged ROC curve
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC and AUC')
    plt.legend(loc="lower right")
    plt.show()


# In[16]:


preds, y_true = learn.get_preds(dl=test_dl)
class_names = data['labels']
plot_average_roc(y_true.numpy(), preds.numpy(), class_names, len(class_names))

plot_multi_class_roc_auc(y_true.numpy(), preds.numpy(), class_names)


# In[ ]:




