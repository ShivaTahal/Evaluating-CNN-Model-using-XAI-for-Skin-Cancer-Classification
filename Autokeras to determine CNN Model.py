"""
This file is used to determine CNN model using AutoKeras

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(42)
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

skin_dataframe = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')


SIZE=32

# label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_dataframe['dx'])
LabelEncoder()
print(list(le.classes_))
 
skin_dataframe['label'] = le.transform(skin_dataframe["dx"]) 
print(skin_dataframe.sample(10))


# Data distribution visualization
fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(221)
skin_dataframe['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type');

ax2 = fig.add_subplot(222)
skin_dataframe['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex');

ax3 = fig.add_subplot(223)
skin_dataframe['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count',size=12)
ax3.set_title('Localization')


ax4 = fig.add_subplot(224)
sample_age = skin_dataframe[pd.notnull(skin_dataframe['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red');
ax4.set_title('Age')

plt.tight_layout()
plt.show()


# Distribution of data into various classes 
from sklearn.utils import resample
print(skin_dataframe['label'].value_counts())

#Balance data.
#Separate each classes, resample, and combine back into single dataframe

dataframe_0 = skin_dataframe[skin_dataframe['label'] == 0]
dataframe_1 = skin_dataframe[skin_dataframe['label'] == 1]
dataframe_2 = skin_dataframe[skin_dataframe['label'] == 2]
dataframe_3 = skin_dataframe[skin_dataframe['label'] == 3]
dataframe_4 = skin_dataframe[skin_dataframe['label'] == 4]
dataframe_5 = skin_dataframe[skin_dataframe['label'] == 5]
dataframe_6 = skin_dataframe[skin_dataframe['label'] == 6]

n_samples=50 # Since my environment is not capable of handling all the  samples in one go. I used sample size to be 50
df_0_balanced = resample(dataframe_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(dataframe_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(dataframe_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(dataframe_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(dataframe_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(dataframe_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(dataframe_6, replace=True, n_samples=n_samples, random_state=42)

skin_dataframe_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])

# Read images based on image ID from the CSV file
#This is the safest way to read images as it ensures the right image is read for the right ID
print(skin_dataframe_balanced['label'].value_counts())


image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('data/HAM10000/', '*', '*.jpg'))}
#Define the path and add as a new column
skin_dataframe_balanced['path'] = skin_dataframe['image_id'].map(image_path.get)
#Use the path to read images.
skin_dataframe_balanced['image'] = skin_dataframe_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


n_samples = 5  

# Plot
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_dataframe_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')


#Convert dataframe column of images into numpy array
X = np.asarray(skin_dataframe_balanced['image'].tolist())
X = X/255. # Scale values to 0-1.
Y=skin_dataframe_balanced['label'] #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem

#Split to training and testing. Get a very small dataset for training as we will be 
# fitting it to many potential models. 
x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X, Y_cat, test_size=0.95, random_state=42)

#Further split data into smaller size to get a small test dataset. 
x_unused, x_valid, y_unused, y_valid = train_test_split(x_test_auto, y_test_auto, test_size=0.05, random_state=42)

#Define classifier for autokeras. Here I will check 25 different models, each model 25 epochs
clf = ak.ImageClassifier(max_trials=25) #MaxTrials - max. number of keras models to try
clf.fit(x_train_auto, y_train_auto, epochs=25)


#Evaluate the classifier on test data
_, acc = clf.evaluate(x_valid, y_valid)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
model = clf.export_model()
print(model.summary())

#Save the model
model.save('cifar_model.h5')


score = model.evaluate(x_valid, y_valid)
print('Test accuracy:', score[1])
