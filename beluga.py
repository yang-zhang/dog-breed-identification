
# coding: utf-8

# https://www.kaggle.com/gaborfodor/use-pretrained-keras-models-lb-0-3

# In[2]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import display
from IPython.display import Image

from os import listdir
from os.path import join
from datetime import datetime

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.utils.np_utils import to_categorical

from keras.applications.vgg16 import preprocess_input, decode_predictions

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
import utils_ds

from secrets import KAGGLE_USER, KAGGLE_PW


# ### preprocessing

# In[9]:

INPUT_SIZE = 224
NUM_CLASSES = 120
SEED = 1987


# In[10]:

competition_name = 'dog-breed-identification'
competition_dir = '/opt/notebooks/data/' + competition_name
raw_dir = competition_dir + '/raw'
data_dir = competition_dir + '/unzipped'


# In[11]:

# !mkdir $data_dir
# !cp $raw_dir/*.zip $data_dir
# !unzip '$data_dir/*.zip' -d $data_dir 
# !mkdir $data_dir/results
# rm $data_dir/*.zip


# In[12]:

labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))
print(len(listdir(join(data_dir, 'train'))), len(labels))
print(len(listdir(join(data_dir, 'test'))), len(sample_submission))


# In[7]:

random_img = np.random.choice(listdir(join(data_dir,'train')))
random_img = join(data_dir, 'train', random_img)
mimg = mpimg.imread(random_img)
plt.imshow(mimg)


# In[37]:

from IPython.display import Image
Image(random_img)


# In[13]:

selected_breed_list = labels.breed.value_counts().index.values[:NUM_CLASSES]
selected_breed_list = list(selected_breed_list)


# In[14]:

labels = labels[labels.breed.isin(selected_breed_list)]


# In[15]:

# labels['target'] = 1
# labels['rank'] = labels.groupby('breed').rank()['id']
# labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
# y_train = labels_pivot[selected_breed_list].values

y_train = pd.get_dummies(labels.breed).values


# In[16]:

le = LabelEncoder()
y_train_encode = le.fit_transform(labels.breed.values)


# In[17]:

np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))

train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
ytr = y_train[train_idx]
yv = y_train[valid_idx]


# In[18]:

def read_img(img_id, train_or_test, size):
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img


# ### ResNet50 class predictions for example images

# In[17]:

model = ResNet50(weights='imagenet')


# In[18]:

for row in labels.iloc[np.random.choice(len(labels), 5)].iterrows():
    img_id = row[1]['id']
    breed = row[1]['breed']
    img = read_img(img_id, 'train', (224, 224))
    x = np.expand_dims(img, axis=0)
    preds = model.predict(x)
    
    _, imagenet_class_name, prob = decode_predictions(preds, top=1)[0][0]
    print(breed, imagenet_class_name, prob)
    plt.imshow(-img)
    plt.show()


# ### Extract VGG16 bottleneck features

# In[19]:

INPUT_SIZE = 224
POOLING = 'avg'
x_train = np.empty((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, img_id in enumerate(labels['id']):
    img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x = preprocess_input(img)
    x_train[i] = x

Xtr = x_train[train_idx]
Xv = x_train[valid_idx]


# In[ ]:

print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))


# In[21]:

vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling=POOLING)
train_vgg_bf = vgg_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_vgg_bf = vgg_bottleneck.predict(Xv, batch_size=32, verbose=1)


# In[22]:

train_vgg_bf.shape, valid_vgg_bf.shape


# ### LogReg on VGG bottleneck features

# In[23]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_vgg_bf, y_train_encode[train_idx])

valid_probs = logreg.predict_proba(valid_vgg_bf)
valid_preds = logreg.predict(valid_vgg_bf)


# In[24]:

log_loss(yv, valid_probs)


# In[25]:

accuracy_score(y_train_encode[valid_idx], valid_preds)


# ### Extract Xception bottleneck features

# In[19]:

INPUT_SIZE = 299
POOLING = 'avg'
x_train = np.empty((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, img_id in enumerate(labels['id']):
    img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(img)
    x_train[i] = x


# In[ ]:

Xtr = x_train[train_idx]
Xv = x_train[valid_idx]


# In[25]:

xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)


# In[26]:

train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)


# ### LogReg on Xception bottleneck features

# In[27]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_x_bf, y_train_encode[train_idx])

valid_probs = logreg.predict_proba(valid_x_bf)
valid_preds = logreg.predict(valid_x_bf)


# In[28]:

log_loss(yv, valid_probs)


# In[29]:

accuracy_score(y_train_encode[valid_idx], valid_preds)


# ### Extract Inception bottleneck features

# In[ ]:

inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=POOLING)
train_i_bf = inception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_i_bf = inception_bottleneck.predict(Xv, batch_size=32, verbose=1)


# ### LogReg on Inception bottleneck features

# In[ ]:

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_i_bf, y_train_encode[train_idx])
valid_probs = logreg.predict_proba(valid_i_bf)
valid_preds = logreg.predict(valid_i_bf)


# In[ ]:

log_loss(yv, valid_probs)


# In[ ]:

accuracy_score(y_train_encode[valid_idx], valid_preds)


# ### LogReg on all bottleneck features

# In[ ]:

X = np.hstack([train_x_bf, train_i_bf])
V = np.hstack([valid_x_bf, valid_i_bf])
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(X, y_train_encode[train_idx])
valid_probs = logreg.predict_proba(V)
valid_preds = logreg.predict(V)


# In[44]:

log_loss(yv, valid_probs)


# In[45]:

accuracy_score(y_train_encode[valid_idx], valid_preds)


# ### Check errors

# In[ ]:

valid_idx_num = np.where(valid_idx)[0]


# In[ ]:

valid_breeds = y_train_encode[valid_idx]
error_idx = (valid_breeds != valid_preds)


# In[ ]:

error_idx_num = np.where(error_idx)[0]


# In[ ]:

error_idx_global_num = valid_idx_num[error_idx]


# In[ ]:

for i, j in zip(error_idx_num, error_idx_global_num):
    pred_breed = le.inverse_transform(valid_preds[i])
    row = labels.iloc[j]
    img_id = row['id']
    breed = row['breed']
    img = read_img(img_id, 'train', (224, 224))
    print(breed, pred_breed, prob)
    plt.imshow(-img)
    plt.show()


# ### predict test data

# In[62]:

test_ids = [file.split('.')[0] for file in listdir(data_dir+'/test')]


# In[105]:

INPUT_SIZE = 299
POOLING = 'avg'
x_test = np.empty((len(test_ids), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, img_id in enumerate(test_ids):
    img = read_img(img_id, 'test', (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(img)
    x_test[i] = (x)


# In[106]:

test_x_bf = xception_bottleneck.predict(x_test, batch_size=32, verbose=1)
test_i_bf = inception_bottleneck.predict(x_test, batch_size=32, verbose=1)
X_test = np.hstack([test_x_bf, test_i_bf])
valid_probs = logreg.predict_proba(X_test)
valid_preds = logreg.predict(X_test)


# ### submit

# In[107]:

ls $data_dir


# In[108]:

get_ipython().system('wc -l $data_dir/sample_submission.csv')


# In[109]:

subm=pd.DataFrame(np.hstack([np.array(test_ids).reshape(-1, 1), valid_probs]))


# In[110]:

cols = ['id']+list(le.inverse_transform(range(NUM_CLASSES)))

subm.columns = cols

submission_file_name = data_dir+'/results/belugatest%s.csv' % datetime.now().strftime('%Y-%m-%d-%H-%M')

subm.to_csv(submission_file_name, index=False)


# In[111]:

get_ipython().system('kg config -g -u $KAGGLE_USER -p $KAGGLE_PW -c $competition_name')
get_ipython().system('kg submit $submission_file_name -u $KAGGLE_USER -p $KAGGLE_PW -m "Test"')


# Your submission scored 0.29973

# In[ ]:



