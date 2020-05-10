%matplotlib qt  # NOQA
import warnings  # NOQA
warnings.simplefilter('ignore')  # NOQA
import os  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NOQA
os.environ['AUTOGRAPH_VERBOSITY'] = '0'  # NOQA
import tensorflow as tf  # NOQA
tf.autograph.set_verbosity(0)  # NOQA
import matplotlib.pyplot as plt  # NOQA
from learning_strategy import online_model  # NOQA
import imageio  # NOQA
from skimage.transform import resize  # NOQA
import numpy as np  # NOQA
from pathlib import Path  # NOQA
import cv2  # NOQA
from sklearn.neighbors import RadiusNeighborsClassifier  # NOQA
from sklearn.metrics import *  # NOQA
import matplotlib.pyplot as plt  # NOQA
import gc  # NOQA
from fast_learner import NN_fl_train  # NOQA

# %%
method = "NN_fl"
if method == "NN_fl":
    NN_fl_path = "./weights/fast_learner_weights_1.h5"
    w = NN_fl_train("/home/touchdown/vggface2_test_preprocessed",
                    "./weights/siamese_weights_3.h5",
                    "./weights/fast_learner_weights_1.h5")
    model = w.fast_learner_model
    data = w._get_data(3000)
    x = np.swapaxes(data[0], 0, 1)
    prediction = model.predict([x[0], x[1]])
    t = prediction.T[0] > 0.8
    num_img = data[1].T[0]
    pred = prediction.T[0]
    test = num_img
else:
    model = online_model(shape=96, use_trained="Mobile")
    model.build(input_shape=[None, 96, 96, 3])
    status = model.load_weights("./weights/siamese_weights_3.h5")

    path2 = "/home/touchdown/vggface2_train_preprocessed"
    path = Path(path2)

    gc.collect()
    file_dir = [x for x in path.iterdir() if x.is_dir()]

    num_people = 1500
    per = 25
    num_img = list()
    people = np.ones((num_people * per, 96, 96, 3), dtype=np.float16)

    for person in range(num_people):
        counter = 0
        for image in list(file_dir[person].glob('*.jpg'))[0:per]:
            people[person * per + counter] = cv2.cvtColor(
                cv2.imread(image.as_posix()), cv2.COLOR_RGB2BGR)/255
            num_img.append(person)
            counter += 1

    gc.collect()
    num_img = np.array(num_img)
    embeddings = model.predict(people)
    del people
    gc.collect()
    neigh = RadiusNeighborsClassifier(radius=0.53, n_jobs=7)
    neigh.fit(embeddings, num_img)
    pred = neigh.predict(embeddings)
    test = num_img
# %%

print("Accuracy is {}".format(
    accuracy_score(pred, test) * 100))
print("Precision is {}".format(
    precision_score(pred, test, average='weighted')))
print("Recall is {}".format(
    recall_score(pred, test, average='macro')))

# %%
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test, pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

roc_auc = roc_auc_score(test, pred)
print(roc_auc)
plt.figure()
plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
# %%
