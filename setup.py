import tensorflow_addons as tfa
import numpy as np
from pathlib import Path
import enum

class LearningStrategy(enum.Enum):
    online_learning = 1
    offline_learning = 2

class Architecture(enum.Enum):
    base = 1
    mobile_net = 2
    inception = 3


## %%
class Config:
    def __init__(self, path_dataset, loss=tfa.losses.TripletSemiHardLoss(0.2), pre_process=True, as_npy=False,
                 path_preprocessed_dataset=None, img_shape=96, architecture=Architecture.mobile_net, use_weights=True):
        path_dataset = Path(path_dataset)
        assert isinstance(img_shape, int)
        self.img_shape = img_shape
        self.loss = loss
        self.use_weights = use_weights
        try:
            assert path_dataset.exists()
        except AssertionError:
            raise(ValueError("The dataset path doesnt exist"))
        try:
            assert 'train','test' in path_dataset.iterdir()
        except AssertionError:
            raise(ValueError("no directory train or test in dataset path"))
        self.path_dataset = path_dataset
        try:
            assert architecture in Architecture
            self.architecture = architecture
        except AssertionError:
            raise(ValueError("Incorrect Architecture type provided"))
        if not pre_process and as_npy:
            raise(ValueError("Cannot specify as numpy without pre-processing"))
        if pre_process:
            self.pre_process = True
            self.as_npy = as_npy
            if path_preprocessed_dataset == None:
                print("Creating pre-process directory in base of training")
                path_preprocessed_dataset = path_dataset.parent.joinpath(path.name + '_pre_processssed')
                path_preprocessed_dataset.mkdir()
                self.path_preprocessed_dataset = path_preprocessed_dataset
            else:
                path_preprocessed_dataset = Path(path_preprocessed_dataset)
                assert path_preprocessed_dataset.exists()
                if(len(list(path_preprocessed_dataset.iterdir())) > 1):
                    raise(ValueError("The pre-process path has files, please ensure that the path is empty"))
                else:
                    self.path_preprocessed_dataset = path_preprocessed_dataset
        else:
            self.pre_process = False

## %%
class OnlineLearning(Config):
    def __init__(self, k=10, num_people=15, *args, **kwargs):
        self.k = k
        self.num_people = 15
        self.strategy = LearningStrategy.online_learning
        super().__init__(*args, **kwargs)

class OfflineLearning(Config):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        self.strategy = LearningStrategy.offline_learning
        super().__init__(*args, **kwargs)

t = OnlineLearning(10, 12, "/home/touchdown/vggface2_train_preprocessed/", loss=tfa.losses.TripletHardLoss(0.4), pre_process=False, architecture=Architecture.mobile_net)
