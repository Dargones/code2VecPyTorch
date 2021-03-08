#!/usr/bin/env python
# coding: utf-8

# # Training a code2vec model on a custom dataset

# In[1]:


from vocabularies import Vocab
from config import *
from dataset import IterableBaseC2VDataset, ShuffleDataset
from torch.utils.data import DataLoader
import torch as tt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import BaseCode2Vec
from catalyst.runners import SupervisedRunner
from catalyst.dl import utils, EarlyStoppingCallback
from catalyst.utils import plot_metrics, load_checkpoint, unpack_checkpoint
from metrics import *


# ## Creating token\path\target vocabularies and initializing the datasets:

# In[2]:


Config.DATASET = "/home/sasha/Desktop/Research/code2vec/data/java14mMed/java14mMed"
Config.TRAIN_DATA = Config.DATASET + ".train.c2v"
Config.TEST_DATA = Config.DATASET + ".test.c2v"
Config.VAL_DATA = Config.DATASET + ".val.c2v"
Config.MAX_TOKEN_VOCAB_SIZE = 0.9  # 90% of all tokens are retained (most frequent tokens are kept)
Config.MAX_PATH_VOCAB_SIZE = 0.9
Config.MAX_TARGET_VOCAB_SIZE = 0.99


# In[3]:


Vocab.prepare_for_file(Config.TRAIN_DATA, override=False)
token_voc = Vocab.tokens(Config.TRAIN_DATA, Config.MAX_TOKEN_VOCAB_SIZE)
path_voc = Vocab.paths(Config.TRAIN_DATA, Config.MAX_PATH_VOCAB_SIZE)
target_voc = Vocab.targets(Config.TRAIN_DATA, Config.MAX_TARGET_VOCAB_SIZE)


# In[4]:


Config.PROPERTIES = 0
Config.BATCH_SIZE = 256


# In[5]:


train_dataset = ShuffleDataset(
    IterableBaseC2VDataset(Config.TRAIN_DATA, token_voc, path_voc, target_voc, Config.PROPERTIES),
    Config.SHUFFLE_BUFFER_SIZE
)
val_dataset = IterableBaseC2VDataset(Config.VAL_DATA, token_voc, path_voc, target_voc, Config.PROPERTIES)
test_dataset = IterableBaseC2VDataset(Config.TEST_DATA, token_voc, path_voc, target_voc, Config.PROPERTIES)

test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)


# ## Creating the model

# In[6]:


Config.DEFAULT_EMBEDDINGS_SIZE = 128
Config.LR_START = 0.01


# In[7]:


model = BaseCode2Vec(Config.DEFAULT_EMBEDDINGS_SIZE,
                 len(token_voc),
                 len(path_voc),
                 len(target_voc), Config.PROPERTIES)
Config.PROPERTIES = 0
criterion = CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=Config.LR_START)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=6)  # TODO: criteria of plateau seem murky here.


# ## Training the model
# ## NOTE: Change tt.device('cuda') to tt.device('cpu') if CUDA is not configured!

# In[8]:


runner = SupervisedRunner(device=tt.device('cuda'))


# In[9]:


Config.NUM_TRAIN_EPOCHS = 100
Config.PATIENCE = 3
Config.PATIENCE_DELTA = 0.001


# In[10]:


runner.train(
    model=model,
    main_metric = "f-score",
    minimize_metric = False,
    criterion=criterion,
    optimizer=optimizer,
    loaders={"train": train_loader, "valid": val_loader}, # loaders={"train": train_loader},
    scheduler=scheduler,
    logdir='./logdir',
    num_epochs=Config.NUM_TRAIN_EPOCHS,
    verbose=True,
    callbacks=[
        EarlyStoppingCallback(patience=Config.PATIENCE, min_delta=Config.PATIENCE_DELTA),
        SubtokenFScoreallback(target_vocab=target_voc)
    ]
)


# ## Training graphs and results on the testing set

# In[ ]:


plot_metrics(
    logdir='./logdir',
    metrics=["f-score"],
    step="epoch"
)


# In[ ]:


plot_metrics(
    logdir='./logdir',
    metrics=["loss"],
    step="epoch"
)


# In[ ]:


checkpoint = load_checkpoint("logdir/checkpoints/best.pth")
unpack_checkpoint(checkpoint=checkpoint, model=model)


# In[ ]:


predictions = runner.predict_loader(model=model, loader=test_loader)  # loader = test_loader
precision, recall, f_score = get_metrics_dataset(predictions, test_loader, target_voc)  # test_loader
print("precision = %.3f, recall = %.3f, f-score = %.3f" %(precision, recall, f_score))
