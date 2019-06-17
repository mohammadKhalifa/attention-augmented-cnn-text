import logging
import os

import sys
sys.path.insert(0,'../')
from my_library.encoders.attention_cnn import AttnCnnEncoder
from my_library.dataset_readers.ar_lm_dataset_reader import ArLMDatasetReader
from my_library.models.ar_language_model import ArLanguageModel

from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, CnnEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
import torch
from allennlp.models.archival import archive_model
from allennlp.training.util import create_serialization_dir, evaluate
import numpy as np

if __name__=='__main__':


    #creating sample tokens tensor of shape (1, 5, 100)
    tokens = torch.FloatTensor(np.random.randint(1, 100, size=(30,100)))
    tokens = tokens.unsqueeze(0)
    
    cnn_attn_encoder = AttnCnnEncoder(100, num_filters=10, output_dim=None, use_self_attention=True)

    cnn_output = cnn_attn_encoder(tokens, None)
    print(cnn_output.size())
    