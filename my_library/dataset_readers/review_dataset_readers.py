from typing import Dict, Optional
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import re


@DatasetReader.register("ReviewDatasetReader")
class ReviewDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 padding_length: int = None,
                 char_level: bool =False) -> None:
        super().__init__(lazy=False)
        
        self._padding_length = padding_length
        self._tokenizer = tokenizer or (CharacterTokenizer() if char_level else WordTokenizer())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(
            token_min_padding_length=(self._padding_length or 0))}

    @overrides
    def _read(self, file_path):
        
        logger.info("Reading instances from: %s", file_path)
        cnt=0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cnt+=1
                if cnt==1:
                    continue
                
                text, label = line.split('\t')
                assert (len(text) > 0)
                assert(len(label) > 0)
                #text = clean_str(text)
                yield self.text_to_instance(text, label)

    @overrides

    def text_to_instance(self, text: str, target: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_text = self._tokenizer.tokenize(text)
        if self._padding_length and len(tokenized_text) > self._padding_length:
            tokenized_text = tokenized_text[:self._padding_length] # truncate 


        text_field = TextField(tokenized_text, self._token_indexers)

        fields = {'text': text_field}
        if target is not None:
            fields['label'] = LabelField(target)
        return Instance(fields)


