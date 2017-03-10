#!/usr/bin/env python3
"""
File: datasets.py
Author: Lukas Galke
Email: github@lpag.de
Github: https://github.com/lgalke
Description: Parsing and loading for all the data sets.
"""

import pandas as pd
import numpy as np
import os
from html.parser import HTMLParser
from abc import abstractmethod, ABC
from collections import defaultdict
from .thesaurus_reader import ThesaurusReader
from .base import harvest
import csv

# NTCIR_ROOT_PATH = # think about this
DEFAULT_CACHEDIR = os.path.expanduser("~/.cache")


class IRDataSetBase(ABC):
    @property
    @abstractmethod
    def docs(self):
        pass

    @property
    @abstractmethod
    def rels(self):
        pass

    @property
    @abstractmethod
    def topics(self):
        pass

    def load(self, verbose=False):

        labels, docs = self.docs
        if verbose:
            print(len(docs), "documents.")

        queries = self.topics
        if verbose:
            n_queries = len(queries)
            print(n_queries, "queries.")

        rels = self.rels
        if verbose:
            n_rels = np.asarray([len([r for r in harvest(rels, qid) if r > 0])
                                 for qid, __ in queries])
            print("{:2f} ({:2f}) relevant documents per query."
                  .format(n_rels.mean(), n_rels.std()))

        return docs, labels, queries, rels


def mine_gold(path, verify_integrity=False):
    """ returns a dict of dicts label -> docid -> 1"""
    def zero_default():
        return defaultdict(int)
    gold = defaultdict(zero_default)
    with open(path, 'r') as f:
        rd = csv.reader(f, delimiter='\t')
        for line in rd:
            doc_id = int(line[0])
            labels = line[1:]
            for label in labels:
                gold[label][doc_id] = 1
    return gold


def _first_preflabel(node):
    return node['prefLabel'][0]


def synthesize_topics(gold, thesaurus, accessor=_first_preflabel):
    """ Returns a list of (query_id, querystring) pairs"""
    topics = [(label, accessor(thesaurus[label])) for label in
              set(gold.keys())]
    return topics


def harvest_docs(path, verify_integrity):
    if os.path.isdir(path):
        fnames = os.listdir(path)
        data = dict()
        for fname in fnames:
            with open(os.path.join(path, fname), 'r') as f:
                label, __ = os.path.splitext(fname)
                data[int(label)] = f.read()
        # fulltext documents
        docs = pd.DataFrame.from_dict(data, orient='index')
        labels, docs = docs.index.values, docs.iloc[:, 0].values

    elif os.path.isfile(path):
        # title doucments
        docs = pd.read_csv(path, sep='\t', names=["title"], index_col=0)
        labels, docs = docs.index.values, docs["title"].values

    else:
        raise UserWarning("No symlinks allowed.")
    print("labels of type {}, docs of type {}".format(labels.dtype,
                                                      docs.dtype))

    return labels, docs


class QuadflorLike(IRDataSetBase):
    """The famous quadflor-like dataset specification"""
    def __init__(self,
                 y=None,
                 thes=None,
                 X=None,  # Path to dir of documents
                 verify_integrity=False):
        self.__docs = None
        self.__rels = None
        self.__topics = None
        self.gold_path = y
        self.thesaurus_reader = ThesaurusReader(thes, normalize=False)
        self.doc_path = X
        self.verify_integrity = verify_integrity

    @property
    def docs(self):
        # in memory cache
        if self.__docs is not None:
            return self.__docs
        path = self.doc_path
        labels, docs = harvest_docs(path,
                                    verify_integrity=self.verify_integrity)
        self.__docs = docs
        return labels, docs

    @property
    def rels(self):
        if self.__rels is not None:
            return self.__rels
        # acquire rels
        path = self.gold_path
        rels = mine_gold(path)
        self.__rels = rels
        return rels

    @property
    def topics(self):
        """ Synthesizes the topics for the dataset, rels will be computed
        first."""
        if self.__topics is not None:
            return self.__topics
        rels, thesaurus = self.rels, self.thesaurus_reader.thesaurus
        # acquire topics
        topics = synthesize_topics(rels, thesaurus)
        self.__topics = topics
        return topics


class NTCIRTopicParser(HTMLParser):
    def __init__(self, *args, record_tag="topic", tags=["title"], **kwargs):
        self.tags = tags
        self.record_tag = record_tag
        self.records = []
        super().__init__(*args, **kwargs)

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == self.record_tag:
            self.current_record = {}
            if len(attrs) > 1:
                raise ValueError
            self.current_record["qid"] = int(attrs[0][1])

    def handle_data(self, data):
        ctag = self.current_tag
        if ctag is None:
            return
        if ctag in self.tags:
            self.current_record[ctag] = data.strip()

    def handle_endtag(self, tag):
        if tag == self.record_tag:
            self.records.append(self.current_record)
        self.current_tag = None


class NTCIRParser(HTMLParser):
    def __init__(self,
                 *args,
                 paragraph_sep="\n",
                 record_tag="rec",
                 id_tag="accn",
                 title_tag="tite",
                 content_tag="abse",
                 paragraph_tag="abse.p",
                 **kwargs):
        self.records = []
        self.record_tag = record_tag
        self.id_tag = id_tag
        self.title_tag = title_tag
        self.content_tag = content_tag
        self.paragraph_tag = paragraph_tag
        self.paragraph_sep = paragraph_sep
        super().__init__(*args, **kwargs)

    def handle_starttag(self, tag, attrs):
        if tag == self.record_tag:
            self.current_record = {}
        elif tag == self.content_tag:
            self.current_paragraphs = []
        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == self.content_tag:
            s = self.paragraph_sep
            self.current_record['content'] = s.join(self.current_paragraphs)
        elif tag == self.record_tag:
            self.records.append(self.current_record)
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag is None:  # we are not inside any tag
            return
        elif self.current_tag == self.paragraph_tag:
            self.current_paragraphs.append(data)
        elif self.current_tag == self.id_tag:
            self.current_record['docid'] = data
        elif self.current_tag == self.title_tag:
            self.current_record['title'] = data


class NTCIR(IRDataSetBase):
    def __init__(self,
                 root_path,
                 kaken=True,
                 gakkai=True,
                 rels=2,
                 topic="title",
                 field="title",
                 verify_integrity=False,
                 cache_dir=os.path.join(DEFAULT_CACHEDIR, "vec4ir", "ntcir"),
                 verbose=0):
        self.__gakkai = gakkai
        self.__kaken = kaken
        self.__rels = int(rels)
        self.__topic = topic
        self.__verify_integrity = verify_integrity
        self.__verbose = verbose
        self.__field = field
        self.root_path = root_path
        if not cache_dir:
            print(UserWarning("No cachedir specified"))
        else:
            print("Using cache:", cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def _read_docs(path, title_tag, verify_integrity=False):
        parser = NTCIRParser(title_tag=title_tag)
        with open(path, 'r') as f:
            parser.feed(f.read())
        df = pd.DataFrame(parser.records)
        df.set_index("docid", inplace=True, verify_integrity=verify_integrity)
        return df

    def _read_rels(path, verify_integrity=False):
        df = pd.read_csv(path,
                         names=["qid", "rating", "docid", "relevance"],
                         sep='\t')
        df.set_index(["qid", "docid"], inplace=True, drop=True,
                     verify_integrity=verify_integrity)
        return df

    def _read_topics(path, names, verify_integrity=False):
        parser = NTCIRTopicParser(tags=names)
        with open(path, 'r') as f:
            parser.feed(f.read())
        df = pd.DataFrame(parser.records)
        df.set_index("qid", inplace=True, verify_integrity=verify_integrity)
        return df

    @property
    def docs(self):
        """ Method to access NTCIR documents with caching """
        kaken = self.__kaken
        gakkai = self.__gakkai
        verify_integrity = self.__verify_integrity
        verbose = self.__verbose
        field = self.__field
        if not kaken and not gakkai:
            raise ValueError("So... you call me and want no documents?")

        if self.cache_dir:
            identifier = {
                (False, True): "kaken.pkl",
                (True, False): "gakkai.pkl",
                (True, True): "gakkeikaken.pkl"
            }[(gakkai, kaken)]
            cache = os.path.join(self.cache_dir, identifier)
        else:
            cache = None

        if cache:
            try:
                if verbose > 0:
                    print("Cache hit:", cache)
                df = pd.read_pickle(cache)
                return df.index.values, df[field].values
            except FileNotFoundError:
                if verbose > 0:
                    print("Cache miss.")
                pass

        docs = []
        if kaken:
            kaken_docs = self.kaken(verify_integrity=verify_integrity,
                                    verbose=verbose)
            docs.append(kaken_docs)

        if gakkai:
            gakkai_docs = self.gakkai(verify_integrity=verify_integrity,
                                      verbose=verbose)
            docs.append(gakkai_docs)

        df = pd.concat(docs, verify_integrity=verify_integrity)
        if cache:
            if verbose > 0:
                print("Writing cache: ", self.cache)
            df.to_pickle(cache)

        labels, documents = df.index.values, df[field].values
        return labels, documents

    def kaken(self, verify_integrity=False, verbose=0):
        path = os.path.join(self.root_path, "e-docs", "ntc2-e1k")
        if verbose > 0:
            print("Loading: ", path)
        df = NTCIR._read_docs(path, "pjne", verify_integrity=verify_integrity)
        return df

    def gakkai(self, verify_integrity=False, verbose=0):
        path = os.path.join(self.root_path, "e-docs", "ntc2-e1g")
        if verbose > 0:
            print("Loading: ", path)
        df = NTCIR._read_docs(path, "tite", verify_integrity=verify_integrity)
        return df

    @property
    def rels(self):
        verify_integrity = self.__verify_integrity
        number = self.__rels
        path = os.path.join(self.root_path, "rels")
        fname = "rel" + str(number) + "_ntc2-e2_0101-0149.nc"
        path = os.path.join(path, fname)
        rels_df = NTCIR._read_rels(path, verify_integrity=verify_integrity)
        return rels_df['relevance']

    @property
    def topics(self):
        desired_topic_field = self.__topic
        names = [desired_topic_field]
        verify_integrity = self.__verify_integrity
        key = "e0101-0149"  # could be global
        path = os.path.join(self.root_path, "topics", "topic-" + key)
        topics_df = NTCIR._read_topics(path, names,
                                       verify_integrity=verify_integrity)
        topics = topics_df[desired_topic_field]
        return list(zip(topics.index, topics))


if __name__ == '__main__':
    ntcir2 = NTCIR('/home/lpag/git/vec4ir/data/NTCIR2/', verify_integrity=True,
                   rels=2)
    docs = ntcir2.docs
    print(docs)
    print(docs.columns)
    del docs
    topics = ntcir2.topics
    print(topics)
    del topics
    rels = ntcir2.rels
    print(rels)
    del rels
