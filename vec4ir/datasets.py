#!/usr/bin/env python3
import pandas as pd
import os
from html.parser import HTMLParser
from abc import abstractmethod, ABC

# NTCIR_ROOT_PATH = # think about this


class IRDataSetBase(ABC):
    # def __subclasshook__(subclass):
    #     if hasattr(subclass, "docs") and hasattr(subclass, "rels") and hasattr(subclass, "topics"):
    #         return True
    #     else:
    #         return False

    @property
    @abstractmethod
    def docs():
        pass

    @property
    @abstractmethod
    def rels():
        pass

    @property
    @abstractmethod
    def topics():
        pass


class Econ62k(IRDataSetBase):
    """The famous econ62k dataset"""
    def __init__(self, gold_path, thesaurus_path, fulltext_path, verify_integrity=False):
        """inits the data set with paths and integrity checks..."""
        self.__docs = None
        self.__rels = None
        self.__topics = None

    @property
    def docs(self):
        # in memory cache
        if self.__docs is not None:
            return self.__docs

        self.__docs = docs
        return docs

    @property
    def rels(self):
        if self.__rels is not None:
            return self.__rels

        self.__rels = rels
        return rels

    @property
    def topics(self):
        if self.__topics is not None:
            return self.__topics
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
                 topics=["title"],
                 verify_integrity=False,
                 cache_dir=None,
                 verbose=0):
        self.__kaken = kaken
        self.__gakkai = gakkai
        self.__rels = int(rels)
        self.__topics = topics
        self.__verify_integrity = verify_integrity
        self.__verbose = verbose
        self.root_path = root_path
        if not cache_dir:
            print(UserWarning("No cachedir specified"))
        else:
            try:
                os.mkdir(cache_dir)
            except FileExistsError:
                pass
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
                return df
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
        return df

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
        path = os.path.join(path, "rel" + str(number) + "_ntc2-e2_0101-0149.nc")
        return NTCIR._read_rels(path, verify_integrity=verify_integrity)

    @property
    def topics(self):
        names = self.__topics
        verify_integrity = self.__verify_integrity
        key = "e0101-0149"  # could be global
        path = os.path.join(self.root_path, "topics", "topic-" + key)
        return NTCIR._read_topics(path, names, verify_integrity=verify_integrity)


if __name__ == '__main__':
    ntcir2 = NTCIR('/home/lpag/git/vec4ir/data/NTCIR2/', verify_integrity=True, rels=2)
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
