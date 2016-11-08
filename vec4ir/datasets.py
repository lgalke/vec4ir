#!/usr/bin/env python3

import pandas as pd
import os
from html.parser import HTMLParser

# NTCIR_ROOT_PATH = # think about this


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


class NTCIR(object):
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

    def __init__(self, root_path, cache_dir=None):
        self.root_path = root_path
        try:
            os.mkdir(cache_dir)
        except FileExistsError:
            pass
        self.cache_dir = cache_dir

    def docs(self, kaken=True, gakkai=True, verify_integrity=False):
        if not kaken and not gakkai:
            raise ValueError

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
                df = pd.read_pickle(cache)
                return df
            except FileNotFoundError:
                pass

        docs = []
        if kaken:
            kaken_docs = self.kaken(verify_integrity=verify_integrity)
            docs.append(kaken_docs)

        if gakkai:
            gakkai_docs = self.gakkai(verify_integrity=verify_integrity)
            docs.append(gakkai_docs)

        df = pd.concat(docs, verify_integrity=verify_integrity)
        if cache:
            df.to_pickle(cache)
        return df

    def kaken(self, verify_integrity=False):
        path = os.path.join(self.root_path, "e-docs", "ntc2-e1k")
        df = NTCIR._read_docs(path, "pjne", verify_integrity=verify_integrity)
        return df

    def gakkai(self, verify_integrity=False):
        path = os.path.join(self.root_path, "e-docs", "ntc2-e1g")
        df = NTCIR._read_docs(path, "tite", verify_integrity=verify_integrity)
        return df

    def rels(self, number, verify_integrity=False):
        number = int(number)
        path = os.path.join(self.root_path, "rels")
        path = os.path.join(path, "rel"+str(number)+"_ntc2-e2_0101-0149.nc")
        return NTCIR._read_rels(path, verify_integrity=verify_integrity)

    def topics(self,
               names=["title"],
               key="e0101-0149",
               verify_integrity=False):
        path = os.path.join(self.root_path, "topics", "topic-" + key)
        return NTCIR._read_topics(path, names,
                                  verify_integrity=verify_integrity)


if __name__ == '__main__':
    ntcir2 = NTCIR('/home/lpag/git/vec4ir/data/NTCIR2/')
    docs = ntcir2.docs(verify_integrity=True)
    print(docs)
    print(docs.columns)
    del docs
    topics = ntcir2.topics(verify_integrity=True)
    print(topics)
    del topics
    rels = ntcir2.rels(2, verify_integrity=True)
    print(rels)
    del rels
