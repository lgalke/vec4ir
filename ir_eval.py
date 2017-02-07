#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from datetime import timedelta
from vec4ir.datasets import NTCIR, QuadflorLike
from argparse import ArgumentParser, FileType
from vec4ir.base import TfidfRetrieval
from vec4ir.word2vec import Word2VecRetrieval, WordCentroidRetrieval
from vec4ir.doc2vec import Doc2VecRetrieval
from vec4ir.eqlm import EQLM
from vec4ir.utils import collection_statistics
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from textwrap import indent
from nltk.tokenize import word_tokenize
import sys
import os
import pprint
import numpy as np
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # compat on non-gui uis
# import matplotlib.patches as mpatches
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)
MODEL_KEYS = ['tfidf', 'wcd', 'wmd', 'pvdm', 'eqlm', 'swcd']


def mean_std(array_like):
    array_like = np.asarray(array_like)
    return array_like.mean(), array_like.std()


def plot_precision_recall_curves(results, path=None, plot_f1=False):
    """
    Plots a precision recall curve to `path`.
    :results:
        dict of dicts containing strategies as first-level keys and
        'precision', 'recall' as second level keys.
    :path:
        Write plot to this file.
    """
    colors = "b g r c m y k".split()

    keys = sorted(results.keys())
    # patches = []
    for name, c in zip(keys, colors):
        values = results[name]
        # patches.append(mpatches.Patch(color=c, label=name))
        precision_recall = zip(values["precision"], values["recall"])
        precision, recall = zip(*list(sorted(precision_recall,
                                             key=itemgetter(0), reverse=True)))
        recall = [0.] + list(recall) + [1.]
        precision = [1.] + list(precision) + [0.]
        plt.plot(recall, precision, color=c, label=name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower left')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def is_embedded(sentence, embedding, analyzer):
    """
    >>> embedding = ["a", "b", "c"]
    >>> queries =  ["a b c", "a", "b", "c", "a b c d", "d", "a b c"  ]
    >>> analyzer = lambda x: x.split()
    >>> [query for query in queries if is_embedded(query, embedding, analyzer)]
    ['a b c', 'a', 'b', 'c', 'a b c']
    >>> analyzer = CountVectorizer().build_analyzer()
    >>> [query for query in queries if is_embedded(query, embedding, analyzer)]
    ['a b c', 'a', 'b', 'c', 'a b c']
    """
    for word in analyzer(sentence):
        if word not in embedding:
            print("Dropping:", sentence, file=sys.stderr)
            return False

    return True


def ir_eval(irmodel, documents, labels, queries, rels, metrics=None, k=20,
            verbose=3, replacement=0):
    """
    irmodel
    X : iterator of documents
    Xtest  : iterator of queries
    Ytest  : iterator of doc,relevance pairs
    """
    if verbose > 0:
        print("=" * 79)
        t0 = timer()
        print("Fit", irmodel.name, "... ")
        irmodel.fit(documents, labels)
        t1 = timer()
        print("took {} seconds.".format(timedelta(seconds=t1 - t0)))
        print("Evaluating", irmodel.name, "...")
    if verbose > 1:
        print("-" * 79)
    values = irmodel.evaluate(queries, rels, verbose=verbose - 1, k=k,
                              replacement=replacement)
    if verbose > 1:
        print("-" * 79)
    if verbose > 0:
        pprint.pprint(pd.DataFrame(values).describe())
        print("=" * 79)

    return values


def smart_load_word2vec(model_path):
    print("Smart loading", model_path)
    if model_path is None:
        return None
    _, ext = os.path.splitext(model_path)
    if ext == ".gnsm":  # Native format
        print("Loading embeddings in native gensim format: {}"
              .format(model_path))
        model = Word2Vec.load(model_path)
    else:  # either word2vec text or word2vec binary format
        binary = ".bin" in model_path
        print("Loading embeddings in word2vec format: {}".format(model_path))
        model = Word2Vec.load_word2vec_format(model_path, binary=binary)
    return model


def _ir_eval_parser(config):
    valid_embedding_keys = config["embeddings"].keys()  # TODO document it
    valid_data_keys = config["data"].keys()
    parser = ArgumentParser()
    parser.add_argument("--doctest", action='store_true',
                        help="Perform doctest on this module")
    parser.add_argument("-d", "--dataset", type=str, default="ntcir2",
                        choices=valid_data_keys,
                        help="Specify dataset to use (as in config file)")
    parser.add_argument("-e", "--embedding", type=str, default=None,
                        choices=valid_embedding_keys,
                        help="Specify embedding to use (as in config file)")
    parser.add_argument("-f", "--focus", nargs='+',
                        choices=MODEL_KEYS, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=-1,
                        help="How many jobs to use, default=-1 (one per core)")

    # OPTIONS FOR OUTPUT
    output_options = parser.add_argument_group("Output options")
    output_options.add_argument("-o", "--outfile", default=sys.stdout,
                                type=FileType('a'))
    output_options.add_argument("-v", "--verbose", default=2, type=int,
                                help="verbosity level")
    output_options.add_argument('-s',
                                '--stats',
                                default=False,
                                action='store_true',
                                help="Print statistics for analyzed tokens")
    output_options.add_argument("-p",
                                "--plot",
                                default=None,
                                type=str,
                                metavar="PLOTFILE",
                                help="Save P-R curves in PLOTFILE")

    # OPTIONS FOR EVALUATION
    evaluation_options = parser.add_argument_group("Evaluation options")
    evaluation_options.add_argument("-k", dest='k', default=20, type=int,
                                    help="number of documents to retrieve")
    evaluation_options.add_argument("-Q",
                                    "--filter-queries",
                                    action='store_true',
                                    help="Filter queries w/o compl. embedding")
    evaluation_options.add_argument("-R", "--replacement",
                                    dest='repstrat', default="zero",
                                    choices=['drop', 'zero'],
                                    help=("Out of relevancy file document ids"
                                          "default is to use zero relevancy"))

    # OPTIONS FOR THE MATCHING OPERATION
    matching_options = parser.add_argument_group('Matching options')
    matching_options.add_argument("-C",
                                  "--cased",
                                  dest="lowercase",
                                  default=True,
                                  action='store_false',
                                  help="Case sensitive analysis")
    matching_options.add_argument("-T",
                                  "--tokenizer",
                                  default='sword',
                                  type=str,
                                  help=("Specify tokenizer for the matching"
                                        "operation" "defaults to 'sword'"
                                        "which removes"
                                        "punctuation but keeps single"
                                        "characterwords."
                                        "'sklearn' is similar but requires"
                                        "at" "least 2 characters for a word"
                                        "'nltk' retains punctuation as"
                                        "seperate" "tokens (use 'nltk' for"
                                        "glove models)"),
                                  choices=['sklearn', 'sword', 'nltk'])
    matching_options.add_argument("-S", "--dont-stop", dest='stop_words',
                                  default=True, action='store_true',
                                  help="Do NOT use stopwords")

    parser.add_argument("-u", "--train", default=False, action='store_true',
                        help="Train a whole new word2vec model")
    return parser


def load_ntcir2(config):
    raise DeprecationWarning("Use data sets load method instead")
    ntcir2 = NTCIR("../data/NTCIR2/",
                   rels=config['rels'],
                   topic=config['topic'],
                   field=config['field'])
    print("Loading NTCIR2 documents...")
    labels, documents = ntcir2.docs
    print("Loaded {:d} documents.".format(len(documents)))

    print("Loading topics...")
    queries = ntcir2.topics
    n_queries = len(queries)
    print("Using {:d} queries".format(n_queries))

    print("Loading relevances...")
    rels = ntcir2.rels
    n_rels = len(rels)
    print("With {:.1f} relevant docs per query".format(n_rels / n_queries))
    return documents, labels, queries, rels


def load_econ62k(cfg):
    raise DeprecationWarning
    dataset = QuadflorLike(y=cfg['y'],
                           thes=cfg['thes'],
                           X=cfg['X'],
                           verify_integrity=cfg['verify_integrity'])
    print("Loading econ62k documents...")
    labels, docs = dataset.docs
    print("Loaded {:d} documents.".format(len(docs)))

    print("Loading topics...")
    queries = dataset.topics
    n_queries = len(queries)
    print("Using {:d} queries".format(n_queries))

    print("Loading relevances...")
    rels = dataset.rels
    n_rels = sum(len(acc) for acc in rels.values())
    print("with {:.1f} relevant docs per query".format(n_rels / n_queries))
    return docs, labels, queries, rels


def init_dataset(data_config, default='quadflorlike'):
    """
    Given some dataset configuguration ("type" and **kwargs for the
    initializer), return the initialized data set object.  The returned object
    provides the properties `docs`, `rels`, and `topics`.
    """
    kwargs = dict(data_config)  # we assume dict
    # special type value to determine constructor
    T = kwargs.pop('type', default).lower()
    constructor = {"quadflorlike": QuadflorLike, "ntcir": NTCIR}[T]
    dataset = constructor(**kwargs)  # expand dict to kwargs
    return dataset


def build_analyzer(tokenizer=None, stop_words=None, lowercase=True):
    """
    A wrapper around sklearns CountVectorizers build_analyzer, providing an
    additional keyword for nltk tokenization.

    :tokenizer:
        None or 'sklearn' for default sklearn word tokenization,
        'sword' is similar to sklearn but also considers single character words
        'nltk' for nltk's word_tokenize function,
        or callable.
    :stop_words:
         False, None for no stopword removal, or list of words, 'english'/True
    :lowercase:
        Lowercase or case-sensitive analysis.
    """
    # some default options for tokenization
    if not callable(tokenizer):
        tokenizer, token_pattern = {
            'sklearn': (None, r"(?u)\b\w\w+\b"),  # mimics default
            'sword': (None, r"(?u)\b\w+\b"),   # specifically for GoogleNews
            'nltk': (word_tokenize, None)  # uses punctuation for GloVe models
        }[tokenizer]

    # allow binary decision for stopwords
    sw_rules = {True: 'english', False: None}
    if stop_words in sw_rules:
        stop_words = sw_rules[stop_words]

    # employ the cv to actually build the analyzer from the components
    analyzer = CountVectorizer(analyzer='word',
                               tokenizer=tokenizer,
                               token_pattern=token_pattern,
                               lowercase=lowercase,
                               stop_words=stop_words).build_analyzer()
    return analyzer


def print_dict(d, header=None, stream=sys.stdout, commentstring="% "):
    if header:
        print(indent(header, commentstring), file=stream)
    s = pprint.pformat(d, 2, 80 - len(commentstring))
    print(indent(s, commentstring), file=stream)
    return


def main():
    """TODO: Docstring for main.
    :returns: TODO
    """
    # parse command line arguments and read config file
    meta_parser = ArgumentParser(add_help=False)
    meta_parser.add_argument('-c',
                             '--config',
                             type=FileType('r'),
                             default='config.yml',
                             help="Specify configuration file")
    meta_args, remaining_args = meta_parser.parse_known_args()
    config = yaml.load(meta_args.config)  # we need meta parser
    parser = _ir_eval_parser(config)
    args = parser.parse_args(remaining_args)
    print(args)
    if args.doctest:
        import doctest
        doctest.testmod()
        exit(int(0))

    # load concrete data (old way)
    # dsc = config['data'][args.dataset]
    # dsc['verify_integrity'] = config.pop('verify_integrity', False)
    # loader = {'econ62k' : load_econ62k,
    #           'ntcir2' : load_ntcir2}[args.dataset]

    # documents, labels, queries, rels = loader(dsc)

    print("Selecting data set: {}".format(args.dataset))
    dataset = init_dataset(config['data'][args.dataset])
    print("Loading Data...", end="")
    documents, labels, queries, rels = dataset.load(verbose=args.verbose)
    print("Done")

    # Set up embedding specific analyzer
    print("Selecting embedding: {}".format(args.embedding))
    embedding_config = config["embeddings"][args.embedding]
    embedding = smart_load_word2vec(embedding_config["path"])
    embedding_analyzer_config = embedding_config["analyzer"]
    embedding_analyzer = build_analyzer(**embedding_analyzer_config)
    if args.stats:
        print("Computing collection statistics...")
        stats = collection_statistics(embedding=embedding,
                                      analyzer=embedding_analyzer,
                                      documents=documents)
        header = ("Statistics: {} x {}"
                  " x {tokenizer} x lower: {lowercase}"
                  " x stop_words: {stop_words}")
        header = header.format(args.dataset, args.embedding,
                               **embedding_analyzer_config)
        print_dict(stats, header=header, stream=args.outfile)
    embedding_oov_token = embedding_config["oov_token"]

    # Set up matching analyzer
    matching_analyzer = build_analyzer(tokenizer=args.tokenizer,
                                       stop_words=args.stop_words,
                                       lowercase=args.lowercase)

    focus = [f.lower() for f in args.focus] if args.focus else None
    repl = {"drop": None, "zero": 0}[args.repstrat]

    # TODO we do not train at all at the moment
    # if not embedding:
    #     print("Training word2vec model on all available data...")
    #     sentences = StringSentence(documents, analyzer)
    #     embedding = Word2Vec(sentences,
    #                          min_count=1,
    #                          iter=20)
    #     embedding.init_sims(replace=True)  # embedding becomes read-only

    if args.filter_queries is True:
        old = len(queries)
        queries = [(nb, query) for nb, query in queries if
                   is_embedded(query, embedding, embedding_analyzer)]
        print("Retained {} (of {}) queries".format(len(queries), old))

    def evaluation(m):
        return ir_eval(m,
                       documents,
                       labels,
                       queries,
                       rels,
                       verbose=args.verbose,
                       k=args.k,
                       replacement=repl)

    results = dict()
    tfidf = TfidfRetrieval(analyzer=matching_analyzer)
    matching = {"analyzer": matching_analyzer} if args.matching else None

    RMs = {"tfidf": tfidf,
           "wcd": Word2VecRetrieval(embedding, wmd=False,
                                    analyzer=matching_analyzer,
                                    vocab_analyzer=embedding_analyzer,
                                    oov=embedding_oov_token,
                                    verbose=args.verbose),
           "swcd": WordCentroidRetrieval(embedding, name="SWCD",
                                         matching=matching,
                                         analyzer=embedding_analyzer,
                                         oov=embedding_oov_token,
                                         verbose=args.verbose,
                                         normalize=False,
                                         algorithm='brute',
                                         metric='cosine',
                                         n_jobs=args.jobs),
           "wmd": Word2VecRetrieval(embedding, wmd=True,
                                    analyzer=matching_analyzer,
                                    vocab_analyzer=embedding_analyzer,
                                    oov=embedding_oov_token,
                                    verbose=args.verbose),
           "pvdm": Doc2VecRetrieval(analyzer=embedding_analyzer,
                                    matching=matching,
                                    n_jobs=args.jobs,
                                    metric="cosine",
                                    alpha=0.25,
                                    min_alpha=0.05,
                                    n_epochs=20,
                                    verbose=args.verbose),
           "eqlm": EQLM(tfidf, embedding, m=10, eqe=1,
                        analyzer=embedding_analyzer, verbose=args.verbose)
           }

    if focus:
        print("Focussing on:", " ".join(focus))
        for f in focus:
            RM = RMs[f]
            results[RM.name] = evaluation(RMs[f])
            del RM, RMs[f]  # clean up
    else:
        for key, RM in RMs.items():
            results[RM.name] = evaluation(RM)
            del RM, RMs[key]  # clean up

    if args.plot:
        plot_precision_recall_curves(results, path=args.plot)
    results = {name: {metric: mean_std(values) for metric, values in
               scores.items()} for name, scores in results.items()}
    print_dict(config, header="CONFIG", stream=args.outfile)
    print_dict(args, header="ARGS", stream=args.outfile)

    pd.DataFrame(results).to_latex(args.outfile)

    print("Done.")


if __name__ == "__main__":
    main()
