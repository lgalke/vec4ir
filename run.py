import os
import fileinput
from argparse import ArgumentParser, FileType

from gensim.models import KeyedVectors, Doc2Vec

from vec4ir.base import Matching, Tfidf
from vec4ir.core import Retrieval
from vec4ir.doc2vec import Doc2VecInference
from vec4ir.utils import build_analyzer
from vec4ir.word2vec import WordCentroidDistance, WordMoversDistance

VALID_RETRIEVAL_MODELS = ('tfidf', 'wcd', 'wmd', 'd2v')

def load_documents(path, with_identifiers=True, sep='\t'):
    documents = []
    if os.path.isdir(path):
        # Read all files of directory
        root, dirs, files = next(os.walk(path))
        for fname in files:
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as fhandle:
                fcontent = fhandle.read()
            if with_identifiers:
                documents.append((fname, fcontent))
            else:
                documents.append(fcontent)
    elif os.path.isfile(path):
        with open(path, 'r') as fhandle:
            for line in fhandle:
                line = line.strip()
                if with_identifiers:
                    # use first column as identifier
                    identifier, content = line.split(sep)
                    documents.append((identifier, content))
                else:
                    # just use full line as content
                    documents.append(line)

    else:
        raise ValueError("{} seems to be neither directory nor file".format(path))

    if with_identifiers:
        ids, docs = tuple(zip(*documents))
        return docs, ids
    else:
        return documents


def run(args, inputs):
    analyzer = build_analyzer(tokenizer=args.tokenizer,
                                       stop_words=args.stop_words,
                                       lowercase=args.lowercase)
    match_op = Matching(analyzer=analyzer)

    if args.retrieval_model == 'tfidf':
        # we do not need an embedding for tf-idf
        retrieval_model = Tfidf(analyzer=analyzer, use_idf=args.idf)
    else:
        # could try-except to guess binary
        embedding = Doc2Vec.load(args.embedding) if args.retrieval_model == 'd2v' else KeyedVectors.load_word2vec_format(args.embedding)

        retrieval_model = {
            'wcd': WordCentroidDistance(embedding=embedding,
                                        analyzer=analyzer,
                                        use_idf=args.idf),
            'wmd': WordMoversDistance(embedding, analyzer,
                                      complete=args.wmd,
                                      use_idf=args.idf),
            'd2v': Doc2VecInference(embedding, analyzer)
        }[args.retrieval_model]



    # retrieval model defined
    retrieval = Retrieval(retrieval_model, name=args.retrieval_model,
                          matching=match_op)

    documents, ids = load_documents(args.data)
    retrieval.fit(documents, ids)


    for line in fileinput.input(inputs):
        results = retrieval.query(line, k=args.k)
        print(results)



def main():
    """ Parse command line arguments """
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default=None,
                        help="Path to data directory or csv file")
    parser.add_argument("-r", "--retrieval-model", type=str, default='tfidf',
                        choices=VALID_RETRIEVAL_MODELS)
    parser.add_argument("-e", "--embedding", type=str, default=None,
                        help="Specify path to word embedding.")
    matching_options = parser.add_argument_group('Matching options')
    matching_options.add_argument("-C",
                                  "--cased",
                                  dest="lowercase",
                                  default=True,
                                  action='store_false',
                                  help="Case sensitive analysis")
    matching_options.add_argument("-T",
                                  "--tokenizer",
                                  default='sklearn',
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
                                  default=True, action='store_false',
                                  help="Do NOT use stopwords")
    matching_options.add_argument("-M", "--no-matching", dest='matching',
                                  default=True, action='store_false',
                                  help="Do NOT apply matching operation.")
    ret_opt = parser.add_argument_group("Retrieval Options")
    ret_opt.add_argument("-I", "--no-idf", action='store_false', dest='idf',
                         default=True,
                         help='Do not use IDF when aggregating word vectors')
    ret_opt.add_argument("-w", "--wmd", type=float, dest='wmd', default=1.0,
                         help="WMD Completeness factor, defaults to 1.0")
    evaluation_options = parser.add_argument_group("Evaluation options")
    evaluation_options.add_argument("-k", dest='k', default=20, type=int,
                                    help="number of documents to retrieve")

    args, inputs = parser.parse_known_args()

    if args.retrieval_model is not 'tfidf' and args.embedding is None:
        raise ValueError("An embedding (-e) is mandatory when retrieval model is not tfidf")

    run(args, inputs)



if __name__ == '__main__':
    main()
