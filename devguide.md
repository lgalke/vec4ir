Developer’s guide
=================

The developer guide is targeted to people that want to extend or understand the functionality of the `vec4ir` package. We will go through the implementation details in a top-down manner (See Figure ). We start with the [core functionality](#core-functionality) (Section ). Then, we describe the implementation of the [matching operation](#matching) (Section ), [provided retrieval models](#retrieval-models) (Section ), [query expansion techniques](#query-expansion) (Section ) and helpful [utilities](#utilities) (Section ) in detail. The API design of `vec4ir` is heavily inspired by the one of `sklearn` (Buitinck et al. 2013).

<embed src="figures/devguide-organization.pdf" id="devguide-structure" style="width:75.0%" />

Core functionality
------------------

The core functionality includes all functionality required for setting up a retrieval model and evaluating it. It consists of [a generic information retrieval pipeline implementation](#retrieval), [embedded vectorisation](#EmbeddedVectorizer) (i.e. word vector aggregation) along with the [evaluation](#RetriEvalMixin) of retrieval models.

### The retrieval class

The `Retrieval` class wraps the functionality of the matching operation, query expansion and similarity scoring.

#### Constructor

In the constructor, the references to the retrieval model, the matching operation and the query expansion instances are stored. Additionally, users may set an identifier name for their composition of retrieval models. The `labels_` property will be filled on invoking `fit`.

``` python
def __init__(self, retrieval_model, matching=None,
              query_expansion=None, name='RM'):
    BaseEstimator.__init__(self)

    self._retrieval_model = retrieval_model
    self._matching = matching
    self._query_expansion = query_expansion
    self.name = name
    self.labels_ = None
```

#### The fit method

Upon `fit(X[, y])`, the respective `Retrieval` instance (as a whole) fits the documents. The `fit` method is called on the delegates of retrieval model instance as well as the matching or query expansion instances (if provided). If additional document identifiers are provided as argument `y` to `fit`, then these are stored inside the `labels_` property, such that in all further computation, the indices of `X` may be used for accessing the documents.

``` python
def fit(self, X, y=None):
    assert y is None or len(X) == len(y)
    self.labels_ = np.asarray(y) if y is not None else np.arange(len(X))
    if query_expansion:
        self._query_expansion.fit(X)

    if matching:
        self._matching.fit(X)

    self._retrieval_model.fit(X)
```

#### The query method

The `query(q[, k])` methods allows to query the `Retrieval` instance for `k` relevant documents to the query string `q`. First, in case a query expansion delegate is provided, it is called to transform the `q` according to the respective query expansion strategy. In a second step the matching operation is conducted (formally also optional) by invocation of its `predict` method. The `predict` method is expected to return a set of indices that matched the query `q`. The result of the matching operation `ind` is used in two ways. On the one hand, it is passed to the `query` method of the delegate of the retrieval model, such that it may (and should) reduce its internal representation according to the matching indices. On the other hand the view on the stored labels is reduced by `labels = labels[ind]`. The benefit of this reduction is that the relative indices `retrieved_indices`, returned by the retrieval model, can be used for accessing the labels array directly. Hence, the original document identifiers are restored. In between, we assert that the retrieval model does not return more than *k* documents by slicing `retrieved_indices = retrieved_indices[:k]` the top *k* indices.

``` python
def query(self, q, k=None):
    labels = self.labels_

    if self._query_expansion:
        q = self._query_expansion.transform(q)

    if self._matching:
        ind = self._matching.predict(q)
        if len(ind) == 0:
            return []
        labels = labels[ind]  # Reduce our own view
    else:
        ind = None

    retrieved_indices = self._retrieval_model.query(q, k=k, indices=ind)
    if k is not None:
        retrieved_indices = retrieved_indices[:k]

    return labels[retrieved_indices]
```

### Embedded vectorisation

Since `vec4ir` is dedicated to embedding-based retrieval models, a class is provided that aggregates word vectors according to the respective term-document frequencies. We call this process “embedded vectorisation”. The `EmbeddedVectorizer` extends the behaviour of `sklearn.feature_extration.text.TfidfVectorizer` as a subclass.

#### Constructor

In the constructor, the `index2word` property of the word vectors `embedding` is used to initialise the vocabulary of its super class `TfidfVectorizer`. This results in the crucial property that the indices of the embedding model and the transformed term-document matrix of the `TfidfVectorizer` correspond to each other. All additional arguments of the constructor are passed directly to the `TfidfVectorizer` so that its functionality is completely retained. Most notably, passing `use_idf=False` disables the discounting of frequent terms over the collection (which is enabled by default). If `normalize=True` is given, the word frequencies are normalised to unit L2-norm for each document.

``` python
def __init__(self, embedding, **kwargs):
    vocabulary = embedding.index2word
    self.embedding = embedding
    TfidfVectorizer.__init__(self, vocabulary=vocabulary, **kwargs)
```

#### Fitting and transforming

The fit method call is passed to the delegate of the super class `TfidfVectorizer`.

``` python
def fit(self, raw_docs, y=None):
    super().fit(raw_docs)
    return self
```

In the `transform(raw_documents, y=None)` method, we also start with the invocation of the `transform` method of the superclass. The obtained `Xt` are the term frequencies of the documents. Depending on the parameters passed to the `TfidfRetrieval` super class in the constructor, the values of `Xt` are already re-weighted and normalised to unit L2-norm. In the final step we aggregate the word vectors with a single matrix multiplication `Xt @ syn0`. Thus, word frequency with unit L2-norm will result in the centroid of the respective embedding word vectors.

``` python
def transform(self, raw_documents, y=None):
    Xt = super().transform(raw_documents)
    # Xt is sparse matrix of word frequencies
    syn0 = self.embedding.syn0
    # syn0 are the word vectors
    return (Xt @ syn0)
```

### Evaluation

For convenient evaluation, we provide the `RetriEvalMixin` class. A retrieval model class, that implements a `query(q[, k])` method, may inherit the provided `evaluate` method from the `RetriEvalMixin` class. The `evaluate` method is an interface for executing a set of queries by a single call and computing various ranking metrics mean average precision, mean reciprocal rank, normalised discounted cumulative gain, precision, recall, F1-score, precision@5, precision@10 as well as the respond time in seconds. The `evaluate` method expects the following arguments:

-   `X`  
    list of query identifier and query string pairs

-   `Y`  
    the gold standard, either a dictionary of dictionaries, a `pandas.DataFrame` with hierarchical index or an `numpy` / `scipy.sparse` array-like. The outer index is expected to be the query identifiers while the inner index needs to correspond to the document identifiers. A value greater than zero indicates relevance. Greater values indicate more relevance.

-   `k`  
    The desired amount of documents to retrieve. Except for precision@5 and precision@10, all metrics are computed with respect to this parameter.

-   `verbose`  
    Controls whether intermediate results should be printed to `stdout`.

-   `replacement`  
    The value to use for documents which could not be found in `Y` for the specific query identifier. The default value is zero. Hence missing query-document pairs are regarded non-relevant.

-   `n_jobs`  
    If greater than 1, the evaluation will be executed in parallel (not supported by all retrieval models).

The evaluate method of the `RetriEvalMixin` class is implemented as follows:

``` python
def evaluate(self, X, Y, k=20, verbose=0, replacement=0, n_jobs=1):
    # ... multi-processing code ...
    values = defaultdict(list)
    for qid, query in X:
        t0 = timer()
        result = self.query(query, k=k)
        values["time_per_query"].append(timer() - t0)
        scored_result = [harvest(Y, qid, docid, replacement)
                          for docid in result]
        r = scored_result[:k] if k is not None else scored_result

        # NDCG
        gold = harvest(Y, qid)
        idcg = rm.dcg_at_k(gold[argtopk(gold, k)], k)
        ndcg = rm.dcg_at_k(scored_result, k) / idcg
        values["ndcg"].append(ndcg)

        # MAP@k
        ap = rm.average_precision(r)
        values["MAP"].append(ap)

        # MRR 
        ind = np.asarray(r).nonzero()[0]
        mrr = (1. / (ind[0] + 1)) if ind.size else 0.
        values["MRR"].append(mrr)

        # ... other metrics ... #
    return values
```

Matching operation
------------------

The matching operation is implemented in the `Matching` class and designed to be employed as a component of the [`Retrieval`](#retrieval) class. The interface to implement for a custom matching operation is:

-   In the `fit(raw_documents)` method, an internal representation of the documents is stored.
-   The `predict(query)` returns the index set of matching documents.

### A generic matching class

The generic `Matching` class reduces the complexity of adding a matching algorithm to the implementation a single function. The default representation is a binary term-document matrix which should suffice for all boolean models. The representation is computed by an `sklearn.feature_extraction.text.CountVectorizer`. This process can be further customised by users, since additional keyword arguments are passed directly to the `CountVectorizer`.

``` python
def __init__(self, match_fn=TermMatch, binary=True, dtype=np.bool_,
              **cv_params):
    self._match_fn = match_fn
    self._vect = CountVectorizer(binary=binary, dtype=dtype,
                                  **cv_params)
```

Upon calling `predict`, the matching function `match_fn` is employed to compute the index set of the matching documents. Hence, the user can customise the behaviour (conjunctive or disjunctive) of each `Matching` instance by passing a function to the constructor. The `match_fn` function object is expected to return the matching indices, when provided with the term-document matrix `X` and the query string: `match_fn(X, q)` → `matching_indices`.

### Disjunctive Matching

Disjunctive (`OR`) matching of single query terms is the default query parsing method for most information retrieval systems. Thus, the framework provides a dedicated function for it, which is also the default value of the `match_fn` argument for the constructor of the `Matching` class.

``` python
def TermMatch(X, q):
    inverted_index = X.transpose()
    query_terms = q.nonzero()[1]
    matching_terms = inverted_index[query_terms, :]
    matching_doc_indices = np.unique(matching_terms.nonzero()[1])
    return matching_doc_indices
```

First, the index is inverted and the indices of the query tokens are extracted by finding the `np.nonzero` entries in the second dimension `[1]`. Second, we look up the indices of the query tokens in the inverted index. Finally, we only keep one index for each the matching document by calling `np.unique`.

Retrieval models
----------------

### TF-IDF

The `Tfidf` class implements the popular retrieval model based on TF-IDF re-weighting introduced by Salton and Buckley (1988). The term frequencies of a document are scaled by the inverse document frequency of the specific terms in the corpus *D* (F. Pedregosa et al. 2011):

tfidf(*t*, *d*, *D*)=tf(*t*, *d*)⋅idf(*t*, *D*)

The *t**f*(*t*, *d*) is the number of occurrences of the word *t* in the document *d*:

tf(*t*, *d*)=freq(*t*, *d*)

The inverse document frequency is a measure for the fraction of documents that contain some term *t*:

$idf(t, D) = \\log \\frac{N}{\\left|\\left\\{d \\in D : t \\in d\\right\\}\\right|}$

The `Tfidf` retrieval model extends the `TfidfVectorizer` of `sklearn` the `Tfidf`. The documents are stored as an L2-normalised matrix of IDF re-weighted word frequencies. After transforming the query in the same way, we can compute the cosine similarity to all matching documents. As both the query and the documents are L2-normalised, the linear kernel yields the desired cosine similarity. Finally we use the function [`argtopk`](#argtopk) (See Section ) to retrieve the top-*k* documents with respect to the result of the linear kernel.

``` python
class Tfidf(TfidfVectorizer):
    def __init__(self, analyzer='word', use_idf=True):
        TfidfVectorizer.__init__(self,
                                 analyzer=analyzer,
                                 use_idf=use_idf,
                                 norm='l2')
        self._fit_X = None

    def fit(self, X):
        Xt = super().fit_transform(X)
        self._fit_X = Xt
        return self

    def query(self, query, k=None, indices=None):
        if self._fit_X is None:
            raise NotFittedError
        q = super().transform([query])
        if indices is not None:
            fit_X = self._fit_X[indices]
        else:
            fit_X = self._fit_X
        # both fit_X and q are l2-normalised
        D = linear_kernel(q, fit_X)
        ind = argtopk(D[0], k)
        return ind
```

### Word centroid similarity

The word centroid similarity aggregates the word vectors of the documents to their centroids. It is implemented in the `WordCentroidSimilarity` class. The implementation makes extensive use of the [`EmbeddedVectorizer`](#EmbeddedVectorizer) (See Section ) class for aggregation of word vectors. Hence, it is possible to choose between using IDF re-weighted word frequencies or plain word frequencies for aggregation. The parameters passed to the `EmbeddedVectorizer` delegate could be further extended. The current implementation limits the possible configuration to the single parameter `use_idf`. Providing `use_idf=True` results in IDF re-weighted word centroid similarity.

``` python
class WordCentroidSimilarity(BaseEstimator):
    def __init__(self, embedding, analyzer='word', use_idf=True):
        self.vect = EmbeddedVectorizer(embedding,
                                       analyzer=analyzer,
                                       use_idf=use_idf)
        self.centroids = None

    def fit(self, X):
        Xt = self.vect.fit_transform(X)
        Xt = normalize(Xt, copy=False)
        self.centroids = Xt

    def query(self, query, k=None, indices=None):
        centroids = self.centroids
        if centroids is None:
            raise NotFittedError
        if indices is not None:
            centroids = centroids[indices]
        q = self.vect.transform([query])
        q = normalize(q, copy=False)
        # l2 normalised, so linear kernel
        D = linear_kernel(q, centroids)
        ret = argtopk(D[0], k=k)
        return ret
```

Please note, that the aggregated centroids need to be re-normalised to unit L2-norm (even if the word frequencies were normalised in the first place), so that the `linear_kernel` corresponds the cosine similarity. Finally, the results of the linear kernel are ranked by the [`argtopk`](#argtopk) (See Section ) function.

### Word Mover’s distance

In order to compute the Word Mover’s distance (Kusner et al. 2015), we employ the `gensim` implementation `Word2Vec.wmdistance` (Řehuřek and Sojka 2010). Additionally, we make the analysis function `analyze_fn` as well as a completeness parameter `complete` available as arguments to the constructor. The `analyze_fn` argument is expected to be a function that returns a list of tokens, given a string. The parameter `complete` allows specification whether the full Word Mover’s distance to all documents or only to the top *k* documents returned by `WordCentroidSimilarity` should be computed. The `WordCentroidSimilarity` is in turn customisable via the `use_idf` constructor argument. The `complete` parameter is expected to be a float value between 1 (compute the Word Mover’s distance to all documents) and 0 (only compute the Word Mover’s distance to the documents returned by WCS). A fraction in between takes the corresponding amount of additional documents into account. The crucial functionality can be summarised in the following lines of code.

``` python
incomplete = complete < 1.0

# inside fit method
docs = np.array([self.analyze_fn(doc) for doc in raw_docs])
if incomplete:
    self.wcd.fit(raw_docs)

# inside query method
if incomplete:
    wcd_ind = self.wcd.query(query, k=n_req, indices=indices)
    docs = docs[wcd_ind]
q = self.analyze_fn(query)
dists = np.array([self.embedding.wmdistance(q, d) for d in docs])
ind = np.argsort(dists)[:k]
if incomplete:
    # stay relative to the matching indices
    ind = wcd_ind[ind]
```

### Doc2Vec inference

The `Doc2VecInference` class implements a retrieval model based on a paragraph vector model (Le and Mikolov 2014), i.e., `Doc2Vec` from `gensim` (Řehuřek and Sojka 2010). Given an existing `Doc2Vec` model, it is used to infer the document vector of a new document. We store these inferred document vectors and also infer a document vector for the query at query time. Afterwards, we compute the cosine similarity of all matching documents to the query and return the respective indices in descending order. The inference step considers three hyper-parameters:

-   `alpha`  
    The initial learning rate

-   `min_alpha`  
    The final learning rate

-   `steps`  
    The number of training epochs

The inference process itself (provided by [gensim](https://radimrehurek.com/gensim/)) runs `steps` training epochs with fixed weights and a linearly decaying learning rate from `alpha` to `min_alpha`. As the model does operate on a list of tokens, an analyser is required to split the documents into tokens. We compute the representation of the documents (as well as the query) as follows:

``` python
analyzed_docs = (analyzed(doc) for doc in docs)
representation = [doc2vec.infer_vector(doc,
                                       steps=self.epochs,
                                       alpha=self.alpha,
                                       min_alpha=self.min_alpha)
                  for doc in analyzed_docs]
representation = normalize(np.asarray(representation), copy=False)
```

The final normalisation step prepares the computation of the cosine similarity with a linear kernel. At query time, the cosine similarity between the vectors of the query and the documents is computed. The documents are then ranked in descending order.

Query expansion
---------------

All query expansion techniques should implement the following interface:

-   `fit(raw_documents)`  
    Fits the query expansion technique to the raw documents.

-   `tranform(query)`  
    Expands the query string.

As two implementations of this interface, we present the naive centroid based expansion as well as embedding-based query language models by Zamani and Croft (2016) . Please note, that for research it is strongly recommended to reduce the employed word embedding to the tokens that actually appear in the collection. Otherwise, words could be added to the query, that never appear in the collection and thus, do not affect the matching operation.

### Centroid Expansion

Centroid expansion is a query expansion technique that computes the (possibly IDF re-weighted) centroid `v` of the query. It expands the query by the *m* nearest words with respect to the cosine distance to `v`. As a first step, we once again compute the centroid vector of the query using the [`EmbeddedVectorizer`](#EmbeddedVectorizer) after fitting the collection (to build a vocabulary). Then, we employ the `similar_by_vector` method provided by the `gensim.models.Word2Vec` class to obtain the nearest tokens.

``` python
## inside transform method
v = vect.transform([query])[0]
exp_tuples = wv.similar_by_vector(v, topn=self.m)
words, __scores = zip(*exp_tuples)
expanded_query = query + ' ' + ' '.join(words)
```

### Embedding-based query language models

The embedding-based query language models proposed by Zamani and Croft (2016) includes two techniques for query expansion (`EQE1` and `EQE2`), both rely on the underlying probability distribution based on the (weighted) sigmoid of the cosine similarity between terms. While `EQE1` assumes statistical independence of the query terms, `EQE2` assumes query-independent term similarities. The probabilities of the respective query language models are defined with respect to a `delta` function. The `delta` function transforms the cosine distance between two word vectors by a parametrised sigmoid function. We implement this behaviour in a dedicated `delta` function:

``` python
def delta(X, Y=None, n_jobs=-1, a=1, c=0):
    D = pairwise_distances(X, Y, metric="cosine", n_jobs=n_jobs)
    D -= c
    D *= a
    D = expit(D)
    return D
```

We compute the pairwise distances for all word vectors `D = delta(E,E)`. While both variants are implemented in the framework, we only present the computation of the `EQE1` variant:

``` python
prior = np.sum(D, axis=1)
conditional = D[q] / prior
posterior = prior * np.product(conditional, axis=0)
topm = np.argpartition(posterior, -m)[-m:]
expansion = [wv.index2word[i] for i in topm]
```

Utilities
---------

### Sorting only the top *k* documents

When dealing with retrieval models, a common operation is to retrieve the top *k* indices from a list of scores in sorted order. Sorting the complete list and then slicing the top *k* documents would lead to unnecessary computation. Thus, `vec4ir` includes a useful helper function for this specific operation: `argtopk`. The function makes extensive of `numpy.argpartion`, which performs one step of the quicksort algorithm. On invocation by `np.argpartition(A, k)`, the element at position `k` of the returned index set is at the correct (with respect to sorted order) position. All indices *i* = 0, …, *k* − 1 smaller than `k` also have smaller values `A[i] <= A[k]`. Since we are interested in the *k* highest values, we invoke `argpartition` with `-k` and slice the top `k` values by `[-k:]`.

``` python
def argtopk(A, k=None):
    A = np.asarray(A)
    if k is None or k >= A.size:
        # if A contains too few elements or k is None,
        # return all indices in sort order
        return np.argsort(A, axis=axis)[::-1]

    ind = np.argpartition(A, -k, axis=-1)[-k:]
    ind = ind[np.argsort(A[ind], axis=-1)][::-1]
    return ind
```

### Harvesting the gold standard

The `RetriEvalMixin` (as described in Section ) allows several different data types for the gold standard `Y`. We provide a unifying function `harvest` to extract the relevance score for the desired respective query-document pair for all of the supported data types. The immediate benefit is that the data neither has to be transformed nor copied before calling `evaluate` on the implementing class. Additionally, the function can be used to obtain the whole set of values for one specific query (`docid=None`). On the one hand, the desired behaviour is to raise an exception, if a query identifier is not found in the gold standard. On the other hand, missing values for the document identifier for one specific query should not raise an exception but return the `default` value (typically zero). The latter behaviour can also be achieved by using a `defaultdict` or a sparse matrix as inner data structure. However, we want to relief the user from those issues. The `harvest` function takes the following arguments:

-   `source`  
    The gold standard of query-document relevance, a two-level data structure with type either `DataFrame`, `dict` of `dict`s or two-dimensional `ndarray`,

-   `query_id`  
    The query identifier to use as index for the first level of `source`.

-   `doc_id`  
    If `None` return a list of relevance scores for the query given by `query_id`, else look up the `doc_id` in the second level of `source`.

-   `default`  
    The default value in case a document identifier is not found on the second level (typically zero).

In the implementation, we first access the `query_id` of the `source` and raise an exception if `query_id` is not prevalent in `source`. In a second step, we look up `doc_id` on the second level of the data structure and return the value. If the second step fails, the `default` value is returned instead.

### Datasets

#### An interface for datasets

We introduce `IRDataSetBase` as an abstract base class for the data sets. Subclasses should implement the following properties:

-   `docs` : Returns the documents of the corpus.

-   `topics` : Returns the topics, i.e. the queries.

-   `rels` : Returns the relevance judgements for the queries either as a `dict` of `defaultdict`s or as a `pandas.Series` object with a hierarchical index (multi-index).

All other options for the data set (such as the path to its root directory and caching) should be placed in the respective constructor. As an example subclass of the `IRDataSetBase`, we present the Quadflor-like dataset format.

#### Quadflor-like Dataset Format

For convenience we adopt the data set format and specification from [Quadflor](https://github.com/quadflor/Quadflor). Quadflor is a framework for multi-label classification. A Quadflor-like data set consists of

-   `X`  
    The documents, either a path to a directory containing `<id>.txt` documents or a tsv file with columns `id` and `content`.

-   `y`  
    The gold standard: label annotations for the documents. A tsv file with document id in the first column, and subsequent columns resemble label identifier.

-   `thes`  
    A thesaurus consisting of a hierarchy of concepts which are used as query. A minimal format would be `{'<query_id>' :  { 'prefLabel': [ "Here is the query" ] }, '<query_id>:' { 'prefLabel': [ "Yet another query string."] } }`
    The query ids should match the ones in `y`.

In our prior work on Quadflor, the gold standard `y` was used as target labels in a multi-label classification task. We employ the data set in a different manner. We extract the preferred labels of each concept from the thesaurus `thes` and use them as queries. When the label annotations (`y`) for some document contain the specific concept identifier, we consider the document as relevant to the query.

Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae, et al. 2013. “API Design for Machine Learning Software: Experiences from the Scikit-Learn Project.” In *ECML Pkdd Workshop: Languages for Data Mining and Machine Learning*, 108–22.

Kusner, Matt J., Yu Sun, Nicholas I. Kolkin, and Kilian Q. Weinberger. 2015. “From Word Embeddings to Document Distances.” In *Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015*, edited by Francis R. Bach and David M. Blei, 37:957–66. JMLR Workshop and Conference Proceedings. JMLR.org. <http://jmlr.org/proceedings/papers/v37/kusnerb15.html>.

Le, Quoc V., and Tomas Mikolov. 2014. “Distributed Representations of Sentences and Documents.” In *Proceedings of the 31th International Conference on Machine Learning, ICML 2014, Beijing, China, 21-26 June 2014*, 32:1188–96. JMLR Workshop and Conference Proceedings. JMLR.org. <http://jmlr.org/proceedings/papers/v32/le14.html>.

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in Python.” *Journal of Machine Learning Research* 12: 2825–30.

Řehuřek, Radim, and Petr Sojka. 2010. “Software Framework for Topic Modelling with Large Corpora.” In *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, 45–50. Valletta, Malta: ELRA.

Salton, Gerard, and Chris Buckley. 1988. “Term-Weighting Approaches in Automatic Text Retrieval.” *Inf. Process. Manage.* 24 (5): 513–23. doi:[10.1016/0306-4573(88)90021-0](https://doi.org/10.1016/0306-4573(88)90021-0).

Zamani, Hamed, and W. Bruce Croft. 2016. “Embedding-Based Query Language Models.” In *Proceedings of the 2016 ACM on International Conference on the Theory of Information Retrieval, ICTIR 2016, Newark, de, Usa, September 12- 6, 2016*, edited by Ben Carterette, Hui Fang, Mounia Lalmas, and Jian-Yun Nie, 147–56. ACM. doi:[10.1145/2970398.2970405](https://doi.org/10.1145/2970398.2970405).
