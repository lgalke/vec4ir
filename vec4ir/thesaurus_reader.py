import json
import string
from collections import defaultdict
from warnings import warn

import networkx as nx
import rdflib as rdf

from .nltk_normalization import NltkNormalizer

_alphabet = set(string.ascii_lowercase + string.digits + ' ')


class ThesaurusReader:
    """ Read nt or json thesauruses, saved in field 'thesaurus'.

        Persist the thesaurus to json to increase speed significantly.

        Json thesauruses must have the format:
        {'id': {
        'prefLabel': [],
        'broader': [],
        'narrower': [],
        'altLabel': []
        }, ...
        }
        With deprecated labels collected in the altLabel field.

        Parameters
        ----------
        resource_path: str
            Path to thesaurus nt or json file.
    """

    def __init__(self, resource_path, normalize=True):
        self._nx_root = None
        self.resource_path = resource_path
        self._thesaurus = None
        self._nx_graph = None
        self._vocabulary = None
        self._nodename_index = None
        self._index_nodename = None
        self.normalizer = NltkNormalizer() if normalize else None
        self._query_prefix = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ' \
                             'PREFIX skos: <http://www.w3.org/2004/02/skos/core#> ' \
                             'PREFIX zbwext: <zbw.eu/namespaces/zbw-extensions/> ' \
                             'PREFIX dcterms: <http://purl.org/dc/terms/> ' \
                             'PREFIX owl: <http://www.w3.org/2002/07/owl#> ' \
                             'PREFIX dcterms: <http://purl.org/dc/terms/> '

    @property
    def thesaurus(self):
        """
        Calling the property for the first time parses the nt graph or json
        file.

        Returns
        -------
        dict[str,dict[str,list[str]]]
            Thesaurus as dict with format:
                {'id': {
                'prefLabel': [],
                'broader': [],
                'narrower': [],
                'altLabel': []
                }, ...
                }

        """
        if self._thesaurus is None:
            self._thesaurus = {}
            self._read_thesaurus(self.resource_path)
        return self._thesaurus

    @property
    def nx_graph(self):
        """
        Calling the property for the first time parses the nt graph or json
        file and creates the nx_graph.

        Returns
        -------
        nx_graph: networkx.DiGraph
            The thesaurus as networkx graph.

        """
        if self._nx_graph is None:
            self._create_nx_graph()
        return self._nx_graph

    @property
    def nx_root(self):
        """
        Calling the property for the first time
        parses the nt graph or json file and creates the nx_graph.

        Returns
        -------
        nx_root: str
            The id of the root node.
        """
        if self._nx_root is None:
            self._find_root()
        return self._nx_root

    @property
    def vocabulary(self):
        """
        Calling the property for the first time
        parses the nt graph or json file and creates the mappings.

        Returns
        -------
        vocabulary: dict[str,int]
            The vocabulary, mapping preflabels and altlabels to a unique index.
        """
        if self._vocabulary is None:
            self._create_vocabulary_and_mappings()
        return self._vocabulary

    @property
    def index_nodename(self):
        """
        Calling the property for the first time parses
        the nt graph or json file and creates the mappings.

        Returns
        -------
        index_nodename: dict[int,str]
            Maps the unique index to the preflabel.
        """
        if self._index_nodename is None:
            self._create_vocabulary_and_mappings()
        return self._index_nodename

    @property
    def nodename_index(self):
        """
        Calling the property for the first time
        parses the nt graph or json file and creates the mappings.

        Returns
        -------
        index_nodename: dict[str,int]
            Maps the preflabel to the index.
        """
        if self._nodename_index is None:
            self._create_vocabulary_and_mappings()
        return self._nodename_index

    def persist(self, persistence_path):
        """Persist to json file.

        Parameters
        ----------
        persistence_path: str
            The path to write the thesaurus to.
        """
        if not persistence_path.endswith('.json'):
            persistence_path += '.json'
        with open(persistence_path, 'w') as f:
            json.dump(self.thesaurus, f)

    def normalize_thesaurus(self):
        """Normalizes the thesaurus entries using nltk lemmatizer, lower
        casing, stop word removal and reduces the characters to
        set(string.ascii_lowercase + string.digits + ' ') """
        for entry in self._thesaurus.values():
            entry['prefLabel'] = self._normalize_labels(entry['prefLabel'])
            if 'altLabel' in entry:
                entry['altLabel'] = self._normalize_labels(entry['altLabel'])

    def _create_vocabulary_and_mappings(self):
        self._vocabulary = {}
        self._nodename_index = {}
        self._index_nodename = {}
        for index, item in enumerate(self.thesaurus.items()):
            labels = item[1]
            pref_labels = labels['prefLabel']
            for label in pref_labels:
                self._vocabulary[label] = index
            nodename = item[0]
            self._nodename_index[nodename] = index
            self._index_nodename[index] = nodename
            if 'altLabel' in labels:
                for alt in labels['altLabel']:
                    self._vocabulary[alt] = index

    def _normalize_labels(self, labels):
        return [self.normalizer.normalize(l) for l in labels]

    def _read_thesaurus(self, resource_path):
        if resource_path.rsplit('.')[-1] == 'nt':
            self._read_nt(resource_path)
        else:
            self._read_json(resource_path)

    def _read_json(self, resource_path):
        with open(resource_path) as f:
            self._thesaurus = json.load(f)
        if self.normalizer:
            self.normalize_thesaurus()

    def _read_nt(self, resource_path):
        self._graph = rdf.Graph()
        self._graph.parse(resource_path, format="nt")
        self._build_thesaurus_dict()
        if self.normalizer:
            self.normalize_thesaurus()

    def _build_thesaurus_dict(self):
        top_concepts = self._get_top_concepts()
        self._thesaurus = defaultdict(
            lambda: {'prefLabel': [],
                     'broader': [],
                     'narrower': [],
                     'altLabel': []})
        if top_concepts:
            # Use R00T for virtual root so as not to be found as concept.
            self.thesaurus['root']['prefLabel'] = ['root']
            self.thesaurus['root']['narrower'] = top_concepts
            for top_concept in top_concepts:
                self.thesaurus[top_concept]['broader'].append('root')
        nodes = self._get_nodes()
        keys = ['prefLabel', 'altLabel', 'narrower', 'broader']
        for c in nodes:
            for key in keys:
                label = c.split('/')[-1]
                entries = self._get_relation(self._normalize_uri(c), key)
                if key in {'broader', 'narrower'}:
                    entries = [entry.split('/')[-1] for entry in entries]
                # In the case of self loops at broader relation:
                # Add relation to virtual root node.
                if label in entries and key == 'broader':
                    self.thesaurus[label]['broader'].append('root')
                    self.thesaurus['root']['narrower'].append(label)
                    entries.remove(label)
                self.thesaurus[label][key].extend(entries)

    def _query_rdf(self, query):
        q = self._query_prefix + query
        return list(self._graph.query(q))

    def _get_top_concepts(self):
        results = self._query_rdf("SELECT ?o WHERE {?o skos:topConceptOf ?p.}")
        return [r[0].split('/')[-1] for r in results]

    def _get_nodes(self):
        results = self._query_rdf(
            "SELECT DISTINCT ?s WHERE {?s skos:prefLabel ?a. }")
        nodes = [r[0] for r in results]
        return nodes

    def _get_relation(self, c, relation):
        f = "FILTER (lang(?o) = 'en')." if 'label' in relation.lower() else ''
        q = "SELECT ?o WHERE {""" + c + " skos:" + relation + " ?o. " + f + "}"
        results = self._query_rdf(q)
        return [str(result[0]) for result in results]

    def _get_label(self, uri):
        quer = "SELECT ?o WHERE {" + uri
        + " rdfs:label ?o. FILTER (lang(?o) = 'en').}"
        return str(self._query_rdf(quer)[0][0])

    @staticmethod
    def _normalize_uri(uri):
        if uri[0] == '<' and uri[-1] == '>':
            return str(uri)
        else:
            return '<' + str(uri) + '>'

    def _create_nx_graph(self):
        self._nx_graph = nx.DiGraph()
        for thesaurus_entry in self.thesaurus.items():
            node = self.nodename_index[thesaurus_entry[0]]
            self._nx_graph.add_node(node)
            for child_entry in thesaurus_entry[1]['narrower']:
                child = self.nodename_index[child_entry]
                self._nx_graph.add_edge(node, child, weight=0)

            for parent_entry in thesaurus_entry[1]['broader']:
                parent = self.nodename_index[parent_entry]
                self._nx_graph.add_edge(parent, node, weight=0)
        # TODO add weights
        for n in self._nx_graph.nodes():
            edges = self._nx_graph.edges(n, data=True)
            n_edges = len(edges)
            for _, _, d in edges:
                d['weight'] = 1 / n_edges

    def _find_root(self):
        # TODO error on two nodes
        roots = self._get_roots()
        if len(roots) > 1:
            warn('More than one root, namely: '
                 + str([self.index_nodename[r] for r in roots]))
        self._nx_root = roots[0]

    def _get_roots(self):
        return [n for n in self.nx_graph.nodes()
                if not self.nx_graph.predecessors(n)]

    def fix_multiple_roots(self, root):
        roots = self._get_roots()
        for r in roots:
            if self.index_nodename[r] == root:
                continue
            r_label = self.index_nodename[r]

            self.thesaurus[r_label]['broader'].append(root)
            self.thesaurus[root]['narrower'].append(r_label)
        self._create_nx_graph()
        assert len(self._get_roots()) == 1,\
            "Still " + str(self._get_roots()) + " roots."

    def _get_nt_root(self):
        print('Searching for root')
        q = 'select distinct ?s where {?s skos:hasTopConcept ?o .}'
        results = self._query_rdf(q)
        if not results:
            q = 'select ?s where  {?s skos:prefLabel ?l. \
                filter not exists {?s skos:broader ?o .} ' \
                'filter not exists {?s owl:deprecated true .}}'
            results = self._query_rdf(q)
        print('Roots found: ')
        print(results)
        return results[0][0]
