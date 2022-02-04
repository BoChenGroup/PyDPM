import numpy as np
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel

"""
Examples
---------
One way of using this feature is through providing a trained topic model. A dictionary has to be explicitly provided
if the model does not contain a dictionary already

.. sourcecode:: pycon
	#
	# >>> from gensim.test.utils import common_corpus, common_dictionary
	# >>> from gensim.models.ldamodel import LdaModel
	# >>> from gensim.models.coherencemodel import CoherenceModel
	# >>>
	# >>> model = LdaModel(common_corpus, 5, common_dictionary)
	# >>>
	# >>> cm = CoherenceModel(model=model, corpus=common_corpus, coherence='u_mass')
	# >>> coherence = cm.get_coherence()  # get coherence value

Another way of using this feature is through providing tokenized topics such as:

.. sourcecode:: pycon

	# >>> from gensim.test.utils import common_corpus, common_dictionary
	# >>> from gensim.models.coherencemodel import CoherenceModel
	# >>> topics = [
	# ...     ['human', 'computer', 'system', 'interface'],
	# ...     ['graph', 'minors', 'trees', 'eps']
	# ... ]
	# >>>
	# >>> cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
	# >>> coherence = cm.get_coherence()  # get coherence value

（Please visit https://radimrehurek.com/gensim/models/coherencemodel.html for more usage.）

"""

class Topic_Coherence(object):
	def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None,
					 window_size=None, keyed_vectors=None, coherence='c_v', topn=20, processes=-1):
		'''
		Inputs:
			model : :class:`~gensim.models.basemodel.BaseTopicModel`, optional
				Pre-trained topic model, should be provided if topics is not provided.
				Currently supports :class:`~gensim.models.ldamodel.LdaModel`,
				:class:`~gensim.models.ldamulticore.LdaMulticore`, :class:`~gensim.models.wrappers.ldamallet.LdaMallet` and
				:class:`~gensim.models.wrappers.ldavowpalwabbit.LdaVowpalWabbit`.
				Use `topics` parameter to plug in an as yet unsupported model.
			topics : list of list of str, optional
				List of tokenized topics, if this is preferred over model - dictionary should be provided.
			texts : list of list of str, optional
				Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)
				probability estimator .
			corpus : iterable of list of (int, number), optional
				Corpus in BoW format.
			dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
				Gensim dictionary mapping of id word to create corpus.
				If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.
			window_size : int, optional
				Is the size of the window to be used for coherence measures using boolean sliding window as their
				probability estimator. For 'u_mass' this doesn't matter.
				If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.
			coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
				Coherence measure to be used.
				Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.
				For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus
				using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)
			topn : int, optional
				Integer corresponding to the number of top words to be extracted from each topic.
			processes : int, optional
				Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as
				num_cpus - 1.

		Outputs:
			topic_coherence : [float], The topic coherence with model

		'''
		self.model = model
		self.topics = topics
		self.texts = texts
		self.corpus = corpus
		self.dictionary = dictionary
		self.window_size = window_size
		self.keyed_vectors = keyed_vectors
		self.coherence = coherence
		self.topn = topn
		self.processes = processes

		self._get()
		print(f'The topic coherence score is: {self._topic_coherence:.4f}')

	def _get(self):

		cm = CoherenceModel(model=self.model, topics=self.topics, texts=self.texts, corpus=self.corpus,
							dictionary=self.dictionary, window_size=self.window_size, keyed_vectors=self.keyed_vectors,
							coherence=self.coherence, topn=self.topn, processes=self.processes)

		self._topic_coherence = cm.get_coherence()

