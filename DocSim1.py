import numpy as np
import nltk  
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math

CONST_PHI = 0.2
CONST_BETA = 0.45
CONST_ALPHA = 0.2
CONST_PHI = 0.2
#CONST_DELTA = 0.875
CONST_DELTA = 0.82
CONST_ETA = 0.4
total_words = 0

class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        #words = [w for w in doc.split(" ") if w not in self.stopwords]
        words = [w for w in doc.split(" ")]
        word_vecs = []
        #print "the sentence is =",doc
        #print "the words list = ",words
        for word in words:
            #print "current word = ",word
            try:
                vec = self.w2v_model[word]
                #print "for word =",word,", corresponding vec = ",vec
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                #print "for word =",word,", dic is empty"
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        #print "word vecs = ",word_vecs
        vector = np.mean(word_vecs, axis=0)
        return vector


    def proper_synset(self, word_one , word_two):
	    pair = (None,None)
	    maximum_similarity = -1
	    synsets_one = wn.synsets(word_one)
	    synsets_two = wn.synsets(word_two)
	    if(len(synsets_one)!=0 and len(synsets_two)!=0):
		for synset_one in synsets_one:
		    for synset_two in synsets_two:
		        similarity = wn.path_similarity(synset_one,synset_two)
		        if(similarity == None):
		            sim = -2
		        elif(similarity > maximum_similarity):
		            maximum_similarity = similarity
		            pair = synset_one,synset_two
	    else:
		pair = (None , None)
	    return pair
    
    def length_between_words(self, synset_one , synset_two):
	    length = 100000000
	    if synset_one is None or synset_two is None:
		return 0
	    elif(synset_one == synset_two):
		length = 0
	    else:
		words_synet1 = set([word.name() for word in synset_one.lemmas()])
		words_synet2 = set([word.name() for word in synset_two.lemmas()])
		if(len(words_synet1) + len(words_synet2) > len(words_synet1.union(words_synet2))):
		    length = 0
		else:
		    #finding the actual distance
		    length = synset_one.shortest_path_distance(synset_two)
		    if(length is None):
		        return 0
	    return math.exp( -1 * CONST_ALPHA * length)
    
    def depth_common_subsumer(self,synset_one,synset_two):
	    height = 100000000
	    if synset_one is None or synset_two is None:
		return 0
	    elif synset_one == synset_two:
		height = max([hypernym[1] for hypernym in synset_one.hypernym_distances()])
	    else:
		#get the hypernym set of both the synset.
		hypernym_one = {hypernym_word[0]:hypernym_word[1] for hypernym_word in synset_one.hypernym_distances()}
		hypernym_two = {hypernym_word[0]:hypernym_word[1] for hypernym_word in synset_two.hypernym_distances()}
		common_subsumer = set(hypernym_one.keys()).intersection(set(hypernym_two.keys()))
		if(len(common_subsumer) == 0):
		    height = 0
		else:
		    height = 0
		    for cs in common_subsumer:
		        val = [hypernym_word[1] for hypernym_word in cs.hypernym_distances()]
		        val = max(val)
		        if val > height : height = val

	    return (math.exp(CONST_BETA * height) - math.exp(-CONST_BETA * height))/(math.exp(CONST_BETA * height) + math.exp(-CONST_BETA * height))
	    
    
    def word_similarity(self,word1,word2):
	    synset_wordone,synset_wordtwo = self.proper_synset(word1,word2)
	    return self.length_between_words(synset_wordone,synset_wordtwo) * self.depth_common_subsumer(synset_wordone,synset_wordtwo)
    
    def most_similar_word(self,word,sentence):
	    most_similarity = 0
	    most_similar_word = ''
	    for w in sentence:
		#compute the word similarity using the already defined function
		sim  =  self.word_similarity(w,word)
		if sim > most_similarity:
		    most_similarity = sim
		    most_similar_word = w
	    if most_similarity <= CONST_PHI:
		most_similarity = 0
	    return most_similar_word,most_similarity 
    
    def word_order_similarity(self,sentence_one , sentence_two):
	    #print("Sentence one :",sentence_one)
	    token_one  = word_tokenize(sentence_one)
	    #print("Sentence two : ",sentence_two)
	    token_two = word_tokenize(sentence_two)
	    joint_word_set = list(set(token_one).union(set(token_two)))
	    r1 = np.zeros(len(joint_word_set))
	    r2 = np.zeros(len(joint_word_set))
	    #filling for the first one
	    en_joint_one = {x[1]:x[0] for x in enumerate(token_one)}
	    en_joint_two = {x[1]:x[0] for x in enumerate(token_two)}
	    set_token_one = set(token_one)
	    set_token_two = set(token_two)
	    i = 0
	    #print(en_joint)
	    for word in joint_word_set:
		if word in set_token_one:
		    r1[i] = en_joint_one[word]#so wrong.
		else:
		    #get best word and check if its greater then a preset threshold
		    sim_word , sim = self.most_similar_word(word , list(set_token_one))
		    if sim > CONST_ETA : 
		        r1[i] = en_joint_one[sim_word]
		    else:
		        r1[i] = 0
		i+=1
	    j = 0
	    for word in joint_word_set:
		if word in set_token_two:
		    r2[j] = en_joint_two[word]
		else:
		    #get best word and check if its greater then a preset threshold
		    sim_word , sim = self.most_similar_word(word , list(set_token_two))
		    if sim > CONST_ETA : 
		        r2[j] = en_joint_two[sim_word]
		    else:
		        r2[j] = 0
		j+=1
	    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1+r2))
    
    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        results = []
        for doc in target_docs:
            target_vec = self.vectorize(doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            order_score = self.word_order_similarity(source_doc,doc)
            print "cosine score = ",sim_score
	    print "order score = ",order_score
            sim_score = (CONST_DELTA * sim_score) + ((1.0 - CONST_DELTA) * order_score)
            if sim_score > threshold:
                results.append({
                    'score' : sim_score,
                    'doc' : doc
                })
            # Sort results by score in desc order
            results.sort(key=lambda k : k['score'] , reverse=True)

        return results

        
