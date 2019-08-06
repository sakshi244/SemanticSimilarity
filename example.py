from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim
#from gingerit.gingerit import GingerIt

googlenews_model_path = './data/GoogleNews-vectors-negative300.bin'
stopwords_path = "./data/stopwords_en.txt"

model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

while(1) :
	source_doc = raw_input("Enter first sentence : ")


	sentence_two = raw_input("Enter second sentence : ")
			
	target_docs = [sentence_two]

	sim_scores = ds.calculate_similarity(source_doc, target_docs)

	print "SIMILARITY SCORE = ",sim_scores[0]
	print ""
	check = raw_input("Do you want to continue(y/n) : ")
	if check == 'n':
		break

# Prints:
##   [ {'score': 0.99999994, 'doc': 'delete a invoice'}, 
##   {'score': 0.79869318, 'doc': 'how do i remove an invoice'}, 
##   {'score': 0.71488398, 'doc': 'purge an invoice'} ]
