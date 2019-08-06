from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim
from gingerit.gingerit import GingerIt

googlenews_model_path = './data/GoogleNews-vectors-negative300.bin'
stopwords_path = "./data/stopwords_en.txt"

model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

while(1) :
	source_doc = raw_input("Enter first sentence : ")

	parser = GingerIt()
	corrected_text = parser.parse(source_doc)['result']
	while source_doc != corrected_text:
		if len(corrected_text.strip())!=len(source_doc.strip())-1:
			#print "Did you mean : \"",corrected_text,"\""
			print "Did you mean : \"",corrected_text,"\""
			print ("Press 0 to continue with suggested version ")
			print ("Press 1 to re-enter the sentence ")
			print ("Press 2 to continue with your version ")
			cond = input(" --> ")
			if cond == 1:
				source_doc = raw_input("Enter first sentence : ")
				corrected_text = parser.parse(source_doc)['result']
			elif cond == 0:
				source_doc = corrected_text.strip()
				break
			else :
				break
		else:
			break

	sentence_two = raw_input("Enter second sentence : ")
	parser = GingerIt()
	corrected_text = parser.parse(sentence_two)['result']
	while sentence_two != corrected_text:
		if len(corrected_text.strip())!=len(sentence_two.strip())-1:
			print "Did you mean : \"",corrected_text,"\""
			print ("Press 0 to continue with suggested version ")
			print ("Press 1 to re-enter the sentence ")
			print ("Press 2 to continue with your version ")
			cond = input(" --> ")
			if cond == 1:
				sentence_two = raw_input("Enter second sentence : ")
				corrected_text = parser.parse(sentence_two)['result']
			elif cond == 0:
				sentence_two = corrected_text.strip()
				break
			else :
				break
		else:
			break
			
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
