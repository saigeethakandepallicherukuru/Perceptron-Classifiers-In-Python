#For binary classification (ham/spam) files based on perceptron model data
import sys
import os
import json
import ast

class PerceptronClassifier:
	input_data_dir=sys.argv[1]
	def classify(self):
		alpha=0
		output_file=sys.argv[2]
		fp_w=open(output_file,'w')
		num_spam_classified, num_ham_classified = 0, 0
		num_spam_correct, num_ham_correct = 0, 0
		total_spam_files, total_ham_files = 0, 0
		with open("per_model.txt","r",encoding="latin1") as fp:
			contents=fp.read()
			vocab_dict, bias=contents.split("; ")
			vocab_dict=ast.literal_eval(vocab_dict)
		for root, subdirs, files in os.walk(PerceptronClassifier.input_data_dir):
			for file in files:
				alpha=0
				if(not file.startswith('.')):
					# count total number of ham and spam files
					if('spam' in file):
						total_spam_files += 1
					if('ham' in file):
						total_ham_files += 1
					fp=open(root+str('/'+file),"r",encoding="latin1")
					for line in fp.readlines():
						words=line.split()
						for word in words:
							word=word.rstrip()
							if(len(word)!=0):
								if(word in vocab_dict):
									alpha+=vocab_dict[word]
					if('.' in bias):
						alpha+=float(bias)
					else:
						alpha+=int(bias)
					if(alpha>0):
						fp_w.write("spam "+root+str('/'+file))
						fp_w.write("\n")
						num_spam_classified += 1
						if('spam' in file):
							num_spam_correct += 1
					if(alpha<=0):
						fp_w.write("ham "+root+str('/'+file))
						fp_w.write("\n")
						num_ham_classified += 1
						if('ham' in file):
							num_ham_correct += 1
		fp_w.close()
		#calculate precision, recall and f1 score
		if(num_spam_classified != 0 and total_spam_files != 0):
			spam_precision = (num_spam_correct) / (num_spam_classified)
			spam_recall = (num_spam_correct) / (total_spam_files)
			spam_f1_score = (2*spam_precision*spam_recall) / (spam_precision+spam_recall)
			print("spam precision: {0:.2f}".format(spam_precision))
			print("spam recall: {0:.2f}".format(spam_recall))
			print("spam F1 score: {0:.2f}".format(spam_f1_score))
		if(num_ham_classified != 0 and total_ham_files != 0):	
			ham_precision = (num_ham_correct) / num_ham_classified
			ham_recall = (num_ham_correct) / (total_ham_files)
			ham_f1_score = (2*ham_precision*ham_recall) / (ham_precision+ham_recall)
			print("ham precision: {0:.2f}".format(ham_precision))
			print("ham recall: {0:.2f}".format(ham_recall))
			print("ham F1 score: {0:.2f}".format(ham_f1_score))
def main():
	classifier=PerceptronClassifier()
	classifier.classify()

if __name__ == '__main__':
	main()