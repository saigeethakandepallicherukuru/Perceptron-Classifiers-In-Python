#To learn a perceptron model from labeled data using standard perceptron algorithm 
import sys
import os,random
from random import shuffle
import json

class AveragedPerceptron:
	input_data_dir=sys.argv[1]
	def generatePerceptronModel(self):
		count=0
		vocab_dict, avg_weights_dict={},{}
		alpha,bias,y_alpha,counter, avg_bias=0, 0, 0, 1, 0
		output_file="per_model.txt"
		fp_w=open(output_file,'w')
		total_files=[]
		for root, subdirs, files in os.walk(AveragedPerceptron.input_data_dir):
			if('/ham' in root or '/spam' in root):
				for file in files:
					if(not file.startswith('.')):
						#To add the unique list of words to vocabulary
						fp=open(root+"/"+str(file),"r",encoding="latin1")
						file_content=""
						file_dict={}
						file_name=root+"/"+str(file)
						for line in fp.readlines():
							file_content+=line
							words=line.split()
							for word in words:
								if(not word in vocab_dict):
									vocab_dict[word]=0
									avg_weights_dict[word]=0
						if(not file_name in file_dict):			
							file_dict[file_name]=file_content
						total_files.append(file_dict.copy())
						fp.close()
		#number of iterations on training data
		while(count<30):
			shuffle(total_files)
			#To calculate activation and adjust weights, bias if the activation value is less than or equal to zero
			for file in total_files:
				for key, value in file.items():
					alpha, y_alpha=0, 0
					words=value.split()
					for word in words:
						alpha+=vocab_dict[word]
					alpha+=bias

					if('/spam' in key):
						y_alpha=1*alpha
					else:
						y_alpha=(-1)*alpha
				
					if(y_alpha<=0):
						if('/spam' in key):
							words=value.split()
							for word in words:
								if(word in vocab_dict):
									vocab_dict[word]+=1
									avg_weights_dict[word]+=(1*counter)
							bias+=1	
							avg_bias+=(1*counter)
						if('/ham' in key):
							words=value.split()
							for word in words:
								if(word in vocab_dict):
									vocab_dict[word]-=1
									avg_weights_dict[word]+=(-1*counter)
							bias-=1 
							avg_bias+=(-1*counter)
					counter+=1
			count=count+1

		for word in vocab_dict:
			avg_weights_dict[word]=round((vocab_dict[word]-((1/counter)*avg_weights_dict[word])),2)
		avg_bias=round((bias-((1/counter)*avg_bias)),2)

		json.dump(avg_weights_dict,fp_w)
		fp_w.write("; ")
		json.dump(avg_bias,fp_w)
		fp_w.close()

def main():		
	perceptron=AveragedPerceptron()
	perceptron.generatePerceptronModel()

if __name__ == '__main__':
	main()