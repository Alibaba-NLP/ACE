import numpy as np
import pdb
import re

# from argparse import ArgumentParser
# argparser = ArgumentParser('train sentence generator')
# argparser.add_argument('--src', default='data/SemEval15/ptb_3.3.0/test.gold.conllu')
# argparser.add_argument('--tgt', default='results/test.conllu')
# argparser.add_argument('--end', default='\n')
# argparser.add_argument('--no_punct', action='store_true')
# args = argparser.parse_args()
def is_uni_punctuation(word):
    match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
    return match is not None


def is_punctuation(word, pos, punct_set=None):
    if punct_set is None:
        return is_uni_punctuation(word)
    else:
        return pos in punct_set

def evaluate(filename, target_filename, punct=['``', "''", ':', ',', '.', 'PU', 'PUNCT']):
		""""""
		
		correct = {'UAS': [], 'LAS': []}
		with open(target_filename) as f_tar:
			with open(filename) as f:
				lem=0
				uem=0
				total_tree=0
				label_match=0
				unlabel_match=0
				tree_lines=0
				for line in f:
					line_tar = f_tar.readline()
					line = line.strip().split('\t')
					line_tar = line_tar.strip().split('\t')
					# pdb.set_trace()
					# if len(line)==10:
					# 	if line[4] in punct:
					# 		pdb.set_trace()
					# pdb.set_trace()
					# if line[1] in punct:
					# 	pdb.set_trace()
					if len(line_tar) == 10 and not is_punctuation(line[1],line_tar[3],punct) and not is_punctuation(line[1],line_tar[4],punct) and not line[1] in punct:
						correct['UAS'].append(0)
						correct['LAS'].append(0)
						tree_lines+=1
						assert line_tar[1]==line[1], "two files are not equal!"
						if line[6] == line_tar[6]:
							correct['UAS'][-1] = 1
							unlabel_match+=1
							if line[7] == line_tar[7]:
								correct['LAS'][-1] = 1
								label_match+=1
					elif tree_lines>0:
						total_tree+=1
						if label_match==tree_lines:
							lem+=1
						if unlabel_match==tree_lines:
							uem+=1
						label_match=0
						unlabel_match=0
						tree_lines=0

		#correct = {k:np.array(v) for k, v in correct.iteritems()}
		# pdb.set_trace()
		# print(len(correct['UAS']))
		correct['UAS']=np.mean(correct['UAS']) * 100
		correct['LAS']=np.mean(correct['LAS']) * 100
		correct['UEM']=uem/(total_tree+1e-12) * 100
		correct['LEM']=lem/(total_tree+1e-12) * 100
		return correct
# if args.no_punct:
# 	correct=evaluate(args.src, args.tgt, punct=[])
# else:
# 	correct=evaluate(args.src, args.tgt)
# print(correct['UAS'],correct['LAS'],correct['UEM'],correct['LEM'],end=args.end)