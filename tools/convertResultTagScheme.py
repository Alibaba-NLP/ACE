# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-11-27 16:53:36
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-09 21:39:10


"""
    convert NER/Chunking tag schemes, i.e. BIO->BIOES, BIOES->BIO, IOB->BIO, IOB->BIOES
"""
from __future__ import print_function

import sys

def BIOES2BIO(input_file, output_file):
    print("Convert BIOES -> BIO for file:", input_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    predicts = []
    golds = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in golds[idx]:
                    lines = words[idx] + " " + golds[idx] + " "
                else:
                    label_type = golds[idx].split('-')[-1]
                    if "E-" in golds[idx]:
                        lines = words[idx] + " I-" + label_type + " "
                    elif "S-" in golds[idx]:
                        lines = words[idx] + " B-" + label_type + " "
                    else:
                        lines = words[idx] + " " + golds[idx] + " "
                if "-" not in predicts[idx]:
                    lines += predicts[idx]
                else:
                    label_type = predicts[idx].split('-')[-1]
                    if "E-" in predicts[idx]:
                        lines += "I-" + label_type
                    elif "S-" in predicts[idx]:
                        lines += "B-" + label_type
                    else:
                        lines += predicts[idx]
                fout.write(lines + '\n')
            fout.write('\n')
            words = []
            predicts = []
            golds = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            golds.append(pair[1])
            predicts.append(pair[2].upper())

    fout.close()
    print("BIO file generated:", output_file)


def choose_label(input_file, output_file):
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    with open(output_file,'w') as fout:
        for line in fins:
            if len(line) < 3:
                fout.write(line)
            else:
                pairs = line.strip('\n').split(' ')
                fout.write(pairs[0]+" "+ pairs[-1]+"\n")


if __name__ == '__main__':
    '''Convert NER tag schemes among IOB/BIO/BIOES.
        For example: if you want to convert the IOB tag scheme to BIO, then you run as following:
            python convertResultTagScheme.py input_iob_file output_bio_file
        Input data format is the standard CoNLL 2003 data format.
    '''

    BIOES2BIO(sys.argv[1],sys.argv[2])
