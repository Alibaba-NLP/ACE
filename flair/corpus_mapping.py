corpus_map={'ner':{'eng':'CONLL_03_ENGLISH','en':'CONLL_03','nl':'CONLL_03_DUTCH','es':'CONLL_03_SPANISH','de':'CONLL_03_GERMAN'},
			'newner':{'en':'CONLL_03_NEW','nl':'CONLL_03_DUTCH_NEW','es':'CONLL_03_SPANISH_NEW','de':'CONLL_03_GERMAN_NEW'},
			'docner':{'en':'CONLL_03_ENGLISH_DOC'},
			'ner_dp':{'en':'CONLL_03_DP','nl':'CONLL_03_DUTCH_DP','es':'CONLL_03_SPANISH_DP','de':'CONLL_03_GERMAN_DP','deu':'CONLL_06_GERMAN_DP'},
			'casedner':{'en':'CONLL_03_ENGLISH_CASED'},
			'caseddocner':{'en':'CONLL_03_ENGLISH_DOC_CASED'},
			'06ner':{'de':'CONLL_06_GERMAN'},
			'upos':{'twitter':'TWITTER','ark':'ARK','tweebank':'TWEEBANK_NEW','ritter':'RITTER_NEW','ptb':'WSJ_POS','en':'UD_ENGLISH','nl':'UD_DUTCH','es':'UD_SPANISH','de':'UD_GERMAN','fr':'UD_FRENCH','it':'UD_ITALIAN','pt':'UD_PORTUGUESE','zh':'UD_CHINESE','ja':'UD_JAPANESE','ta':'UD_TAMIL','eu':'UD_BASQUE','fi':'UD_FINNISH','he':'UD_HEBREW','ar':'UD_ARABIC','id':'UD_INDONESIAN','cs':'UD_CZECH','fa':'UD_PERSIAN'},
			'chunk':{'en':'CONLL_03','de':'CONLL_03_GERMAN','conll00':'CONLL_2000'},
			'panx':{'en':'PANX-EN','ta':'PANX-TA','fi':'PANX-FI','eu':'PANX-EU','he':'PANX-HE','ar':'PANX-AR','id':'PANX-ID','cs':'PANX-CS','it':'PANX-IT','fa':'PANX-FA','ja':'PANX-JA','sl':'PANX-SL','fr':'PANX-FR','pt':'PANX-PT','de':'PANX-DE','es':'PANX-ES','nl':'PANX-NL'},
			'mixedner':{'en':'CONLL_03','nl':'CONLL_03_DUTCH','es':'CONLL_03_SPANISH','de':'CONLL_03_GERMAN','eu':'MIXED_NER-EU','fa':'MIXED_NER-FA','fi':'MIXED_NER-FI','fr':'MIXED_NER-FR','he':'MIXED_NER-HE','hi':'MIXED_NER-HI','hr':'MIXED_NER-HR','id':'MIXED_NER-ID','ja':'MIXED_NER-JA','no':'MIXED_NER-NO','pl':'MIXED_NER-PL','pt':'MIXED_NER-PT','sl':'MIXED_NER-SL','sv':'MIXED_NER-SV','ta':'MIXED_NER-TA'},
			'lowmixedner':{'en':'MIXED_NER-EN','nl':'MIXED_NER-NL','es':'MIXED_NER-ES','de':'MIXED_NER-DE','eu':'MIXED_NER-EU','fa':'MIXED_NER-FA','fi':'MIXED_NER-FI','fr':'MIXED_NER-FR','he':'MIXED_NER-HE','hi':'MIXED_NER-HI','hr':'MIXED_NER-HR','id':'MIXED_NER-ID','ja':'MIXED_NER-JA','no':'MIXED_NER-NO','pl':'MIXED_NER-PL','pt':'MIXED_NER-PT','sl':'MIXED_NER-SL','sv':'MIXED_NER-SV','ta':'MIXED_NER-TA'},
			'richmixedner':{'en':'MIXED_NER-EN','nl':'MIXED_NER-NL','es':'MIXED_NER-ES','de':'MIXED_NER-DE'},
			'indoeuro1':{'cs':'MIXED_NER-CS','fa':'MIXED_NER-FA','fr':'MIXED_NER-FR','hi':'MIXED_NER-HI','hr':'MIXED_NER-HR'},
			'indoeuro2':{'no':'MIXED_NER-NO','pl':'MIXED_NER-PL','pt':'MIXED_NER-PT','sl':'MIXED_NER-SL','sv':'MIXED_NER-SV'},
			'difffam':{'ce':'MIXED_NER-CE','vi':'MIXED_NER-VI','zh':'MIXED_NER-ZH','ka':'MIXED_NER-KA','eu':'MIXED_NER-EU'},
			'turkic':{'az':'MIXED_NER-AZ','kk':'MIXED_NER-KK','tr':'MIXED_NER-TR','ky':'MIXED_NER-KY','tt':'MIXED_NER-TT'},
			'austronesian':{'ms':'MIXED_NER-MS','su':'MIXED_NER-SU','tl':'MIXED_NER-TL','id':'MIXED_NER-ID','mg':'MIXED_NER-MG'},
			'lowner':{'eu':'MIXED_NER-EU','fa':'MIXED_NER-FA','fi':'MIXED_NER-FI','fr':'MIXED_NER-FR','he':'MIXED_NER-HE','hi':'MIXED_NER-HI','hr':'MIXED_NER-HR','id':'MIXED_NER-ID','ja':'MIXED_NER-JA','no':'MIXED_NER-NO','pl':'MIXED_NER-PL','pt':'MIXED_NER-PT','sl':'MIXED_NER-SL','sv':'MIXED_NER-SV','ta':'MIXED_NER-TA'},
			'low10ner':{'en':'CONLL_03','nl':'CONLL_03_DUTCH','es':'CONLL_03_SPANISH','de':'CONLL_03_GERMAN','eu':'LOW10_NER-EU','fa':'LOW10_NER-FA','fi':'LOW10_NER-FI','fr':'LOW10_NER-FR','he':'LOW10_NER-HE','hi':'LOW10_NER-HI','hr':'LOW10_NER-HR','id':'LOW10_NER-ID','ja':'LOW10_NER-JA','no':'LOW10_NER-NO','pl':'LOW10_NER-PL','pt':'LOW10_NER-PT','sl':'LOW10_NER-SL','sv':'LOW10_NER-SV','ta':'LOW10_NER-TA'},
			'commner':{'en':'COMMNER-EN','es':'COMMNER-ES','fr':'COMMNER-FR','ru':'COMMNER-RU'},
			'query':{'fr':'FRQUERY'},
			'icbu':{'en':'ICBU'},
			'semeval':{'tr':'SEMEVAL16-TR','es':'SEMEVAL16-ES','nl':'SEMEVAL16-NL','en':'SEMEVAL16-EN','ru':'SEMEVAL16-RU','sem14lap':'SEMEVAL14_LAPTOP','sem14res':'SEMEVAL14_RESTAURANT','sem15res':'SEMEVAL15_RESTAURANT'},
			'smallud':{'en':'UD_English-EWT','he':'UD_Hebrew-HTB','ja':'UD_Japanese-GSD','sl':'UD_Slovenian-SST','fr':'UD_French-Sequoia','id':'UD_Indonesian-GSD','fa':'UD_Persian-Seraji','ta':'UD_Tamil-TTB','nl':'UD_Dutch-LassySmall','de':'UD_German-GSD','sv':'UD_Swedish-LinES','it':'UD_Italian-PoSTWITA','es':'UD_Spanish-GSD','cs':'UD_Czech-FicTree','ar':'UD_Arabic-PADT'},
			'ontonote':{'en':'ONTONOTE_ENG'},
			'srl':{'en':'SRL-EN','de':'SRL-DE','es':'SRL-ES','zh':'SRL-ZH','cs':'SRL-CS','ca':'SRL-CA'},
			'atis':{'en':'ATIS-EN','hi':'ATIS-HI','tr':'ATIS-TR'},
			'enhancedud':{'ar':'UD_Arabic','bg':'UD_Bulgarian','cs':'UD_Czech','nl':'UD_Dutch','en':'UD_English','et':'UD_Estonian','fi':'UD_Finnish','fr':'UD_French','it':'UD_Italian','lv':'UD_Latvian','lt':'UD_Lithuanian','pl':'UD_Polish','ru':'UD_Russian','sk':'UD_Slovak','sv':'UD_Swedish','ta':'UD_Tamil','uk':'UD_Ukrainian','dm':'DM:DM_OOD','pas':'PAS:PAS_OOD','psd':'PSD:PSD_OOD'},
			'dependency':{'ptb':'PTB','ctb':'CTB','en':'UD_English-EWT','he':'UD_Hebrew-HTB','ja':'UD_Japanese-GSD','sl':'UD_Slovenian-SST','fr':'UD_French-Sequoia','id':'UD_Indonesian-GSD','fa':'UD_Persian-Seraji','ta':'UD_Tamil-TTB','nl':'UD_Dutch-LassySmall','de':'UD_German-GSD','sv':'UD_Swedish-LinES','it':'UD_Italian-PoSTWITA','es':'UD_Spanish-GSD','cs':'UD_Czech-FicTree','ar':'UD_Arabic-PADT'},
			}

corpus_map['mixed_data']={}
corpus_map['mixed_data'].update(corpus_map['indoeuro1'])
corpus_map['mixed_data'].update(corpus_map['indoeuro2'])
corpus_map['mixed_data'].update(corpus_map['difffam'])
corpus_map['mixed_data'].update(corpus_map['turkic'])
corpus_map['mixed_data'].update(corpus_map['austronesian'])

def set_reverse_corpus_map(corpus_map):
	reverse_corpus_map={}
	for key in corpus_map:
		reverse_corpus_map[key]={}
		for lang in corpus_map[key]:
			reverse_corpus_map[key][corpus_map[key][lang]]=lang
	return reverse_corpus_map
reverse_corpus_map=set_reverse_corpus_map(corpus_map)