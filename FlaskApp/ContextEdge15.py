# author: aaroncraig
# date: 2018-08-08

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import os 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, "static")
raw_path = os.path.join(APP_STATIC, "dream_data_raw.csv")
clean_path = os.path.join(APP_STATIC, "cleaned_dreams_dataset.csv")
#read data

def ContextEdgePreprocessor(filename = 'dream_data_raw.csv', 
								raw_input_file = True, 
								min_proportion = 0.1, 
								max_proportion = 0.6, 
								ngram_span = (1,1), 
								mode = 'stem_tfidf_matrix', 
								perform_vectoriz_ops = True,
								text_features = ['all_scenes_text'],
								return_df = True):
	

	# Import statements
	import pandas as pd
	import numpy as np
	import re
	import time
	import string
	from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer, TfidfTransformer
	from nltk.stem import PorterStemmer
	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import stopwords
	

	# import spacy
	# en_nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])
	# import en_core_web_lg
	# nlp = en_core_web_lg.load()

	

	#-------------------------------------------------------------------------------------------
	# Reading in the raw dream data
	df_raw = pd.read_csv(raw_path, encoding = 'latin1')
	#-------------------------------------------------------------------------------------------
	
	#-------------------------------------------------------------------------------------------
	# Making a copy of the raw data
	df = df_raw
	#-------------------------------------------------------------------------------------------

	#-------------------------------------------------------------------------------------------
	if raw_input_file == True:

		print('Processing the raw file.')

		
		# Renaming columns
		if ('name' in df.columns.tolist()) and ('scene_one' in df.columns.tolist()) and \
							 ('scene_two' in df.columns.tolist()) and ('scene_three' in df.columns.tolist()) and \
							 ('scene_four' in df.columns.tolist()):
			# Renaming columns
			df = df.rename(columns = {'Contact State and Country': 'location', 'Scene One': 'scene_one', 'Scene Two': 'scene_two',
		                         'Scene Three': 'scene_three', 'Created Date': 'date', 'Journal Entry Name': 'name', 
		                         'Journal Profile Number': 'id', 'Context': 'context', 'Scene Four': 'scene_four'})
		
		elif ('Event' in df.columns.tolist()):
			df = df.rename(columns = {'Event': 'event_summary'})

		

		if 'date' in df.columns.tolist():

			# Dropping the three records with null date values
			df  = df[pd.notnull(df['date'])].reset_index(drop = True)

			# Reformatting date column, extracting year, month, and day features
			df['date'] = pd.to_datetime(df['date'])
			df['year'] = df['date'].dt.year
			df['year'] = df['year'].astype(int)
			df['month'] = df['date'].dt.month	
			df['month'] = df['month'].astype(int)
			df['day'] = df['date'].dt.day
			df['day'] = df['day'].astype(int)


		if 'location' in df.columns.tolist():

			# Cleaning up null location values
			df.loc[df.location == 'Not Currently Available', 'location'] = np.nan

			# df['is_in _USA'] = np.nan
			df['country'] = df['location'].apply(lambda x: str(x).split(', ')[-1])

			# Making a state_region abbre. dictionary
			state_region_dict = {'South Carolina': 'SC', 'Ohio': 'OH', 'Texas': 'TX', 'Massachusetts': 'MA', \
	              'Pennsylvania': 'PA','Cherkas': 'Cherkas','Wyoming': 'WY','Tennessee': 'TN', \
	              'Alabama': 'AL', 'Maine': 'ME', 'Florida': 'FL', 'Missouri': 'MO', 'Oklahoma': 'OK',\
	              'Georgia': 'GA','New Mexico': 'NM', 'Alaska': 'AK', 'Alberta': 'Alberta', 'North Carolina': 'NC', \
	              'Ontario': 'Ontario', 'Michigan': 'MI', 'Illinois': 'IL', 'Washington': 'WA', 'Kentucky': 'KY',\
	               'Louisiana': 'LA', 'California': 'CA', 'Iowa': 'IA', 'Wisconsin': 'WI', 'nan': np.nan, \
	               'TX': 'TX', 'Virginia': 'VA'}


			# Adding state/region feature
			df['state_region'] = [state_region_dict[str(str(x).split(', ')[0])] for x in df['location'].tolist()]

			# Adding in special cases for country feature
			df.loc[df.country == 'United States', 'country'] = 'USA'
			df.loc[df.country == 'Ukr', 'country'] = 'Ukraine'
			df.loc[(df.state_region == 'Alberta') | (df.state_region == 'Ontario'), 'country'] = 'Canada'
			df.loc[df.country == 'nan', 'country'] = np.nan



		# sorting by increasing date
		df = df.sort_values('date').reset_index(drop = True)

		
		#-------------------------------------------------------------------------------------------
		if ('name' in df.columns.tolist()) and ('scene_one' in df.columns.tolist()) and \
							 ('scene_two' in df.columns.tolist()) and ('scene_three' in df.columns.tolist()) and \
							 ('scene_four' in df.columns.tolist()):

			# Text cleanup and creation of the 'all_scenes_text' column
			original_text_colnames = ['name', 'scene_one', 'scene_two', 'scene_three', 'scene_four']

			# Cleaning up each of the text features in the raw dataset
			for colname in original_text_colnames:
				
				# Removing all non-letter characters
				df[colname] = df[colname].fillna('').apply(lambda x: re.sub('[ ](?=[ ])|[^A-Za-z ]+', '', x))
				df[colname] = df[colname].str.replace('\d+', '').str.replace('  ', ' ')


			# Combining all the text from each scene into a new column
			df['all_scenes_text'] = df[['scene_one', 'scene_two', 'scene_three', 'scene_four']].fillna('').sum(axis = 1)

			# Write the processed file to disk
			df.to_csv("data/processed_contextedge_input_file_v2.csv", index = False)

		else:
			# Case where input dataframe is not the dream dataset or one of its derivatives 
			# and special case: input dataframe is the Wikipedia events dataset
			
			if 'text_features' == ['event_summary']:
				
				# Removing all non-letter characters
				df['event_summary'] = df['event_summary'].fillna('').apply(lambda x: re.sub('[ ](?=[ ])|[^A-Za-z ]+', '', x))
				df['event_summary'] = df['event_summary'].str.replace('\d+', '', regex = True).str.replace('  ', ' ')

				# # Making a list of all numbers between 0 and 100 for removal (trouble with replace() calls below)
				# numbers_list = [str(x) for x in range(101)]

				# df['event_summary'] = [[x.replace(number_str, '') for number_str in numbers_list] for x in df['event_summary'].tolist()]
				# df['event_summary'] = [[x.replace('\d+','', regex = True) for x in df ]



	elif raw_input_file == False:
		print('No data cleaning was performed, since "raw_input_file" was set to False.')
		print(' ')
	#-------------------------------------------------------------------------------------------


	#-------------------------------------------------------------------------------------------
	# Creating empty dictionaries to hold the tfidf matrices and feature name lists for each text
	# feature. Note that if perform_tfidf_ops is set to False, the function returns empty
	# dictionaries.

	matrices_dict = {}
	features_dict = {}
	#-------------------------------------------------------------------------------------------
	

	# Vectorization Operations
	if perform_vectoriz_ops == True:

		#-------------------------------------------------------------------------------------------
		# Looping over the text features specified in the function call, generating tfidf outputs

		for colname in text_features:
			#-------------------------------------------------------------------------------------------
			# Perform either lemmatization or stemming when generating tfidf objects
			#-------------------------------------------------------------------------------------------
			
			if mode == 'lemma_tfidf_matrix':
				t0 = time.time()
				
				# Getting a list of all documents in the text feature's column
				all_docs = df[colname].tolist()
				
				#-------------------------------------------------------------------------------------------
				# Defining custom lemmatizing function
				def lemmatizer(text):
					sent = []

					# numbers_list = [str(x) for x in range(101)]


					# text = text.replace('\d+', '')
					sent = [x.lemma_ for x in nlp(text)]

					return " ".join(sent)

    			#-------------------------------------------------------------------------------------------
			
				print('Lemmatizing the text feature: {}.'.format(colname))

				# Getting the lemmatized documents
				lemmatized_docs = list(map(lemmatizer, all_docs))


    			#-------------------------------------------------------------------------------------------
				# Generating the tfidf

				lemma_tfidf_fit = TfidfVectorizer(min_df = min_proportion, max_df = max_proportion, \
																		stop_words = 'english', ngram_range = ngram_span).fit(lemmatized_docs)

				#-------------------------------------------------------------------------------------------

				print('Generating the tfidf matrix for text feature: {}.'.format(colname))
				
				# Generating the tfidf matrix
				lemma_tfidf = lemma_tfidf_fit.transform(lemmatized_docs)


				# Inserting tfidf matrix into the tfidf dictionary
				matrices_dict[colname] = lemma_tfidf

				# Getting the list of feature names
				lemma_feature_names = lemma_tfidf_fit.get_feature_names()
				
				# Inserting the list of feature names into the features dictionary
				features_dict[colname] = lemma_feature_names
				
				print('There are: {} token features in the tfidf model.'.format(len(lemma_feature_names)))


				print('The tfidf matrix has shape: {}'.format(lemma_tfidf.shape))		

				model_out = lemma_tfidf_fit

				t1 = time.time()
				print('The tfidf operations for the feature: {} took {} seconds to complete.'.format(colname, np.round(t1-t0, decimals = 1)))


			
			elif mode == 'stem_tfidf_matrix':
				t0 = time.time()
				
				# Getting a list of all documents in the text feature's column
				all_docs = df[colname].tolist()

				# Getting the stemmed documents for this feature
				stemmed_docs = list(map(lambda x: PorterStemmer().stem(str(x)), all_docs))

				print('The tfidf fit for the stemmed text feature: {} is being generated.'.format(colname))
				
				#-------------------------------------------------------------------------------------------
				# Generating the tfidf fit for the stemmed documents

				stem_fit = TfidfVectorizer(min_df = min_proportion, max_df = max_proportion, \
																		stop_words = 'english', ngram_range = ngram_span).fit(stemmed_docs)
				
				#-------------------------------------------------------------------------------------------

				print('The tfidf fit for the stemmed text feature: {} is complete.'.format(colname))

				# Making the tfidf bag of words
				stem_tfidf = stem_fit.transform(stemmed_docs)

				# Getting the tfidf feature names
				stem_feature_names = stem_fit.get_feature_names()

				print('There are: {} token features in the tfidf model.'.format(len(stem_feature_names)))

				print('The tfidf matrix has shape: {}'.format(stem_tfidf.shape))

				# Inserting results into the tfidf dictionaries
				matrices_dict[colname] = stem_tfidf
				features_dict[colname] = stem_feature_names

				model_out = stem_fit

				t1 = time.time()
				print('The tfidf operations for the feature: {} took {} seconds to complete.'.format(colname, np.round(t1-t0, decimals = 1)))


			elif mode == 'lemma_dtf_matrix':

				t0 = time.time()

				print('The dtf fit for the lemmatized text feature: {} is being generated.'.format(colname))

				# Getting a list of all documents in the text feature's column
				all_docs = df[colname].tolist()

				#-------------------------------------------------------------------------------------------
				# Defining custom lemmatizing function
				def lemmatizer(text):
					# sent = []
					# doc = nlp(text)
					# for word in doc:
					# 	sent.append(word.lemma_)

					sent = [x.lemma_ for x in nlp(text)]

					return " ".join(sent)

    			#-------------------------------------------------------------------------------------------

				# Getting the lemmatized documents
				lemmatized_docs = list(map(lemmatizer, all_docs))

    			#-------------------------------------------------------------------------------------------
				# Generating the document-term-frequency matrix for the text feature

				lemma_count_fit = CountVectorizer(min_df = min_proportion, max_df = max_proportion, \
															stop_words = 'english', ngram_range = ngram_span).fit(lemmatized_docs)

				print('The dtf fit for the lemmatized text feature: {} is complete.'.format(colname))

				lemma_dtf_matrix = lemma_count_fit.transform(lemmatized_docs)
				#-------------------------------------------------------------------------------------------

				print('The dtf fit for the lemmatized text feature: {} is complete.'.format(colname))

				# Inserting the dtf matrix into the appropriate dictionary
				matrices_dict[colname] = lemma_dtf_matrix

				# Getting the bag-of-words feature names
				lemma_feature_names = lemma_dtf_fit.get_feature_names()

				print('There are: {} features in the dtf/bag-of-words model for the feature: {}.'.format(len(stem_feature_names, colname)))

				# Inserting the list of features into the appropriate dictionary
				features_dict[colname] = lemma_feature_names

				model_out = lemma_count_fit

				t1 = time.time()

				print('The dtf operations for the feature: {} took {} seconds to complete.'.format(colname, np.round(t1-t0, decimals = 1)))


			elif mode == 'stem_dtf_matrix':

				t0 = time.time()

				# Getting a list of all documents in the text feature's column
				all_docs = df[colname].tolist()

				# Getting the stemmed documents for this feature
				stemmed_docs = list(map(lambda x: PorterStemmer().stem(str(x)), all_docs))

				print('The dtf fit for the stemmed text feature: {} is being generated.'.format(colname))

				#-------------------------------------------------------------------------------------------
				# Generating the document-term-frequency matrix for the text feature

				stem_dtf_fit = CountVectorizer(min_df = min_proportion, max_df = max_proportion, \
																stop_words = 'english', ngram_range = ngram_span).fit(stemmed_docs)

				#-------------------------------------------------------------------------------------------

				print('The dtf fit for the stemmed text feature: {} is complete.'.format(colname))

				print('Generating the document-term-frequency matrix for the text feature: {}.'.format(colname))
				# Making the document-term-frequency matrix
				stem_dtf_matrix = stem_dtf_fit.transform(stemmed_docs)

				# Inserting the dtf matrix into the appropriate dictionary
				matrices_dict[colname] = stem_dtf_matrix

				# Getting the bag-of-words feature names
				stem_feature_names = stem_dtf_fit.get_feature_names()

				print('There are: {} features in the dtf/bag-of-words model for the feature: {}.'.format(len(stem_feature_names, colname)))

				# Inserting the list of features into the appropriate dictionary
				features_dict[colname] = stem_feature_names


				model_out = stem_dtf_fit

				t1 = time.time()

				print('The dtf operations for the feature: {} took {} seconds to complete.'.format(colname, np.round(t1-t0, decimals = 1)))


		#-------------------------------------------------------------------------------------------
		#-------------------------------------------------------------------------------------------
	elif perform_vectoriz_ops == False:

		print('No tfidf operations were performed, because perform_tfidf_ops is set to False.')

	

	if return_df == True:

		if mode == 'stem_dtf_matrix' or mode == 'stem_tfidf_matrix':
		
			return df, stemmed_docs, matrices_dict, features_dict, model_out

		if mode == 'lemma_dtf_matrix' or mode == 'lemma_tfidf_matrix':
		
			return df, lemmatized_docs, matrices_dict, features_dict, model_out


	elif return_df == False:

		if mode == 'stem_dtf_matrix' or mode == 'stem_tfidf_matrix':
		
			return stemmed_docs, matrices_dict, features_dict, model_out

		if mode == 'lemma_dtf_matrix' or mode == 'lemma_tfidf_matrix':
		
			return lemmatized_docs, matrices_dict, features_dict, model_out





