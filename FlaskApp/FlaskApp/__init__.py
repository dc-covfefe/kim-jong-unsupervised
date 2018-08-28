import dash
from xlutils import copy
import flask
from dash.dependencies import Input, Output, State, Event
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
import base64
import datetime
import io
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# import en_core_web_lg
import scipy

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist, pdist
from nltk.cluster.kmeans import KMeansClusterer
import nltk
from sklearn.metrics import jaccard_similarity_score
import seaborn as sns
import pickle
from sklearn import metrics
import random
import time

# Import ContextEdgePreprocessor 
from ContextEdge15 import ContextEdgePreprocessor



# os.chdir("C:/Users/aacraig/Documents/ContextEdge/data/")

# df = pd.read_csv("C:/Users/aacraig/Documents/ContextEdge/data/cleaned_dreams_dataset.csv", encoding = 'latin1')


# Read in static dataframes
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, "static")
raw_path = os.path.join(APP_STATIC, "dream_data_raw.csv")
clean_path = os.path.join(APP_STATIC, "cleaned_dreams_dataset.csv")
#read data
raw_data = pd.read_csv(raw_path, encoding="latin1")
cleaned_data = pd.read_csv(clean_path, encoding="latin1")
# copy to reflect aaron's df 
df = cleaned_data.copy()
# Initialize empty output matrix

matrix = np.empty((1,1))


ngram_values_dict = {1 : (1,1), 2: (1,2), 3: (1,3)}



# App Front-end


# app = dash.Dash()
# server = flask.Flask(__name__)
# app = dash.Dash(__name__, server=server)


server = flask.Flask(__name__)
app = dash.Dash(server=server)
# Generating a 'unique' key
word_bank1 = ['guardian', 'pioneer', 'integrator', 'driver', 'nihilist', 'postnihilist', 'neonihilist']
word_bank2 = ['cognitive', 'rshiny', 'robotics', 'edge', 'neocognitive', 'postcognitive', 'levelsetting', 'touchpoint']
word_bank3 = ['consultant', 'associate', 'analyst', 'professor', 'datascientist', 'drone', 'robot', 'chatbot']


word1 = random.choice(word_bank1)
word2 = random.choice(word_bank2)
word3 = random.choice(word_bank3)


timestamp = time.time()

temporary_key = word1 + '_' + word2 + '_' + word3 + '_' + str(timestamp)[-7:]

print('Your temporary key is: {}'.format(temporary_key))


app.scripts.config.serve_locally = True
app.config.supress_callback_exceptions = True

app.layout = html.Div([


				dcc.Tabs(
						id = 'tabs',
						value = 'tab-0',
						children = [


							dcc.Tab(
									label = "NMF", 
		                           	children=[


		                           	# Column of drop-downs, etc.


		                           	html.Div([

		                           		html.Label('Configure NMF', style = {
		                           											'font-size': '36',
		                           											'font-family': 'Arial',
		                           											'text-align': 'center',
		                           											'display': 'inline-block',
		                           											'margin-bottom': '8px'}),

		                           		html.Br(),

		                           		html.Br(),

		                           		html.Div([
					                                        # Upload button
					                                        dcc.Upload(
					                                                id='upload-data',
					                                                filename = '',
					                                                children=html.Div([
					                                                
					                                                    html.A('load dataset')
					                                                ]),
					                                                style={
					                                                    'width': '56%',
					                                                    'height': '60px',
					                                                    'lineHeight': '60px',
					                                                    'borderWidth': '1px',
					                                                    'borderStyle': 'solid',
					                                                    'borderRadius': '5px',
					                                                    'borderColor': '#37383A',
					                                                    'display': 'inline-block',
					                                                    'text-align': 'center',
					                                                    'margin-top': '20px',
					                                                    'margin-bottom': '8px',
					                                                    'background-color': '#eff1f7',
					                                                    # 'margin-right': '24px',
					                                                    # 'margin-left': '24px',
					                                                    'font-size': '32',
					                                                    # 'font-weight': 'bold'
					                                                },
					                                                # Allow multiple files to be uploaded
					                                                multiple=True
					                                            )
					                                    
					                                    ],
					                                    style = {
					                                        'display': 'inline-block',
					                                        'text-align': 'center',
					                                        'width': '100%',
					                                        'height': '100%',
					                                        # 'margin-left': '36px',
					                                        # 'margin-right': '36px'
					                                        'margin-bottom': '16px'
					                                        }
					                                    ),

		                         

		                           		html.Br(),


		                           		html.Label('Select preprocessing method:',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),


		                           		dcc.Dropdown(
                                                    id = 'preprocessing-dropdown',
                                                    options = [
                                                    			{'label': 'Stemming', 'value': 'stemming'},
                                                    			{'label': 'Lemmatization', 'value': 'lemmatization'}

                                                    ],
                                                    value = 'stemming',

                                                    multi=False,

                                            ),


		                           		html.Br(),
		                           		

		                           		html.Label('Select ngram range:',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),


		                           		dcc.Dropdown(
                                                    id = 'ngram-range-dropdown',
                                                    options = [
                                                    			{'label': '(1,1)', 'value': 1},
                                                    			{'label': '(1,2)', 'value': 2},
                                                    			{'label': '(1,3)', 'value': 3}

                                                    ],
                                                    value = 1,
                                                    multi=False,

                                            ),



		                           		html.Br(),


		                           		html.Label('Select min_df:',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),

		                           		dcc.Input(
		                           				id = 'mindf-input',
		                           				value = 0.01,
		                           				min = 0,
		                           				max = 10000000,
		                           				style = {
		                           				'width': '80%',
		                           				'height': '32px',
		                           				'text-align': 'center',
		                           				'margin-bottom': '28px',
		                           				'font-size': '20'
		                           				}

		                           			),

		                           		html.Br(),

		                           		html.Label('Select max_df:',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),

		                           		html.Br(),

		                           		dcc.Input(
		                           				id = 'maxdf-input',
		                           				value = 0.99,
		                           				min = 0,
		                           				max = 10000000,
		                           				style = {
		                           				'width': '80%',
		                           				'height': '32px',
		                           				'text-align': 'center',
		                           				'margin-bottom': '20px',
		                           				'font-size': '20'
		                           				}

		                           			),

		                           		html.Br(),

		                           		html.Button(
													id='nmf-button1',
													n_clicks=0,
													children='>',
			                                        style={
		                                                'width': '56%',
		                                                'height': '60px',
		                                                'lineHeight': '60px',
		                                                'borderWidth': '1px',
		                                                'borderStyle': 'solid',
		                                                'borderRadius': '5px',
		                                                'borderColor': '#37383A',
		                                                'display': 'inline-block',
		                                                'text-align': 'center',
		                                                'margin-top': '24px',
		                                                'margin-bottom': '8px',
		                                                'background-color': '#eff1f7',
		                                                # 'margin-right': '24px',
		                                                # 'margin-left': '24px',
		                                                'font-size': '32',
		                                                'font-family': 'Arial'
		                                                # 'font-weight': 'bold'
			                                            }
			                                    ),




		                           		html.Br(),


		                           		html.Br(),

		                           		html.Label('Select no. of topics:',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),

		                           		html.Br(),

		                           		dcc.Input(
		                           				id = 'no-topics-input',
		                           				value = 2,
		                           				min = 2,
		                           				max = 100,
		                           				style = {
		                           				'width': '80%',
		                           				'height': '32px',
		                           				'text-align': 'center',
		                           				'margin-bottom': '28px',
		                           				'font-size': '20'
		                           				}

		                           			),

		                           		html.Br(),

		                           		html.Label('Select no. of components:',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),

		                           		html.Br(),

		                           		dcc.Input(
		                           				id = 'nmf-components-input',
		                           				value = 2,
		                           				min = 2,
		                           				max = 10000,
		                           				style = {
		                           				'width': '80%',
		                           				'height': '32px',
		                           				'text-align': 'center',
		                           				'margin-bottom': '28px',
		                           				'font-size': '20'
		                           				}

		                           			),

		                           		html.Br(),


		                           		html.Label('Select a value for "alpha":',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),

		                           		html.Br(),

		                           		dcc.Input(
		                           				id = 'nmf-alpha-input',
		                           				value = 0.5,
		                           				min = 2,
		                           				max = 10000,
		                           				style = {
		                           				'width': '80%',
		                           				'height': '32px',
		                           				'text-align': 'center',
		                           				'margin-bottom': '28px',
		                           				'font-size': '20'
		                           				}

		                           			),

		                           		html.Br(),

		                           		html.Label('Select a value for "l1-ratio":',
		                           			style = {
		                           				'font-size': '20',
		                           				'text-align': 'center',
		                           				'font-family': 'Arial'
		                           				}),

		                           		html.Br(),

		                           		dcc.Input(
		                           				id = 'nmf-l1ratio-input',
		                           				value = 0.5,
		                           				min = 0.000001,
		                           				max = 10000,
		                           				style = {
		                           				'width': '80%',
		                           				'height': '32px',
		                           				'text-align': 'center',
		                           				'margin-bottom': '28px',
		                           				'font-size': '20'
		                           				}

		                           			),		                          



		                           		html.Button(
													id='nmf-button2',
													n_clicks=0,
													children='>',
			                                        style={
		                                                'width': '56%',
		                                                'height': '60px',
		                                                'lineHeight': '60px',
		                                                'borderWidth': '1px',
		                                                'borderStyle': 'solid',
		                                                'borderRadius': '5px',
		                                                'borderColor': '#37383A',
		                                                'display': 'inline-block',
		                                                'text-align': 'center',
		                                                'margin-top': '12px',
		                                                'margin-bottom': '8px',
		                                                'background-color': '#eff1f7',
		                                                # 'margin-right': '24px',
		                                                # 'margin-left': '24px',
		                                                'font-size': '32',
		                                                # 'font-weight': 'bold'
			                                            }
			                                    ),


		                           		],
		                           		style = {
		                           			'margin-left': '38px',
		                           			'text-align': 'center',
		                           			'display': 'inline-block',
		                           			'float': 'left',
		                           			'width': '16%'

		                           			}
		                           		),

		                           	######
		                           	# End column

		                           	html.Div([

		                           		dcc.Graph(id = 'nmf-graph1')




		                           		],
		                           		style = {'float': 'left',
		                           				'width': '82%'}
		                           		)









		                           	],

								)





						],
						style = {
							'font-size': '32',
							'font-family': 'Arial'
							}

					)






	])



#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


# CALLBACKS

@app.callback(Output('nmf-button1', 'n_clicks'), [Input('mindf-input', 'value'), Input('maxdf-input', 'value'),
	Input('preprocessing-dropdown', 'value'), Input('ngram-range-dropdown', 'value')], 
	[State('nmf-button1', 'n_clicks')],
	[Event('nmf-button1', 'click')])
def step_one_callback(min_df_value, max_df_value, preprocessing_type, ngram_range_value, button_clicks):
    
	
	if button_clicks == 1:

		# Translate ngram range dropdown's value into a usable tuple
		ngram_selection = ngram_values_dict[ngram_range_value]

		print(button_clicks)


		# Generate mode string
		if preprocessing_type == 'stemming':

			print('stemming mode activated')
			mode_selection = 'stem_tfidf_matrix'



		elif preprocessing_type == 'lemmatization':

			mode_selection = 'lemma_tfidf_matrix'



		processed_docs, output_matrices, feature_lists, output_models = ContextEdgePreprocessor(filename = 'processed_contextedge_input_file_v2.csv', 
																			raw_input_file = False, 
																			min_proportion = min_df_value, 
																			max_proportion = max_df_value, 
																			ngram_span = ngram_selection, 
																			mode = mode_selection, 
																			perform_vectoriz_ops = True,
																			text_features = ['all_scenes_text'],
																			return_df = False)


		out_name1 = 'temp_data/' + temporary_key + '_output_matrix.csv'
		out_name2 = 'temp_data/' + temporary_key + '_processed_docs.csv'
		out_name3 = 'temp_data/' + temporary_key + '_features_list.csv'
		out_name4 = 'temp_data/' + temporary_key + '_vectorizer_model.pickle'


		dense_matrix = output_matrices['all_scenes_text'].todense()
		# Write the output_matrix to disk
		# output_matrix['all_scenes_text'].to_csv(out_name1, index = False)

		np.savetxt(out_name1, dense_matrix , delimiter = ',')

		
		# print(processed_docs[0])



		# Writting processed dreams to disk
		pd.DataFrame(processed_docs, columns = ['processed_doc']).to_csv(out_name2, index = False)

		# numpy.savetxt("foo.csv", a, delimiter=",")

		# Writ the feature list to disk
		pd.DataFrame(feature_lists['all_scenes_text'], columns = ['feature_list']).to_csv(out_name3, index = False)

		# Save the tfidf model to disk
		# pickle.dump(output_models['all_scenes_text'], open(out_name4, "wb"))	



		return 0

	else:

		return 0




#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

@app.callback(Output('nmf-graph1', 'figure'), [Input('no-topics-input', 'value'), Input('nmf-components-input', 'value'),
	Input('nmf-alpha-input', 'value'), Input('nmf-l1ratio-input', 'value'), Input('mindf-input', 'value'), Input('maxdf-input', 'value'),
	Input('ngram-range-dropdown', 'value')], [State('nmf-button2', 'n_clicks')],
	[Event('nmf-button2', 'click')])
def update_nmf_graph1(no_topics, nmf_components_value, nmf_alpha_value, nmf_l1ratio_value, min_df_value, max_df_value, ngram_range_value, num_clicks):


	if num_clicks > 0:

		# Getting the filenames
		matrix_filename = 'temp_data/' + temporary_key + '_output_matrix.csv'
		processed_docs_filename = 'temp_data/' + temporary_key + '_processed_docs.csv'
		features_list_filename = 'temp_data/' + temporary_key + '_features_list.csv'
		tfidf_fit_filename = 'temp_data/' + temporary_key + '_vectorizer_model.pickle'

		print('loading nmf input objects')
		# Read in tfidf

		dense_tfidf_matrix = pd.read_csv(matrix_filename)
		print('The shape of the tfidf_matrix is: {}.'.format(dense_tfidf_matrix.shape))

		# Reading in the processed documents
		processed_docs = pd.read_csv(processed_docs_filename, encoding = 'latin1')
		processed_docs = processed_docs['processed_doc'].tolist() 

		print(processed_docs[0])


		features_df = pd.read_csv(features_list_filename)
		features_list = features_df['feature_list'].tolist()
		print('The first five token features are: {}.'.format(features_list[:5]))


		sparse_tfidf_matrix = scipy.sparse.csr_matrix(dense_tfidf_matrix.values)

		# print(sparse_tfidf_matrix)
		print('the sparse tfidf matrix is loaded')

		# Defining the NMF object
		nmf = NMF(n_components=no_topics, random_state=42, alpha=0.1, l1_ratio=.2, \
          max_iter = 500, verbose = False, shuffle = True, init='nndsvd', solver = 'cd')


		print('Computing the NMF for the sparse tfidf matrix')
		nmf_model = nmf.fit(sparse_tfidf_matrix)


		print(nmf_model)
		#--------------------------------------------------------------------------------------------------
		#--------------------------------------------------------------------------------------------------
		def generate_topic_table(model, feature_names, n_top_words):
		    topics = {}
		    for topic_idx, topic in enumerate(model.components_):
		        t = ("topic_%d" % topic_idx)
		        topics[t] = [feature_names[i] for i in top_words(topic, n_top_words)]
		        
		    out_df = pd.DataFrame(topics)
		    out_df = out_df[list(topics.keys())]
		    
		    return out_df
		#--------------------------------------------------------------------------------------------------
		#--------------------------------------------------------------------------------------------------



		print(processed_docs[0])


		# tfidf_vectorizer = TfidfVectorizer(min_df = min_df_value, max_df = max_df_value, ngram_range = ngram_range_value, stop_words = 'english').fit(processed_docs)




		# Computing the weights for each dream
		# docs_weights = nmf_model.transform(tfidf_vectorizer.transform(processed_docs))

		# print(doc_weights[0])


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


























#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# CSS Setup
# css_directory = "C:/Users/aacraig/Dash/ContextEdge/app"
css_directory = APP_STATIC        
stylesheets = ['stylesheet.css']
static_css_route = '/static/'

@app.server.route('{}<stylesheet>'.format(static_css_route))
def serve_stylesheet(stylesheet):
    if stylesheet not in stylesheets:
        raise Exception(
            '"{}" is excluded from the allowed static files'.format(
                stylesheet
            )
        )
    return flask.send_from_directory(css_directory, stylesheet)


for stylesheet in stylesheets:
    app.css.append_css({"external_url": "/static/{}".format(stylesheet)})


# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------








if __name__ == '__main__':
    app.run_server(debug=True)