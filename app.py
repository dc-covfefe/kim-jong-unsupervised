import dash
import os
from xlutils import copy
import flask
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from loremipsum import get_sentences
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
import base64
import datetime
import io


app = dash.Dash()

app.scripts.config.serve_locally = True


# Dictionary to convert tab numbers to page names
values_pagenames = {0 : 'Main', 1 : 'Topic Models', 2 : 'Classification Models', 3 : 'Explore New Data' }
tab_nos_pagenames = {'tab-0': 'Main', 'tab-1': 'Topic Models', 'tab-2': 'Classification Models', 'tab-3': 'Explore New Data'}



# Read in static dataframes
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, "static")
raw_path = os.path.join(APP_STATIC, "dream_data_raw.csv")
clean_path = os.path.join(APP_STATIC, "cleaned_dreams_dataset.csv")
#read data
raw_data = pd.read_csv(raw_path, encoding="latin1")
cleaned_data = pd.read_csv(clean_path, encoding="latin1")


# 1. Raw dreams dataset
# raw_data = pd.read_csv("C:/Users/aacraig/Documents/ContextEdge/data/raw_dreams_dataset.csv", encoding = 'latin1', low_memory = False)

# 2. Processed dreams dataset 
# cleaned_data = pd.read_csv("C:/Users/aacraig/Documents/ContextEdge/data/cleaned_dreams_dataset.csv", encoding = 'latin1', low_memory = False)

# Dictionary for storing dataframes with keys/names
dataframes_dict = {'cleaned_data': cleaned_data,
                    'raw_data': raw_data}




# Layout
app.layout = html.Div([
    dcc.Tabs(id = 'tabs',\
                value = 'tab-0',
                children=[
                    # Main tab        
                    dcc.Tab(label = values_pagenames[0], 
                        children=[
                            html.Div([
                                
                                html.Div([
                                            html.P('ContextEdge'),
                                        ],
                                        style = {
                                            'margin-top': '140px',
                                            'margin-bottom': '60px',
                                            'font-size': '102',
                                            'text-align': 'center',
                                            'font-weight': 'bold'
                                            # 'text-shadow': {'h-shadow':'20px', 'v-shadow':'20px', 'color':'#808080'}
                                                    
                                            }
                                        ),

                                html.Div([
                                    # Upload button
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div([
                                            
                                            html.A('Load Dataset')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '80px',
                                            'lineHeight': '80px',
                                            'borderWidth': '2px',
                                            'borderStyle': 'solid',
                                            'borderRadius': '6px',
                                            'textAlign': 'center',
                                            'margin-top': '74px',
                                            'margin-bottom': '16px',
                                            'margin-right': '2px',
                                            'margin-left': '2px',
                                            'font-size': '40',
                                            'font-weight': 'bold'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple=True
                                        )
                                ],
                                style = {
                                    'display': 'inline-block',
                                    'text-align': 'center',
                                    'width': '20%',
                                    'height': '100%'
                                    }
                                ),

                                html.Div(id='output-data-upload',
                                        style = {
                                            'margin-top': '16px',
                                            'margin-bottom': '16px'
                                            }
                                        ),


                                html.Div([
                                        dt.DataTable(rows=[{}])
                                        ],
                                        style = {
                                            'display': 'none',
                                            'margin-top': '16px',
                                            'margin-bottom': '16px'
                                            }
                                        )
                            

                            ])
                        ],
                    style = {
                        'fontFamily': 'Arial',
                        'margin-bottom': '0px',
                        'display':'inline-block',
                        'text-align': 'center',
                        'font-size': '32'
                        }
                    ),


                    #--------------------------------------------------------------------------------------------------
                    # Topic Models tab
                    dcc.Tab(label = values_pagenames[1], 
                            children=[
                                   html.Div([
                                        
                                            html.Div([
                                                        html.H2('Topic Models'),
                                                    ],
                                                    style = {
                                                        'margin-top': '2px',
                                                        'margin-left': '2px',
                                                        'margin-bottom': '14px',
                                                        'margin-top': '60px',
                                                        'font-size': '30',
                                                        'text-align': 'Center'
                                                        }
                                                    ),

                                            html.Br(),

                                            # Hidden div to hold dataset/dataframe name
                                            html.Div(id ='topic-models-plot-dataframe-name',                                                
                                                style = {
                                                'display':'none'
                                                }),

                                            html.Br(),

                                            html.Div([
                                                    # First drop-down
                                                    html.Label('Select dataset'),
                                                ],
                                                style = {
                                                    'text-align': 'left',
                                                    'font-weight': 'bold',
                                                    'font-size': '18'
                                                    }
                                                ),

                                            html.Br(),

                                            # Pre-existing datasets dropdown
                                            dcc.Dropdown(
                                                id = 'topic-model-datasets-dropdown',
                                                options = [
                                                    {'label': 'Dreams Dataset (Cleaned)', 'value': 'cleaned_dreams_dataset'},
                                                    {'label': 'Dreams Dataset (Raw)', 'value': 'raw_dreams_dataset'}
                                                ],
                                                value = 'cleaned_dreams_dataset',

                                                multi=False
                                            ),

                                            html.Br(),

                                            html.Br(),

                                            html.Div([
                                                    # Label for second dropdown
                                                    html.Label('Select algorithm'),
                                                ],
                                                style = {
                                                    'text-align': 'left',
                                                    'font-weight': 'bold',
                                                    'font-size': '18'
                                                    }
                                                ),

                                                    

                                            html.Br(),

                                            dcc.Dropdown(
                                                id = 'topic-model-algorithms-dropdown',
                                                options=[
                                                    {'label': 'Nonnegative Matrix Factorization (NMF)', 'value': 'NMF'},
                                                    {'label': 'Latent Dirichlet Allocation (LDA)', 'value': 'LDA'},
                                                    {'label': 'Latent Sentiment Analysis (LSA)', 'value': 'LSA'}
                                                ],
                                                value='NMF',
                                                multi=False
                                            ),

                                            html.Br(),
                                            html.Br(),

                                            html.Div([
                                                    # Label for third dropdown
                                                    html.Label('Select preprocessing method'),
                                                ],
                                                style = {
                                                    'text-align': 'left',
                                                    'font-weight': 'bold',
                                                    'font-size': '18'
                                                    }
                                                ),

                                            html.Br(),
                                            
                                            # Third drop-down
                                                dcc.Dropdown(
                                                    id = 'topic-model-preprocessing-dropdown',
                                                    options = [
                                                        {'label': 'Stemming', 'value': 'stemming'},
                                                        {'label': 'Lemmatization', 'value': 'Lemmatization'}
                                                    ],
                                                    value = 'stemming',

                                                    multi=False
                                                ),


                                            html.Br(),

                                            html.Br(),
                                            
                                            html.Div([
                                                    # Label for fourth dropdown
                                                    html.Label('Select visualization level'),
                                                ],
                                                style = {
                                                    'text-align': 'left',
                                                    'font-weight': 'bold',
                                                    'font-size': '18'
                                                    }
                                                ),

                                            html.Br(),

                                            # Fourth drop-down
                                            dcc.Dropdown(
                                                id = 'topic-model-viz-level-dropdown',
                                                options=[
                                                    {'label': 'Topic', 'value': 'tab2_topic_mode'},
                                                    {'label':'Dream' , 'value': 'tab2_dream_mode'},
                                                    {'label': 'Word', 'value': 'tab2_word_mode'},
                                                    {'label': 'Date', 'value': 'tab2_date_mode'},
                                                    {'label': 'Location', 'value': 'tab2_location_mode'}
                                                ],
                                                value='tab2_topic_mode',
                                                multi=False
                                            ),


                                            html.Br(),
                                            html.Br(),

                                            html.Div([
                                                    # Label for additional dropdowns
                                                    html.Label('Select additional options'),
                                                ],
                                                style = {
                                                    'text-align': 'left',
                                                    'font-weight': 'bold',
                                                    'font-size': '18'
                                                    }
                                                ),

                                            html.Br(),

                                            # Fifth drop-down                                        
                                            dcc.Dropdown(
                                                id = 'topic-model-viz-option2',
                                                multi=False
                                            ),

                                            html.Br(),

                                            # Sixth drop-down                                        
                                            dcc.Dropdown(
                                                id = 'topic-model-viz-option3',
                                                multi=False
                                            ),

                                            html.Br(),

                                            # Seventh drop-down                                        
                                            dcc.Dropdown(
                                                id = 'topic-model-viz-option4',
                                                multi=False
                                            ),

                                            html.Br(),

                                            html.Button(id='submit-button', n_clicks=0, children='GO',
                                                style = {
                                                'background-color': '#f2f5f9',
                                                'width': '34%',
                                                'height': '62px',
                                                'border': '4px solid #37383a',
                                                'border-radius': '4px',
                                                'color': 'black',
                                                'padding': '0px' '0px',
                                                'text-align': 'center',
                                                'text-decoration': 'none',
                                                'display': 'inline-block',
                                                'font-size': '32px',
                                                'font-weight': 'bold',
                                                'margin-top': '46px'

                                                }
                                            ),


                                            
                                            html.Div(id = 'selected-topic-model-dataset-output'),

                                            # Hiden div to hold datatable for selected dataset
                                            html.Div(children = [
                                                                dt.DataTable(rows=[{}])
                                                            ],
                                                    style = {
                                                        'display': 'none',
                                                        'margin-top': '16px',
                                                        'margin-bottom': '16px'
                                                        }
                                                            ),


                                        
                                    ],
                                    style = {
                                        'width': '16%',
                                        'height': '100%',
                                        'margin-top': '24px',
                                        'margin-bottom': '0px',
                                        'margin-left': '24px',
                                        'margin-right': '0px'
                                    })
                            ],
                            style = {
                                'fontFamily': 'Arial',
                                'font-size': '32'
                                }
                            ),

                    #--------------------------------------------------------------------------------------------------
                    # Unsupervised Models tab
                    dcc.Tab(label = values_pagenames[2], 
                            children=[
                                   html.Div([
                                        
                                        html.Div([
                                                    html.H2('Classification Models'),
                                                    ],
                                                    style = {
                                                        'margin-top': '2px',
                                                        'margin-left': '2px',
                                                        'margin-bottom': '14px',
                                                        'margin-top': '240px',
                                                        'font-size': '20',
                                                        'text-align': 'Center'
                                                        }
                                                    ),

                                        html.Br(),

                                        # First drop-down
                                        html.Label('Algorithm'),

                                        html.Br(),
                                        
                                        dcc.Dropdown(
                                            id = 'unsupervised-model-dropdown',
                                            options=[
                                                {'label': 'K-Means', 'value': 'kmeans'},
                                                {'label': 'DBSCAN', 'value': 'dbscan'},
                                                {'label': 'Hierarchical Clustering', 'value': 'hierarchical'}
                                            ],
                                            value='kmeans',
                                            multi=False
                                        ),

                                        html.Br(),
                                        html.Br(),

                                        # Label for second drop-down
                                        html.Label('Preprocessing Method'),

                                        html.Br(),
                                        
                                        # Second drop-down
                                        dcc.Dropdown(
                                            id = 'topic-viz-option1',
                                            options = [
                                                {'label': 'Stemming', 'value': 'stemming'},
                                                {'label': 'Lemmatization', 'value': 'Lemmatization'}
                                            ],
                                            value = 'stemming',

                                            multi=False
                                        ),

                                        html.Br(),

                                        html.Br(),

                                        # Label for third drop-down
                                        html.Label('Visualization Level'),

                                        html.Br(),

                                        # Third drop-down
                                        dcc.Dropdown(
                                            id = 'topic-viz-level-dropdown',
                                            options=[
                                                {'label': 'Cluster' , 'value': 'tab3_cluster_mode'},
                                                {'label': 'Word', 'value': 'tab3_word_mode'}
                                            ],
                                            value='tab3_cluster_mode',
                                            multi=False
                                        ),


                                        html.Br(),
                                        html.Br(),

                                        html.Label('Additional Options'),

                                        html.Br(),

                                        # Fourth drop-down                                        
                                        dcc.Dropdown(
                                            id = 'topic-viz-option2',
                                            multi=False
                                        ),

                                        html.Br(),

                                        # Fifth drop-down                                        
                                        dcc.Dropdown(
                                            id = 'topic-viz-option3',
                                            multi=False
                                        ),

                                        html.Br(),

                                        # Sixth drop-down                                        
                                        dcc.Dropdown(
                                            id = 'topic-viz-option4',
                                            multi=False
                                        ),


                                    
                                    ],
                                    style = {
                                        'width': '16%',
                                        'height': '100%'
                                    })
                            ],
                            style = {
                                'fontFamily': 'Arial',
                                'font-size': '32'
                                }
                            ),

                    
                    #--------------------------------------------------------------------------------------------------
                    # Explore New Data tab
                    dcc.Tab(label = values_pagenames[3], 
                            children=[
                                   html.Div([
                                        
                                        html.Div([

                                            html.Div([
                                                    html.H2('Explore New Data'),
                                                ],
                                                style = {
                                                    'margin-top': '2px',
                                                    'margin-left': '2px',
                                                    'margin-bottom': '14px',
                                                    'margin-top': '240px',
                                                    'font-size': '20',
                                                    'text-align': 'Center'
                                                    }
                                                ),
                                            ],
                                            style = {
                                            'text-align': 'center'
                                                }

                                            ),

                                        
                                    
                                    ],
                                    style = {
                                        'width': '16%',
                                        'height': '100%'
                                    })
                            ],
                            style = {
                                'fontFamily': 'Arial',
                                'font-size': '32'
                                }
                            ),


        ]),



], 

style={
    'width': '100%',
    'fontFamily': 'Arial',
    # 'font-weight': 'bold',
    'margin-left': '0px',
    'margin-right': '0px',
    'margin-top': '0px',
    'margin-bottom': '0px',
    # 'font-size': '32',
    'display': 'inline-block',
    'text-align': 'center',
    # 'background-color': '#f4f6f9'
    }
)


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# Callbacks

# 1. Topic Models callbacks


# 1a. callback to update which dataframe is used as the input for topic models tab
@app.callback(
    dash.dependencies.Output('selected-topic-model-dataset-output', 'children'),
    [dash.dependencies.Input('topic-model-datasets-dropdown', 'value')]
)
def update_topic_models_dataset(selected_dataset_name):
    
    # Create dictionary from dropdown values to filenames
    values_filepaths = {'cleaned_dreams_dataset': clean_path,
                        'raw_dreams_dataset': raw_path}
    values_filenames = {'cleaned_dreams_dataset': 'cleaned_dreams_dataset.csv',
                        'raw_dreams_dataset': 'raw_dreams_dataset.csv'}

    filepath = values_filepaths[selected_dataset_name]

    topic_models_df = pd.read_csv(filepath, encoding = 'latin1', low_memory = False)

    return html.Div([

                html.H3([
                            'Selected dataset:'
                        ],
                        style = {
                            'margin-top' : '36px',
                            'margin-bottom': '0px',
                            'font-size': '20'
                            }
                        ),

                html.Br(),
                
                html.H3([
                            values_filenames[selected_dataset_name]
                            ],
                            style = {
                            'margin-top' : '0px',
                            'margin-bottom': '14px',
                            'font-size': '16',
                            'font-weight': 'lighter'
                                }
                            ),

                # Use the DataTable prototype component:
                # github.com/plotly/dash-table-experiments
                # dt.DataTable(rows=df.to_dict('records')),

               
            ])


@app.callback(
    dash.dependencies.Output('topic-models-plot-dataframe-name', 'children'),
    [dash.dependencies.Input('topic-model-datasets-dropdown', 'value')]
)
def update_topic_models_df_name(selected_dataset_name):
    
    df_names_dict = {'raw_dreams_dataset': 'raw_data',
                    'cleaned_dreams_dataset': 'cleaned_data'}

    return df_names_dict[selected_dataset_name]









# @app.callback(
#     dash.dependencies.Output('topic-models-option2', 'options'),
#     [dash.dependencies.Input('topic-viz-level-dropdown', 'value'),
#        dash.dependencies.Input('topic-viz-option1', 'value')]
# )
# def update_topic_option2_dropdown(selected_model):
#     # return [{'label': lab, 'value': val} for (lab, val) in zip(all_y_feature_names[selected_dataset], all_y_feature_options[selected_dataset])]

# @app.callback(
#     dash.dependencies.Output('topic-models-option3', 'options'),
#     [dash.dependencies.Input('topic-viz-level-dropdown', 'value')
#        dash.dependencies.Input('topic-viz-option1', 'value')]
# )
# def update_topic_option3_dropdown(selected_model):
#     # return [{'label': lab, 'value': val} for (lab, val) in zip(all_y_feature_names[selected_dataset], all_y_feature_options[selected_dataset])]

# @app.callback(
#     dash.dependencies.Output('topic-models-option4', 'options'),
#     [dash.dependencies.Input('topic-viz-level-dropdown', 'value'),
#        dash.dependencies.Input('topic-viz-option1', 'value')]
# )
# def update_topic_option4_dropdown(selected_model):
#     # return [{'label': lab, 'value': val} for (lab, val) in zip(all_y_feature_names[selected_dataset], all_y_feature_options[selected_dataset])]







#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------










def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([

        html.H3([
                    'Selected dataset:'
                ],
                style = {
                    'margin-top' : '48px',
                    'margin-bottom': '8px',
                    'font-size': '24'
                    }
                ),

        html.Br(),
        
        html.H3([
                    filename
                    ],
                    style = {
                    'margin-top' : '8px',
                    'margin-bottom': '20px',
                    'font-size': '24',
                    'font-weight': 'lighter'
                        }
                    ),

        # Use the DataTable prototype component:
        # github.com/plotly/dash-table-experiments
        dt.DataTable(rows=df.to_dict('records')),

       
    ])






#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
## CSS Setup
#css_directory = "C:/Users/aacraig/Dash/ContextEdge/app"
#stylesheets = ['stylesheet.css']
#static_css_route = '/static/'
#
#@app.server.route('{}<stylesheet>'.format(static_css_route))
#def serve_stylesheet(stylesheet):
#    if stylesheet not in stylesheets:
#        raise Exception(
#            '"{}" is excluded from the allowed static files'.format(
#                stylesheet
#            )
#        )
#    return flask.send_from_directory(css_directory, stylesheet)
#
#
#for stylesheet in stylesheets:
#    app.css.append_css({"external_url": "/static/{}".format(stylesheet)})
#
#
#

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------








if __name__ == '__main__':
    app.run_server(debug=True)