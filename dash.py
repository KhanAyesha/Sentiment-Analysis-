# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:18:35 2021

@author: ayesha
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:22:56 2021

@author: ayesha
"""

# Importing the libraries
import pickle
import pandas as pd
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objs as go


app=dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiment Analysis with Insights"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")



def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    
    
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
    
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def create_app_ui():
    global project_name
    df_s=pd.read_csv(r'D:\internship\scrappedReview_updated.csv')
    values=[sum(df_s['Possitivity']), len(df_s)-sum(df_s['Possitivity'])]
    colors=['green', 'red']

    layout= html.Div([
        html.H1(id = 'heading', children = project_name,
                     style = {
                "textAlign" : "center"
             
                
                }),
             #html.Hr(),
   
        dbc.Row([
            dbc.Col(
                html.Div([
                   # html.H4(children='Overall Distribution of Positive & Negative Reviews '),
                dcc.Graph(
                    id='chart',
                    
                           figure=go.Figure(
                               data=[
                                   go.Pie
                                     (
                                   labels=["Postive", 'Negative'],
                                   hole=.6,
                                   hoverinfo='label+value+percent', textinfo='value+label',
                                      
                                            values=values,
                                            marker=dict(colors=['green', 'red'])
                                            )],
                               layout=go.Layout(title='Overall Distribution of Positive & Negative Reviews',
                                     template='seaborn'          )
                           )
                           )
                ])
                    ),
            dbc.Col(
                html.Div([
                    html.H4(children='Word Cloud for customer reviews', style={'margin-top':'15px'}),
                       html.Img(src=app.get_asset_url('p.png'), style={'height': '300px', 'width':'700px'}),
                       
                      
                    ]))
                ]),
        dbc.Row([
            dbc.Col(
                html.Div([
                      dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = { 'width':'1000px', 'margin-left':'1px'}
                    
               )
                   
          
                    ])
                
                ),
         
                dbc.Col(
                    html.Div([
                          html.Div(id ='result1', style={'margin-left':'5px'})
                 ]  )
       )    
               
                    
                ]),
     
        dbc.Row([
            dbc.Col(
                html.Div([
                    
               dcc.Textarea(
                id='textarea',
                 value = 'its crap, plastic, not wood with an inlay and worth maybe 10 bucks.  Shady business,will not order from again.',
                placeholder="Enter Your review text here",
                rows=4,
                # cols=8,
                style={'width':'1000px','height':'200', 'margin-top':'0.5%', 'margin-left':'1px'}
            ),      ])
                
                ),
            dbc.Col(
                 html.Div(id ='result', style={'margin-top':'7%'})
                )

            ]),
        dbc.Row([
            dbc.Col(
                html.Div([ dbc.Button("Submit", color="success", className="mt-2 mb-3", id = 'button', style = {'width': '100px', 'margin':'0 50%'},
                                   )
                           
          
                    ])
         
           
                )
            ])
        ])
    return layout
   

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
def main():
    global app
    global project_name
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None
if __name__ == '__main__':
    main()


