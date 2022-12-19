#===========================================
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from chart_studio import plotly
import scipy
import os
from tabulate import tabulate
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

import dash as dash
from dash import dcc, dash_table
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_daq as daq
import dash_bootstrap_components as dbc
import warnings
from scipy import signal
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from datetime import date
warnings.filterwarnings('ignore')

style={'textAlign':'center','background': 'rgb(220, 220, 220)','color': 'black'}
style2={'textAlign':'center'}
steps=0.1
marks= lambda min,max:{i:f"{i}" for i in range(min,max)}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
BS= ["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"]

#============Helper Function===============
def qqp(values):
    qqplot_data = qqplot(values, line='s').gca().lines
    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata()})
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines'})
    fig['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })
    return fig

def shapiro_test(x, title):
    stats, p = shapiro(x)
    alpha = 0.01
    if p > alpha :
        return f'Shapiro test:\n statistics = {stats:.2f} p-vlaue of ={p:.2f} \n{title} dataset is Normal'
    else:
        return f'Shapiro test:\n statistics = {stats:.2f} p-vlaue of ={p:.2f} \n {title} dataset is NOT Normal'


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)

    alpha = 0.01
    if p > alpha :
        return f'K-S test:\n statistics = {stats:.2f} p-vlaue of ={p:.2f} \n  {title} dataset is Normal'
    else:
        return f'K-S test :\n statistics = {stats:.2f} p-vlaue of ={p:.2f} \n {title} dataset is Not Normal'


def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    alpha = 0.01
    if p > alpha :
        return f'da_k_squaredtest:\n statistics = {stats:.2f} p-vlaue of ={p:.2f} \n  {title} dataset is Normal'
    else:
        return f'da_k_squared test :\n statistics = {stats:.2f} p-vlaue of ={p:.2f} \n {title} dataset is Not Normal'

#==========================================


intro= """ In this project, the pay of public servants in San Francisco,\n 
United States,\n with respect to different Job titles,\n 
status are going to be analyzed with visual representation.
 """
#=======================================================
#path="C:/Users/oseme/Desktop/Data Visualization Class/Project/"
df= pd.read_csv("san-francisco-payroll_2011-2019.csv",low_memory=False)

#==cleaning
df['Status'].fillna(value=df['Status'].mode()[0],inplace=True)
df= df[~(df["Base Pay"]=="Not Provided")]  #Remove Base pay rows with Not Provided
df['Benefits'][df.Benefits=="Not Provided"]=0
df['Overtime Pay'][df["Overtime Pay"]=="Not Provided"]=0
df['Other Pay'][df["Other Pay"]=="Not Provided"]=0
df['Status']=df['Status'].map(lambda x:1 if x== 'FT' else 0)
df=df[~(df['Total Pay & Benefits']==0)]
# change data type
# print(df.iloc[:,2:6])
df_clean=df.astype(dict(zip(df.columns[2:6],[float]*4)))
#== Reverse dataframe
df_clean=df_clean.iloc[::-1,:].reset_index().drop(columns=["index"])

#== Make pay positive where negative
df_clean['Total Pay & Benefits']=np.abs(df_clean['Total Pay & Benefits'])
df_clean['Base Pay']=np.abs(df_clean['Base Pay'])
df_clean['Overtime Pay']=np.abs(df_clean['Overtime Pay'])
df_clean['Other Pay']=np.abs(df_clean['Other Pay'])
df_clean['Total Pay']=np.abs(df_clean['Total Pay'])
df_clean['Benefits']=np.abs(df_clean['Benefits'])

print(df_clean.info())
def outlier(data):
 global Q1,Q3
 sorted(data)
 Q1,Q3 = np.percentile(data , [25,75])
 IQR = Q3-Q1
 lower_range = Q1 - (1.5 * IQR)
 upper_range = Q3 + (1.5 * IQR)
 return lower_range,upper_range

lower,upper= outlier(df_clean['Total Pay & Benefits'])
df_no_outlier= df_clean[(df_clean['Total Pay & Benefits']<upper)]
print(tabulate(pd.DataFrame(df_no_outlier.describe().iloc[:,-3]),headers='keys',tablefmt="fancy_grid"))

print("#============Transformation Using Quantile Transformer======================")
#============Transformation Using Quantile Transformer======================

quantile = QuantileTransformer(output_distribution='normal')
data_trans = quantile.fit_transform(df_no_outlier['Total Pay & Benefits'].values.reshape(-1,1))

print("#============Statistics======================")
df_no_outlier['Status']=df_no_outlier['Status'].map(lambda x:'FT' if x== 1 else 'PT')
features=['mean','median']
dfc= df_no_outlier.select_dtypes(include='float64')
cols=dfc.columns
stat_df= pd.DataFrame(columns=features,index=cols)
stat_df.loc[cols[0]]=[dfc[cols[0]].mean(),dfc[cols[0]].median()]
stat_df.loc[cols[1]]=[dfc[cols[1]].mean(),dfc[cols[1]].median()]
stat_df.loc[cols[2]]=[dfc[cols[2]].mean(),dfc[cols[2]].median()]
stat_df.loc[cols[3]]=[dfc[cols[3]].mean(),dfc[cols[3]].median()]
stat_df.loc[cols[4]]=[dfc[cols[4]].mean(),dfc[cols[4]].median()]
stat_df.loc[cols[5]]=[dfc[cols[5]].mean(),dfc[cols[5]].median()]
stat_df=stat_df.round(2)
stat_df.index.name= "Statistics"
#print(tabulate(stat_df,headers='keys',tablefmt="fancy_grid"))
stat_df= stat_df.round({'mean':2,'median':2})
stat_df.insert(0,'Features',list(stat_df.index))
statistics= stat_df

cat= ['Job Title','Status']
cat2= ['Base Pay', 'Overtime Pay', 'Other Pay','Benefits', 'Total Pay', 'Total Pay & Benefits']
df_clean['Status']=df_clean['Status'].map(lambda x:'FT' if x== 1 else 'PT')

countdf= df_clean.groupby('Year').count()
figcount=px.bar(countdf, y='Status', title="    Employee Count Over The Years ",
                      template="plotly_dark",text='Status')
figcount.update_traces(texttemplate='%{text:.2s}', textposition='outside')
figcount.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')










#===========================================

main_df = df_no_outlier

my_app= dash.Dash(__name__,external_stylesheets=[dbc.themes.SOLAR]) #dbc.themes.MORPH





my_app.layout= html.Div(children=[
    html.H3("Visualization of  Payroll in San Francisco",style=style),
    html.H6(intro,style=style2),
    html.Br(),


    #Pie Seciton
    html.H3("Overview of Pay with Respect to Job-Type",style=style),
    dcc.RadioItems(id='overradio',
                                   options=[{"label": "Dataset with Outliers", "value": "with-outliers"},
                                            {"label": "Dataset without Outliers", "value": "without-outliers"}],
                                   value='without-outliers', labelStyle={'display': 'in-line'}),
    html.P("Select a Numeric Feature:"),
    dcc.Dropdown(id='overcheck',
                  options=[{'label':i,'value':i} for i in cat2],
                  value='Base Pay',placeholder='Select one...'),html.Br(),
    html.P("Select a Categorical Feature:"),
    dcc.Dropdown(id='overcheck2',
                  options=[{'label': i, 'value': i} for i in cat],
                  value='Status',placeholder='Select one...'), html.Br(),
    html.P("Input value for Doughnut size:"),
    dcc.Input(id='overin',type='number',value=0.1,min=0.01, max=0.9, step=0.01),
    html.Div(id='overview',style={'width':'80%','height':'80%','horizontal-align': 'middle'}),html.Br(),
    #Pie Section Ends

    html.Br(),
    #Outlier Section
    html.H3("Outlier Detection",style=style), html.Br(),
    html.Div([
                html.P("Filter:"),html.Br(),
                dcc.Tabs(id='filter',children=[
                    dcc.Tab(label='No Filter',value='No filter'),
                    dcc.Tab(label='IQR',value='IQR')
                ]),
                html.Br(),
                dcc.Graph(id='g1')
    ],id='outlier'),
    #Outlier End

    #Normality section
    html.H3("Normality Test | Data Transformation",style=style), html.Br(),
    html.Div([
                dcc.RadioItems(id='normradio',
                                   options=[{"label": "Dataset with Outliers", "value": "with-outliers"},
                                            {"label": "Dataset without Outliers", "value": "without-outliers"}],
                                   value='with-outliers', labelStyle={'display': 'block'}),
                    html.Br(),
                html.P("Transformation Type:"),
                dcc.RadioItems(id='transform',
                                   options=[{"label": "Quantile Transformation", "value": "quantile"},
                                            {"label": "Box-Cox", "value": "box"},
                                            {"label": "Reciprocal Transform", "value": "reciprocal"},
                                            {"label": "Square Root Transform", "value": "sqrt"}],
                                   value='with-outliers', labelStyle={'display': 'block'}),html.Br(),
                    html.P("Select Testing Procedure:"),
                    dcc.Dropdown(id='normdrop',options=[
                                    {'label': 'Shapiro"s Test', 'value': 'shapiro'},
                                    {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'},
                                    {'label': 'D"Agostino-Pearson Test', 'value': 'dap'},
                                    {'label': 'Histogram', 'value': 'hist'},
                                    {'label': 'QQ-Plot', 'value': 'qq'}
                                ],value='shapiro',clearable=False),
                    html.Br(),
                    html.Div(id='normal-out')
    ],id='normality'), html.Br(),
    #Normality End
    html.Br(),
    html.H3("Statistics",style=style), html.Br(),
    html.Div([
        html.Div(dash_table.DataTable(statistics.to_dict('records'),
                                      [{"name": i, "id": i} for i in statistics.columns],
                                      style_cell={'textAlign': 'left','padding':'5px'},
                                      style_header={'backgroundColor':  'rgb(220, 220, 220)','fontWeight': 'bold'}))
    ],id='statistics'), html.Br(),

    #Viz Start
    html.H3("Visualizations",style=style), html.Br(),
    html.P('Histogram Plot'),
    html.Div(children=[
                        dcc.RadioItems(id='vizradio',
                                   options=[{"label": "Dataset with Outliers", "value": "with-outliers"},
                                            {"label": "Dataset without Outliers", "value": "without-outliers"}],
                                   value='with-outliers', labelStyle={'display': 'block'}),
                        html.Br(),
                        html.P("Select a Numeric Feature:"),
                        dcc.Dropdown(id='vizdrop',
                                      options=[{'label':i,'value':i} for i in cat2],
                                      value='Base Pay',placeholder='Select one...'),html.Br(),
                        dcc.Slider(id='vizslide',
                                        min=10,
                                        max=200,
                                        value=50,
                                        step=1,
                                        marks={f'{i}':i for i in range(10,200,20)},
                                        tooltip={"placement": "bottom", "always_visible": False}),
                        html.Div(id='vizout'),
                        ],id='visuals'), html.Br(),

                        html.P('Bar Plot'),
                        html.Div([
                                html.P('Pick the Date Range'),
                               html.Div([ html.P('From:'),dcc.Dropdown(id='date1',
                                         options=[{'label': i, 'value': i} for i in range(2011,2020,1)],
                                         value=2011),
                                          html.P('To:'),
                                          dcc.Dropdown(id='date2',
                                                       options=[{'label': i, 'value': i} for i in range(2011, 2020, 1)],
                                                       value=2012)
                                          ],style={'width': '20%','display': 'inline-block'}),
                                html.P("Select Job Title"),
                                dcc.Dropdown(id='bardrop',
                                  options=[{'label': i, 'value': i} for i in df_clean['Job Title'].unique()],
                                  placeholder='Select one...',value='FIREFIGHTER'), html.Br(),
                        ]),
                        html.Div(id='barout'),html.Br(),

                        html.P('Heatmap Plot'),
                        html.Div(children=[
                        dcc.Checklist(id = "heatcheck",
                            options=[{'label': i, 'value': i} for i in cat2],
                           value=["Base Pay","Benefits"]),
                        html.Br(),
                        ]),
                        html.Div(id='heatmap'),html.Br(),

                        html.P('Correlation Matrix Plot'),
                        dcc.Checklist(id = "corrcheck",
                            options=[{'label': i, 'value': i} for i in cat2],
                           value=["Base Pay","Benefits"]),
                        html.Div(id='corrout'),html.Br(),

                        html.P('LINE PLOT'),html.Br(),
                        html.P('Select Feature:'),
                        dcc.Dropdown(id='linedrop',
                                     options=[{'label': i, 'value': i} for i in cat2],
                                     value='Base Pay', placeholder='Select one...',multi=False), html.Br(),

                        html.P('Pick Year'),
                               html.Div([ dcc.Dropdown(id='date3',
                                         options=[{'label': i, 'value': i} for i in range(2011,2020,1)],
                                         value=2011),
                        html.Div(id='lineout')]),html.Br(),

                        html.P('COUNT PLOT'),html.Br(),
                        html.Div(
                            dcc.Graph(figure=figcount)
                        ),

                        html.P('BOX PLOT'),html.Br(),
                        html.P('Select Feature:'),
                        dcc.Dropdown(id='boxdrop',
                                     options=[{'label': i, 'value': i} for i in cat2],
                                     value='Base Pay', placeholder='Select one...',multi=False), html.Br(),
                        html.Div(id='boxxout'),html.Br(),

                        html.P('VIOLIN PLOT'),html.Br(),
                        html.P('Select Feature:'),
                        dcc.Dropdown(id='viodrop',
                                     options=[{'label': i, 'value': i} for i in cat2],
                                     value='Base Pay', placeholder='Select one...',multi=False), html.Br(),
                        html.Div(id='vioout'),

                        html.P('REGRESSION PLOT'),html.Br(),
                        html.P('Select Feature to regress on:'),
                        dcc.Dropdown(id='regdrop',
                                     options=[{'label': i, 'value': i} for i in cat2],
                                     value='Base Pay', placeholder='Select one...',multi=False),
                        html.Div(id='regout'),

])

@my_app.callback(
    Output('g1','figure'),
    [Input('filter','value')]
)
def update(f):
    if f== 'IQR':
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Histogram(x=df_no_outlier['Total Pay & Benefits'],name='Total Pay & Benefits'),
            row=1, col=1)
        fig.add_trace(
            go.Box(y=df_no_outlier['Total Pay & Benefits'],name='Total Pay & Benefits'),
            row=1, col=2)
        fig.update_layout(
            title_text='Histogram & Box Plot of Total Pay & Benefits',  # title of plot
        )
        return fig
    else:
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Histogram(x=df_clean['Total Pay & Benefits'],name='Total Pay & Benefits'),
            row=1, col=1)
        fig.add_trace(
            go.Box(y=df_clean['Total Pay & Benefits'],name='Total Pay & Benefits'),
            row=1, col=2)
        fig.update_layout(
            title_text='Histogram & Box Plot of Total Pay & Benefits',  # title of plot
        )
        return fig


@my_app.callback(
    Output('normal-out','children'),
    [Input("normdrop","value"),
     Input('normradio','value'),Input('transform','value')]
)
def update(d,r,t):
    quantil = QuantileTransformer(output_distribution='normal')
    r= df_clean if r=="with-outliers" else df_no_outlier
    quantile_t=quantil.fit_transform(r['Total Pay & Benefits'].values.reshape(-1,1))
    box,_=stats.boxcox(r['Total Pay & Benefits'].values)
    sqr= np.sqrt(r['Total Pay & Benefits'].values)
    reciprocal= 1/r['Total Pay & Benefits'].values
    if d== 'shapiro':
        if t== 'quantile':
            return html.Div(str(shapiro_test(quantile_t, 'Total Pay & Benefits')))
        elif t== 'box':
            return html.Div(str(shapiro_test(box, 'Total Pay & Benefits')))
        elif t== 'sqrt':
            return html.Div(str(shapiro_test(sqr, 'Total Pay & Benefits')))
        elif t == 'reciprocal':
            return html.Div(str(shapiro_test(reciprocal, 'Total Pay & Benefits')))
        else:
            return html.Div(str(shapiro_test(r['Total Pay & Benefits'], 'Total Pay & Benefits')))
    elif d== 'ks':
        if t== 'quantile':
            return html.Div(str(ks_test(quantile_t, 'Total Pay & Benefits')))
        elif t== 'box':
            return html.Div(str(ks_test(box, 'Total Pay & Benefits')))
        elif t== 'sqrt':
            return html.Div(str(ks_test(sqr, 'Total Pay & Benefits')))
        elif t== 'reciprocal':
            return html.Div(str(ks_test(reciprocal, 'Total Pay & Benefits')))
        else:
            return html.Div(str(ks_test(r['Total Pay & Benefits'], 'Total Pay & Benefits')))
    elif d== 'dap':
        if t== 'quantile':
            return html.Div(str(da_k_squared_test(quantile_t, 'Total Pay & Benefits')))
        elif t== 'box':
            return html.Div(str(da_k_squared_test(box, 'Total Pay & Benefits')))
        elif t== 'sqrt':
            return html.Div(str(da_k_squared_test(sqr, 'Total Pay & Benefits')))
        elif t == 'reciprocal':
            return html.Div(str(da_k_squared_test(reciprocal, 'Total Pay & Benefits')))
        else:
            return html.Div(str(da_k_squared_test(r['Total Pay & Benefits'], 'Total Pay & Benefits')))
    elif d=='hist':
        if t== 'quantile':
            return dcc.Graph(figure=px.histogram(x=quantile_t.ravel(),nbins=50,template="plotly_dark"))
        elif t== 'box':
            return dcc.Graph(figure=px.histogram(x=box,nbins=50,template="plotly_dark"))
        elif t== 'sqrt':
            return dcc.Graph(figure=px.histogram(x=sqr,nbins=50,template="plotly_dark"))
        elif t == 'reciprocal':
            return dcc.Graph(figure=px.histogram(x=reciprocal,nbins=50,template="plotly_dark"))
        else:
            return dcc.Graph(figure=px.histogram(x=r['Total Pay & Benefits'], nbins=50,template="plotly_dark"))
    else:
        if t== 'quantile':
            return dcc.Graph(figure=qqp(quantile_t.ravel()))
        elif t== 'box':
            return dcc.Graph(figure=qqp(box))
        elif t== 'sqrt':
            return dcc.Graph(figure=qqp(sqr))
        elif t == 'reciprocal':
            return dcc.Graph(figure=qqp(reciprocal))
        else:
            return dcc.Graph(figure=qqp(r['Total Pay & Benefits']))
@my_app.callback(
    Output('overview','children'),
    [Input('overcheck','value'),
     Input('overcheck2','value'),
     Input('overradio','value'),
     Input('overin','value')]
)
def update(a,b,c,d):
    c = df_clean if c == "with-outliers" else df_no_outlier
    fig = px.pie(c[(c['Year']==2011) | (c['Year']==2012) |(c['Year']==2013) ], values=a, names=b,facet_col='Year',
                 hole=float(d) ,title=f"Pie Chart for {a} with respect to {b}",
                 color_discrete_sequence=px.colors.sequential.RdBu,template="plotly_dark")
    fig2 = px.pie(c[(c['Year'] == 2014) | (c['Year'] == 2015) | (c['Year'] == 2016)], values=a, names=b, facet_col='Year',
                 hole=float(d),
                 color_discrete_sequence=px.colors.sequential.RdBu,template="plotly_dark")
    fig3 = px.pie(c[(c['Year'] == 2017) | (c['Year'] == 2018) | (c['Year'] == 2019)], values=a, names=b,
                  facet_col='Year',
                  hole=float(d),
                  color_discrete_sequence=px.colors.sequential.RdBu,template="plotly_dark")
    return html.Div([dcc.Graph(figure=fig),dcc.Graph(figure=fig2),dcc.Graph(figure=fig3)])

@my_app.callback(
    Output('vizout','children'),
    [Input('vizradio','value'),Input('vizdrop','value'),
     Input('vizslide','value')]
)
def update(a,b,c):
    dftitle=a
    a = df_clean if a == "with-outliers" else df_no_outlier
    fig = px.histogram(x=a[b], nbins=int(c),title=f"Histogram Plot for {b} {[str(dftitle)]}",template="plotly_dark")
    return dcc.Graph(figure=fig)

@my_app.callback(
    Output('barout','children'),
    [Input('vizradio','value'),
     Input('bardrop','value'),
     Input('date1','value'),
     Input('date2','value')]
)
def update(a,b,c,d):
    dftitle = a
    a = df_clean if a == "with-outliers" else df_no_outlier
    df=a[a['Job Title']==b]
    df= df[(df['Year']==int(c)) | (df['Year']==int(d))]
    df=df[cat2].sum()
    fig = px.bar(x=df.index, y=df, text=df,title=f"Bar-Plot for {b} {[dftitle]}",template="plotly_dark")
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return dcc.Graph(figure=fig)

@my_app.callback(
    Output('heatmap','children'),
    [Input('vizradio','value'),Input('heatcheck','value')]
)
def update(a,b):
    dftitle = a
    a = df_clean if a == "with-outliers" else df_no_outlier
    df=a[b].corr()
    fig = px.imshow(df, text_auto=True,
                    title=f"Heatmap on Numeric Features {[dftitle]}",
                    color_continuous_scale=px.colors.sequential.Cividis_r,
                    template="plotly_dark")
    return dcc.Graph(figure=fig)

@my_app.callback(
    Output('corrout','children'),
    [Input('vizradio','value'),Input('corrcheck','value')]
)
def update(a,b):
    dftitle = a
    a = df_clean if a == "with-outliers" else df_no_outlier
    # fig = px.scatter_matrix(a[b],color=a.Status)
    fig = px.scatter_matrix(a,dimensions=b, color='Status',symbol='Status',
                            title=f"Scatter Matrix on Numeric Features Per Job Status{[dftitle]}",template="plotly_dark")
    fig.update_traces(diagonal_visible=False)

    return dcc.Graph(figure=fig)


@my_app.callback(
    Output('lineout','children'),
    [Input('vizradio','value'),Input('linedrop','value'),
     Input('date3','value')]
)
def update(a,b,year):
    dftitle = a
    a = df_clean if a == "with-outliers" else df_no_outlier
    a = a[a.Year == int(year)]
    fig = px.line(a,y=b,title=f"Line Plot for {b} for Year {year}",
                  template="plotly_dark")
    # fig= go.Figure(data=go.Scatter(x=a.Year, y=a[cat2]))
    return dcc.Graph(figure=fig)


@my_app.callback(
    Output('boxxout','children'),
    [Input('vizradio','value'),Input('boxdrop','value')]
)
def update(a,b):
    dftitle = a
    a = df_clean if a == "with-outliers" else df_no_outlier
    fig = px.box(a, x="Year", y=b, color="Status",template="plotly_dark",
                 title=f"Box Plot for {b} {[dftitle]}")
    return dcc.Graph(figure=fig)

@my_app.callback(
    Output('vioout','children'),
    [Input('vizradio','value'),Input('viodrop','value')]
)
def update(a,b):
    dftitle = a
    a = df_clean if a == "with-outliers" else df_no_outlier
    fig = px.violin(a, x="Year", y=b, color="Status",template="plotly_dark",
                 title=f"Box Plot for {b} {[dftitle]}")
    return dcc.Graph(figure=fig)

@my_app.callback(
    Output('regout','children'),
    [Input('vizradio','value'),Input('regdrop','value')]
)
def update(a,b):
    dftitle = a
    a = df_clean if a == "with-outliers" else df_no_outlier
    fig = px.scatter(a, y="Total Pay & Benefits", x=b,
                     trendline="ols",template="plotly_dark",
                     title=f"Regression Plot for Total Pay & Benefits Vs {b}")
    return dcc.Graph(figure=fig)




# Please Note: host='127.0.0.1' works for me else host='0.0.0.0' Thank you.
if __name__ == '__main__':
    my_app.run_server(
        port = random.randint(8000,9999), #8080
        host = "127.0.0.1"
    )

#df_clean[df_clean['Job Title']=="Electrical Transit System Mech"][cat2].sum()
#df_clean['Job Title'].unique()
#int(g[:4])