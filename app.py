
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import openpyxl



# Read Data
path = 'https://raw.githubusercontent.com/catarinacoelho1/ProjectDV/main/Source1.xlsx'
df = pd.read_excel(path)

# Filter the dataset by year to ensure accuracy of results
df = df.loc[(df['Year'] >= 2010) & (df['Year'] <= 2019)]


# Interactive Components

country_options = ['Portugal', 'Spain', 'Greece', 'Germany', 'Finland', 'Italy', 'Denmark', 'Austria',
                   'France', 'Norway', 'Poland','Belgium', 'Luxembourg', 'Sweden', 'Switzerland']

mental_indicators = ['Mental disorders', '% Anxiety disorders', '% Depressive disorders',
                     '%HappyPeople', 'Life satisfaction']


corr_indicators = ['Mental disorders', '% Anxiety disorders', '% Depressive disorders',
                   '%HappyPeople', 'Life satisfaction', 'Annual working hours per worker', 'Income']

factors = ['Annual working hours per worker', 'Productivity per hour worked']

indicator_options = [dict(label=indicator, value=indicator) for indicator in mental_indicators]

corr_options = [dict(label=influencing, value=influencing) for influencing in corr_indicators]

factors_options = [dict(label=factor, value=factor) for factor in factors]

slider_year = dcc.Slider(
        id='year_slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        marks={str(i): '{}'.format(str(i)) for i in
               [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]},
        value=df['Year'].min(),
        step=1)

dropdown_country = dcc.Dropdown(
        id='country_drop',
        options=country_options,
        value='Portugal'
    )

dropdown_country1 = dcc.Dropdown(
        id='country1',
        options=country_options,
        value='Germany'
    )

dropdown_country2 = dcc.Dropdown(
        id='country2',
        options=country_options,
        value='Portugal'
    )

dropdown_scope = dcc.Dropdown(
        id='scopes_option',
        clearable=False,
        searchable=False,
        options=[{'label': 'World', 'value': 'world'},
                 {'label': 'Europe', 'value': 'europe'},
                 {'label': 'Asia', 'value': 'asia'},
                 {'label': 'Africa', 'value': 'africa'},
                 {'label': 'North america', 'value': 'north america'},
                 {'label': 'South america', 'value': 'south america'}
                 ],
        value='world',
    )

dropdown_indicators = dcc.Dropdown(
        id='dropdown_indicator',
        options=indicator_options,
        value='Mental disorders'
    )

dropdown_influences = dcc.Dropdown(
        id='dropdown_influence',
        options=corr_options,
        value='Mental disorders'
    )


# THE APP ITSELF

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div((

    html.Div([
        html.Img(
            src=app.get_asset_url('Nova_IMS_Icon.png'),
            id="image", style={'width': '3%', 'margin': '0%'})
    ], className="header"),

    html.Div([
        html.H1(children='The Productivity-Mental Health Trade-off:'
                         ' An Exploration of the Effects of Working Hours on Employee Well-being'),
        html.H2(children='The Impact of Work Hours on Mental Health: A Study on Employee Productivity')
    ]),

    html.Div([
        html.Div([
            html.H3(id='choropleth_title'),
            html.Div([
                html.H3('Map Scope'),
                dropdown_scope
            ], className='six columns'),
            html.Div([
                html.H3('Year'),
                slider_year
            ], className='six columns')
        ], className='row'),
        html.Div([
            html.P('The interactive map below shows the distribution of average weekly working hours across countries worldwide. '
                   'You can filter the data by year using the dropdown menu above. '
                   'In the map, countries with missing or unavailable data are represented in black, '
                   'with darker greens indicating higher values and lighter greens indicating lower values. '
                   'It is clear countries in Asia have the highest average weekly working hours, which may be due to'
                   'poor working conditions. In contrast, European countries generally have lower average weekly working hours, '
                   'typically ranging from 30 to 40 hours per week (equivalent to 7-8 hours per day), '
                   'reflecting stronger labor laws and regulations.',
                   className='six columns'),
            html.Div([
                dcc.Graph(id='choropleth')
            ], className='six columns'),
        ], className='row')
    ], className='map'),

    html.Div([
        html.H4(id='Year_selected_1'),
        html.Div([
            html.P(
                'The graph below shows evidence that higher productivity does not correspond to more working hours. '
                'In fact, countries with the lowest annual working hours tend to have higher productivity, indicating a '
                'better performance at work. Countries like Germany, Denmark and Norway are known for providing good '
                'working environments, while "poorer" countries such as Portugal and Greece may have less favorable '
                'working conditions. This information is clearly displayed on the plot.', className='six columns'),
            dcc.Graph(id='line_graph'),
        ], id='Graph1', style={'width': '70%', 'padding-right': '15%', 'padding-left': '15%', 'height': '10%'}),
    ], id='2nd row', className='row2back'),

    html.Div([
        html.H5(id='Year_selected_2'),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Choose one indicator to compare', style={'color': '#000000'}),
                    dropdown_indicators,
                ], className='container'),
            ], style={'width': '30%', 'padding-right': '35%', 'padding-left': '35%', 'padding-bottom': '0%',}),
            html.Div((
                dcc.Graph(id='bar_graph'),
            ), style={'width': '50%', 'padding-right': '25%', 'padding-left': '25%'}),
        ]),
    ], id='3rd row', className='row1back'),

    html.Div([
        html.H6(id='Year_selected_3'),
        html.Div([
           html.Div([
                html.Label('Choose one indicator to compare', style={'color': '#000000'}),
                dropdown_influences,
            ], className='container'),], style={'width': '50%','padding-right':'25%','padding-left':'25%','padding-bottom':'2%'}),
        html.Div([
            html.Div([
                dcc.Graph(id='box_graph'),
            ], style={'display': 'flex', 'width': '100%', 'padding-right': '0%', 'padding-left': '0%'}),
          #'width': '50%', 'padding-right': '1%', 'padding-left': '1%', 'float': 'left'
            html.Div([
                dcc.Graph(id='cor_graph'),
            ], style={'width': '50%', 'padding-right': '1%', 'padding-left': '1%', 'float': 'right'})
        ], style={'display': 'flex', 'width': '100%', 'padding-right': '0%', 'padding-left': '0%'}),
    ], id='4th row', className='row2back'),

    html.Div([
        html.H6(id='Year_selected_4'),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Select the first country', style={'color': '#000000'}),
                    html.Br(),
                    dropdown_country1,
                    html.Br(),
                ]),
                html.Div([
                    html.Label(id='Working_1'),
                    html.Br(),
                ], style={'color': '#000000'}, className='ranks_box'),
                html.Div([
                    html.Label(id='Working_2'),
                    html.Br(),
                ], style={'color': '#000000'}, className='ranks_box'),
            ], id='c1', style={'width': '25%', 'padding-right': '1%', 'padding-left': '25%'}),
            html.Div([
                html.Div([
                    html.Label('Select the second country', style={'color': '#000000'}, ),
                    html.Br(),
                    dropdown_country2,
                    html.Br(),
                ]),
                html.Div([
                    html.Label(id='Working_3'),
                    html.Br(),
                ], style={'color': '#000000'}, className='ranks_box'),
                html.Div([
                    html.Label(id='Working_4'),
                    html.Br(),
                ], style={'color': '#000000'}, className='ranks_box')
            ], id='c2', style={'width': '25%', 'padding-right': '25%', 'padding-left': '1%'}),
        ], style={'display': 'flex', 'padding-bottom': '5%'}),
        html.Div([
            dcc.Graph(id='polar-graph')
        ], id='polar', style={'width': '50%', 'padding-right': '25%', 'padding-left': '25%', }),
    ], id='5th row', className='row1back'),

    html.Div([
        html.Div([
            html.H6("Authors"),
            dcc.Markdown("""\
              Catarina Arrimar Coelho (r20191239@novaims.unl.pt)    
        """, style={"text-align": "center", "font-size": "13pt", 'padding-right': '25%', 'padding-left': '25%'}),
        ]),
        html.Div([
            html.H6("Sources"),
            dcc.Markdown(
                """Dash Enterprise App Gallery: https://dash.gallery/Portal/ \n  \n DataSource https://ourworldindata.org/""",
                style={"text-align": "center", "font-size": "13pt", "white-space": "pre", 'padding-right': '25%',
                       'padding-left': '25%'}),
        ]),
    ], className='lastrow')

))

# Scatter Plot
@app.callback(
    Output("line_graph", "figure"),
    Input('year_slider', 'value')
)

def plots(year):

    df_scatter = df.loc[df['Year'] == year]

    df_scatter = df_scatter.loc[
        (df['Country'] == 'Portugal') | (
                df_scatter['Country'] == 'Spain') | (df_scatter['Country'] == 'Norway') |
                (df_scatter['Country'] == 'Finland') | (
                df_scatter['Country'] == 'France') | (df_scatter['Country'] == 'Germany') | (
                df_scatter['Country'] == 'Greece') | (df_scatter['Country'] == 'Denmark') | (
                df_scatter['Country'] == 'Italy') | (df_scatter['Country'] == 'Austria') |
                (df_scatter['Country'] == 'Luxembourg') | (df_scatter['Country'] == 'Belgium') |
                (df_scatter['Country'] == 'Switzerland') | (
                df_scatter['Country'] == 'Sweden')
        ]

    color_map = {
        'Portugal': '#009E73',
        'Spain': '#F0E442',
        'Norway': '#0072B2',
        'Finland': '#D55E00',
        'France': '#CC79A7',
        'Germany': '#56B4E9',
        'Greece': '#E69F00',
        'Denmark': '#999999',
        'Italy': '#A1C9F4',
        'Austria': '#FDBF6F',
        'Luxembourg': '#B3DE69',
        'Belgium': '#FCCDE5',
        'Switzerland': '#D2B48C',
        'Sweden': '#EFEFEF'
    }

    df_scatter = df_scatter[df_scatter['Country'].isin(color_map.keys())]

    x_scatter = df_scatter['Productivity per hour worked']
    y_scatter = df_scatter['Annual working hours per worker']

    fig = px.scatter(df_scatter,
                     x=x_scatter, y=y_scatter,
                     animation_group='Country', text='Country',
                     color='Country',
                     color_discrete_map=color_map
                     )

    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    fig.update_traces(marker=dict(size=15))

    return fig

# Map Choropleth

@app.callback(
        [Output("choropleth", "figure"),
         Output('choropleth_title', "children")],
        [
         Input('year_slider', 'value'),
         Input('scopes_option', 'value')
        ]
)

def map(year, continent):
    #df_happy_0 = df.loc[df['Year'] <= 2019]
    df_happy_0 = df.loc[df['Year'] == year]

    data_choropleth = dict(type='choropleth',
                           locations=df_happy_0['Country'],
                           locationmode='country names',
                           z=df_happy_0['Annual working hours per worker']/52,
                           text=df_happy_0['Country'],
                           colorscale='greens',
                           hovertemplate='Country: %{text} <br>' + 'Weekly working hours per worker' + ': %{z}',
                           )

    layout_choropleth = dict(geo=dict(scope=continent,
                                      projection=dict(type='equirectangular'),
                                      landcolor='black',
                                      lakecolor='black',
                                      showocean=True,
                                      oceancolor='azure',
                                      bgcolor='#ffffff'
                                      ))

    return go.Figure(data=data_choropleth, layout=layout_choropleth),  \
           ' Average Weekly Working Hours Across Countries Worldwide ' + str(year)

#Bar chart

@app.callback(
    Output("bar_graph", "figure"),
    Input('year_slider', 'value'),
    Input('dropdown_indicator', 'value')
)

def barchart(year, indicator):

  #df_box = df[(df['Year'] <= 2021)]
  df_box = df

  df_box = df_box[df_box['Year'] == year]
  df_box = df_box.loc[
        (df_box['Country'] == 'Portugal') | (
                    df_box['Country'] == 'Spain') | (df_box['Country'] == 'Norway') | (
                    df_box['Country'] == 'Finland') | (
                    df_box['Country'] == 'France') | (df_box['Country'] == 'Germany') | (
                    df_box['Country'] == 'Greece') | (df_box['Country'] == 'Denmark') | (
                    df_box['Country'] == 'Italy') | (df_box['Country'] == 'Austria') | (
                    df_box['Country'] == 'Luxembourg') | (df_box['Country'] == 'Belgium') | (
                    df_box['Country'] == 'Switzerland') | (
                    df_box['Country'] == 'Sweden')
        ]

  names = df_box['Country']
  values = df_box[indicator]

  df_box1 = df_box.sort_values(indicator, ascending=False)

  fig_box = px.bar(df_box1, x=names, y=values, color=values,
                 color_continuous_scale='YlGnBu')

  fig_box.update_layout(
      plot_bgcolor='white',
      xaxis=dict(showgrid=False, title = "Countries", categoryorder='total descending'),
      yaxis=dict(showgrid=False, title = indicator)
  )


  return fig_box


# Box scatter plot


@app.callback(
    Output('box_graph', 'figure'),
    Input('year_slider', 'value'),
    Input('dropdown_influence', 'value')
)
def box_graph_function(year, indicator):
    df_box = df
    df_box = df_box[df_box['Year'] == year]
    df_box = df_box.loc[
        (df_box['Country'] == 'Portugal') | (
                df_box['Country'] == 'Spain') | (df_box['Country'] == 'Norway') | (
                df_box['Country'] == 'Finland') | (
                df_box['Country'] == 'France') | (df_box['Country'] == 'Germany') | (
                df_box['Country'] == 'Greece') | (df_box['Country'] == 'Denmark') | (
                df_box['Country'] == 'Italy') | (df_box['Country'] == 'Austria') | (
                df_box['Country'] == 'Luxembourg') | (df_box['Country'] == 'Belgium') | (
                df_box['Country'] == 'Switzerland') | (
                df_box['Country'] == 'Sweden')]

    x_box = df_box['Productivity per hour worked']
    y_box = df_box[indicator]

    fig_box = px.scatter(df_box, x=x_box, y=y_box, 
                         animation_group='Country', text="Country", trendline = "OLS",
                         marginal_x='box', marginal_y='box', template="simple_white",
                         color_discrete_sequence=["#4CB5F5"]
                         )

    #fig_box.update_layout(legend=dict(orientation="h", xanchor='center', x=0.8, yanchor='top', y=-0.4))

    fig_box.update_layout(title={
        'text': "Relationship between working hours and a chosen Indicator", 'x': 0.45, 'y': 0.97,
        'xanchor': 'center', 'yanchor': 'top'},
         margin={'l': 40, 'b': 40, 't': 30, 'r': 0},      
         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', titlefont=dict(color='black'),
         xaxis=dict(titlefont=dict(color='black'), tickfont=dict(color='black')),
         yaxis=dict(titlefont=dict(color='black'), tickfont=dict(color='black')),
         legend=dict(orientation="h", xanchor='center', x=0.8, yanchor='top', y=-0.4))
    
    fig_box.update_traces(marker=dict(size=7.5))

    return fig_box


#Corr heatmap

@app.callback(
    Output('cor_graph', 'figure'),
    Input('year_slider', 'value')
)

def cor_graph_function(year):

    corr_list = ['Productivity per hour worked', 'Mental disorders', '% Anxiety disorders',
                 '% Depressive disorders', '%HappyPeople', 'Life satisfaction', 'Paid work',
                 'Personal care', 'Sports', 'Seeing friends', 'TV and Radio',
                 'Annual working hours per worker','Income']

    df_corr_r = df[df['Year'] == year][corr_list]
    df_corr_round = df_corr_r.corr()[['Productivity per hour worked']].T[corr_indicators].T.round(2)

    fig_cor = ff.create_annotated_heatmap(
        z=df_corr_round.to_numpy(),
        x=df_corr_round.columns.tolist(),
        y=df_corr_round.index.tolist(),
        zmax=1, zmin=-1,
        showscale=True,
        hoverongaps=True,
        ygap=3,
        colorscale='YlGnBu'
    )

    fig_cor.update_layout(yaxis=dict(showgrid=False, autorange='reversed'), xaxis=dict(showgrid=False),

                          legend=dict(
                              orientation="h",
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=1
                          ))
    fig_cor.update_layout(xaxis_tickangle=0)
    fig_cor.update_layout(title={
        'text': "Correlation Heatmap of Working Hours", 'x': 0.57, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', titlefont=dict(color='black'),
                          xaxis=dict(color='black'), yaxis=dict(color='black'))

    return fig_cor

# Polar plot

@app.callback(
    Output('polar-graph', 'figure'),
    [Input("country1", "value"),
     Input("country2", "value")]
)
def polar_function(country1, country2):
    df_country1 = df.loc[df['Year'] == 2017]
    df_country1 = df_country1.loc[df['Country'] == country1]
    df_country1 = df_country1[
        ['Shopping', 'Personal care', 'TV and Radio', 'Seeing friends', 'Other leisure activities', 'Sports',
         'Paid work']]
    time_use1 = pd.DataFrame(df_country1)
    time_use1 = time_use1.T

    r1 = [time_use1.iat[0, 0], time_use1.iat[1, 0], time_use1.iat[2, 0], time_use1.iat[3, 0], time_use1.iat[4, 0],
          time_use1.iat[5, 0]]

    df_country2 = df.loc[df['Year'] == 2017]
    df_country2 = df_country2.loc[df['Country'] == country2]
    df_country2 = df_country2[
        ['Shopping', 'Personal care', 'TV and Radio', 'Seeing friends', 'Other leisure activities', 'Sports',
         'Paid work']]
    time_use2 = pd.DataFrame(df_country2)
    time_use2 = time_use2.T

    r2 = [time_use2.iat[0, 0], time_use2.iat[1, 0], time_use2.iat[2, 0], time_use2.iat[3, 0], time_use2.iat[4, 0],
          time_use2.iat[5, 0]]

    n = ('Shopping', 'Personal care', 'TV and Radio', 'Seeing friends', 'Other leisure activities', 'Sports')

    fig = go.Figure(data=go.Scatterpolar(
        r=r1,
        theta=n,
        fill='toself',
        marker_color='#4CB5F5',
        opacity=1,
        hoverinfo="text",
        name=country1
    ))

    fig.add_trace(go.Scatterpolar(
        r=r2,
        theta=n,
        fill='toself',
        marker_color='#6AB187',
        opacity=1,
        hoverinfo="text",
        name=country2
    ))

    return fig

# Titles Definition

@app.callback(
    [
        Output("Working_1", "children"),
        Output("Working_2", "children")
    ],
    [
        Input("country1", "value"),
        Input("year_slider", "value")
    ]
)
def indicator1(country, year):
    df_loc = df.loc[df['Country'] == country].groupby('Year').sum().reset_index()

    value_1 = round(df_loc.loc[df_loc['Year'] == year][factors[0]].values[0], 2)
    value_2 = round(df_loc.loc[df_loc['Year'] == year][factors[1]].values[0], 2)


    return str(year)+' '+str(factors[0])+' of '+str(country)+': ' + str(value_1), \
           str(year)+' '+str(factors[1])+' of '+str(country)+': ' + str(value_2),

@app.callback(
    [
        Output("Working_3", "children"),
        Output("Working_4", "children"),
        Output("Year_selected_1", "children"),
        Output("Year_selected_2", "children"),
        Output("Year_selected_3", "children"),
        Output("Year_selected_4", "children"),
    ],
    [
        Input("country2", "value"),
        Input("year_slider", "value"),
    ]
)
def indicator2(country, year):
    df_loc = df.loc[df['Country'] == country].groupby('Year').sum().reset_index()

    value_1 = round(df_loc.loc[df_loc['Year'] == year][factors[0]].values[0], 2)
    value_2 = round(df_loc.loc[df_loc['Year'] == year][factors[1]].values[0], 2)

    return str(year) + ' ' + str(factors[0]) + ' of ' + str(country) + ': ' + str(value_1), \
           str(year) + ' ' + str(factors[1]) + ' of ' + str(country) + ': ' + str(value_2), \
           str(year) + ' Relationship between weekly working hours and productivity at work ',\
           str(year) + ' What are the countries with the highest number of records of mental illnesses? \n ' \
           'Are they the ones where people work more hours per year? ', \
           str(year) + ' What factors influence productivity at work? ', \
           ' Compare the time spent (min per day) in others activities besides work of two Countries in ' + str(year), \


if __name__ == '__main__':
    app.run_server(debug=True)
