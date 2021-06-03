# -*- coding: utf-8 -*-
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from functools import partial

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from dash.exceptions import PreventUpdate

external_stylesheets = [dbc.themes.LUX]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#For initial run
freeVariables = {'tau0': 0.1,
 'T0': 2000,
 'iz_params': np.array([[ 1.0e+02, -6.0e+01, -4.0e+01,  3.5e+01,  3.0e-02, -2.0e+00,
         -5.0e+01,  1.0e+02,  7.0e-01],
        [ 5.0e+01, -8.0e+01, -2.5e+01,  4.0e+01,  1.0e-02, -2.0e+01,
         -5.5e+01,  1.5e+02,  1.0e+00],
        [ 5.0e+01, -8.0e+01, -2.5e+01,  4.0e+01,  1.0e-02, -2.0e+01,
         -5.5e+01,  1.5e+02,  1.0e+00],
        [ 1.0e+02, -6.0e+01, -4.0e+01,  3.5e+01,  3.0e-02, -2.0e+00,
         -5.0e+01,  1.0e+02,  7.0e-01],
        [ 1.0e+02, -6.0e+01, -4.0e+01,  3.5e+01,  3.0e-02, -2.0e+00,
         -5.0e+01,  1.0e+02,  7.0e-01],
        [ 1.0e+02, -6.0e+01, -4.0e+01,  3.5e+01,  3.0e-02, -2.0e+00,
         -5.0e+01,  1.0e+02,  7.0e-01],
        [ 1.0e+02, -6.0e+01, -4.0e+01,  3.5e+01,  3.0e-02, -2.0e+00,
         -5.0e+01,  1.0e+02,  7.0e-01]]),
    'E': np.array([  0,   0,   0, 300, 300, 500,   0]),

 'psp_amp0': 1000.0,
 'psp_decay0': 100,
 'w': np.zeros((9, 9)),

}

freeVariables['w'][0, 1] = 1 * 100
freeVariables['w'][1, 3] = -1 * 125


# hyperdirect pathway
freeVariables['w'][0, 6] = 1 * 90
freeVariables['w'][6, 3] = 1 * 50


# plotting functions
t = np.arange(0, freeVariables['T0'], freeVariables['tau0'])
n_steps = t.shape[0]
n_cells = freeVariables['iz_params'].shape[0]

# memory allocation for neurons
v = np.zeros((n_cells, n_steps))
u = np.zeros((n_cells, n_steps))
g = np.zeros((n_cells, n_steps))
spike = np.zeros((n_cells, n_steps))
v[:, 0] = freeVariables['iz_params'][:, 1] + np.random.rand(n_cells) * 100
I_net = np.zeros((n_cells, n_steps))
I_in = np.zeros(n_steps)
I_in[5000:] = 5e1

for i in range(1, n_steps):

    dt = t[i] - t[i - 1]

    I_net = np.zeros((n_cells, n_steps))
    for jj in range(n_cells):
        for kk in range(n_cells):
            if jj != kk:
                I_net[jj, i - 1] += freeVariables['w'][kk, jj] * g[kk, i - 1]
            if jj == 0:
                I_net[jj, i - 1] += I_in[i - 1]

        C = freeVariables['iz_params'][jj, 0]
        vr = freeVariables['iz_params'][jj, 1]
        vt = freeVariables['iz_params'][jj, 2]
        vpeak = freeVariables['iz_params'][jj, 3]
        a = freeVariables['iz_params'][jj, 4]
        b = freeVariables['iz_params'][jj, 5]
        c = freeVariables['iz_params'][jj, 6]
        d = freeVariables['iz_params'][jj, 7]
        k = freeVariables['iz_params'][jj, 8]

        dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) - u[jj, i - 1] +
                I_net[jj, i - 1] + freeVariables['E'][jj]) / C
        dudt = a * (b * (v[jj, i - 1] - vr) - u[jj, i - 1])


        dgdt = (-g[jj, i - 1] + freeVariables['psp_amp0'] * spike[jj, i - 1]) / freeVariables['psp_decay0']

        v[jj, i] = v[jj, i - 1] + dvdt * dt
        u[jj, i] = u[jj, i - 1] + dudt * dt
        g[jj, i] = g[jj, i - 1] + dgdt * dt

        if v[jj, i] >= vpeak:
            v[jj, i - 1] = vpeak
            v[jj, i] = c
            u[jj, i] = u[jj, i] + d
            spike[jj, i] = 1




#iz_param_matrix = None

# tau = 0.1
# T = 2000
# t = np.arange(0, T, tau)
# n_steps = t.shape[0]
# izparam = None
# izRow = 0
# izCol = 0
# iz_param_indices = []



#Helper functions:

#makes an item for a single variable
def make_item(i):
    v = i[0:len(i)-1]
    return html.Div(
        [
        dbc.Button(
            v,
            id=f"{i}-toggle",

            color="primary",
            outline= True,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody([html.Div("Value: "),
                                   html.Div(freeVariables[i],id=f"{i}label"),
                                   dbc.Collapse([
                                       #input_array(v, 1,1,np.array([freeVariables[i]]),'number'),
                                       dbc.Input(placeholder=f"{freeVariables[i]}", type="number", min=0,
                                                                                                  step=1,
                                                 id=f"{i}val", ),
                                       dbc.Button("submit", id=f"{i}valbtn",color="success", className="mr-1")
                                       ]
                                       ,id=f"{i}_changeVal_Collapse",
                                      className="w-50",
                                   ),
                                   dbc.Button(
                                       "Edit",
                                       id=f"{i}_changeVal_btn",
                                       className="mb-3",
                                       color="secondary ",
                                       outline=True,
                                   ),

                                  ]),className="w-50",),
                id=f"collapse-{i}",
        ),
    ])

#Changes a flatten dictionary to a matrix
def changeDictToMat(d, maxCol):
    if(not maxCol):
        j = np.zeros(len(d))
        col = 0
        for i in d.values():
            j[col] = i
            col += 1
        return j

    maxRow = int(len(d) / maxCol)
    j = np.zeros((maxRow , maxCol ))
    col = 0
    row = 0

    for i in d.values():
        j[row][col] = i
        col += 1
        if col == maxCol:
            col = 0
            row += 1
    return j

#makes a two dimensional item
def make_twoDimMatrix_item(i,maxRow,maxCol):
    v = i
    return html.Div(
        [
            dbc.Button(
                v,
                id=f"{i}-toggle",

                color="primary",
                outline=True,
            ),
            dbc.Collapse(
                dbc.Card(dbc.CardBody([html.Div("This variable is a matrix "),
                                       html.Div(dbc.Table.from_dataframe(pd.DataFrame(freeVariables[i])), id = f"{v}label"),
                                       dbc.Collapse([
                                           dbc.Input(placeholder="Enter Row Number", type="number", min=0, max=maxRow-1,
                                                     step=1,
                                                     id = f"{i}row",),
                                           dbc.Input(placeholder="Enter Column Number", type="number", min=0,
                                                     max=maxCol-1,
                                                     step=1,
                                                     id = f"{i}col",)
                                           ,
                                           dbc.Button(
                                               "Enter",
                                               id=f"{i}_matVal_btn",
                                               className="mb-3",
                                               color="secondary ",
                                               outline=True,
                                           ),

                                       dbc.Collapse(
                                           # input_array(i,maxRow,maxCol,freeVariables[i]),
                                           # html.Div(id = f"intputVal_{i}"),
                                                    id=f"{i}_matVal",
                                                    ),
                                           html.Div(id=f"intputVal_{i}"),
                                           dbc.Button("submit",id=f"{i}valbtn", color="success", className="mr-1"),

                                       ],
                                           id=f"{i}_changeVal_Collapse",
                                           className="w-50",
                                       ),

                                       dbc.Button(
                                           "Edit",
                                           id=f"{i}_changeVal_btn",
                                           className="mb-3",
                                           color="secondary ",
                                           outline=True,
                                       ),

                                       ]), ),
                id=f"collapse-{i}",
            ),
        ])

#makes an one dimensional matrix item
def make_oneDimMatrix_item(i,maxCol):
    v = i
    return html.Div(
        [
            dbc.Button(
                v,
                id=f"{i}-toggle",

                color="primary",
                outline=True,
            ),
            dbc.Collapse(
                dbc.Card(dbc.CardBody([html.Div("This variable is a matrix "),
                                       html.Div(dbc.Table.from_dataframe(pd.DataFrame(freeVariables[i])), id = f"{v}label"),
                                       dbc.Collapse([

                                           dbc.Input(placeholder="Enter Column Number", type="number", min=0,
                                                     max=maxCol-1,
                                                     step=1,
                                                     id = f"{i}col",)
                                           ,
                                           dbc.Button(
                                               "Enter",
                                               id=f"{i}_OnematVal_btn",
                                               className="mb-3",
                                               color="secondary ",
                                               outline=True,
                                           ),

                                       dbc.Collapse(
                                           # input_array(i,maxRow,maxCol,freeVariables[i]),
                                           # html.Div(id = f"intputVal_{i}"),
                                                    id=f"{i}_matVal",
                                                    ),
                                           html.Div(id=f"intputOneVal_{i}"),
                                           dbc.Button("submit", id=f"{i}valbtn", color="success", className="mr-1"),

                                       ],
                                           id=f"{i}_changeVal_Collapse",
                                           className="w-50",
                                       ),

                                       dbc.Button(
                                           "Edit",
                                           id=f"{i}_changeVal_btn",
                                           className="mb-3",
                                           color="secondary ",
                                           outline=True,
                                       ),

                                       ]), ),
                id=f"collapse-{i}",
            ),
        ])


#Renders front end:
app.layout = html.Div(children=[

    html.H1(children='Neural Activity Ploter'),
    html.Div(children='''
        An interactive graphing app for the basal ganglia functions.
    '''),
    html.Div(id='output-div'),
    html.Br(),



html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(make_item("T0")), width=6),
                dbc.Col(html.Div(make_item("tau0")), width=6),

            ]
        ),
        html.Br(),
        dbc.Row(
            [

                dbc.Col(html.Div(make_item("psp_amp0")), width=6),
                dbc.Col(html.Div(make_item("psp_decay0")), width=6),
     ]),

        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.Div(make_twoDimMatrix_item("iz_params",7,9)), ),
                dbc.Col(html.Div(make_twoDimMatrix_item("w",7,9)), ),
     ]),

        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.Div(make_oneDimMatrix_item("E",7)), ),
    ]),
    ]),

    html.Br(),
    html.Br(),
    dbc.Button('Submit Changes', id='submitChange'),
    dcc.Dropdown(
        id='ChooseGraph',
        options=[
            {'label': 'CTX', 'value': 'ctx'},
            {'label': 'STN', 'value': 'stn'},
            {'label': 'D1', 'value': 'd1'},
            {'label': 'D2', 'value': 'd2'},
            {'label': 'GPI', 'value': 'gpi'},
            {'label': 'GPE', 'value': 'gpe'},
            {'label': 'THAL', 'value': 'thal'}
        ],
        placeholder="Select a graph",
    ),
    html.Div(id='output-graph-div'),


    #allocate memory on client side to store changes in values

    html.Div([dcc.Store(id=i , data = freeVariables[i]) for i in ['tau0','T0','psp_amp0','psp_decay0']]),
    html.Div([dcc.Store(id=i, data=dict(enumerate(freeVariables[i].flatten(), 1))) for i in ['iz_params', 'E', 'w']]),
    html.Div([dcc.Store(id=j, data=dict(enumerate(i.flatten(), 1))) for i,j in zip([t,v,u,g,spike,I_net,I_in],["t","v","u","g","spike","I_net","I_in"])]),
    html.Div([dcc.Store(id=j, data=i) for i,j in zip([n_steps,n_cells],["n_steps","n_cells"])]),
])

#Store changes in single variables in the memory
def storeChangedVals(n,val,oldVal):
    return val if n and val else oldVal

storeChangedVals = \
        {f'storeChangedVals{i}': app.callback(Output(component_id=i, component_property='data'),
                                            Input(component_id=f'{i}valbtn', component_property='n_clicks'),
                                            State(component_id=f'{i}val', component_property='value'),
                                            State(component_id=i, component_property='data'),)(partial(storeChangedVals))
        for i in ["T0","tau0","psp_amp0","psp_decay0"]}


#Stores changes in matrices
def storeOneDimChangedVals(n,newVal,oldVal):
    data = oldVal
    if(n and (type(newVal) is list)):
        for i in newVal:
            if ('value' in i['props'].keys()):
                id = str(int(i['props']['id'][-1])+1)
                val = i['props']['value']
                data[id] = val

        return data
    return oldVal
storeOneDimChangedVals = \
        {f'storeOneDimChangedVals{i}': app.callback(Output(component_id=i, component_property='data'),
                                                    Input(f"{i}valbtn", "n_clicks"),
                                                    State(f"intputOneVal_{i}", "children"),
                                                    State(component_id=i, component_property='data'),
                                                    prevent_initial_call=True)(partial(storeOneDimChangedVals))
         for i in ["E"]}

def storeTwoDimChangedVals(n,newVal,oldVal,maxCol):
    numCol = maxCol+1
    data = oldVal
    if(n and (type(newVal) is list)):
        for i in newVal:
            if('value' in i['props'].keys()):
                row = int(i['props']['id'][-2])
                col = int(i['props']['id'][-1]) + 1
                id = (row*numCol) + col
                val = i['props']['value']
                data[id] = val
        return data
    return oldVal

storeTwoDimChangedVals = \
        {f'storeTwoDimChangedVals{i}': app.callback(Output(component_id=i, component_property='data'),
                                                    Input(f"{i}valbtn", "n_clicks"),
                                                    State(f"intputVal_{i}", "children"),
                                                    State(component_id=i, component_property='data'),
                                                    State(f"{i}col", "max"),
                                                    prevent_initial_call=True)(partial(storeTwoDimChangedVals))
         for i in ["iz_params","w"]}

#Render row/column input values for matrices
def makeOneDimMatrix(n,m,col,maxCol,theDiv,i,data):
    ctx = dash.callback_context

    mat = i[13:]

    if ctx.triggered[0]['prop_id'].split('.')[0] == mat:
        return 'please enter values'

    id = col + 1 if col is not None else None

    if(n):
        mat = i[13:]

        if col is None or col>maxCol or col<0 or not isinstance(col, int):
            raise PreventUpdate

        elif theDiv is None or theDiv == "please enter values":
            return [dbc.Input(placeholder=f"column {col} : {data[str(id)]}",
                         id=f"{mat}{col}",
                              type = "number", )]

        else:
            for i in theDiv:
                rowCol = i['props']['id'][len(mat):]
                if(rowCol == str(col)):
                    raise PreventUpdate

            x = theDiv
            x.append(dbc.Input(placeholder=f"column {col} : {data[str(id)]}",
                                               id=f"{mat}{col}",
                               type = "number",
                                  ))

            return x

    else:
        return 'please enter values'

makeOneDimMatrixValues = \
        {f'makeOneDimMatrix{i}': app.callback(Output(f"intputOneVal_{i}", "children"),
                                Input(f"{i}_OnematVal_btn", "n_clicks"),
                                Input(component_id= i , component_property='data'),
                                State(f"{i}col", "value"),
                                State(f"{i}col", "max"),
                                State(f"intputOneVal_{i}", "children"),
                                State(f"intputOneVal_{i}", "id"),
                                State(i , "data"),prevent_initial_call=True)(partial(makeOneDimMatrix))
         for i in ['E']}

def makeTwoDimMatrix(n,m,row,maxRow,col,maxCol,theDiv,i,data):
    ctx = dash.callback_context

    mat = i[10:]
    id = (row * (maxCol+1)) + (col+1)
    if ctx.triggered[0]['prop_id'].split('.')[0] == mat:
        return 'please enter values'
    id = (row * (maxCol + 1)) + (col + 1) if row is not None and col is not None else None
    if(n):
        mat = i[10:]

        if row is None or col is None or row<0 or col<0 or row>maxRow or col>maxCol or not isinstance(row, int) or not isinstance(col, int):
            raise PreventUpdate

        elif theDiv is None or theDiv == "please enter values":
            return [dbc.Input(placeholder=f"row {row} column {col} : {data[str(id)]}",
                         id=f"{mat}{row}{col}",
                              type = "number", )]

        else:
            for i in theDiv:
                rowCol = i['props']['id'][len(mat):]
                if(rowCol == str(row)+str(col)):
                    raise PreventUpdate

            x = theDiv
            x.append(dbc.Input(placeholder=f"row {row} column {col} : {data[str(id)]}",
                                               id=f"{mat}{row}{col}",
                               type = "number",
                                  ))

            return x


    else:
        return 'please enter values'

makeTwoDimMatrixValues = \
        {f'makeTwoDimMatrix{i}': app.callback(Output(f"intputVal_{i}", "children"),
                                            Input(f"{i}_matVal_btn", "n_clicks"),
                                            Input(component_id= i , component_property='data'),
                                            State(f"{i}row", "value"),
                                            State(f"{i}row", "max"),
                                            State(f"{i}col", "value"),
                                            State(f"{i}col", "max"),
                                            State(f"intputVal_{i}", "children"),
                                            State(f"intputVal_{i}", "id"),
                                            State(i , "data"),prevent_initial_call=True)(partial(makeTwoDimMatrix))
         for i in ['iz_params','w']}


#Renders collapse frontend functionality for each parameter
def showSingleInput(n,is_open):
    if n:
        return not is_open
    return is_open

makeEditCollapse = \
        {f'showSingleInput{k}': app.callback(Output(f"{k}_changeVal_Collapse", "is_open"),
                                            Input(f"{k}_changeVal_btn", "n_clicks"),
                                            State(f"{k}_changeVal_Collapse", "is_open"),)(partial(showSingleInput))
         for k in ["T0","tau0","psp_amp0","psp_decay0",'iz_params','w','E']}




#Keeps a single collapse open at a time
@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in ["T0","tau0","psp_amp0","psp_decay0","iz_params","w","E"]],
    [Input(f"{i}-toggle", "n_clicks") for i in ["T0","tau0","psp_amp0","psp_decay0","iz_params","w","E"]],
    [State(f"collapse-{i}", "is_open") for i in ["T0","tau0","psp_amp0","psp_decay0","iz_params","w","E"]],
)
def toggle_accordion(n1, n2,n3,n4,n5,n6,n7, is_open1, is_open2,is_open3,is_open4,is_open5,is_open6,is_open7):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False, False, False, False, False

    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "T0-toggle" and n1:
        return not is_open1, False,False,False,False, False, False
    elif button_id == "tau0-toggle" and n2:
        return False, not is_open2 ,False,False, False, False, False
    elif button_id == "psp_amp0-toggle" and n3:
        return False, False, not is_open3, False, False, False, False
    elif button_id == "psp_decay0-toggle" and n4:
        return False, False, False, not is_open4, False, False, False
    elif button_id == "iz_params-toggle" and n5:
        return False, False, False, False, not is_open5, False, False
    elif button_id == "w-toggle" and n6:
        return False, False, False, False, False, not is_open6, False
    elif button_id == "E-toggle" and n7:
        return False, False, False, False, False, False, not is_open7

    return False, False, False, False, False, False, False


@app.callback(
    Output(component_id='output-graph-div', component_property='children'),
    [Input(component_id='ChooseGraph', component_property='value')],
    Input('v','data'),
    Input('g','data'),
    State('v','data'),
    State('g','data'),
    State('n_steps','data'),
)
def makeGraph(n, y,z,dv,dg,maxCol):
    v = changeDictToMat(dv,maxCol)
    g = changeDictToMat(dg,maxCol)



    cntx = dash.callback_context

    graph_dict = {'ctx': dcc.Graph(
        id='ctx_graph',
        figure={
            'data': [
                {'x': t, 'y': v[0, :],'name' : 'V neuron'},
                {'x': t, 'y': g[0, :],'name' : 'G neuron'},
            ],
            'layout': {
                'title': 'ctx'
            }
        }),
        'stn': dcc.Graph(
            id='stn_graph',
            figure={
                'data': [
                    {'x': t, 'y': v[6, :],'name' : 'V neuron'},
                    {'x': t, 'y': g[6, :],'name' : 'G neuron'},
                ],
                'layout': {
                    'title': 'STN'
                }
            }),
        'd1': dcc.Graph(
            id='d1_graph',
            figure={
                'data': [
                    {'x': t, 'y': v[1, :],'name' : 'v neuron'},
                    {'x': t, 'y': g[1, : ],'name' : 'G neuron'},
                ],
                'layout': {
                    'title': 'd1'
                }
            }),
        'd2': dcc.Graph(
            id='d2_graph',
            figure={
                'data': [
                    {'x': t, 'y': v[2, :],'name' : 'V neuron'},
                    {'x': t, 'y': g[2, :],'name' : 'G neuron'},
                ],
                'layout': {
                    'title': 'd2'
                }
            }),
        'gpi': dcc.Graph(
            id='gpi_graph',
            figure={
                'data': [
                    {'x': t, 'y': v[3, :],'name' : 'V neuron'},
                    {'x': t, 'y': g[3, :],'name' : 'G neuron'},
                ],
                'layout': {
                    'title': 'gpi'
                }
            }),
        'gpe': dcc.Graph(
            id='gpe_graph',
            figure={
                'data': [
                    {'x': t, 'y': v[4, :],'name' : 'V neuron'},
                    {'x': t, 'y': g[4, :],'name' : 'G neuron'},
                ],
                'layout': {
                    'title': 'gpe'
                }
            }),
        'thal': dcc.Graph(
            id='d2_graph',
            figure={
                'data': [
                    {'x': t, 'y': v[5, :],'name' : 'V neuron'},
                    {'x': t, 'y': g[5, :],'name' : 'G neuron'},
                ],
                'layout': {
                    'title': 'thal'
                }
            })}




    return graph_dict[cntx.triggered[0]['value']] if cntx.triggered and cntx.triggered[0]['prop_id'].split('.')[0] == "ChooseGraph" else ''

#makes changes in backend
@app.callback(
    [Output(i,"data")for i in ['t','v','u','g','spike','I_net','I_in','n_steps','n_cells']],
    Input("submitChange","n_clicks"),
    [State(i , "data") for i in ['tau0','T0','psp_amp0','psp_decay0','iz_params','E','w']],
    prevent_initial_call=True
)
def makeChanges(n,tau0,T0,psp_amp0,psp_decay0,iz_params0,E0,w0):
    tau = tau0
    T = T0
    t = np.arange(0, T, tau)
    n_steps = t.shape[0]
    iz_params = changeDictToMat(iz_params0,len(freeVariables["iz_params"][0]))
    E = changeDictToMat(E0,None)
    n_cells = iz_params.shape[0]
    psp_amp = psp_amp0
    psp_decay = psp_decay0
    v = np.zeros((n_cells, n_steps))
    u = np.zeros((n_cells, n_steps))
    g = np.zeros((n_cells, n_steps))
    spike = np.zeros((n_cells, n_steps))
    v[:, 0] = iz_params[:, 1] + np.random.rand(n_cells) * 100
    w = changeDictToMat(w0,len(freeVariables['w'][0]))
    I_net = np.zeros((n_cells, n_steps))
    I_in = np.zeros(n_steps)
    I_in[5000:] = 5e1

    for i in range(1, n_steps):

        dt = t[i] - t[i - 1]

        I_net = np.zeros((n_cells, n_steps))
        for jj in range(n_cells):
            for kk in range(n_cells):
                if jj != kk:
                    I_net[jj, i - 1] += w[kk, jj] * g[kk, i - 1]
                if jj == 0:
                    I_net[jj, i - 1] += I_in[i - 1]

            C = iz_params[jj, 0]
            vr = iz_params[jj, 1]
            vt = iz_params[jj, 2]
            vpeak = iz_params[jj, 3]
            a = iz_params[jj, 4]
            b = iz_params[jj, 5]
            c = iz_params[jj, 6]
            d = iz_params[jj, 7]
            k = iz_params[jj, 8]

            dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) - u[jj, i - 1] +
                    I_net[jj, i - 1] + E[jj]) / C
            dudt = a * (b * (v[jj, i - 1] - vr) - u[jj, i - 1])
            dgdt = (-g[jj, i - 1] + psp_amp * spike[jj, i - 1]) / psp_decay

            v[jj, i] = v[jj, i - 1] + dvdt * dt
            u[jj, i] = u[jj, i - 1] + dudt * dt
            g[jj, i] = g[jj, i - 1] + dgdt * dt

            if v[jj, i] >= vpeak:
                v[jj, i - 1] = vpeak
                v[jj, i] = c
                u[jj, i] = u[jj, i] + d
                spike[jj, i] = 1

    return dict(enumerate(t.flatten(), 1)),dict(enumerate(v.flatten(), 1)),dict(enumerate(u.flatten(), 1)),dict(enumerate(g.flatten(), 1)),dict(enumerate(spike.flatten(), 1)),dict(enumerate(I_net.flatten(), 1)),dict(enumerate(I_in.flatten(), 1)),n_steps, n_cells

#Changes UI variables if change in variable is submitted
def UiChangeVar(n,m):
    return m,m

makeUiChangeVar = \
        {f'UiChangeVar{i}': app.callback(Output(f"{i}label","children"),
    Output(f"{i}val","placeholder"),
    Input(i , "data"),
    State(i , "data"),
    prevent_initial_call=True)(partial(UiChangeVar))
         for i in ['tau0','T0','psp_amp0','psp_decay0']}

def UiChangeMat(n,m,id):
    mat = changeDictToMat(m,len(freeVariables[id][0])) if id != 'E' else changeDictToMat(m,None)

    return dbc.Table.from_dataframe(pd.DataFrame(mat))

makeUiChangeMat = \
        {f'UiChangeMat{i}': app.callback(Output(f"{i}label","children"),
    Input(i , "data"),
    State(i , "data"),
    State(i,"id"),
    prevent_initial_call=True)(partial(UiChangeMat))
         for i in ['iz_params','w','E']}


if __name__ == '__main__':
    app.run_server(debug=True)

