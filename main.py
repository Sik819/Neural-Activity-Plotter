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
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

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

 'I_in': np.zeros(2000)
}

# direct pathway
freeVariables['w'][0, 1] = 1 * 100
freeVariables['w'][1, 3] = -1 * 125

# indirect pathway
# w[0, 2] = 1 * 100
# w[2, 4] = -1 * 100
# w[4, 3] = -1 * 25

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
                                   html.Div(freeVariables[i]),
                                   dbc.Collapse([
                                       #input_array(v, 1,1,np.array([freeVariables[i]]),'number'),
                                       dbc.Input(placeholder=f"{freeVariables[i]}", type="number", min=0,
                                                                                                  step=1,
                                                 id=f"{i}val", ),
                                       dbc.Button("submit", color="success", className="mr-1")
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
                                       html.Div(dbc.Table.from_dataframe(pd.DataFrame(freeVariables[i]))),
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
                                           dbc.Button("submit", color="success", className="mr-1"),

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
                                       html.Div(dbc.Table.from_dataframe(pd.DataFrame(freeVariables[i]))),
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
                                           dbc.Button("submit", color="success", className="mr-1"),

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


def input_array(label, num_rows ,num_column,val,type='number'):
    inputString = html.Label(label),

    if val.any and val.shape[0]:
        try:
            val.any(axis=1)
        except IndexError:
            inputString = input_onedim_array(label, inputString ,num_column,val,'number')
            return html.Div(
                id=label,

                children=[i for i in inputString])


    if val.any and val.shape[1]:

        inputString = input_twodim_array(label,inputString ,num_rows,num_column,val,'number')

    elif not val:
        for i in range(num_rows):
            if i != 0:
                inputString += html.Br(),
            for j in range(num_column):
                inputString += dbc.Input(
                    id=label + "{}".format(str(i) + str(j)),
                    type=type,
                ),
    return html.Div(
        id=label,

        children=[i for i in inputString])


def input_onedim_array(label,inputString ,num_column,val,type='number',):


    for i in range(num_column):
        inputString += (dbc.Input(
            id=label + "{}".format(i),
            type=type,
            value=val[i],
            bs_size = 'sm',
            size = "10"
        ),)

    return inputString

def input_twodim_array(label,inputString ,num_rows,num_column,val,type='number'):


    for i in range(num_rows):
            if i != 0:
                inputString += (html.Br(),)
            for j in range(num_column):
                inputString += (html.Div(dbc.Input(
                id=label + "{}{}".format(str(i) , str(j)),
                type=type,
                value = val[i][j],
                    bs_size='sm',
                    size="10"
                ),
                id = label + "{}{}".format(str(i) , str(j)) + "div",
                hidden = True,)),
    return inputString

def isMatrix(str):
    matrices = ["iz_params", "w", "E"]
    return (str[0:len(str)-2],2) if (str[0:len(str)-2] in matrices) else (str[0:len(str)-1],1) if (str[0:len(str)-1] in matrices) else None



#front end:


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


#     input_array("E", 1, 7,np.array([0, 0, 0, 300, 300, 500, 0]),'number'),

    html.Br(),


     #input_array("w", 9, 9,np.zeros((9, 9)),"number"),

    html.Br(),
    html.Button('Submit Changes', id='makeGraph', n_clicks=0),
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
    #

    html.Div([dcc.Store(id=i , data = freeVariables[i]) for i in ['tau0','T0','psp_amp0','psp_decay0']]),
    html.Div([dcc.Store(id=i, data=dict(enumerate(freeVariables[i].flatten(), 1))) for i in ['iz_params', 'E', 'w']]),

])


def makeOneDimMatrix(n,col,theDiv,i):

    if(n):
        mat = i[13:]
        if not col:
            raise PreventUpdate
            return 'please enter values'

        elif theDiv is None or theDiv == "please enter values":
            print(theDiv)
            return [dbc.Input(placeholder=f"column {col} : {freeVariables[mat][col]}",
                         id=f"{mat}{col}",
                              type = "number", )]

        else:
            for i in theDiv:
                rowCol = i['props']['id'][len(mat):]
                if(rowCol == str(col)):
                    raise PreventUpdate

            print("ey yo")
            print(theDiv)
            x = theDiv
            x.append(dbc.Input(placeholder=f"column {col} : {freeVariables[mat][col]}",
                                               id=f"{mat}{col}",
                               type = "number",
                                  ))

            return x


    else:
        return 'please enter values'

makeOneDimMatrixValues = \
        {f'makeOneDimMatrix{i}': app.callback(Output(f"intputOneVal_{i}", "children"),
        Input(f"{i}_OnematVal_btn", "n_clicks"),
        State(f"{i}col", "value"),
        State(f"intputOneVal_{i}", "children"),
        State(f"intputOneVal_{i}", "id"),)(partial(makeOneDimMatrix))
         for i in ['E']}

def makeTwoDimMatrix(n,row,col,theDiv,i):

    if(n):
        mat = i[10:]
        if not row or not col:
            raise PreventUpdate
            return 'please enter values'

        elif theDiv is None or theDiv == "please enter values":
            print(theDiv)
            return [dbc.Input(placeholder=f"row {row} column {col} : {freeVariables[mat][row][col]}",
                         id=f"{mat}{row}{col}",
                              type = "number", )]

        else:
            for i in theDiv:
                rowCol = i['props']['id'][len(mat):]
                if(rowCol == str(row)+str(col)):
                    raise PreventUpdate

            print("ey yo")
            print(theDiv)
            x = theDiv
            x.append(dbc.Input(placeholder=f"row {row} column {col} : {freeVariables[mat][row][col]}",
                                               id=f"{mat}{row}{col}",
                               type = "number",
                                  ))

            return x


    else:
        return 'please enter values'

makeTwoDimMatrixValues = \
        {f'makeTwoDimMatrix{i}': app.callback(Output(f"intputVal_{i}", "children"),
        Input(f"{i}_matVal_btn", "n_clicks"),
        State(f"{i}row", "value"),
        State(f"{i}col", "value"),
        State(f"intputVal_{i}", "children"),
        State(f"intputVal_{i}", "id"),)(partial(makeTwoDimMatrix))
         for i in ['iz_params','w']}


def showSingleInput(n,is_open):
    if n:
        return not is_open
    return is_open

makeEditCollapse = \
        {f'showSingleInput{k}': app.callback(
            Output(f"{k}_changeVal_Collapse", "is_open"),
        Input(f"{k}_changeVal_btn", "n_clicks"),
        State(f"{k}_changeVal_Collapse", "is_open"),)(partial(showSingleInput))
         for k in ["T0","tau0","psp_amp0","psp_decay0",'iz_params','w','E']}




# @app.callback(
#     [Output(f"{i}_changeVal_Collapse", "is_open") for i in ["T0","tau0","psp_amp0","psp_decay0"]],
#     [Input(f"{i}_changeVal_btn", "n_clicks") for i in ["T0","tau0","psp_amp0","psp_decay0"]],
#     [State(f"{i}_changeVal_Collapse", "is_open") for i in ["T0","tau0","psp_amp0","psp_decay0"]],
# )
# def showSingleInput(n1, n2,n3,n4, is_open1, is_open2, is_open3, is_open4):
#     ctx = dash.callback_context
#
#     if not ctx.triggered:
#         return False, False, False, False
#     else:
#         button_id = ctx.triggered[0]["prop_id"].split(".")[0]
#
#     if button_id == "T0_changeVal_btn" and n1:
#         return not is_open1, False,False,False
#     elif button_id == "tau0_changeVal_btn" and n2:
#         return False, not is_open2 ,False,False
#     elif button_id == "psp_amp0_changeVal_btn" and n3:
#         return False, False, not is_open3, False
#     elif button_id == "psp_decay0_changeVal_btn" and n4:
#         return False, False, False, not is_open4
#     return False, False, False, False


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



def modifyChangeFunction(*iz):

    #enter free variables here


    ctx = dash.callback_context

    #if change is made:
    if ctx.triggered:
        changedID = ctx.triggered[0]['prop_id'].split('.')[0]

        if(isMatrix(changedID) and isMatrix(changedID)[1] == 2): #check if changed value is a matrix variable
            #2dim array

            i = int(changedID[len(changedID)-2])
            j = int(changedID[len(changedID)-1])
            freeVariables[isMatrix(changedID)[0]][i][j] = ctx.triggered[0]['value']

        elif (isMatrix(changedID) and isMatrix(changedID)[1] == 1):
            #one dim array

            i = int(changedID[len(changedID) - 1])
            freeVariables[isMatrix(changedID)[0]][i] = ctx.triggered[0]['value']

        else:
            if(changedID == 'psp_decay0'):
                #safety param for division
                if(ctx.triggered[0]['value'] == 0):
                    freeVariables[changedID] = 1
                else:
                    freeVariables[changedID] = ctx.triggered[0]['value']
            else:
                freeVariables[changedID] = ctx.triggered[0]['value']

    return str(iz) + " "

@app.callback(
    Output(component_id='output-graph-div', component_property='children'),
    Input(component_id='makeGraph', component_property='n_clicks'),
    [Input(component_id='ChooseGraph', component_property='value')]
)
def makeGraph(*iz):

    cntx = dash.callback_context
    # stn-gpe feedback
    freeVariables['w'][6, 4] = 1
    freeVariables['w'][4, 6] = -1 * 50

    # output
    freeVariables['w'][3, 5] = -1 * 100
    freeVariables['w'][5, 0] = 1
    #
    # #update values if values change
    # print(freeVariables['w'])
    # t = np.arange(0, freeVariables['T0'], freeVariables['tau0'])
    # n_steps = t.shape[0]
    # n_cells = freeVariables['iz_params'].shape[0]
    #
    # # memory allocation for neurons
    # v = np.zeros((n_cells, n_steps))
    # u = np.zeros((n_cells, n_steps))
    # g = np.zeros((n_cells, n_steps))
    # spike = np.zeros((n_cells, n_steps))
    # v[:, 0] = freeVariables['iz_params'][:, 1] + np.random.rand(n_cells) * 100
    # I_net = np.zeros((n_cells, n_steps))
    # I_in = np.zeros(n_steps)
    # I_in[5000:] = 5e1
    #
    # for i in range(1, n_steps):
    #
    #     dt = t[i] - t[i - 1]
    #
    #     I_net = np.zeros((n_cells, n_steps))
    #     for jj in range(n_cells):
    #         for kk in range(n_cells):
    #             if jj != kk:
    #                 I_net[jj, i - 1] += freeVariables['w'][kk, jj] * g[kk, i - 1]
    #             if jj == 0:
    #                 I_net[jj, i - 1] += I_in[i - 1]
    #
    #             C = freeVariables['iz_params'][jj, 0]
    #             vr = freeVariables['iz_params'][jj, 1]
    #             vt = freeVariables['iz_params'][jj, 2]
    #             vpeak = freeVariables['iz_params'][jj, 3]
    #             a = freeVariables['iz_params'][jj, 4]
    #             b = freeVariables['iz_params'][jj, 5]
    #             c = freeVariables['iz_params'][jj, 6]
    #             d = freeVariables['iz_params'][jj, 7]
    #             k = freeVariables['iz_params'][jj, 8]
    #
    #             dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) - u[jj, i - 1] +
    #                     I_net[jj, i - 1] + freeVariables['E'][jj]) / C
    #             dudt = a * (b * (v[jj, i - 1] - vr) - u[jj, i - 1])
    #
    #             dgdt = (-g[jj, i - 1] + freeVariables['psp_amp0'] * spike[jj, i - 1]) / freeVariables['psp_decay0']
    #
    #             v[jj, i] = v[jj, i - 1] + dvdt * dt
    #             u[jj, i] = u[jj, i - 1] + dudt * dt
    #             g[jj, i] = g[jj, i - 1] + dgdt * dt
    #
    #             if v[jj, i] >= vpeak:
    #                 v[jj, i - 1] = vpeak
    #                 v[jj, i] = c
    #                 u[jj, i] = u[jj, i] + d
    #                 spike[jj, i] = 1
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



if __name__ == '__main__':
    app.run_server(debug=True)