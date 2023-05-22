# Importing necessary libraries
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Creating the app
app = dash.Dash(__name__)

# Loading the data
df = pd.read_csv('Churn_Modelling.csv')
df1 = pd.read_csv('test.csv')


# Creating a scatter plot for the first page
#Check the class distribution in the given data
total = df['Exited'].count()
counts = df['Exited'].value_counts()
percentages = round((counts / total),4) * 100
frequencies = df['Geography'].value_counts()

num=df['NumOfProducts'].value_counts()


fig = px.bar(num, x=num.index, y=num.values, color = ['#FFA500','#8B0000','#0055A4', '#9370DB'])


fig2 = make_subplots(rows=2, cols=2, subplot_titles=('Credit Score', 'Age', 'Balance', 'Estimated Salary'))

cols=['CreditScore','Age','Balance','EstimatedSalary']

colors = ['blue', 'red']

for i in range(4):
    row = (i // 2) + 1
    col = (i % 2) + 1
    hist1 = go.Histogram(x=df[df['Exited'] == 0][cols[i]], nbinsx=30, marker_color=colors[0], name='Not Exited')
    hist2 = go.Histogram(x=df[df['Exited'] == 1][cols[i]], nbinsx=30, marker_color=colors[1], name='Exited')
    fig2.add_trace(hist1, row=row, col=col)
    fig2.add_trace(hist2, row=row, col=col)

    fig2.update_xaxes(title_text=cols[i], row=row, col=col)
    fig2.update_yaxes(title_text='Count', row=row, col=col)

fig2.update_layout(height=1100, width =1450, title_text='Histogram of Features by Exited')


fig3 = px.bar(frequencies.index[::-1], frequencies.values[::-1], color=['#FFA500','#8B0000','#0055A4'], orientation = 'h', text = frequencies.index[::-1].values.tolist())


fig4 = px.pie(df, names='Tenure', color = 'Age')
fig4.update_traces(name='Tenure', showlegend=True)
fig4.update_layout(
    title={
        'text': "Tenure of customers by age",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)



corr= df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']].corr()

fig20 = px.imshow(corr,
                labels=dict(x='', y=''),
                x=['CreditScore', 'Age', 'Balance', 'EstimatedSalary'],
                y=['CreditScore', 'Age', 'Balance', 'EstimatedSalary'],
                color_continuous_scale='Oranges',
                width=800, height=600,
                zmin=-1, zmax=1)

# Add correlation values on top of the squares
annotations = []
for i, row in enumerate(corr.values):
    for j, value in enumerate(row):
        if i==j:
            annotations.append(dict(x=j, y=i, text=f'{value:.0f}', showarrow=False, font=dict(color='black', size=18)))
        else :
            annotations.append(dict(x=j, y=i, text=f'{value:.5f}', showarrow=False, font=dict(color='black', size=18)))
fig20.update_layout(annotations=annotations)

fig20.update_layout(title='<b>Correlation Heatmap of Major Features<b>',
                  title_font_size=20, title_x=0.5, title_y=0.95,
                  xaxis=dict(tickangle=-45,tickfont=dict(size=14)),
                  yaxis=dict(showticklabels=True,tickfont=dict(size=14)),
                 width=750, height=700)















fig5 = px.histogram(df, x="Age", color="Exited", nbins=60,
                   barmode="overlay", opacity=0.7,
                   labels={"Age": "Age", "Exited": "Exited"},
                   histnorm='probability density',
                   color_discrete_sequence=['red', 'blue'])

fig5.update_layout(
    title={
        'text': "Distribution of Age by Exited Status",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)

fig5.update_traces(marker=dict(line=dict(width=1, color='Black')))


# Creating a pie chart for the fourth page
fig6 = px.pie(df, names='Exited', hole=0.5)
fig6.update_traces(name='Exited', showlegend=True)
fig6.update_layout(
    title={
        'text': "Percentage of exited customers",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)

fig5.update_traces(marker=dict(line=dict(width=1, color='Black')))





#cm logistic

cm = confusion_matrix(df1["Exited"],df1["logistic"])

fig8 = px.imshow(cm,
                labels=dict(x="Real", y="Predicted"),
                x=[0,1],
                y=[0,1])
fig8.update_traces(name='CM', showlegend=True)
fig8.update_layout(
    title={
        'text': "Confusion matrix",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)


#
fpr, tpr, thresholds = roc_curve(df1["Exited"], df1["proba_logistic"])
roc_auc = roc_auc_score(df1["Exited"], df1["proba_logistic"])
auc_score=roc_auc_score(df1["Exited"], df1["proba_logistic"])

# create the ROC-AUC curve plot
trace1 = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % auc_score)
trace3 = go.Scatter(x=fpr, y=fpr, mode='lines', name='Reference line')

layout = go.Layout(title='ROC-AUC Curve', xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'))
fig7 = go.Figure(data=[trace1, trace3], layout=layout)





#cm logistic

cm = confusion_matrix(df1["Exited"],df1["randomforest"])

fig9 = px.imshow(cm,
                labels=dict(x="Real", y="Predicted"),
                x=[0,1],
                y=[0,1])

fig9.update_traces(name='CM', showlegend=True)
fig9.update_layout(
    title={
        'text': "Confusion matrix",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)


#
fpr, tpr, thresholds = roc_curve(df1["Exited"], df1["proba_randomf"])
roc_auc = roc_auc_score(df1["Exited"], df1["proba_randomf"])
auc_score=roc_auc_score(df1["Exited"], df1["proba_randomf"])

# create the ROC-AUC curve plot
trace1 = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % auc_score)
trace3 = go.Scatter(x=fpr, y=fpr, mode='lines', name='Reference line')

layout = go.Layout(title='ROC-AUC Curve', xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'))
fig10 = go.Figure(data=[trace1, trace3], layout=layout)



#cm logistic

cm = confusion_matrix(df1["Exited"],df1["svc"])

fig11 = px.imshow(cm,
                labels=dict(x="Real", y="Predicted"),
                x=[0,1],
                y=[0,1])

fig11.update_traces(name='CM', showlegend=True)
fig11.update_layout(
    title={
        'text': "Confusion matrix",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)

#
fpr, tpr, thresholds = roc_curve(df1["Exited"], df1["proba_svc"])
roc_auc = roc_auc_score(df1["Exited"], df1["proba_svc"])
auc_score=roc_auc_score(df1["Exited"], df1["proba_svc"])

# create the ROC-AUC curve plot
trace1 = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % auc_score)
trace3 = go.Scatter(x=fpr, y=fpr, mode='lines', name='Reference line')

layout = go.Layout(title='ROC-AUC Curve', xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'))
fig12 = go.Figure(data=[trace1, trace3], layout=layout)



#cm logistic

cm = confusion_matrix(df1["Exited"],df1["hyp_logistic"])

fig13 = px.imshow(cm,
                labels=dict(x="Real", y="Predicted"),
                x=[0,1],
                y=[0,1])

fig13.update_traces(name='CM', showlegend=True)
fig13.update_layout(
    title={
        'text': "Confusion matrix",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)


#
fpr, tpr, thresholds = roc_curve(df1["Exited"], df1["proba_hyp_logistic"])
roc_auc = roc_auc_score(df1["Exited"], df1["proba_hyp_logistic"])
auc_score=roc_auc_score(df1["Exited"], df1["proba_hyp_logistic"])

# create the ROC-AUC curve plot
trace1 = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % auc_score)
trace3 = go.Scatter(x=fpr, y=fpr, mode='lines', name='Reference line')

layout = go.Layout(title='ROC-AUC Curve', xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'))
fig14 = go.Figure(data=[trace1, trace3], layout=layout)





#cm logistic

cm = confusion_matrix(df1["Exited"],df1["hyp_random_forest"])

fig15 = px.imshow(cm,
                labels=dict(x="Real", y="Predicted"),
                x=[0,1],
                y=[0,1])

fig15.update_traces(name='CM', showlegend=True)
fig15.update_layout(
    title={
        'text': "Confusion matrix",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)


#
fpr, tpr, thresholds = roc_curve(df1["Exited"], df1["hyp_proba_randomf"])
roc_auc = roc_auc_score(df1["Exited"], df1["hyp_proba_randomf"])
auc_score=roc_auc_score(df1["Exited"], df1["hyp_proba_randomf"])

# create the ROC-AUC curve plot
trace1 = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % auc_score)
trace3 = go.Scatter(x=fpr, y=fpr, mode='lines', name='Reference line')

layout = go.Layout(title='ROC-AUC Curve', xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'))
fig16 = go.Figure(data=[trace1, trace3], layout=layout)




#cm logistic

cm = confusion_matrix(df1["Exited"],df1["hyp_svc"])

fig17 = px.imshow(cm,
                labels=dict(x="Real", y="Predicted"),
                x=[0,1],
                y=[0,1])

fig17.update_traces(name='CM', showlegend=True)
fig17.update_layout(
    title={
        'text': "Confusion matrix",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Age",
    yaxis_title="Density",
    legend_title="Exited",
    bargap=0.1)


#
fpr, tpr, thresholds = roc_curve(df1["Exited"], df1["hyp_proba_svc"])
roc_auc = roc_auc_score(df1["Exited"], df1["hyp_proba_svc"])
auc_score=roc_auc_score(df1["Exited"], df1["hyp_proba_svc"])

# create the ROC-AUC curve plot
trace1 = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % auc_score)
trace3 = go.Scatter(x=fpr, y=fpr, mode='lines', name='Reference line')

layout = go.Layout(title='ROC-AUC Curve', xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'))
fig18 = go.Figure(data=[trace1, trace3], layout=layout)




# Creating the layout for the dashboard
app.layout = html.Div([
    html.H1('Customer Churn Classification'),
    dcc.Tabs([
        dcc.Tab(label='Product count and customers per country', children=[
            html.Div([
                html.P("These are bar plots of the products seperated by Exit status."),
                html.P("Given graphs provides valuable insights for the company in selecting target audience from the particular country or in efforts to extend subscription durations."),
                html.P("Obviously France is leading in the list as per the number of the customers"),
                dcc.Graph(id='bar-chart0', figure=fig),
                dcc.Graph(id='bar-chart2', figure=fig3)
                        
        ])]),

        dcc.Tab(label='Relation between exit and other factors', children=[
            html.Div([
                html.P("Here are histograms of features colored by Exited status."),
                html.P("From the age graph, it seems that younger people tend to keep their customer loyalty. Only the shape of the estimated salary graph is uniformly distributed."),
                dcc.Graph(id='bar-chart1', figure=fig2)
                        
        ])]),
        dcc.Tab(label='Customer segmentation', children=[
            html.Div([
                dcc.Graph(id='bar-chart3', figure=fig4),
                dcc.Graph(id='bar-chart4', figure=fig5),
                html.P("Balance problem can be easily observed in the pie chart."),
                html.P("Based on this pattern one can expect poor prediction results while classifying the customers."),
                dcc.Graph(id='bar-chart5', figure=fig6)

                        
        ])]),
	dcc.Tab(label='Correlation heatmap', children=[
            html.Div([
                dcc.Graph(id='bar-chart111', figure=fig20)
                       
	])]),
        dcc.Tab(label='Logistic Regression model', children=[
            html.Div([
                dcc.Graph(id='bar-chart6', figure=fig8),
                dcc.Graph(id='bar-chart7', figure=fig7)
                        
        ])]),
        dcc.Tab(label='Random Forest model', children=[
            html.Div([
                dcc.Graph(id='bar-chart8', figure=fig9),
                dcc.Graph(id='bar-chart9', figure=fig10)
                        
        ])]),
        dcc.Tab(label='SVC model', children=[
            html.Div([
                dcc.Graph(id='bar-chart10', figure=fig11),
                dcc.Graph(id='bar-chart11', figure=fig12)
                        
        ])]),
        dcc.Tab(label='Logistic Regression hyperparameter tuning', children=[
            html.Div([
                dcc.Graph(id='bar-chart12', figure=fig13),
                dcc.Graph(id='bar-chart13', figure=fig14)
                        
        ])]),
        dcc.Tab(label='Random Forest hyperparameter tuning', children=[
            html.Div([
                dcc.Graph(id='bar-chart14', figure=fig15),
                dcc.Graph(id='bar-chart15', figure=fig16)
                        
        ])]),
        dcc.Tab(label='SVC hyperparameter tuning', children=[
            html.Div([
                dcc.Graph(id='bar-chart16', figure=fig17),
                dcc.Graph(id='bar-chart17', figure=fig18)
                        
        ])])
       
       
    ])
])

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)