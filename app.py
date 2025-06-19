from dash import Dash, html, dcc, dash_table, Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression
import numpy as np


# Carga de datos
df = pd.read_excel("analfabetismo_mundial_2000_2025.xlsx")

# Conversión de columnas necesarias
df['Analfabetas'] = pd.to_numeric(df['Analfabetas'], errors='coerce')
df['Alfabetas'] = pd.to_numeric(df['Alfabetas'], errors='coerce')
df['Total_Hombres'] = pd.to_numeric(df['Total_Hombres'], errors='coerce')
df['Total_Mujeres'] = pd.to_numeric(df['Total_Mujeres'], errors='coerce')
df['5-9_Total'] = pd.to_numeric(df.get('5-9_Total', 0), errors='coerce')
df['10-15_Total'] = pd.to_numeric(df.get('10-15_Total', 0), errors='coerce')

# App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout
app.layout = dbc.Container([
    html.H1("📊 Evolución del Analfabetismo Mundial (2000–2025)", className="text-center mt-4 mb-4"),
    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="📘 Introducción", value="tab1"),
        dcc.Tab(label="📑 Base de Datos", value="tab2"),
        dcc.Tab(label="📈 Análisis Global", value="tab3"),
        dcc.Tab(label="🗺 Información por País", value="tab4"),
        dcc.Tab(label="📊 Gráficas Avanzadas", value="tab5"),
        dcc.Tab(label="🎂 Distribución por Edad", value="tab6"),
        dcc.Tab(label="📉 Predicción 2030", value="tab7"),

    ]),
    html.Div(id="contenido-tab")
], fluid=True, style={
    "background": "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)",
    "minHeight": "100vh",
    "padding": "20px"
})

# Callback principal para renderizar pestañas
@app.callback(
    Output("contenido-tab", "children"),
    Input("tabs", "value")
)
def renderizar_contenido(tab):
    if tab == "tab1":
        return html.Div([
            html.H5("🧠 ¿Qué es el analfabetismo?"),
            html.P("Se define como la falta de capacidad para leer y escribir. Es la incapacidad de una persona para leer y escribir, incluso mensajes cortos, lo que puede limitar su desarrollo y la adquisición de nuevos conocimientos."),
            html.P("Además, el analfabetismo funcional se refiere a la dificultad para aplicar estas habilidades en la vida cotidiana."),
            html.H6("📌 En detalle:"),
            html.Ul([
                html.Li([
                    html.Strong("Analfabetismo básico: "),
                    "La falta de conocimientos básicos de lectura y escritura. Una persona analfabeta no puede comprender textos ni expresarse por escrito."
                ]),
                html.Li([
                    html.Strong("Analfabetismo funcional: "),
                    "Se refiere a la incapacidad de aplicar las habilidades de lectura y escritura en situaciones cotidianas, como entender un contrato, llenar una solicitud o seguir instrucciones."
                ]),
                html.Li([
                    html.Strong("Consecuencias del analfabetismo: "),
                    "El analfabetismo puede llevar a la exclusión social, limitar las oportunidades laborales y dificultar el acceso a la información y la participación en la sociedad."
                ]),
                html.Li([
                    html.Strong("Analfabetismo digital: "),
                    "Un concepto relacionado es el analfabetismo digital, que se refiere a la falta de habilidades para utilizar las nuevas tecnologías, especialmente internet."
                ])
            ]),
            html.H5("📅 Contexto del análisis"),
            html.P("Este proyecto analiza los datos de analfabetismo global desde el año 2000 hasta 2025."),
            html.Img(
                src="/assets/mapa_analfabetismo.jpeg",
                style={"width": "100%", "maxWidth": "700px", "margin": "20px auto", "display": "block", "borderRadius": "10px"}
            )
        ])

    elif tab == "tab2":
        return html.Div([
            html.H3("📑 Base de Datos"),
            dcc.RadioItems(
                id='modo-tabla',
                options=[
                    {'label': 'Paginado (10 filas por página)', 'value': 'paginado'},
                    {'label': 'Mostrar todos', 'value': 'todo'}
                ],
                value='paginado',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}
            ),
            html.Br(),
            dash_table.DataTable(
                id='tabla-datos',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                page_current=0,
                page_size=10,
                page_action='custom',
                filter_action='native',
                sort_action='native',
                style_table={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '100px', 'whiteSpace': 'normal'}
            )
        ])

    elif tab == "tab3":
        resumen = df.groupby('Año')[['Analfabetas', 'Total_Hombres', 'Total_Mujeres']].mean().reset_index()
        fig_total = px.line(resumen, x="Año", y="Analfabetas", markers=True, title="Analfabetismo Total Mundial")
        fig_hombres = px.line(resumen, x="Año", y="Total_Hombres", markers=True, title="Analfabetismo Hombres")
        fig_mujeres = px.line(resumen, x="Año", y="Total_Mujeres", markers=True, title="Analfabetismo Mujeres")
        return html.Div([
            html.H3("📈 Análisis Global del Analfabetismo"),
            dcc.Graph(figure=fig_total),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_hombres), width=6),
                dbc.Col(dcc.Graph(figure=fig_mujeres), width=6)
            ])
        ])

    elif tab == "tab4":
        paises = df['País'].unique()
        return html.Div([
            html.H4("🔎 Información por país"),
            dcc.Dropdown(
                id='dropdown-pais',
                options=[{'label': pais, 'value': pais} for pais in paises],
                placeholder="Selecciona un país"
            ),
            dcc.Graph(id='grafico-pais'),
            html.Br(),
            html.H4("🏅 Top 5 países con mayor analfabetismo en un año"),
            dcc.Slider(
                id='slider-anio',
                min=df['Año'].min(),
                max=df['Año'].max(),
                step=1,
                value=df['Año'].min(),
                marks={str(a): str(a) for a in range(df['Año'].min(), df['Año'].max()+1, 5)}
            ),
            dcc.Graph(id='grafico-top5')
        ])

    elif tab == "tab5":
        fig1 = px.scatter(df, x="Año", y="Analfabetas", color="País", title="Analfabetismo por país a lo largo del tiempo")
        fig2 = px.histogram(df, x="Analfabetas", nbins=30, title="Distribución del Analfabetismo")
        fig3 = px.box(df, x="Año", y="Analfabetas", title="Analfabetismo por Año")
        return html.Div([
            html.H3("📊 Análisis avanzado"),
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3)
        ])

    elif tab == "tab6":
        paises = df['País'].unique()
        return html.Div([
            html.H3("🎂 Distribución de Analfabetas y Alfabetas por País y Año"),
            dcc.Dropdown(
                id='dropdown-pais-pastel',
                options=[{'label': p, 'value': p} for p in paises],
                placeholder="Selecciona un país",
            ),
            dcc.Slider(
                id='slider-anio-pastel',
                min=df['Año'].min(),
                max=df['Año'].max(),
                step=1,
                value=df['Año'].min(),
                marks={str(a): str(a) for a in range(df['Año'].min(), df['Año'].max()+1, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.Graph(id='grafico-pastel'),
        ])
    elif tab == "tab7":
        resumen = df.groupby('Año')[['Analfabetas']].mean().reset_index()
        modelo = LinearRegression()
        X = resumen['Año'].values.reshape(-1, 1)
        y = resumen['Analfabetas'].values
        modelo.fit(X, y)
        anio_pred = np.array([[2030]])
        prediccion = modelo.predict(anio_pred)[0]
        resumen_pred = resumen.copy()
        resumen_pred.loc[len(resumen_pred.index)] = [2030, prediccion]

        fig_pred = px.line(resumen_pred, x="Año", y="Analfabetas", markers=True,
                           title="Predicción de Analfabetismo Mundial hasta 2030")
        fig_pred.add_scatter(x=[2030], y=[prediccion], mode='markers+text',
                             marker=dict(color='red', size=12),
                             text=[f"{int(prediccion):,}"], textposition="top center",
                             name="Predicción 2030")

        return html.Div([
            html.H3("📉 Predicción Global y por País para el año 2030"),
            html.H5("🌍 Predicción Global"),
            html.P(f"Se estima que en el año 2030 habrá aproximadamente {int(prediccion):,} personas analfabetas en el mundo."),
            dcc.Graph(figure=fig_pred),
            html.Hr(),
            html.H5("🌎 Predicción por País"),
            dcc.Dropdown(
                id='dropdown-pais-pred',
                options=[{'label': p, 'value': p} for p in df['País'].unique()],
                placeholder="Selecciona un país para ver la predicción"
            ),
            dcc.Graph(id='grafico-pred-pais')
        ])

# Callback para manejar paginación o mostrar todos (fuera de la función renderizar_contenido)
@app.callback(
    Output('tabla-datos', 'data'),
    Output('tabla-datos', 'page_size'),
    Output('tabla-datos', 'page_current'),
    Output('tabla-datos', 'page_action'),
    Input('modo-tabla', 'value'),
    Input('tabla-datos', 'page_current'),
    Input('tabla-datos', 'page_size'),
)
def actualizar_tabla(modo, pagina_actual, tamano_pagina):
    if modo == 'todo':
        return df.to_dict('records'), len(df), 0, 'none'
    else:
        start = pagina_actual * tamano_pagina
        end = start + tamano_pagina
        datos_pagina = df.iloc[start:end].to_dict('records')
        return datos_pagina, tamano_pagina, pagina_actual, 'custom'

# Callbacks para tabs 4 y 6 (por país y pastel)
@app.callback(
    Output('grafico-pais', 'figure'),
    Input('dropdown-pais', 'value')
)
def actualizar_grafico_pais(pais):
    if pais:
        df_pais = df[df['País'] == pais]
        fig = px.line(df_pais, x='Año', y='Analfabetas', markers=True,
                      title=f"Evolución del Analfabetismo en {pais}")
        return fig
    return px.line(title="Selecciona un país para ver los datos")

@app.callback(
    Output('grafico-top5', 'figure'),
    Input('slider-anio', 'value')
)
def actualizar_grafico_top5(anio):
    if anio:
        df_anio = df[df['Año'] == anio]
        top5 = df_anio.nlargest(5, 'Analfabetas')
        fig = px.bar(top5, x='País', y='Analfabetas',
                     title=f"Top 5 países con mayor analfabetismo en {anio}",
                     color='País')
        return fig
    return px.bar(title="Selecciona un año")

@app.callback(
    Output('grafico-pastel', 'figure'),
    [Input('dropdown-pais-pastel', 'value'),
     Input('slider-anio-pastel', 'value')]
)
def actualizar_grafico_pastel(pais, anio):
    if pais and anio:
        df_sel = df[(df['País'] == pais) & (df['Año'] == anio)]
        if df_sel.empty:
            return px.pie(values=[1], names=["Sin datos"], title=f"No hay datos para {pais} en {anio}")

        analfabetas = df_sel['Analfabetas'].values[0]
        alfabetas = df_sel['Alfabetas'].values[0]

        if pd.isna(analfabetas) or pd.isna(alfabetas) or analfabetas < 0 or alfabetas < 0:
            return px.pie(values=[1], names=["Dato inválido"], title=f"Dato inválido para {pais} en {anio}")

        total = analfabetas + alfabetas
        if total == 0:
            return px.pie(values=[1], names=["Datos vacíos"], title=f"Datos vacíos para {pais} en {anio}")

        valores_pct = [analfabetas / total * 100, alfabetas / total * 100]

        fig = px.pie(
            names=["Analfabetas", "Alfabetas"],
            values=valores_pct,
            title=f"Distribución de Analfabetas y Alfabetas en {pais} - {anio}",
            color_discrete_map={"Analfabetas": "red", "Alfabetas": "green"}
        )
        fig.update_traces(
            textinfo='percent+label',
            hovertemplate='%{label}: %{value:.2f}%<extra></extra>',
            textfont_size=16,
            marker=dict(line=dict(color='#000000', width=2))
        )
        return fig
    return px.pie(values=[1], names=["Seleccione país y año"], title="Esperando selección...")
@app.callback(
    Output('grafico-pred-pais', 'figure'),
    Input('dropdown-pais-pred', 'value')
)
def prediccion_por_pais(pais):
    if pais:
        df_pais = df[df['País'] == pais]
        if df_pais.empty:
            return px.line(title=f"No hay datos suficientes para {pais}")

        resumen = df_pais.groupby('Año')[['Analfabetas']].mean().reset_index()
        if resumen.shape[0] < 2:
            return px.line(title=f"No hay suficientes datos históricos para predecir en {pais}")

        modelo = LinearRegression()
        X = resumen['Año'].values.reshape(-1, 1)
        y = resumen['Analfabetas'].values
        modelo.fit(X, y)

        anio_pred = np.array([[2030]])
        prediccion = modelo.predict(anio_pred)[0]

        resumen.loc[len(resumen.index)] = [2030, prediccion]

        fig = px.line(resumen, x='Año', y='Analfabetas', markers=True,
                      title=f"Predicción de Analfabetismo en {pais} hasta 2030")
        fig.add_scatter(x=[2030], y=[prediccion], mode='markers+text',
                        marker=dict(color='red', size=12),
                        text=[f"{int(prediccion):,}"], textposition="top center",
                        name="Predicción 2030")
        return fig
    return px.line(title="Selecciona un país para ver su predicción")

# Ejecutar servidor
import os

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
