import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
from numpy.linalg import eig, inv
from numpy import linalg as LA
import plotly.io as pio
from datetime import datetime
import matplotlib.pyplot as plt


class MarkovChain:
    def __init__(self):
        self.df = self.load_data()

    def load_data(self):
        # File path can be changed to your specific path
        file_path = r"C:\Users\32917\Downloads\ProyectoMarkov\tec_estocasticos.parquet"
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        fechas_convertidas = []
        for fecha_str in df['periodo']:
            mes, anio = map(int, fecha_str.split('-'))
            fecha = datetime(anio, mes, 1)
            fechas_convertidas.append(fecha)

        df['periodo'] = fechas_convertidas
        df.sort_values(by='periodo', inplace=True)
        df.dropna(inplace=True)
        return df

    def distribucion_limite(self, matriz_transicion):
        P = np.array(matriz_transicion)
        Lambda, Q = LA.eig(P)
        for idx in range(len(Lambda)):
            if Lambda[idx] < 1:
                Lambda[idx] = 0

        Lambda = np.diag(Lambda)
        Q_1 = LA.inv(Q)
        P_n = np.matmul(np.matmul(Q, Lambda), Q_1).round(decimals=4)
        return pd.DataFrame(P_n)

    def transition_matrix(self, new_df1, client_id):
        fechas = pd.DataFrame({
            'periodo': pd.date_range(start='2021-01-01', end='2023-09-01', freq='MS'),
        })

        new_df = new_df1[new_df1['cliente_id'] == client_id].copy()
        new_df['compra'] = 1
        new_df = pd.merge(fechas, new_df, on='periodo', how='left')
        new_df['compra'].fillna(0, inplace=True)
        new_df = new_df.loc[:, ["compra", "periodo"]]
        new_df = new_df.rename(columns={'compra': 'Xt', 'periodo': 'Date'})

        Xt = new_df['Xt'][0:-1].reset_index(drop=True).rename('X_t')
        Xt_1 = new_df['Xt'][1::].reset_index(drop=True).rename('X_t+1')
        indice = list(new_df['Xt']).index(1.0)
        new_df = new_df.iloc[indice::, ::]
        new_data = pd.concat((Xt, Xt_1), axis=1)
        matriz_transicion = new_data.groupby('X_t').value_counts(normalize=True).unstack(level='X_t+1').fillna(0)
        return matriz_transicion

    def crear_Cadena(self, material_id):
        print("---------------Consumo del producto ", material_id, "---------------")
        new_df1 = self.df[self.df['material_id'] == material_id]
        idCliente = set(new_df1['cliente_id'])
        probCompra = {}
        probVolverCompra = {}
        desactivar = {}
        tiempoMedio = {}
        
        for client_id in idCliente:
            matriz_transicion = self.transition_matrix(new_df1, client_id)

            if matriz_transicion.shape == (1, 2):
                continue
            if matriz_transicion.shape == (2, 1):
                city = [0.0, 0.0]
                matriz_transicion['1.0'] = city
            if matriz_transicion.shape == (1, 1) and matriz_transicion.iloc[0, 0] == 1:
                print("Cliente ", client_id, "compra con probabilidad : ", matriz_transicion.iloc[0, 0])
                continue
            if matriz_transicion.iloc[1, 1] == 1:
                print("El tiempo en que mi cliente", client_id, "deja de comprar es indefinido")
                continue

            if matriz_transicion.shape[0] != matriz_transicion.shape[1]:
                print(f"Transition matrix for material {material_id} and client {client_id} is not square!")
                continue

            probCompra[client_id] = matriz_transicion.iloc[0, 1]
            probVolverCompra[client_id] = matriz_transicion.iloc[1, 1]
            matriz_convergida = self.distribucion_limite(matriz_transicion)
            desactivar[client_id] = matriz_convergida.iloc[0, 0]
            tiempoMedio[client_id] = (1 / (1 - matriz_transicion.iloc[1, 1]))

        figs_html = self.generate_plots(desactivar, tiempoMedio, probCompra, probVolverCompra, material_id)
        return figs_html



    def generate_plots(self, desactivar, tiempoMedio, probCompra, probVolverCompra, material_id):
        # Probabilidad de desactivación
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=list(desactivar.keys()),
            y=list(desactivar.values()),
            mode='markers',
            text=list(desactivar.keys()),
            marker=dict(color=list(desactivar.values()), colorscale='plasma', size=10, opacity=0.7)
        ))
        fig1.update_layout(
            title="Probabilidad de que los clientes se desactiven del producto " + str(material_id),
            xaxis_title="ID del Cliente",
            yaxis_title="Probabilidad de desactivación",
            hovermode="closest"
        )

        # Tiempo medio para desactivación
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(tiempoMedio.keys()),
            y=list(tiempoMedio.values()),
            mode='markers',
            text=list(tiempoMedio.keys()),
            marker=dict(color=list(tiempoMedio.values()), colorscale='viridis', size=10, opacity=0.7)
        ))
        fig2.update_layout(
            title="Tiempo medio para que un cliente se desactive del producto " + str(material_id),
            xaxis_title="ID del Cliente",
            yaxis_title="Tiempo Medio",
            hovermode="closest"
        )

        # Probabilidad de Compra
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=list(probCompra.keys()),
            y=list(probCompra.values()),
            mode='markers',
            text=list(probCompra.keys()),
            marker=dict(color=list(probCompra.values()), colorscale='sunset', size=10, opacity=0.7)
        ))
        fig3.update_layout(
            title="Probabilidad de compra del producto " + str(material_id),
            xaxis_title="ID del Cliente",
            yaxis_title="Probabilidad de Compra",
            hovermode="closest"
        )

        # Probabilidad de Volver a Comprar
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=list(probVolverCompra.keys()),
            y=list(probVolverCompra.values()),
            mode='markers',
            text=list(probVolverCompra.keys()),
            marker=dict(color=list(probVolverCompra.values()), colorscale='earth', size=10, opacity=0.7)
        ))
        fig4.update_layout(
            title="Probabilidad de volver a comprar el producto " + str(material_id),
            xaxis_title="ID del Cliente",
            yaxis_title="Probabilidad de Volver a Comprar",
            hovermode="closest"
        )
        figs_html = {
            'fig1_html': pio.to_html(fig1, full_html=False),
            'fig2_html': pio.to_html(fig2, full_html=False),
            'fig3_html': pio.to_html(fig3, full_html=False),
            'fig4_html': pio.to_html(fig4, full_html=False)
        }
        return figs_html






    def get_client_ids(self, material_id):
        new_df1 = self.df[self.df['material_id'] == material_id]
        return list(set(new_df1['cliente_id']))


    def analyze_clients(self, material_id, client_list=None):
        new_df1 = self.df[self.df['material_id'] == material_id]
        
        if client_list is None:
            idCliente = set(new_df1['cliente_id'])
        else:
            idCliente = client_list

        probCompra = {}
        probVolverCompra = {}
        desactivar = {}
        tiempoMedio = {}
        
        for client_id in idCliente:
            matriz_transicion = self.transition_matrix(new_df1, client_id)

        
        

            if matriz_transicion.shape == (1, 2):
                continue
            if matriz_transicion.shape == (2, 1):
                city = [0.0, 0.0]
                matriz_transicion['1.0'] = city
            if matriz_transicion.shape == (1, 1) and matriz_transicion.iloc[0, 0] == 1:
                print("Cliente ", client_id, "compra con probabilidad : ", matriz_transicion.iloc[0, 0])
                continue
            if matriz_transicion.iloc[1, 1] == 1:
                print("El tiempo en que mi cliente", client_id, "deja de comprar es indefinido")
                continue

            if matriz_transicion.shape[0] != matriz_transicion.shape[1]:
                print(f"Transition matrix for material {material_id} and client {client_id} is not square!")
                continue

            probCompra[client_id] = matriz_transicion.iloc[0, 1]
            probVolverCompra[client_id] = matriz_transicion.iloc[1, 1]
            matriz_convergida = self.distribucion_limite(matriz_transicion)
            desactivar[client_id] = matriz_convergida.iloc[0, 0]
            tiempoMedio[client_id] = (1 / (1 - matriz_transicion.iloc[1, 1]))

        # Plots
        self.generate_plots_cliente(desactivar, tiempoMedio, probCompra, probVolverCompra, material_id)



    def generate_plots_cliente(self, desactivar, tiempoMedio, probCompra, probVolverCompra, material_id):
        # Probabilidad de desactivación
        print("Valores de desactivar:", desactivar)
        print("Valores de tiempoMedio:", tiempoMedio)
        print("Valores de probCompra:", probCompra)
        print("Valores de probVolverCompra:", probVolverCompra)


        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=list(desactivar.keys()),
            y=list(desactivar.values()),
            mode='markers',
            text=list(desactivar.keys()),
            marker=dict(color=list(desactivar.values()), colorscale='plasma', size=10, opacity=0.7)
        ))
        fig1.update_layout(
            title="Probabilidad de que los clientes se desactiven del producto " + str(material_id),
            xaxis_title="ID del Cliente",
            yaxis_title="Probabilidad de desactivación",
            hovermode="closest"
        )
        
        # Tiempo medio para desactivación
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(tiempoMedio.keys()),
            y=list(tiempoMedio.values()),
            mode='markers',
            text=list(tiempoMedio.keys()),
            marker=dict(color=list(tiempoMedio.values()), colorscale='viridis', size=10, opacity=0.7)
        ))
        fig2.update_layout(
            title="Tiempo medio para que un cliente se desactive del producto " + str(material_id),
            xaxis_title="ID del Cliente",
            yaxis_title="Tiempo Medio",
            hovermode="closest"
        )
        
        # Histograma de desactivación
        fig3 = go.Figure(data=[go.Histogram(x=list(desactivar.values()), marker=dict(colorscale='blues'))])
        fig3.update_layout(
            title="Distribución de probabilidades de desactivación del producto " + str(material_id),
            xaxis_title="Probabilidad de desactivación",
            yaxis_title="Número de clientes",
        )
        
        # Gráfico de dispersión entre probCompra y probVolverCompra
        fig4 = go.Figure(data=go.Scatter(
            x=list(probCompra.values()),
            y=list(probVolverCompra.values()),
            mode='markers',
            marker=dict(color='red', size=10, opacity=0.7)
        ))
        fig4.update_layout(
            title="Relación entre probabilidad de comprar y probabilidad de volver a comprar " + str(material_id),
            xaxis_title="Probabilidad de comprar",
            yaxis_title="Probabilidad de volver a comprar",
            hovermode="closest"
        )

        # Mostrar plots
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()



    def analyze_cliente(self, material_id, client_id):
        print(f"--------------- Análisis del cliente {client_id} con el producto {material_id} ---------------")
        
        new_df1 = self.df[self.df['material_id'] == material_id]
        
        matriz_transicion = self.transition_matrix(new_df1, client_id)

        # Check the shape and conditions of the matrix
        if matriz_transicion.shape == (1, 2):
            return "Matriz de transición no válida para este cliente."
        if matriz_transicion.shape == (2, 1):
            city = [0.0, 0.0]
            matriz_transicion['1.0'] = city
        if matriz_transicion.shape == (1, 1) and matriz_transicion.iloc[0, 0] == 1:
            return f"Cliente {client_id} compra con probabilidad: {matriz_transicion.iloc[0, 0]}"
        if matriz_transicion.iloc[1, 1] == 1:
            return f"El tiempo en que mi cliente {client_id} deja de comprar es indefinido."

        if matriz_transicion.shape[0] != matriz_transicion.shape[1]:
            return f"La matriz de transición para el material {material_id} y el cliente {client_id} no es cuadrada."

        probCompra = matriz_transicion.iloc[0, 1]
        probVolverCompra = matriz_transicion.iloc[1, 1]
        matriz_convergida = self.distribucion_limite(matriz_transicion)
        desactivar = matriz_convergida.iloc[0, 0]
        tiempoMedio = 1 / (1 - matriz_transicion.iloc[1, 1])

        figs_html = self.generate_single_client_plots(desactivar, tiempoMedio, probCompra, probVolverCompra, material_id, client_id)
        
        return figs_html

    def generate_single_client_plots(self, desactivar, tiempoMedio, probCompra, probVolverCompra, material_id, client_id):
        # Probabilidad de Compra
        fig1 = go.Figure(go.Indicator(
            mode="number+gauge",
            value=probCompra,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Probabilidad de Compra del Producto {material_id} por el Cliente {client_id}"}
        ))

        # Probabilidad de Volver a Comprar
        fig2 = go.Figure(go.Indicator(
            mode="number+gauge",
            value=probVolverCompra,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Probabilidad de Volver a Comprar el Producto {material_id} por el Cliente {client_id}"}
        ))

        # Probabilidad de Desactivación
        fig3 = go.Figure(go.Indicator(
            mode="number+gauge",
            value=desactivar,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Probabilidad de Desactivación del Producto {material_id} por el Cliente {client_id}"}
        ))

        # Tiempo Medio para Desactivación
        fig4 = go.Figure(go.Indicator(
            mode="number+gauge",
            value=tiempoMedio,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Tiempo Medio para Desactivación del Producto {material_id} por el Cliente {client_id}"}
        ))

        figs_html = {
            'fig1_html': pio.to_html(fig1, full_html=False),
            'fig2_html': pio.to_html(fig2, full_html=False),
            'fig3_html': pio.to_html(fig3, full_html=False),
            'fig4_html': pio.to_html(fig4, full_html=False)
        }
        
        return figs_html

