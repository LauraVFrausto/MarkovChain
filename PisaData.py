import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.offline as pyo

class PisaData:
    def __init__(self):
        self.df = self.load_data()  
    # DATA
    
    def load_data(self):
        """
        Read the data.
        """
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

    def groupby_data(self, groupby_column):
        """
        Group data by material_id and the column specified.

        Args:
        - groupby_column (str): The column to group by. Either 'client_id' or 'tipo_cliente'.

        Returns:
        - DataFrame
        """
        assert groupby_column in ['cliente_id', 'tipo_cliente'], "Invalid groupby_column"
        self.df.index = self.df['periodo']
        
        if groupby_column == 'cliente_id':
            grouped = self.df.groupby('material_id')[['cliente_id','periodo']].agg(list).reset_index()
        else:
            grouped = self.df.groupby('material_id')[['tipo_cliente','periodo']].agg(list).reset_index()

        return grouped


    def matrix_with_dummies(self, filtered_data, x_id):

        # Use crosstab to generate a matrix where cliente_id are columns and periodo are rows
        matrix = pd.crosstab(filtered_data['periodo'], filtered_data['cliente_id'])

        # Drop the second level of multi-index if it exists
        if isinstance(matrix.columns, pd.MultiIndex):
            matrix.columns = matrix.columns.get_level_values(0)

        # Convert values greater than 1 to 1 (in case there are duplicates)
        matrix[matrix > 1] = 1

        # Create a date range for all periods
        fechas = pd.DataFrame({
            'periodo': pd.date_range(start='2021-01-01', end='2023-09-01', freq='MS').astype(str),  # Convert periodo to str
        })

        # Reindex the matrix to include all periods in the given date range
        matrix = matrix.reindex(fechas['periodo']).fillna(0).astype(int)
        # Remove the index name
        matrix.index.name = None
        matrix.index = matrix.index.astype(str)

        return matrix
    
    # PLOTS
    
    def plot_periodo_histogram(self):
        """
        Plot histogram of periodo frequencies per month.
        """
        self.df['year_month'] = self.df['periodo'].dt.to_period('M').astype(str)  # Convert Period to String
        fig = px.histogram(self.df, x='year_month', nbins=len(self.df['year_month'].unique()))
        fig.update_layout(xaxis_title="Month-Year", xaxis_type='category')
        return fig

    def plot_top_clients_or_materials(self, column_name, title, colors):
        """
        Helper function to plot top 15 of clients or materials.
        """
        top_data = self.df[column_name].value_counts().head(15)
        fig = px.bar(top_data, x=top_data.index, y=top_data.values, color_discrete_sequence=colors)
        fig.update_layout(xaxis_title=column_name, yaxis_title='Frequency', xaxis_type='category')
        return fig
    
    def plot_lesstop_clients_or_materials(self, column_name, title, colors):
        """
        Helper function to plot the last 15 items (excluding the last one) of clients or materials.
        """
        sorted_data = self.df[column_name].value_counts()
        top_data = sorted_data.iloc[-50:]  # Use .iloc for position-based indexing
        fig = px.bar(top_data, x=top_data.index, y=top_data.values, color_discrete_sequence=colors)
        fig.update_layout(title=title, xaxis_title=column_name, yaxis_title='Frequency', xaxis_type='category')
        return fig


    def plot_top_five_clients(self):
        """
        Plot top fifteen most frequent client_id.
        """
        colors = px.colors.qualitative.Set1
        return self.plot_top_clients_or_materials('cliente_id', "Top 15 Clients", colors)

    def plot_top_five_materials(self):
        """
        Plot top fifteen most frequent material_id.
        """
        colors = px.colors.qualitative.Set2
        return self.plot_top_clients_or_materials('material_id', "Top 15 Materials", colors)
    
    def plot_lesstop_five_clients(self):
        """
        Plot top fifteen most frequent client_id.
        """
        colors = px.colors.qualitative.Set3
        return self.plot_lesstop_clients_or_materials('cliente_id', "Top 30 Clients", colors)

    def plot_lesstop_five_materials(self):
        """
        Plot top fifteen most frequent material_id.
        """
        colors = px.colors.qualitative.Set3_r
        return self.plot_lesstop_clients_or_materials('material_id', "Top 30 Materials", colors)

    def plot_pie_tipo_cliente(self):
        """
        Plot pie chart of tipo_cliente.
        """
        fig = px.pie(self.df, names='tipo_cliente')
        return fig
    
    @staticmethod
    def plot_dummy_matrix(matrix_with_dummies):
        """
        Plot bar chart (as a histogram) for summed values per column in the dummy_matrix.
        """
        summed_values = matrix_with_dummies.sum()  # Sum the values per column

        # Convert Series to DataFrame for clearer data handling in plotly express
        df_summed = summed_values.reset_index()
        df_summed.columns = ['Category', 'Frequency']

        fig = px.bar(df_summed, 
                    x='Category', 
                    y='Frequency', 
                    title="Frequency per Category", 
                    labels={'Frequency': 'Frequency', 'Category': 'Category'},
                    color='Category')  # Assigning color based on the Category column

        # Updating the layout to ensure the x-axis is categorical, bars are adjacent, and setting the size.
        fig.update_layout(xaxis_title="Category", 
                        yaxis_title='Frequency', 
                        xaxis_type='category', 
                        bargap=0,
                        width=1500,  # For example, setting the width to 1200 pixels
                        height=800)  # For example, setting the height to 800 pixels

        plot_div = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        return plot_div










