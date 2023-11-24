
from MarkovChain import MarkovChain
from PisaData import PisaData
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import plotly.io as pio
from plotly.offline import plot
import matplotlib.pyplot as plt

app = Flask(__name__)
data = PisaData()
@app.route('/', methods=['GET', 'POST'])
def index():
    mc = MarkovChain()
    material_ids = set(mc.df['material_id'])
    figs_html = None
    client_ids = []
    material_id_selected = None

    if request.method == 'POST':
        material_id_selected = int(float(request.form['material_id']))
        figs_html = mc.crear_Cadena(material_id_selected)
        client_ids = mc.get_client_ids(material_id_selected)  # Asume que tienes esta función


    # Assuming data is an object that is defined and provides plotting methods
    # Initialize data object here if needed
    plots = {
        "periodo_histogram": pio.to_html(data.plot_periodo_histogram(), full_html=False),
        "top_five_clients": pio.to_html(data.plot_top_five_clients(), full_html=False),
        "top_five_materials": pio.to_html(data.plot_top_five_materials(), full_html=False),
        "top_five_lessclients": pio.to_html(data.plot_lesstop_five_clients(), full_html=False),
        "top_five_lessmaterials": pio.to_html(data.plot_lesstop_five_materials(), full_html=False),
        "pie_tipo_cliente": pio.to_html(data.plot_pie_tipo_cliente(), full_html=False),
    }

    if figs_html:
        return render_template('graphs.html', material_ids=material_ids, material_id=material_id_selected, client_ids=client_ids, plots=plots, figs_html=figs_html)
    
    return render_template('index.html', material_ids=material_ids, plots=plots)


@app.route('/cliente_analysis', methods=['GET', 'POST'])
def cliente_analysis():
    mc = MarkovChain()

    if request.method == 'POST':
        material_id_selected = int(float(request.form['material_id']))
        cliente_id_selected = int(float(request.form['cliente_id']))

        analysis_results = mc.analyze_cliente(material_id_selected, cliente_id_selected)
        return render_template('cliente_analysis.html', analysis=analysis_results)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

    