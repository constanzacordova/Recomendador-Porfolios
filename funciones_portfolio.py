import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objs as go

# -------------------------------------------------------------------
def perfil_riesgo(betas, caso):
    
    betas_caso = betas.loc[caso]
    conservador = []
    neutro = []
    arriesgado = []
    
    for accion, beta in betas_caso.items():
        if beta <1:
            conservador.append(accion)
            
        if beta > 0.85 and beta < 1.15:
            neutro.append(accion)
            
        if beta > 1:
            arriesgado.append(accion)
            
    #conservador.append('IPSA')
    #neutro.append('IPSA')
    #arriesgado.append('IPSA')
            
    return conservador, neutro, arriesgado
# ------------------------------------------------------------------
def historico_acciones_perfil(df, perfil, meses):

    rango_fecha = df.index[-meses:].to_list()
    
    df_perfil = df[perfil].loc[rango_fecha]
    
    return df_perfil

# ------------------------------------------------------------------
def prediccion_caso(dict_models, inputs):
    rent_predict = []
    
    for accion in dict_models.keys():
        key = accion
        prediccion = dict_models[key].predict(inputs)
        rent_predict.append(prediccion)
        
    df_predict = pd.DataFrame(np.array(rent_predict).T, columns= dict_models.keys())
    
    return df_predict

#-----------------------------------------------------------------------


def construccion_portafolio(acciones, rent_pred, rent_real, rent_hist, tasa, metodo = False, var_return = False):
    '''
    construccion_portafolio: Retorna la distribución de pesos de un portafolio de acciones definido
    parámetros:
    acciones : lista con el nombre de las acciones
    rent_pred : serie de rentabilidad predicha, el orden de la serie debe ser igual al orden de las acciones
    rent_real : serie de rentabilidad real para la fecha predicha en caso de querer comparar la predicción, el orden de la serie debe ser igual al orden de las acciones
    rent_hist: dataframe con la rentabilidad de las acciones historicamente para determinar la covarianza entre ellas
    tasa : corresponde a la tasa libre de riesgo que se quiere utilizar para sharpe ratio
    metodo: string, valor por defecto False, estimará en base a los pesos de las acciones igualados
            'max_sharpe': optimizará el portafolio en base a la maximización de sharpe ratio que se encuentra en la frontera eficiente
            'min_risk': optimizará el portafolio con el mínimo riesgo que se encuentre en la frontera eficiente
    var_return: 
            False: valor por defecto
            True: retorna los osiguientes objetos
                    - diccionario de acción y pesos de portafolio 
                    - rent_port_pred: rentabilidad de portafolio predicha
                    - rent_port_real: rentabilidad de portafolio real
                    - volatilidad: riesgo de la cartera
                    - sharpe_ratio: sharpe ratio del portafolio
    '''
    
    def mu(pesos, rendimiento):
        '''
        rendimiento porfolio
        '''
        return sum(pesos * rendimiento)
    #----------------
    def sigma(pesos, covarianza):
        '''
        sigma: Desviación estandar del portafolio 
        '''
        return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
    #-----------------
    def sharpe(pesos, rendimiento, covarianza, tasa):
        '''
        sharpe: Sharpe ratio.
        '''
        return (mu(pesos, rendimiento) - tasa) / sigma(pesos, covarianza)
    #-----------------
    def neg_sharpe(pesos, rendimiento, covarianza, tasa):
        '''
        neg_sharpe: Sharpe ratio negativo para optimizar en la función minimize.
        '''
        return -sharpe(pesos, rendimiento, covarianza, tasa)
    
    def min_var(pesos, covarianza):
        '''
        min_var: Retorna la varianza de sigma.
        '''
        return sigma(pesos, covarianza) ** 2
    

    #Pesos igualados
    pesos = np.array(len(acciones) * [1. / len(acciones)])
    covarianza = np.cov(rent_hist.dropna().transpose())
    volatilidad = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
    rent_port_pred = sum(pesos * rent_pred)
    rent_port_real = sum(pesos * rent_real)
    sharpe_ratio = (rent_port_pred - tasa)/volatilidad
    
    
    #Parámetros función minimize
    cons = ({'type' :'eq', 'fun' : lambda x: np.sum(x) - 1 })
    bnds = tuple((0, 1) for x in range(len(acciones))) 

    #Maximización sharpe ratio
    if metodo == 'max_sharpe':
        opts = sco.minimize(neg_sharpe, pesos, (rent_pred, covarianza , tasa), method = 'SLSQP', bounds = bnds, constraints = cons)
        pesos = opts['x']
        
    #Minimización Riesgo
    if metodo == 'min_risk':
        optv = sco.minimize( min_var, pesos, (covarianza), method = 'SLSQP', bounds = bnds, constraints = cons)
        pesos = optv['x']
    
    df_pesos = pd.DataFrame(pesos).T
    df_pesos.columns = acciones
    
    rent_port_pred = round(sum(pesos * rent_pred)*100,2)
    rent_port_real = round(sum(pesos * rent_real)*100,2)
    volatilidad = round(np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))*100,2)
    sharpe_ratio = (rent_port_pred - tasa*100)/volatilidad

    peso_accion = []
    for accion in acciones:
        peso_accion.append(round(df_pesos[accion].iloc[0]*100))
    
       
    if var_return == True:
        return acciones, peso_accion, rent_port_pred, rent_port_real, volatilidad, sharpe_ratio
    
    
#-----------------------------------------------------

def grafico_dona(peso_accion, acciones):
    data = {"values": peso_accion,
            "labels": acciones,
            "domain": {"column": 0},
            "name": "Acciones",
            "hoverinfo":"label+percent+name",
            "hole": .4,
            "type": "pie"}
    
    layout = go.Layout(
        {"title":"",
         "grid": {"rows": 1, "columns": 1},
         "annotations": [
             {"font": {"size": 10},
              "showarrow": False,
              "text": "Distribución Propuesta"}]})

    fig = go.Figure(data = data, layout = layout)
    return fig

#----------------------------------------------
def grafico_dona_pesos(monto_invertir, acciones):
    data = {"values": monto_invertir,
            "labels": acciones,
            "domain": {"column": 0},
            "name": "Acciones",
            "hoverinfo":"label+value+name",
            "hovertext":"value",
            "hole": .4,
            "type": "pie"}
    
    layout = go.Layout(
        {"title":"Hola",
         "grid": {"rows": 1, "columns": 1},
         "annotations": [
             {"font": {"size": 10},
              "showarrow": True,
              "text": "Distribución Propuesta"}]})

    fig = go.Figure(data = data, layout = layout)
    return fig

#----------------------------------------------

def graficos(acciones, peso_accion, monto):

    montos = list(map(lambda x: x*monto/100, peso_accion ))

    data1 = {
            "values": peso_accion,
            "labels": acciones,
            "domain": {"column": 0},
            "name": "Acciones",
            "hoverinfo":"label+percent+name",
            "hole": .4,
            "type": "pie"
            }

    data2 = {
            "values": montos,
            "labels": acciones,
            "domain": {"column": 1},
            "name": "Montos",
            #"hoverinfo":"label+percent+name",
            "hole": .4,
            "type": "bar"
            }

    data = [data1,data2]

    layout = go.Layout(
        {
            "title":"Portafolio propuesto",
            "grid": {"rows": 1, "columns": 2},
            "annotations": [{"font": {"size": 10},
                             "showarrow": False,
                             "text": "% Propuestos",
                             "x": 0.17,
                             "y": 0.5},
                            {"font": {"size": 10},
                             "showarrow": False,
                             "text": "$ Propuestos",
                             "x": 0.82,
                             "y": 0.5}]})
           
        
    fig = go.Figure(data = data, layout = layout)
    return(fig)

#-----------------------------------------------------
def grafico_precios(df):
    
    # Create figure
    fig = go.Figure()

    for accion in df.columns:

        fig.add_trace(go.Scatter(
            x=list(df.index), 
            y=list(df[accion]),
            name= accion
        ))

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=False
            ),
            type="date"
        )
    )

    return fig


#-------------------------------------
def grid_portafolio(acciones, peso_accion, monto_distribuido, rent_pred):

    column_1 = acciones
    column_2 = list(map(lambda x: str(x)+' %', peso_accion))
    column_3 = list(map(lambda x: '$ '+str(x), monto_distribuido))
    column_4 = list(map(lambda x: str(round(x*100,2))+' %', rent_pred.values))

    df = pd.DataFrame(data = {
        'Accion': column_1,
        'Distribución': column_2,
        'Monto a invertir': column_3,
        'Rentabilidad esperada': column_4
        })
    
    blankIndex=[''] * len(df)
    df.index=blankIndex
    
    return df
