######### IMPORTAR LIBRERIAS ###############
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import scipy.optimize as sco
import plotly.graph_objs as go
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, '')

import warnings
warnings.filterwarnings(action = "ignore")

import funciones_portfolio as fn

######## FIN LIBRERIAS ####################

# ------------------------------------------
########## IMPORTAR DATA SET #################

# Precio de cierre históricos
df_precios_cierre = pd.read_csv('dataset/precios_cierre_universo.csv', parse_dates=[0], index_col=0).drop(['POLPAICO', 'IGPA'], axis=1)

#Rentabilidad histórico
df_rentabilidad = pd.read_csv('dataset/rentabilidad_universo.csv',  parse_dates=[0], index_col=0).drop(['POLPAICO', 'IGPA'], axis=1)

#Rentabilidades para modelamiento los casos
df_rent_caso_1 = pd.read_csv('dataset/df_rentabilidad_case1.csv', parse_dates=[0], index_col=0)
df_rent_caso_2 = pd.read_csv('dataset/df_rentabilidad_case2.csv', parse_dates=[0], index_col=0)
df_rent_caso_3 = pd.read_csv('dataset/df_rentabilidad_case3.csv', parse_dates=[0], index_col=0)

#Betas 
betas = pd.read_excel('dataset/betas_model.xlsx', index_col = 0)

########## FIN DATA SET ###################
#-------------------------------------------

############ IMPORTAR MODELOS #################

# CASO 1 - Modelos todos los datos históricos
COPEC_lr_1 = pkl.load(open('modelos/COPEC_lr_c1.sav', "rb"))
CMPC_lr_1 = pkl.load(open('modelos/CMPC_lr_c1.sav', "rb"))
CCU_lr_1 = pkl.load(open('modelos/CCU_lr_c1.sav', "rb"))
ENEL_lr_1 = pkl.load(open('modelos/ENEL_lr_c1.sav', "rb"))
FALABELLA_lr_1 = pkl.load(open('modelos/FALABELLA_lr_c1.sav', "rb"))
RIPLEY_lr_1 = pkl.load(open('modelos/RIPLEY_lr_c1.sav', "rb"))

model_case_1 = {'COPEC': COPEC_lr_1, 'CMPC': CMPC_lr_1, 'CCU': CCU_lr_1, 'ENEL': ENEL_lr_1, 'FALABELLA': FALABELLA_lr_1, 'RIPLEY': RIPLEY_lr_1}

# CASO 2 - Modelo sin outliers 
COPEC_lr_2 = pkl.load(open('modelos/COPEC_lr_c2.sav', "rb"))
CMPC_lr_2 = pkl.load(open('modelos/CMPC_lr_c2.sav', "rb"))
CCU_lr_2 = pkl.load(open('modelos/CCU_lr_c2.sav', "rb"))
ENEL_lr_2 = pkl.load(open('modelos/ENEL_lr_c2.sav', "rb"))
FALABELLA_lr_2 = pkl.load(open('modelos/FALABELLA_lr_c2.sav', "rb"))
RIPLEY_lr_2 = pkl.load(open('modelos/RIPLEY_lr_c2.sav', "rb"))

model_case_2 = {'COPEC': COPEC_lr_2, 'CMPC': CMPC_lr_2, 'CCU': CCU_lr_2, 'ENEL': ENEL_lr_2, 'FALABELLA': FALABELLA_lr_2, 'RIPLEY': RIPLEY_lr_2}

# CASO 3 - Sin los acontecimientos historicos
COPEC_lr_3 = pkl.load(open('modelos/COPEC_lr_c3.sav', "rb"))
CMPC_lr_3 = pkl.load(open('modelos/CMPC_lr_c3.sav', "rb"))
CCU_lr_3 = pkl.load(open('modelos/CCU_lr_c3.sav', "rb"))
ENEL_lr_3 = pkl.load(open('modelos/ENEL_lr_c3.sav', "rb"))
FALABELLA_lr_3 = pkl.load(open('modelos/FALABELLA_lr_c3.sav', "rb"))
RIPLEY_lr_3 = pkl.load(open('modelos/RIPLEY_lr_c3.sav', "rb"))

model_case_3 = {'COPEC': COPEC_lr_3, 'CMPC': CMPC_lr_3, 'CCU': CCU_lr_3, 'ENEL': ENEL_lr_3, 'FALABELLA': FALABELLA_lr_3, 'RIPLEY': RIPLEY_lr_3}

############ FIN MODELOS ########################

#-------------------------------------------

## FECHAS PARA PRUEBA DE APP ##
#Últimos 5 meses - corresponde a año 2021
list_inputs = df_rentabilidad['IPSA'].iloc[-5:].index.to_list()
fecha = list(map(lambda x: x.strftime("%m/%Y"), list_inputs))

### DESCARGO DE RESPONSABILIDAD ########
disclaimer = open('apoyo/disclaimer.txt')
#-------------------------------------------

st.beta_set_page_config( page_title="Recomendador de Portafolios ", page_icon='📈')
############# APP PORTAFOLIO OPTIMO #############
def main():
    
    op1 = 'Inicio'
    op2 ='Obtener portafolio recomendado'
    op3 ='Ver precios históricos de las acciones'
    
    st.sidebar.markdown("# MENU")
    menu = st.sidebar.selectbox("", (op1, op2, op3))

    if menu == op1:
        st.subheader("Recomendador de Portafolios de Inver$ión")
        video_file = open('apoyo/video_inicio.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, start_time = 0)

        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('Producto creado por Constanza Córdova, Yesenia Lara y Andrea Nuñez - Team Girls ✌ para proyecto Data Science G25')


    if menu == op2:
       
        st.title("Optimiza tu portafolio &#128200;") 

        ####### ENTRADAS #########

        st.subheader('¿Cuánto quieres invertir? &#128176;')
        monto = st.number_input(label='', 
                               min_value = 0,
                               max_value = 100000000,
                               value = 100000,
                               step = 50000,
                               format = '%i')
        # ----------------------------------------------
        st.subheader('Escoge tu estrategia de inversión &#128161;')
        estrategia = st.selectbox('',
        ('Muy conservador', 'Conservador', 'Neutro', 'Arriesgado' ,'Muy Arriesgado'))

        #-----------------------------------------------
        st.subheader('Escoge una fecha para estimar la rentabilidad de tu portafolio &#128198;')
        fecha_input = st.selectbox('', fecha)
        indice = fecha.index(fecha_input)
        fecha_porfolio = list_inputs[indice]

        #rentabilidades realides de los activos para la fecha porfolio
        real = df_rentabilidad.loc[fecha_porfolio]

        #vector para predecir la rentabilidad para la fecha porfolio
        vector_ipsa = df_rentabilidad['IPSA'].loc[fecha_porfolio]
        vector_ipsa = vector_ipsa.reshape(-1, 1)

        #-----------------------------------------------
        escenario1 = 'Quiero estimar la rentabilidad considerando todos los datos históricos'
        escenario2 = 'Quiero estimar la rentabilidad sin considerar desviaciones de precios tan altas'
        escenario3 = 'Quiero estimar la rentabilidad sin considerar las caidas dado los hechos históricos'

        st.subheader('¿Bajo qué escenario quieres estimar la rentabilidad de tu porfatolio? &#127919;')
        escenario = st.selectbox('', (escenario1, escenario2, escenario3 )) 

        #-----------------------------------------------
        st.subheader('Por Último, ¿Con qué tasa quieres comparar?')
        tasa = st.number_input(label='Ingrese el porcentaje de una tasa libre de riesgo', 
                               min_value = 0.0,
                               max_value = 100.0,
                               value = 0.3,
                               step= 0.1,
                               format ='%.1f')

        tasa = tasa/100

        st.write('')
        st.write('')
        st.write('')

        boton1 = st.button("Generar recomendación")

        st.write('')
        st.write('')
        st.write('')
        st.write('')
        #st.info(disclaimer.read())

        ####### FIN ENTRADAS #######

        if boton1:
            
            # Realizamos la prediccion de rentabilidad de acuerdo a escenario escogido

            if escenario == escenario1:
                conservador, neutro, arriesgado = fn.perfil_riesgo(betas, 1)

                #Predicciones de rentabilidad
                df_prediccion = fn.prediccion_caso(model_case_1, vector_ipsa) 
                rent_hist = df_rent_caso_1      


            elif escenario == escenario2:
                conservador, neutro, arriesgado = fn.perfil_riesgo(betas, 2)

                #Predicciones de rentabilidad
                df_prediccion = fn.prediccion_caso(model_case_2, vector_ipsa)
                rent_hist = df_rent_caso_2      


            elif escenario == escenario3:
                conservador, neutro, arriesgado = fn.perfil_riesgo(betas, 3)

                #Predicciones de rentabilidad
                df_prediccion = fn.prediccion_caso(model_case_3, vector_ipsa)
                rent_hist = df_rent_caso_3      

            # ----------------------------------------------
            #Con el output del escenario, generamos los vectores que construiran el portafolio óptimo

            #Data frame historicos, ultimos 36 meses
            df_conservador = fn.historico_acciones_perfil(df_precios_cierre, conservador, 36)
            df_neutro = fn.historico_acciones_perfil(df_precios_cierre, neutro, 36)
            df_arriesgado = fn.historico_acciones_perfil(df_precios_cierre, arriesgado, 36)

            #Portafolio óptimo de acuerdo a perfil de inversión

            #Muy Conservador - min_risk
            if estrategia == 'Muy conservador':
                acciones = conservador
                rent_pred = df_prediccion[acciones].loc[0]
                rent_real = real[acciones]
                df = df_conservador
                metodo = 'min_risk'



            #Conservador - max_sharpe    
            elif estrategia == 'Conservador':
                acciones = conservador
                rent_pred = df_prediccion[acciones].loc[0]
                rent_real = real[acciones]
                df = df_conservador
                metodo = 'max_sharpe'



            #Neutro - min_risk
            elif estrategia == 'Neutro':
                acciones = neutro
                rent_pred = df_prediccion[acciones].loc[0]
                rent_real = real[acciones]
                df = df_neutro
                metodo = 'min_risk'


            #Arriesgado - min_risk
            elif estrategia == 'Arriesgado':
                acciones = arriesgado
                rent_pred = df_prediccion[acciones].loc[0]
                rent_real = real[acciones]
                df = df_arriesgado
                metodo = 'min_risk'



            #Muy Arriesgado - max_sharpe
            elif estrategia == 'Muy Arriesgado':
                acciones = arriesgado
                rent_pred = df_prediccion[acciones].loc[0]
                rent_real = real[acciones]
                df = df_arriesgado
                metodo = 'max_sharpe'

            #------------------------------------------------------
            ######## OUTPUTS ######
            acciones, peso_accion, rent_port_pred, rent_port_real, riesgo, sharpe_ratio =\
            fn.construccion_portafolio(acciones, rent_pred, rent_real, rent_hist[acciones], tasa, metodo = metodo , var_return = True)

            monto_invertir = locale.format('%.0f', monto, grouping = True, monetary = True)

            retorno_esperado = locale.format('%.0f', monto*rent_port_pred/100, grouping = True, monetary = True)

            monto_distribuido = list(map(lambda x: locale.format('%.0f', x*monto/100, grouping = True, monetary = True), peso_accion ))


            ### Camino Feliz
            if rent_port_pred > 0:
                st.balloons()
                
                #titulo
                st.header('&#128083; De acuerdo a nuestro modelo, tu portafolio óptimo es el siguiente: ')

                #gráfico
                st.subheader("Distribución de portafolio")
                st.write(fn.grafico_dona(peso_accion, acciones))

                #Datos Globales
                st.subheader("Detalle de inversión")
                st.write("Monto a invertir $", monto_invertir)
                st.write('Retorno esperado $',retorno_esperado)
                st.write('Se espera una rentabilidad de',rent_port_pred,'% con un riesgo de', riesgo,'% para el peridoo', fecha_input)
                st.write('Sharpe Ratio', round(sharpe_ratio*100,2),'%')
                st.write('La rentabilidad real fue de', rent_port_real,'%')

                #Detalle de las acciones del portafolio
                st.subheader("Detalle del portafolio")
                #st.dataframe(fn.grid_portafolio(acciones, peso_accion, monto_distribuido, rent_pred))
                st.table(fn.grid_portafolio(acciones, peso_accion, monto_distribuido, rent_pred))        
                

                
            ### Camino Triste
            if rent_port_pred < 0:

                #titulo
                st.header(':cold_sweat: UPS!, no encontramos retornos positivos para la estrategia y periodo seleccionado')

                #gráfico
                st.subheader("Distribución de portafolio")
                st.write(fn.grafico_dona(peso_accion, acciones))

                #Datos Globales
                st.subheader("Detalle del análisis")
                st.write("Monto a invertir $", monto_invertir)
                st.write('Retorno esperado $',retorno_esperado)
                st.write('Se espera una rentabilidad de',rent_port_pred,'% con un riesgo de', riesgo,'% para el peridoo', fecha_input)
                st.write('Sharpe Ratio', round(sharpe_ratio*100,2),'%')
                st.write('La rentabilidad real fue de', rent_port_real,'%')
                
                #Detalle de las acciones del portafolio
                st.subheader("Detalle del portafolio")
                #st.dataframe(fn.grid_portafolio(acciones, peso_accion, monto_distribuido, rent_pred)[['Accion', 'Rentabilidad esperada']])
                st.table(fn.grid_portafolio(acciones, peso_accion, monto_distribuido, rent_pred)[['Accion', 'Rentabilidad esperada']])
                
            ## Gráfico de acciones
            st.subheader('Precio de cierre de las acciones del portafolio en los últimos 36 meses')
            st.write(fn.grafico_precios(df))
                
            ## Info
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.info(disclaimer.read())

    if menu == op3:
        st.subheader('Precio de cierre de acciones')
        st.write(fn.grafico_precios(df_precios_cierre))
    

    
    
if __name__ == "__main__":
    main()
