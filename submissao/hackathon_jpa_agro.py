"""
Created on Mon Feb 23 22:25:22 2021


                           Hackathon JPA Agro 2021
                           
    
    Equipe: Sirius
    
    1. Thiago Mantuani de Souza    - Mestrado em Engenharia de Sistemas e Automação
    2. Cecília Ramos de Oliveira   - Doutorado em Botânica Aplicada  
    
    
"""


import numpy               as np
import pandas              as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics       import mean_squared_error
from lightgbm              import LGBMRegressor
from xgboost               import XGBRegressor
from datetime              import timedelta, datetime
from matplotlib.ticker     import FormatStrFormatter
import pickle
import matplotlib.pyplot   as plt
import seaborn             as sns           

def save_pred_txt(values):
    v1 = [str(i) for i in values]
    v2 = ','.join(v1)
    with open('predicted_values.txt','w') as f:
        for row in v2:                                
            f.write(row)    
            
def plot_predict(wdf,y_pred,y_true):
    dplot = wdf.copy()
    plt.clf()
    plt.figure(1)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(y_pred, label='Previsto')
    ax.plot(y_true, label='Real')
    ax.set_xlabel('Período')
    ax.set_ylabel('R$')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))          
    ax.set_xticks(range(0,dplot.shape[0],2))
    ax.set_xticklabels(dplot['negotiation_date'].loc[::2].dt.strftime('%Y-%m-%d'),rotation=90)  
    ax.set_title('Previsão')
    ax.legend()
  
    
def plot_dist_price(df):
    sns.histplot(df.sold_price,kde=True)    
  
  
class HackathonJPAAgro(object):
    
    def __init__(self):
        #self.bd = pd.read_csv('dataset_train.csv',parse_dates=['negotiation_date'],dayfirst=True)
        self.bd = pd.read_csv('https://github.com/dsrg-icet/hackathon_JPAAgro/blob/main/dataset/dataset_train.csv?raw=True',parse_dates=['negotiation_date'],dayfirst=True)
        self.predict_agost = pd.DataFrame()
        

    def ChangeType(self,df):    
        #colocando no formato datetime
        df['negotiation_date'] = pd.to_datetime(df['negotiation_date'])   
        
        #excluindo o atributo -> product, pois é o mesmo para todo conjunto
        df.drop('product',axis=1,inplace=True)    
        
        return df    
        
    def SeriesCorrect(self,df1):
        #corrigindo falha na sequencia da serie
        #os valores NaN apos a correcao foram preenchidos com os valores anteriores a eles
        df_aux = pd.date_range(start='2014-01-07',end='2019-07-31',name='negotiation_date').to_frame().reset_index(drop=True)
        all_data = pd.merge(left=df_aux,right=df1,how='left',left_on='negotiation_date',right_on='negotiation_date')
        all_data['sold_price'].fillna(method='ffill',inplace=True)
        d=all_data[(all_data['negotiation_date']>='2019-07-01') & (all_data['negotiation_date']<='2019-07-31')]
   
        v = d['sold_price'].values.tolist()
        v1 = [str(i) for i in v]
        v2 = ','.join(v1)
        with open('true_values.txt','w') as f:
            for row in v2:                                
                f.write(row)
        
        return df1
    
    def FeatureEngineering(self,df2):
        
        #criando novas features (apenas as que foram selecionadas pelo algoritmo Boruta)
        
        df2['day'] = df2['negotiation_date'].dt.day
        for i in range(1,6):
            df2['sold_price_lag_{}'.format(i)] = df2['sold_price'].shift(i)
        for i in range(1,6):
            df2['sold_price_diff_{}'.format(i)] = df2['sold_price'].diff(i)   
            
        return df2
    
    def DataPreparation(self,df3,train=True):
        #preparando os dados, transformando os atributos numéricos
        #transformação de natureza no atributo dia
                        
        ms = MinMaxScaler()        
        rs = RobustScaler()
                
    
        if train:
            for i in range(1,6):
                df3['sold_price_lag_{}'.format(i)] = ms.fit_transform(df3[['sold_price_lag_{}'.format(i)]])
                pickle.dump(ms,open('./scaler_sold_price_lag_{}'.format(i)+'.pkl','wb'))
                    
            for i in range(1,6):
                df3['sold_price_diff_{}'.format(i)] = rs.fit_transform(df3[['sold_price_diff_{}'.format(i)]])
                pickle.dump(rs,open('./scaler_sold_price_diff_{}'.format(i)+'.pkl','wb'))                
        
        
        #transformando a variavel de saida em uma dist. normal com logaritmo
        df3['sold_price'] = np.log1p(df3['sold_price'])
        
        #transformacao de variaveis ciclicas como o dia        
        df3['day_sen'] = df3['day'].apply(lambda x: np.sin(x * (2*np.pi/30)))
        
        if not train:
            self.scaler_sold_price_lag_1 = pickle.load(open('./scaler_sold_price_lag_1.pkl','rb'))
            self.scaler_sold_price_lag_2 = pickle.load(open('./scaler_sold_price_lag_2.pkl','rb'))
            self.scaler_sold_price_lag_3 = pickle.load(open('./scaler_sold_price_lag_3.pkl','rb'))
            self.scaler_sold_price_lag_4 = pickle.load(open('./scaler_sold_price_lag_4.pkl','rb'))
            self.scaler_sold_price_lag_5 = pickle.load(open('./scaler_sold_price_lag_5.pkl','rb'))            
            self.scaler_sold_price_diff_1 = pickle.load(open('./scaler_sold_price_diff_1.pkl','rb'))
            self.scaler_sold_price_diff_2 = pickle.load(open('./scaler_sold_price_diff_2.pkl','rb'))
            self.scaler_sold_price_diff_3 = pickle.load(open('./scaler_sold_price_diff_3.pkl','rb'))
            self.scaler_sold_price_diff_4 = pickle.load(open('./scaler_sold_price_diff_4.pkl','rb'))
            self.scaler_sold_price_diff_5 = pickle.load(open('./scaler_sold_price_diff_5.pkl','rb'))
            df3['sold_price_lag_1']  = self.scaler_sold_price_lag_1.transform(df3[['sold_price_lag_1']])
            df3['sold_price_lag_2']  = self.scaler_sold_price_lag_1.transform(df3[['sold_price_lag_2']])
            df3['sold_price_diff_1'] = self.scaler_sold_price_diff_1.transform(df3[['sold_price_diff_1']])
            df3['sold_price_diff_2'] = self.scaler_sold_price_diff_1.transform(df3[['sold_price_diff_2']])
            df3['sold_price_diff_3'] = self.scaler_sold_price_diff_1.transform(df3[['sold_price_diff_3']])
            df3['sold_price_diff_4'] = self.scaler_sold_price_diff_1.transform(df3[['sold_price_diff_4']])
            df3['sold_price_diff_5'] = self.scaler_sold_price_diff_1.transform(df3[['sold_price_diff_5']])
            
                        
        #atraves de seleção de features do algoritmo Boruta, os atributos escolhidos foram:
        features = ['sold_price_lag_1',
                    'sold_price_lag_2',
                    'sold_price_diff_1',
                    'sold_price_diff_2',
                    'sold_price_diff_3',
                    'sold_price_diff_4',
                    'sold_price_diff_5',
                    'day_sen']
        features.extend(['negotiation_date','sold_price'])   
        
        df3.dropna(inplace=True)
        df3 =df3[features]
        
        
        self.df_tmp = df3[features]
        
        return df3
    
    
    def getPredict30daysAgo(self,df6):
        #conforme proposto deve-se realizar a previsão de 30 dias a frente após a ultima
        #data presente no arquivo dataset_train.csv
        #a ultima data que temos é 31/07/2019, portanto a previsão será feita a partir dessa data
        
        self.lgbm_model = pickle.load(open('./lgbm_model.pkl','rb'))
        self.xgb_model  = pickle.load(open('./xgb_model.pkl','rb'))
        
        df6['sold_price'] = np.expm1(df6['sold_price'])
        df_pred = df6.copy()
            
        last_date = datetime.strptime('2019-07-31','%Y-%m-%d')
        limit_date = datetime.strptime('2019-08-30','%Y-%m-%d')
        
        while (last_date < limit_date):
            df6 = df6[['negotiation_date','sold_price']].tail(6)
            df_1 = self.FeatureEngineering(df6)
            df_2 = self.DataPreparation(df_1,False)

            X = df_2.drop(['sold_price','negotiation_date'],axis=1)
            
            y_pred1 = self.lgbm_model.predict(X)
            y_pred2 = self.xgb_model.predict(X)
            y_pred1 = np.expm1(y_pred1)
            y_pred2 = np.expm1(y_pred2)
            
            y_pred  = (y_pred1*0.6+y_pred2*0.4)            
            #y_pred  = (y_pred1)            
            y_pred = np.round(y_pred,2)
            date_predict = (last_date + timedelta(days=1))
            sold_price  =  y_pred
            df_tmp = pd.DataFrame(data={'negotiation_date':date_predict,'sold_price':sold_price})            
            self.predict_agost = self.predict_agost.append(df_tmp,ignore_index=True)
            df_pred = df_pred.append(df_tmp,ignore_index= True)
            df6 = df_pred.copy()
            last_date = last_date+timedelta(days=1)                  
            
        v = self.predict_agost['sold_price'].values.tolist()                          
        save_pred_txt(v)
                    
        
    def trainingModels(self,df4):        
                        
        # foi feito um stacking de modelos, utilizando LightGBM e XGBBoost
        # o resultado é uma media ponderada de ambos os modelos
        
        print('Treinando modelos...')
        lgbm = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', learning_rate=0.1, max_depth=-1,
              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
              n_estimators=1000, n_jobs=-1, num_leaves=31, objective=None,
              random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
        
        xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=700,
             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
        
        #separando os atributos de entrada para os modelos e saida
        X = df4.drop(['negotiation_date','sold_price'],axis=1)
        y = df4['sold_price']
        
        lgbm.fit(X,y)
        xgb.fit(X,y)
        
        #salvandos os modelos treinados
        print('Salvando modelos...')
        pickle.dump(lgbm,open('./lgbm_model.pkl','wb'))
        pickle.dump(xgb,open('./xgb_model.pkl','wb'))
        
       
    # realiza as predicoes do mes de agosto/2019
    def predict_month_ago(self):
        print('Realizando previsoes para os 30 dias de agosto/2019...')        
        self.getPredict30daysAgo(self.df_tmp)
                           
    
    def treinar(self):        
        d1 = self.ChangeType(self.bd)
        d2 = self.SeriesCorrect(d1)                
        d3 = self.FeatureEngineering(d2)
        d4 = self.DataPreparation(d3)
        self.trainingModels(d4)
        


modelo = HackathonJPAAgro()
modelo.treinar()
modelo.predict_month_ago()
print('Resultado previsao para agosto!\n')
print( modelo.predict_agost )
print('Arquivo de previsao salvo!')
        
        
        
        
        

        
        
        