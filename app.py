from flask import Flask,render_template,request
import pickle
import tensorflow as tf
import keras
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

merged_df=pd.read_csv('mock-data.csv')
final_df=merged_df.sample(frac=1,ignore_index=True,random_state=42)

def predict_Policy(vm_profile):
    features_encoder=joblib.load('std_scaler.bin')
    model=tf.keras.models.load_model('model.keras')

    vm_prof_standardized=features_encoder.transform(vm_profile)
    policy_pred=model.predict(vm_prof_standardized,verbose=0)
    sorted_policies=np.argsort(policy_pred[0])[::-1]
    return sorted_policies[:3]

# Function to return top 3 highest avg reliability B&R locations for a given user location and policy number
def best_BR_Loc(location,pol):
    temp=final_df.loc[(final_df['Location']==location) & (final_df['Policy_Num']==pol)]
    temp=temp[['B&R_Location','Reliability_%']]
    temp=temp.groupby(['B&R_Location'],as_index=False).mean()
    temp.sort_values('Reliability_%',ascending=False,inplace=True)
    locs=[i[0] for i in temp.values[0:3]]
    loc=', '.join(locs)
    return loc

# Function to find avg cost for a given policy number
def avg_Cost_Policy(pol):
    avg_cost=final_df[final_df.Policy_Num==pol]['Cost_$'].mean()
    return avg_cost   

# Function to find avg oncloud% and onprem% for a given policy number
def avg_Cloud_Onprem(pol):
    temp=final_df.loc[final_df['Policy_Num']==pol]
    avg_cloud=temp['Cloud_%'].mean()
    avg_onprem=temp['OnPrem_%'].mean()
    return avg_cloud,avg_onprem

# Function to return the final output dataframe
def output_Information(predicted_pol,location):
    list_data=[]
    
    for idx,pol in enumerate(predicted_pol):
        new_list=[]
        avg_cost=avg_Cost_Policy(pol)
        loc=best_BR_Loc(location,pol)
        avg_cloud,avg_onprem=avg_Cloud_Onprem(pol)
        
        new_list.append(f'Policy-{pol}')
        new_list.append(round(avg_cost,2))
        new_list.append(loc)
        new_list.append(round(avg_cloud,2))
        new_list.append(round(avg_onprem,2))
        list_data.append(new_list)
    output_df=pd.DataFrame(list_data,columns=['Policy_Num','Average_Cost_$','Best_B&R_Locations','Ideal_On_Cloud_%','Ideal_On_Prem_%'])
    return output_df

app=Flask(__name__)

@app.route('/')
def index():
   
    # return render_template('index.html')
    # User Data
    read_iops=120
    write_iops=76
    read_latency=170
    write_latency=112
    memory=3072
    cpu_cores=43
    location='Michigan'
    # Users OnCloud and OnPrem %
    # Users Cost  

    # Finding recommended policy for the given user
    predicted_pol=predict_Policy([[read_iops,write_iops,read_latency,write_latency,memory,cpu_cores]])
    final_output=output_Information(predicted_pol,location)
    return render_template('index.html',final_df=final_output.to_json(orient='columns'))

@app.route('/VM44')
def VM44():
    render_template('Flask Proj/templates/VM44.html')


if __name__=='__main__':
    app.run(debug=True)