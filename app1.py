from unittest import result
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 隐藏made with streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('A machine learning-based predictive model for predicting bone metastasis in patients with breast cancer') #主栏
st.sidebar.title('Variables') #侧边栏
#st.text('Variables')
#option = st.selectbox(
#   '标题',
#   ('选项1', '选项2', '选项3'))

CE153 = st.sidebar.slider('CA153', 0, 999)
CEA = st.sidebar.slider('CEA', 0, 999)
UA = st.sidebar.slider('UA', 0, 999)
CA125 = st.sidebar.slider('CA125', 0, 999)
Pathological_type_list=['non-invasive', 'invasive']
Pathological_type_name=st.sidebar.selectbox(
    "Pathological type",
    Pathological_type_list
)
if Pathological_type_name=='non-invasive':
    Pathological_type_num=0
else:
     Pathological_type_num=1
# option = st.sidebar.selectbox(
#     'Pathological-type',
#        ('非浸润性', '浸润性'))
T_list=['T1', 'T2','T3','T4']
T_name=st.sidebar.selectbox(
    "T",
    T_list
)
if T_name=='T1':
    T_num=1
elif T_name=='T2':
    T_num=2 
elif T_name=='T3':
    T_num=3    
elif T_name=='T4':
    T_num=4  
# else:
#     T_num=4
LDH = st.sidebar.slider('LDH', 0, 999)
TBIL = st.sidebar.slider('TBIL', 0, 999)
# option = st.sidebar.selectbox(
#     'T',
#        ('T1', 'T2','T3','T4'))

import streamlit as st
import pandas as pd
import joblib
from joblib import dump, load
model = joblib.load(filename='RF.model')
c=[[CE153, CEA, UA, CA125, Pathological_type_num, LDH, TBIL,T_num]]
#a=model.predict(c)
b=model.predict_proba(c)
d=b[0][1]
#e=("%.2f" % d)
e='{:.2%}'.format(d)
#st.write(f"根据你的选择，发生转移的概率为{e}")
run_button=st.button('Predict')

#st.sidebar.title('菜单侧边栏')
if run_button:
    st.write('Probability of BM:',str(e))
    #st.title('Probability of BM:',str(e))
# form joblib import  dump, load
# RF =load(RF.model) 
#If button is pressed
# if st.button("Predict"):
    
#     # Unpickle classifier
#     clf = joblib.load("clf.pkl")
    
#     # Store inputs into dataframe
#     X = pd.DataFrame[[pathological_type, T, CEA, CA125, CE153, LDH, TBIL, UA]], 
#                   #  columns = ['pathological_type','T','CEA','CA125','CE153','LDH','TBIL','UA'])
    
#     # Get prediction
#     prediction = clf.predict(X)[0]
    
#     # Output prediction
#     st.text(f"This instance is a {prediction}")


# from sklearn import feature_selection
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# data=pd.read_csv('D:\临床课题资料\乳腺癌骨转移临床数据\乳腺癌机器学习模型/乳腺癌骨转移数据-标准后.csv')
# data_featureCata=data[['pathological_type','T']]
# data_featureNum=data[['CEA','CA125','CE153','LDH','TBIL','UA']]

# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# data_featureNum=scaler.fit_transform(data_featureNum)

# data_featureCata=np.array(data_featureCata)
# data_feature=np.hstack((data_featureCata,data_featureNum))

# data_targetClass=data['status']
# class_X_train,class_X_test,class_y_train,class_y_test=train_test_split(data_feature,data_targetClass,test_size=0.3,random_state=0)

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score

# def clf=RandomForestClassifier(criterion='entropy', max_depth=9, n_estimators=461,bootstrap=False,random_state=2)
#   clf.fit(class_X_train,class_X_train)
#     result=clf.score(class_X_train,class_X_train)
#     return result


# run_button=st.button('Predict')

# if run_button:
#    result=RandomForestClassifier(criterion='entropy', max_depth=9, n_estimators=461,bootstrap=False,random_state=2)
#    st.write('RandomForestClassifier accuracy:',str(result))
