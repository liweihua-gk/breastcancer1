from unittest import result
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
st.title('雷枪医学信息技术的APP')
st.text('准备开始运行app')
code = '''def hello():
     print("Hello, Streamlit!")'''
st.code(code, language='python')
# image=Image.open('E:/streamlit/3e3d340f-1cca-4cc2-9aa4-64aca42555de.jpg')
# st.image(image, caption='ThunderSpear Tech')
data=pd.read_csv('E:/streamlit/Breast Cancer METABRIC.csv')





st.dataframe(data, 2000,400 )
# st.table(data)
# import matplotlib.pyplot as plt


# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)


st.title('交互随机森林模型')
mydata=st.file_uploader('上传你的数据')


if mydata:
    df=pd.read_csv(mydata)
    st.dataframe(df, 2000,400 )
    
option = st.selectbox(
     '选择criterion',
     ('gini','entropy'))

n_estimator=st.slider('n_estimator',10,30)
max_depth = st.number_input('Insert a number')
st.write('You selected:','criterion:', option,'n_estimator:',n_estimator,'max_depth:',max_depth)

from sklearn.ensemble import RandomForestClassifier

def RF_model(n_estimator,option,max_depth):
    features=df[['Sequece','Patient ID','Type of Breast Surgery','Cancer Type','Cancer Type Detailed']]
    target=df[['Cellularity']]
    clf=RandomForestClassifier(n_estimators=n_estimator,criterion=option,max_depth=max_depth)
    
    clf.fit(features,target)
    result=clf.score(features,target)
    return result

run_button=st.button('run_RFmodel')

if run_button:
    result=RF_model(n_estimator,option,max_depth)
    st.write('RandomForestClassifier accuracy:',str(result))



    
    
