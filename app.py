import streamlit as st
import pandas  as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title('K-Means 클러스터링 앱')
    st.text('asdad')
    # 1. csv 파일을 업로드 할 수 있다.
    csv_file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    # 2. 업로드 한 csv 파일을 데이터 프레임으로 읽는다.
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.dataframe(df)

        st.subheader('Nan 데이터 확인')
        st.dataframe(df.isna().sum())

        st.subheader('결측값 처리한 결과')
        df = df.dropna().reset_index(drop=True)
        st.dataframe(df)

        st.subheader('클러스터링에 사용할 컬럼 선택')
        selected_columns = st.multiselect('X로 사용할 선택할 컬럼', df.columns)
        if len(selected_columns) != 0:
            X = df[selected_columns]
            st.dataframe(X)

            # 새로운 데이터 프레임
            X_new = pd.DataFrame()
            for name in X.columns:
            # 데이터가 문자열이면, 데이터의 종류가 몇개인지 확인한다.
                if X[name].dtype == object:
                    if X[name].nunique() > 2:
                    # 원핫 인코딩
                        ct = ColumnTransformer([
                            ('encoder', OneHotEncoder(), [0])  
                            ], remainder='passthrough')
                        col_names = sorted(X[name].unique())
                        X_new[col_names] = ct.fit_transform( X[name].to_frame())
                    else :
                    # 레이블 인코딩
                        label_encoder = LabelEncoder()
                        X_new[name] = label_encoder.fit_transform(X[name])
                # 숫자 데이터 일때 처리
                else:
                    X_new[name] = X[name]
            
            st.subheader('문자열은 숫자로 바꿔줍니다.')
            st.dataframe(X_new)

            # 값 평균 화 (피처 스케일링)
            st.subheader('피처 스케일링')
            scaler = MinMaxScaler()
            X_new = scaler.fit_transform(X_new)
            st.dataframe(X_new)

            # 유저가 입력한 파일의 데이터 갯수를 세어서
            # 해당 데이터의 갯수가 10보다 작으면, 데이터의 갯수까지만 wcss를 구하고 10보다 크면, 10개로 한다.
            X_new.shape[0]  # 튜플로 나오는 데이터 중 첫번째 값을 가져옴 (데이터의 길이를 나타내니깐)
            if X_new.shape[0] < 10:
                max_count = X_new.shape[0]
            else :
                max_count = 10

            wcss = []
            for k in range(1, max_count+1):
                kmeans = KMeans(n_clusters = k, random_state=5, n_init = 'auto')
                kmeans.fit(X_new) # 학습만 시킨다.
                # wcss 값을 가져옴
                wcss.append(kmeans.inertia_)

            st.write(wcss)

            fig = plt.figure()
            x = np.arange(1, max_count+1)
            plt.plot(
                x,
                wcss,    
            )

            plt.title('The Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            st.pyplot(fig)

            st.subheader('클러스터링 갯수 선택')
            k = st.slider('클러스터링을 실행할 수를 정하세요', min_value=1, max_value=max_count, value=2)

            kmeans = KMeans(n_clusters=k, random_state=5, n_init='auto')
            y_pred = kmeans.fit_predict(X_new)
            df['Group'] = y_pred
            
            st.subheader('그루핑 정보 표시')
            st.dataframe(df)

            st.subheader('보고 싶은 그룹을 선택하세요')
            group_number = st.number_input('그룹 번호 선택', 0, k-1)

            st.dataframe(df.loc[df['Group'] == group_number, ])

            df.to_csv('result.csv', index=False)


            

            




if __name__ == '__main__':
    main()