import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')


df_clean = df.dropna(subset=['age', 'embarked'])


st.header('Survival Count by Passenger Class')
fig1, ax1 = plt.subplots()
sns.countplot(data=df_clean, x='class', hue='survived')
st.pyplot(fig1)

st.header('Age Distribution: Survivors vs Non-Survivors')
fig2, ax2 = plt.subplots()
sns.histplot(data=df_clean, x='age', hue='survived', bins=30, kde=True)
st.pyplot(fig2)

survival_by_gender = df_clean.groupby('sex')['survived'].mean() * 100

st.header('Survival Rate by Gender (%)')
fig3, ax3 = plt.subplots()
sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values)
st.pyplot(fig3)



avg_fare_by_class = df_clean.groupby('class')['fare'].mean()

st.header('Average Fare Paid by Class')
fig4, ax4 = plt.subplots()
avg_fare_by_class.plot(kind='bar', color=['#F18F01', '#C73E1D', '#6A994E'])
st.pyplot(fig4)