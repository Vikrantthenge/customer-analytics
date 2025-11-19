# 📊 Customer Analytics & Segmentation Dashboard  
**AI-Enhanced Customer Insights • RFM Segmentation • K-Means Clustering • Cohort Retention • Streamlit Web App**

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?style=for-the-badge&logo=postgresql)
![ML](https://img.shields.io/badge/Machine%20Learning-KMeans-orange?style=for-the-badge)

---

## 🚀 Live Demo  
🔗 **Streamlit App:** https://customer-analytic.streamlit.app/  
🔗 **GitHub Repo:** https://github.com/Vikrantthenge/customer-analytics  

---

## 🖼️ Project Overview  

This project provides a full customer analytics workflow, integrating data engineering, machine learning, RFM scoring, clustering, and a modern interactive dashboard.

The app supports:  
✔ Exploratory analytics  
✔ RFM scoring  
✔ K-Means clustering  
✔ Cohort retention modeling  
✔ Customer lookup  
✔ Transaction-level filtering  
✔ Data export  

---

# 🏗️ Architecture  

```mermaid
flowchart TD
    A[Raw Online Retail Data] --> B[Python Cleaning & Preprocessing]
    B --> C[PostgreSQL Data Warehouse]
    C --> D[RFM Feature Engineering]
    D --> E[K-Means Segmentation]
    E --> F[Cohort & CLTV Modeling]
    F --> G[Streamlit Dashboard UI]
