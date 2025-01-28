import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Judul aplikasi
st.set_page_config(page_title="Clustering", layout="wide")
st.title("Clustering")

# Membaca dataset secara langsung dari file yang disediakan
dataset_path = "Clustering.csv"
try:
    df = pd.read_csv(dataset_path)
    st.success("Dataset berhasil dimuat!")

    # Menu navigasi
    menu = st.sidebar.selectbox("Menu", ["Beranda", "Visualisasi", "K-Means"])

    if menu == "Beranda":
        st.subheader("Beranda")
        st.write("Dataset memiliki **{} baris** dan **{} kolom**.".format(df.shape[0], df.shape[1]))
        st.write("Berikut adalah dataset yang digunakan:")
        st.dataframe(df)

    elif menu == "Visualisasi":
        st.subheader("Visualisasi")
        show_histogram = st.checkbox("Tampilkan Distribusi Usia")

        if show_histogram:
            st.write("Distribusi Usia")
            column = st.selectbox("Pilih Kolom untuk Histogram:", df.columns)
            bins = st.slider("Jumlah Bins:", min_value=5, max_value=50, value=10)

            fig, ax = plt.subplots()
            sns.histplot(df[column], bins=bins, kde=True, color='skyblue', ax=ax)
            ax.set_title(f"Distribusi {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            st.pyplot(fig)

    elif menu == "K-Means":
        st.subheader("K-Means")

        # Pilih kolom untuk clustering
        st.write("Pilih kolom fitur untuk clustering:")
        features = st.multiselect("Pilih Kolom Fitur", df.columns, default=df.columns[:2])

        if len(features) > 1:
            # Standardisasi data
            st.write("Menstandarkan data untuk K-Means...")
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[features])

            # Tentukan jumlah klaster
            k = st.slider("Pilih jumlah klaster (K):", min_value=1, max_value=10, value=3)

            # Terapkan K-Means
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df_scaled)

            st.write(f"Jumlah Klaster: {k}")
            st.write("Hasil clustering dengan K-Means:")

            # Tampilkan tabel hasil clustering
            st.dataframe(df[['Cluster'] + features])

            # Visualisasi klaster
            st.write("Visualisasi Klaster:")
            fig, ax = plt.subplots()

            sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Cluster'], palette='viridis', ax=ax)
            ax.set_title(f"K-Means(K={k})")
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            st.pyplot(fig)
        else:
            st.warning("Pilih lebih dari satu kolom fitur untuk clustering!")


except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")






# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans

# # Konfigurasi tema warna Streamlit
# st.set_page_config(page_title="Analisis Demografi & K-Means Clustering", layout="wide")
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #f0f8ff;
#         color: #1e3a8a;
#     }
#     .stApp {
#         background-color: #f0f8ff;
#     }
#     .sidebar .sidebar-content {
#         background-color: #1e3a8a;
#         color: #f0f8ff;
#     }
#     .sidebar .sidebar-content .css-1d391kg {
#         color: #f0f8ff;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Judul aplikasi
# st.title("Analisis Demografi & K-Means Clustering")

# # Membaca dataset secara langsung dari file yang disediakan
# dataset_path = "Clustering.csv"
# try:
#     df = pd.read_csv(dataset_path)
#     st.success("Dataset berhasil dimuat!")

#     # Menu navigasi
#     menu = st.sidebar.radio("Navigasi", ["About", "Dataset", "Run Clustering"])

#     if menu == "About":
#         st.subheader("About")
#         st.write("Aplikasi ini dirancang untuk melakukan analisis demografi dan clustering menggunakan algoritma K-Means.")

#     elif menu == "Dataset":
#         st.subheader("Dataset dan Visualisasi")
#         st.write("Berikut adalah dataset yang digunakan:")
#         st.dataframe(df)

#         # Visualisasi Clustering
#         st.subheader("Visualisasi Clustering")
#         show_clustering = st.checkbox("Tampilkan Visualisasi Clustering")

#         if show_clustering:
#             features = st.multiselect("Pilih Kolom untuk Clustering", df.select_dtypes(include=['int64', 'float64']).columns.tolist(), default=['Age', 'Income'])
#             if len(features) == 2:
#                 n_clusters = st.slider("Jumlah Klaster", min_value=2, max_value=10, value=3)

#                 kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#                 clusters = kmeans.fit_predict(df[features])
#                 df['Cluster'] = clusters

#                 fig, ax = plt.subplots()
#                 scatter = ax.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis', s=50)
#                 legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
#                 ax.add_artist(legend1)
#                 ax.set_xlabel(features[0])
#                 ax.set_ylabel(features[1])
#                 ax.set_title(f"Clustering K-Means berdasarkan {features[0]} dan {features[1]}")
#                 st.pyplot(fig)
#             else:
#                 st.warning("Pilih tepat dua kolom untuk visualisasi clustering.")

#     elif menu == "Run Clustering":
#         st.subheader("K-Means Clustering")
#         st.write("Fitur ini sedang dalam pengembangan. Nantikan pembaruan berikutnya!")

# except Exception as e:
#     st.error(f"Gagal memuat dataset: {e}")








# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # Load Dataset
# st.title("Clustering Analysis App")

# uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type="csv")
# if uploaded_file is not None:
#     # Read the uploaded file
#     data = pd.read_csv(uploaded_file)
#     st.subheader("Dataset Overview")
#     st.dataframe(data.head())

#     # Show basic statistics
#     st.subheader("Statistical Summary")
#     st.write(data.describe())

#     # Select columns for clustering analysis
#     st.subheader("Select Columns for Clustering")
#     numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     selected_columns = st.multiselect("Choose numeric columns:", numeric_columns, default=numeric_columns)

#     if selected_columns:
#         # Standardize the data
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(data[selected_columns])

#         # Apply PCA for dimensionality reduction
#         pca = PCA(n_components=2)
#         pca_result = pca.fit_transform(scaled_data)
        
#         # Create a DataFrame for PCA results
#         pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])

#         # Combine PCA results with the original data
#         final_df = pd.concat([pca_df, data], axis=1)

#         st.subheader("PCA Result Visualization")
#         fig, ax = plt.subplots()
#         ax.scatter(final_df['PCA1'], final_df['PCA2'], alpha=0.7)
#         ax.set_xlabel("PCA1")
#         ax.set_ylabel("PCA2")
#         ax.set_title("Clustering Visualization")
#         st.pyplot(fig)

#         # Allow filtering by a specific column
#         st.subheader("Filter Data")
#         filter_column = st.selectbox("Select a column to filter:", data.columns)
#         if filter_column:
#             unique_values = data[filter_column].unique()
#             selected_value = st.selectbox("Select a value to filter by:", unique_values)
#             filtered_data = data[data[filter_column] == selected_value]
#             st.write(f"Filtered Data ({filter_column} = {selected_value})")
#             st.dataframe(filtered_data)

#     else:
#         st.warning("Please select at least one numeric column for analysis.")
# else:
#     st.info("Please upload a CSV file to proceed.")
