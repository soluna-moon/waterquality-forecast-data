import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("💧 Water Quality Analysis & Forecast")

# 1️⃣ Загрузка CSV
uploaded_file = st.file_uploader(
    "/home/soluna/Загрузки/archive/Cities1.csv", type="csv"
)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("The first rows of data:")
    st.dataframe(data.head())
    cities = st.multiselect(
        "Select the cities",
        options=data["City"].unique(),
        default=data["City"].unique(),
    )
    filtered_data = data[data["City"].isin(cities)]
    # 2️⃣ Проверка наличия даты
    date_cols = [col for col in data.columns if "date" in col.lower()]
    if date_cols:
        date_col = st.selectbox("Select the date column", options=date_cols)
        data[date_col] = pd.to_datetime(data[date_col])
        data["Day"] = (data[date_col] - data[date_col].min()).dt.days
    else:
        st.info("The date column was not found. The row index will be used as 'Day'.")
        data["Day"] = range(len(data))
        # 3️⃣ Выбор колонок для анализа
        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "Day"]
        value_cols = st.multiselect(
            "Select numeric columns for analysis", options=numeric_cols
        )
    if value_cols:
        # 4️⃣ Построение интерактивного графика
        fig = go.Figure()
        for col in value_cols:
            X = data[["Day"]]
            y = data[col]

            # Простейшая линейная регрессия
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Добавляем реальные данные
            fig.add_trace(
                go.Scatter(
                    x=data["Day"], y=y, mode="lines+markers", name=f"{col} (real)"
                )
            )
            # Добавляем прогноз
            fig.add_trace(
                go.Scatter(
                    x=data["Day"].iloc[len(X_train) :],
                    y=y_pred,
                    mode="lines",
                    line=dict(dash="dash"),
                    name=f"{col} (forecast)",
                )
            )
            fig.update_layout(
                title="📈 Water Quality Trends & Forecast",
                xaxis_title="Day",
                yaxis_title="Meaning",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Прогноз на будущее
            future_days = st.number_input(
                "Forecast for the following days", min_value=1, max_value=365, value=7
            )
            if future_days > 0:
                future_fig = go.Figure()
                for col in value_cols:
                    X = data[["Day"]]
                    y = data[col]
                    model = LinearRegression()
                    model.fit(X, y)
                    last_day = data["Day"].max()
                    future_X = pd.DataFrame(
                        {"Day": range(last_day + 1, last_day + future_days + 1)}
                    )
                    future_pred = model.predict(future_X)
                    future_fig.add_trace(
                        go.Scatter(
                            x=future_X["Day"],
                            y=future_pred,
                            mode="lines+markers",
                            name=f"{col} (future forecast)",
                        )
                    )
                    future_fig.update_layout(
                        title="📊 Forecast for Future Days",
                        xaxis_title="Day",
                        yaxis_title="Meaning",
                        hovermode="x unified",
                    )
                    st.plotly_chart(future_fig, use_container_width=True)
