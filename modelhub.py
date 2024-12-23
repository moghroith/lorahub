import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import humanize
import logging
import os

class LoRAExplorer:
    MODEL_TYPES = ["SDXL", "SD 1.5", "Pony"]

    def __init__(self):
        st.set_page_config(page_title="LoRA Explorer", page_icon="🎨", layout="wide")
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    @staticmethod
    def convert_file_size(size_in_bytes):
        try:
            return humanize.naturalsize(int(size_in_bytes or 0))
        except (ValueError, TypeError):
            return "0 B"

    @classmethod
    @st.cache_data
    def load_data(cls):
        db_path = "models.db"

        if not os.path.exists(db_path):
            st.error(f"Database file {db_path} not found.")
            return pd.DataFrame()

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sdxl_models', 'sd15_models', 'pony_models');")
                tables = [row[0] for row in cursor.fetchall()]

                if not set(tables) == set(['sdxl_models', 'sd15_models', 'pony_models']):
                    st.error("Required database tables are missing.")
                    return pd.DataFrame()

                query = """
                SELECT 'SDXL' as model_type, model_name, file_uuid, version_name, created_at, file_size FROM sdxl_models
                UNION ALL
                SELECT 'SD 1.5', model_name, file_uuid, version_name, created_at, file_size FROM sd15_models
                UNION ALL
                SELECT 'Pony', model_name, file_uuid, version_name, created_at, file_size FROM pony_models;
                """
                df = pd.read_sql_query(query, conn)

            df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
            df["file_uuid"] = df["file_uuid"].str.replace(".safetensors", "", regex=False)
            df["file_size"] = pd.to_numeric(df["file_size"], errors="coerce")
            df["file_size_r"] = df["file_size"].apply(cls.convert_file_size)
            return df

        except sqlite3.Error as e:
            st.error(f"SQLite error: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Unexpected error loading data: {e}")
            return pd.DataFrame()

    def filter_data(self, df, model_type, search_term):
        if model_type != "All":
            df = df[df["model_type"] == model_type]
        if search_term:
            df = df[
                df["model_name"].str.contains(search_term, case=False, na=False) |
                df["file_uuid"].str.contains(search_term, case=False, na=False)
            ]
        return df

    def show_explorer(self, df):
        st.subheader("🎨 LoRA Explorer")

        st.sidebar.title("Filters")
        model_type = st.sidebar.radio("Select LoRA Type", self.MODEL_TYPES + ["All"])
        search_term = st.sidebar.text_input("Search LoRAs", placeholder="Search Spell Name or LoRA ID")

        cols = st.columns(3)
        for i, model in enumerate(self.MODEL_TYPES):
            model_df = df[df["model_type"] == model]
            with cols[i]:
                st.metric(f"{model} LoRAs", len(model_df))
                st.metric(f"Total {model} Size", self.convert_file_size(model_df["file_size"].sum()))

        st.subheader("📊 Spell Details")
        filtered_df = self.filter_data(df, model_type, search_term)
        display_df = (
            filtered_df[["model_name", "version_name", "file_uuid", "created_at", "model_type"]]
            .rename(
                columns={
                    "model_name": "Spell Name",
                    "version_name": "Version",
                    "file_uuid": "LoRA ID",
                    "created_at": "Upload Date",
                    "model_type": "Model Type",
                }
            )
            .sort_values("Upload Date", ascending=False)
            .reset_index(drop=True)
        )

        if not display_df.empty:
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No models found matching the search criteria.")

    def show_stats(self, df):
        st.subheader("📊 LoRA Statistics")

        cols = st.columns(2)
        with cols[0]:
            st.metric("Total LoRAs", len(df))
            st.metric("Total Size", self.convert_file_size(df["file_size"].sum()))

        st.subheader("LoRA Type Breakdown")
        for model in self.MODEL_TYPES:
            model_df = df[df["model_type"] == model]
            cols = st.columns(2)
            with cols[0]:
                st.metric(f"{model} LoRAs", len(model_df))
            with cols[1]:
                st.metric(f"Total {model} Size", self.convert_file_size(model_df["file_size"].sum()))

        df["Upload Date"] = df["created_at"].dt.date
        daily_uploads = df.groupby(["Upload Date", "model_type"]).size().reset_index(name="Uploads")
        cumulative_uploads = daily_uploads.copy()
        cumulative_uploads["Cumulative Uploads"] = cumulative_uploads.groupby("model_type")["Uploads"].cumsum()

        st.subheader("Daily LoRA Uploads (Logarithmic)")
        fig_daily = px.bar(
            daily_uploads,
            x="Upload Date",
            y="Uploads",
            color="model_type",
            title="Daily LoRA Uploads",
            log_y=True
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        st.subheader("Cumulative LoRA Uploads")
        fig_cumulative = px.line(
            cumulative_uploads,
            x="Upload Date",
            y="Cumulative Uploads",
            color="model_type",
            title="Cumulative LoRA Uploads"
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

    def run(self):
        df = self.load_data()

        if df.empty:
            st.warning("No models found in the database.")
            return

        page = st.sidebar.radio("Navigate", ["Explorer", "Stats"])

        if page == "Explorer":
            self.show_explorer(df)
        else:
            self.show_stats(df)

def main():
    app = LoRAExplorer()
    app.run()

if __name__ == "__main__":
    main()
