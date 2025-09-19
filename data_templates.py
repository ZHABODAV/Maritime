import pandas as pd
import streamlit as st
from io import BytesIO

# --- Шаблонные структуры данных ---
TEMPLATES = {
    "ports": {
        "columns": ["Регион", "Порт", "LOCODE", "Причал", "Переваливаемые грузы", "Накопление максимум"]
    },
    "cargo_dict": {
        "columns": ["Тип груза", "Группа груза", "Наименование груза", "Краткое наименование"]
    },
    "ships": {
        "columns": ["Судно", "ИМО №", "Флаг", "Судовладелец", "LOA", "BEAM", "DM", "DRAFT", "DWT"]
    },
    "voyages": {
        "columns": ["VOY_ID", "Судно", "Тип контракта", "Порты погрузки", "Порты назначения", "Груз", "Тоннаж", "Дата погрузки", "Планируемая дата выгрузки"]
    },
    "voy_route_legs": {
        "columns": ["VOY_ID", "ROUTE.ID", "ROUTE.LEG", "ROUTE.LEG.TYPE", "OPS.RELATE", 
                    "ROUTE POINT DESCRIPTION", "OPS GROUP", "STATUS", "META_REF_OPS",
                    "DT.TIME-BEG", "DT.TIME FIN", "DUR", "COSTS"]
    }
}

# --- Функция генерации Excel шаблона ---
def generate_template(template_name: str):
    if template_name not in TEMPLATES:
        raise ValueError(f"Неизвестный шаблон: {template_name}")
    df = pd.DataFrame(columns=TEMPLATES[template_name]["columns"])
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    return buffer

# --- Streamlit UI для скачивания шаблонов ---
def download_template_ui():
    st.header("📥 Скачать шаблоны Excel для заполнения")
    for name in TEMPLATES:
        buffer = generate_template(name)
        st.download_button(
            label=f"Скачать шаблон {name}.xlsx",
            data=buffer,
            file_name=f"{name}_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- Streamlit UI для загрузки готовых данных ---
def upload_data_ui():
    st.header("📤 Загрузить готовые данные")
    uploaded_files = {}
    for name in TEMPLATES:
        file = st.file_uploader(f"Загрузить файл {name}.xlsx", type=["xlsx", "csv"], key=f"upload_{name}")
        if file:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            uploaded_files[name] = df
            st.success(f"Файл {file.name} успешно загружен для {name}")
    return uploaded_files


# --- Пример интеграции ---
if __name__ == "__main__":
    st.title("Модуль работы с шаблонами данных")
    download_template_ui()
    uploaded = upload_data_ui()
    if uploaded:
        for name, df in uploaded.items():
            st.subheader(f"Предпросмотр: {name}")
            st.dataframe(df)