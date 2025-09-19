import logging

# Безопасный импорт драйвера Neo4j
try:
    from neo4j import GraphDatabase  # pip package: neo4j
except Exception as e:
    GraphDatabase = None

# Логгер для отслеживания операций работы с Neo4j
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jIntegration:
    def __init__(self, uri: str, user: str, password: str):
        """Инициализация драйвера подключения к Neo4j"""
        if GraphDatabase is None:
            raise ImportError("Пакет 'neo4j' не установлен. Добавьте 'neo4j' в requirements.txt")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Подключение к Neo4j инициализировано")

    def close(self):
        """Закрыть соединение с БД"""
        try:
            self.driver.close()
            logger.info("Соединение с Neo4j закрыто")
        except Exception as e:
            logger.warning(f"Ошибка при закрытии соединения Neo4j: {e}")

    def create_voyage_graph(self, voyages: list):
        """
        Загрузка рейсов в Neo4j.
        voyages — массив словарей вида:
        {
            "VOY_ID": "001",
            "ports_load": ["RUAST", "RUPORT"],
            "ports_discharge": ["TRIST", "GRPIR"],
            "cargo": "Железная руда",
            "contract": "Spot"
        }
        """
        if GraphDatabase is None:
            logger.warning("Neo4j недоступен (нет драйвера). Пропускаю загрузку графа.")
            return

        with self.driver.session() as session:
            for voy in voyages:
                logger.info(f"Создание узлов для рейса {voy.get('VOY_ID')}")
                session.run(
                    """
                    MERGE (v:Voyage {id:$voy_id})
                    SET v.cargo = $cargo, v.contract = $contract
                    """,
                    voy_id=voy.get("VOY_ID"),
                    cargo=voy.get("cargo"),
                    contract=voy.get("contract"),
                )
                for p in voy.get("ports_load", []) or []:
                    session.run("MERGE (p:Port {locode:$locode})", locode=p)
                    session.run(
                        """
                        MATCH (v:Voyage {id:$voy_id}), (p:Port {locode:$locode})
                        MERGE (v)-[:LOADS_AT]->(p)
                        """,
                        voy_id=voy.get("VOY_ID"),
                        locode=p,
                    )
                for p in voy.get("ports_discharge", []) or []:
                    session.run("MERGE (p:Port {locode:$locode})", locode=p)
                    session.run(
                        """
                        MATCH (v:Voyage {id:$voy_id}), (p:Port {locode:$locode})
                        MERGE (v)-[:DISCHARGES_AT]->(p)
                        """,
                        voy_id=voy.get("VOY_ID"),
                        locode=p,
                    )

    def get_all_voyages(self):
        """Вытащить все рейсы из БД"""
        if GraphDatabase is None:
            logger.warning("Neo4j недоступен (нет драйвера). Возвращаю пустой список.")
            return []
        with self.driver.session() as session:
            result = session.run(
                "MATCH (v:Voyage) RETURN v.id AS id, v.cargo AS cargo, v.contract AS contract"
            )
            return [record.data() for record in result]


# Упрощённые функции для совместимости с app.py
def neo4j_connection(uri: str, user: str, password: str):
    """
    Возвращает драйвер Neo4j либо None, если драйвер недоступен или подключение не удалось.
    """
    if GraphDatabase is None:
        logger.warning("Пакет 'neo4j' отсутствует. Пропускаю подключение к Neo4j.")
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Проверка подключения (ленивая) — откроем и закроем сессию
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("Успешное подключение к Neo4j")
        return driver
    except Exception as e:
        logger.error(f"Не удалось подключиться к Neo4j: {e}")
        return None


def build_and_query_graph(data: dict):
    """
    Возвращает Plotly Figure для визуализации графа.
    Если Neo4j недоступен, строит простую диаграмму по данным портов без Neo4j.
    """
    try:
        import plotly.graph_objects as go
        import pandas as pd
    except Exception as e:
        logger.error(f"Plotly/Pandas не установлены: {e}")
        return None

    ports = data.get("ports", [])
    if not ports:
        fig = go.Figure().add_annotation(
            text="Нет данных по портам для визуализации графа",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Простая визуализация узлов портов (без Neo4j)
    xs, ys, texts = [], [], []
    for p in ports:
        xs.append(p.get("lon", 0))
        ys.append(p.get("lat", 0))
        texts.append(p.get("name", "Порт"))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=texts,
                textposition="top center",
                marker=dict(size=10, color="blue"),
                name="Порты",
            )
        ]
    )
    fig.update_layout(
        title="Визуализация графа портов (fallback без Neo4j)",
        xaxis_title="Долгота",
        yaxis_title="Широта",
        showlegend=False,
        height=500,
    )
    return fig