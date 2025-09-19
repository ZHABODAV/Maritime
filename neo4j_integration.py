from neo4j import GraphDatabase
import logging

# Логгер для отслеживания операций работы с Neo4j
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jIntegration:
    def __init__(self, uri: str, user: str, password: str):
        """Инициализация драйвера подключения к Neo4j"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Подключение к Neo4j инициализировано")

    def close(self):
        """Закрыть соединение с БД"""
        self.driver.close()
        logger.info("Соединение с Neo4j закрыто")

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
        with self.driver.session() as session:
            for voy in voyages:
                logger.info(f"Создание узлов для рейса {voy['VOY_ID']}")
                session.run(
                    """
                    MERGE (v:Voyage {id:$voy_id, cargo:$cargo, contract:$contract})
                    """,
                    voy_id=voy["VOY_ID"], cargo=voy["cargo"], contract=voy["contract"]
                )
                for p in voy["ports_load"]:
                    session.run("MERGE (p:Port {locode:$locode})", locode=p)
                    session.run(
                        """
                        MATCH (v:Voyage {id:$voy_id}), (p:Port {locode:$locode})
                        MERGE (v)-[:LOADS_AT]->(p)
                        """,
                        voy_id=voy["VOY_ID"], locode=p
                    )
                for p in voy["ports_discharge"]:
                    session.run("MERGE (p:Port {locode:$locode})", locode=p)
                    session.run(
                        """
                        MATCH (v:Voyage {id:$voy_id}), (p:Port {locode:$locode})
                        MERGE (v)-[:DISCHARGES_AT]->(p)
                        """,
                        voy_id=voy["VOY_ID"], locode=p
                    )

    def get_all_voyages(self):
        """Вытащить все рейсы из БД"""
        with self.driver.session() as session:
            result = session.run("MATCH (v:Voyage) RETURN v.id AS id, v.cargo AS cargo, v.contract AS contract")
            return [record.data() for record in result]