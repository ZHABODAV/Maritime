import unittest
from analytics import calculate_voyage_kpis


class TestAnalytics(unittest.TestCase):
    def test_calculate_voyage_kpis_basic(self):
        voyages = [
            {
                "id": "V001",
                "legs": [
                    {"duration": 5, "cost": 10000, "cargo": 2000},
                    {"duration": 3, "cost": 6000, "cargo": 1500}
                ],
                "contract_type": "spot",
                "rate_per_ton": 10
            },
            {
                "id": "V002",
                "legs": [
                    {"duration": 7, "cost": 15000, "cargo": 3000}
                ],
                "contract_type": "time-charter",
                "expenses": {
                    "rent": 7000,
                    "bunker": 5000,
                    "ports": 2000,
                    "canals": 500,
                    "insurance": 300,
                    "other": 200
                }
            }
        ]

        kpis = calculate_voyage_kpis(voyages)

        # Проверка структуры
        self.assertIn("V001", kpis)
        self.assertIn("V002", kpis)

        # Проверка расчетов по спотовому контракту
        v1 = kpis["V001"]
        self.assertEqual(v1["duration"], 8)  # 5 + 3
        self.assertEqual(v1["cargo"], 3500)  # 2000 + 1500
        # Доход = ставка*тоннаж = 10*3500 = 35000; прибыль = доход - cost = 35000 - 16000 = 19000
        self.assertEqual(v1["profit"], 19000)

        # Проверка расчетов по тайм-чартеру
        v2 = kpis["V002"]
        self.assertEqual(v2["duration"], 7)
        self.assertEqual(v2["cargo"], 3000)
        # Стоимость рейса = сумма расходов
        self.assertEqual(v2["cost"], 15000 + sum(voyages[1]["expenses"].values()))


if __name__ == "__main__":
    unittest.main()