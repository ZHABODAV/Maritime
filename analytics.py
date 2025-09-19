import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

def calculate_kpis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate key performance indicators"""
    
    kpis = {}
    
    # Basic counts
    print(f"Data keys: {list(data.keys())}")
    print(f"Ships count: {len(data.get('ships', []))}")
    print(f"Ports count: {len(data.get('ports', []))}")
    print(f"Berths count: {len(data.get('berths', []))}")
    
    kpis["total_vessels"] = len(data.get("ships", []))
    kpis["total_ports"] = len(data.get("ports", []))
    kpis["total_berths"] = len(data.get("berths", []))
    
    # Active operations
    active_statuses = ["Loading", "Discharge", "Transit"]
    kpis["active_operations"] = len([s for s in data.get("ships", []) if s.get("status") in active_statuses])
    
    # Port utilization
    occupied_berths = len([b for b in data.get("berths", []) if b.get("status") == "Occupied"])
    total_berths = len(data.get("berths", []))
    kpis["port_utilization"] = (occupied_berths / total_berths * 100) if total_berths > 0 else 0
    
    # Fleet efficiency (simplified calculation)
    active_vessels = len([s for s in data.get("ships", []) if s.get("status") in active_statuses])
    kpis["fleet_efficiency"] = (active_vessels / kpis["total_vessels"] * 100) if kpis["total_vessels"] > 0 else 0

    # KPI by voyage type (ТЧ / СПОТ)
    charter_groups = {"Тайм-чартер": [], "Спот": []}
    for ship in data.get("ships", []):
        charter_type = ship.get("charter_type", "Спот")
        if charter_type not in charter_groups:
            charter_groups[charter_type] = []
        charter_groups[charter_type].append(ship)

    for charter, vessels_group in charter_groups.items():
        if vessels_group:
            kpis[f"{charter}_кол-во"] = len(vessels_group)
            kpis[f"{charter}_средн_эффективность"] = np.mean([s.get("capacity",0) for s in vessels_group])

    # KPI по текущим рейсам
    voyages = data.get("voyages", [])
    kpis["всего_рейсов"] = len(voyages)
    kpis["выполненные_рейсы"] = len([v for v in voyages if v.get("STATUS") == "completed"])
    kpis["текущие_рейсы"] = len([v for v in voyages if v.get("STATUS") in ["enroute", "idle"]])
    kpis["планируемые_рейсы"] = len([v for v in voyages if v.get("STATUS") == "planned"])

    return kpis

def calculate_vessel_efficiency(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calculate efficiency metrics for each vessel"""
    
    efficiency_data = []
    
    for ship in data["ships"]:
        vessel_metrics = {
            "vessel_name": ship["name"],
            "vessel_id": ship["id"],
            "vessel_type": ship["type"],
            "capacity": ship.get("capacity", 0),
            "current_status": ship.get("status", "Unknown"),
        }
        
        # Calculate schedule efficiency
        schedule = ship.get("schedule", [])
        if schedule:
            # On-time performance (simulated)
            vessel_metrics["on_time_performance"] = np.random.uniform(70, 95)
            
            # Port turnaround time (hours)
            turnaround_times = []
            for sched in schedule:
                if sched.get("arrival") and sched.get("departure"):
                    try:
                        arrival = pd.to_datetime(sched["arrival"])
                        departure = pd.to_datetime(sched["departure"])
                        turnaround = (departure - arrival).total_seconds() / 3600
                        turnaround_times.append(turnaround)
                    except:
                        continue
            
            vessel_metrics["avg_turnaround_time"] = np.mean(turnaround_times) if turnaround_times else 24
            vessel_metrics["min_turnaround_time"] = min(turnaround_times) if turnaround_times else 12
            vessel_metrics["max_turnaround_time"] = max(turnaround_times) if turnaround_times else 48
            
            # Cargo utilization
            total_quantity = sum(s.get("quantity", 0) for s in schedule)
            capacity_utilization = (total_quantity / vessel_metrics["capacity"] * 100) if vessel_metrics["capacity"] > 0 else 0
            vessel_metrics["capacity_utilization"] = min(capacity_utilization, 100)  # Cap at 100%
            
        else:
            # Default values for vessels without schedule
            vessel_metrics["on_time_performance"] = np.random.uniform(60, 85)
            vessel_metrics["avg_turnaround_time"] = np.random.uniform(18, 36)
            vessel_metrics["min_turnaround_time"] = np.random.uniform(8, 18)
            vessel_metrics["max_turnaround_time"] = np.random.uniform(36, 72)
            vessel_metrics["capacity_utilization"] = np.random.uniform(60, 90)
        
        # Fuel efficiency (simulated based on vessel characteristics)
        base_consumption = ship.get("fuel_consumption", np.random.uniform(15, 35))
        speed = ship.get("speed", np.random.uniform(12, 18))
        
        # Calculate fuel efficiency score (higher is better)
        fuel_efficiency = 100 - ((base_consumption - 15) / 20 * 30)  # Scale to 70-100
        vessel_metrics["fuel_efficiency"] = max(70, min(100, fuel_efficiency))
        
        # Speed efficiency
        optimal_speed = 15  # Assume 15 knots is optimal
        speed_variance = abs(speed - optimal_speed) / optimal_speed
        speed_efficiency = 100 - (speed_variance * 50)
        vessel_metrics["speed_efficiency"] = max(70, min(100, speed_efficiency))
        
        # Overall efficiency score (weighted average)
        weights = {
            "on_time_performance": 0.25,
            "capacity_utilization": 0.25,
            "fuel_efficiency": 0.25,
            "speed_efficiency": 0.15,
            "turnaround_efficiency": 0.10
        }
        
        # Turnaround efficiency (inverse of turnaround time, normalized)
        turnaround_efficiency = 100 - min(float((vessel_metrics["avg_turnaround_time"] - 12) / 60 * 100), 50.0)
        vessel_metrics["turnaround_efficiency"] = max(50, turnaround_efficiency)
        
        # Calculate weighted efficiency score
        efficiency_score = (
            vessel_metrics["on_time_performance"] * weights["on_time_performance"] +
            vessel_metrics["capacity_utilization"] * weights["capacity_utilization"] +
            vessel_metrics["fuel_efficiency"] * weights["fuel_efficiency"] +
            vessel_metrics["speed_efficiency"] * weights["speed_efficiency"] +
            vessel_metrics["turnaround_efficiency"] * weights["turnaround_efficiency"]
        )
        
        vessel_metrics["efficiency_score"] = efficiency_score
        
        # Performance rating
        if efficiency_score >= 90:
            vessel_metrics["performance_rating"] = "Excellent"
        elif efficiency_score >= 80:
            vessel_metrics["performance_rating"] = "Good" 
        elif efficiency_score >= 70:
            vessel_metrics["performance_rating"] = "Fair"
        else:
            vessel_metrics["performance_rating"] = "Needs Improvement"
        
        # Add additional operational metrics
        vessel_metrics["maintenance_score"] = np.random.uniform(75, 95)
        vessel_metrics["crew_efficiency"] = np.random.uniform(80, 95)
        vessel_metrics["safety_score"] = np.random.uniform(85, 100)
        
        efficiency_data.append(vessel_metrics)
    
    return efficiency_data

def get_performance_trends(data: Dict[str, Any], days: int = 30) -> Dict[str, Any]:
    """Generate performance trend data"""
    
    # Generate time series data for the last 30 days
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    trends = {
        "dates": [d.strftime('%Y-%m-%d') for d in dates],
        "vessel_utilization": [],
        "port_efficiency": [],
        "fuel_consumption": [],
        "on_time_performance": []
    }
    
    # Generate realistic trend data with some seasonality and noise
    base_utilization = 75
    base_port_efficiency = 82
    base_fuel_consumption = 25
    base_on_time = 88
    
    for i, date in enumerate(dates):
        # Add weekly seasonality (lower on weekends)
        weekly_factor = 1.0 if date.weekday() < 5 else 0.9
        
        # Add random noise
        noise = np.random.normal(0, 2)
        trend_factor = 1 + (i / days) * 0.1  # Slight upward trend
        
        trends["vessel_utilization"].append(
            max(60, min(95, base_utilization * weekly_factor * trend_factor + noise))
        )
        
        trends["port_efficiency"].append(
            max(70, min(95, base_port_efficiency * weekly_factor * trend_factor + noise))
        )
        
        trends["fuel_consumption"].append(
            max(15, min(40, base_fuel_consumption / trend_factor + noise))
        )
        
        trends["on_time_performance"].append(
            max(75, min(98, base_on_time * weekly_factor * trend_factor + noise))
        )
    
    return trends

def calculate_port_performance(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate port-specific performance metrics"""
    
    port_performance = {}
    
    for port in data.get("ports", []):
        port_name = port.get("name", "Unknown")
        port_id = port.get("id", "Unknown")
        
        # Find vessels currently at this port
        vessels_at_port = [s for s in data.get("ships", []) if s.get("current_port") == port_name]
        
        # Find berths at this port
        port_berths = [b for b in data.get("berths", []) if b.get("port") == port_name]
        
        # Calculate metrics
        metrics = {
            "port_name": port_name,
            "port_id": port_id,
            "region": port.get("region", "Unknown"),
            "total_berths": len(port_berths),
            "occupied_berths": len([b for b in port_berths if b.get("status") == "Occupied"]),
            "available_berths": len([b for b in port_berths if b.get("status") == "Available"]),
            "vessels_count": len(vessels_at_port),
            "total_capacity": port.get("capacity", 0)
        }
        
        # Calculate utilization rate
        if metrics["total_berths"] > 0:
            metrics["berth_utilization"] = (metrics["occupied_berths"] / metrics["total_berths"]) * 100
        else:
            metrics["berth_utilization"] = 0
        
        # Calculate average vessel capacity at port
        if vessels_at_port:
            metrics["avg_vessel_capacity"] = np.mean([v.get("capacity", 0) for v in vessels_at_port])
            metrics["total_vessel_capacity"] = sum(v.get("capacity", 0) for v in vessels_at_port)
        else:
            metrics["avg_vessel_capacity"] = 0
            metrics["total_vessel_capacity"] = 0
        
        # Simulated performance metrics
        metrics["avg_waiting_time"] = np.random.uniform(2, 12)  # hours
        metrics["avg_processing_time"] = np.random.uniform(18, 48)  # hours
        metrics["throughput_efficiency"] = np.random.uniform(75, 95)  # percentage
        
        # Грузооборот порта (сумма количества грузов по всем причалам и рейсам в этом порту)
        cargo_turnover = 0
        for ship in data.get("ships", []):
            if "schedule" in ship:
                for sched in ship["schedule"]:
                    if sched.get("port") == port_name:
                        cargo_turnover += sched.get("quantity", 0)
            else:
                print(f"Warning: Ship {ship.get('name')} has no 'schedule' key")
        metrics["cargo_turnover"] = cargo_turnover

        # Грузооборот причалов
        berth_turnover = {}
        for berth in port_berths:
            b_name = berth.get("id", "Unknown")
            total_qty = 0
            for ship in data.get("ships", []):
                if "schedule" in ship:
                    for sched in ship["schedule"]:
                        if sched.get("berth") == b_name:
                            total_qty += sched.get("quantity", 0)
                else:
                    print(f"Warning: Ship {ship.get('name')} has no 'schedule' key")
            berth_turnover[b_name] = total_qty
        metrics["berth_turnover"] = berth_turnover

        port_performance[port_name] = metrics
    
    return port_performance

def analyze_schedule_conflicts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze potential schedule conflicts"""
    
    conflicts = []
    
    # Group schedules by port and time
    port_schedules = {}
    
    for ship in data["ships"]:
        for schedule in ship.get("schedule", []):
            port = schedule.get("port")
            if not port:
                continue
                
            if port not in port_schedules:
                port_schedules[port] = []
                
            schedule_item = schedule.copy()
            schedule_item["vessel_name"] = ship["name"]
            schedule_item["vessel_id"] = ship["id"]
            port_schedules[port].append(schedule_item)
    
    # Check for overlapping schedules at each port
    for port, schedules in port_schedules.items():
        # Sort schedules by arrival time
        valid_schedules = []
        for sched in schedules:
            if sched.get("arrival") and sched.get("departure"):
                try:
                    sched["arrival_dt"] = pd.to_datetime(sched["arrival"])
                    sched["departure_dt"] = pd.to_datetime(sched["departure"])
                    valid_schedules.append(sched)
                except:
                    continue
        
        valid_schedules.sort(key=lambda x: x["arrival_dt"])
        
        # Check for overlaps
        for i in range(len(valid_schedules) - 1):
            current = valid_schedules[i]
            next_sched = valid_schedules[i + 1]
            
            # Check if current departure is after next arrival
            if current["departure_dt"] > next_sched["arrival_dt"]:
                overlap_hours = (current["departure_dt"] - next_sched["arrival_dt"]).total_seconds() / 3600
                
                conflicts.append({
                    "port": port,
                    "vessel1": current["vessel_name"],
                    "vessel2": next_sched["vessel_name"],
                    "conflict_type": "Schedule Overlap",
                    "overlap_hours": abs(overlap_hours),
                    "vessel1_departure": current["departure"],
                    "vessel2_arrival": next_sched["arrival"],
                    "severity": "High" if overlap_hours > 12 else "Medium" if overlap_hours > 4 else "Low"
                })
    
    return conflicts


def get_current_fleet_table(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Формирует таблицу текущего положения флота и следующего шага (плеча маршрута).
    Колонки: Судно, Статус, Текущий порт, Следующая операция, Следующий порт, ETA.
    """
    rows: List[Dict[str, Any]] = []
    now = pd.Timestamp.now(tz=None)
    for ship in data.get("ships", []):
        schedule = ship.get("schedule", [])
        next_op = None
        # Находим ближайшую будущую операцию
        best_dt = None
        for s in schedule:
            arr = s.get("arrival")
            dep = s.get("departure")
            candidate = None
            if dep:
                try:
                    dep_dt = pd.to_datetime(dep)
                    if dep_dt >= now:
                        candidate = dep_dt
                except Exception:
                    pass
            if candidate is None and arr:
                try:
                    arr_dt = pd.to_datetime(arr)
                    if arr_dt >= now:
                        candidate = arr_dt
                except Exception:
                    pass
            if candidate is not None:
                if best_dt is None or candidate < best_dt:
                    best_dt = candidate
                    next_op = s

        rows.append({
            "Судно": ship.get("name", ""),
            "Статус": ship.get("status", ""),
            "Текущий порт": ship.get("current_port", ""),
            "Следующая операция": (next_op or {}).get("operation", ""),
            "Следующий порт": (next_op or {}).get("port", ""),
            "ETA": str(best_dt) if best_dt is not None else ""
        })
    return pd.DataFrame(rows)


def calculate_fleet_efficiency_by_contract(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Расчет эффективности флота по типу контракта (Тайм-чартер и Спот).
    Эффективность = доля судов в активных операциях (Loading/Discharge/Transit) от общего числа судов группы.
    """
    active_statuses = {"Loading", "Discharge", "Transit", "EnRoute", "enroute"}
    groups = {"Тайм-чартер": [], "Спот": []}
    for ship in data.get("ships", []):
        ctype = ship.get("charter_type", "Спот")
        groups.setdefault(ctype, [])
        groups[ctype].append(ship)

    result = {}
    for ctype, ships in groups.items():
        total = len(ships)
        active = sum(1 for s in ships if s.get("status") in active_statuses)
        eff = (active / total * 100) if total > 0 else 0.0
        result[ctype] = {
            "total_ships": total,
            "active_ships": active,
            "fleet_efficiency_pct": round(eff, 2),
        }
    return result


def calculate_timecharter_idle_losses(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Оценка совокупных «потерь» (в часах) для судов ТЧ (Тайм-чартер) в статусе idle.
    Потери = длительность простоя от момента idle_since до планируемого возобновления (planned_resume), иначе до текущего времени.
    Если idle_since отсутствует, пытаемся оценить по последнему факту (departure) из расписания в прошлом.
    """
    now = pd.Timestamp.now(tz=None)
    total_idle_hours = 0.0
    details: List[Dict[str, Any]] = []

    for ship in data.get("ships", []):
        if ship.get("charter_type") != "Тайм-чартер":
            continue
        if str(ship.get("status", "")).lower() != "idle":
            continue

        idle_since = ship.get("idle_since")
        planned_resume = ship.get("planned_resume")

        try:
            idle_since_dt = pd.to_datetime(idle_since) if idle_since else None
        except Exception:
            idle_since_dt = None

        try:
            planned_resume_dt = pd.to_datetime(planned_resume) if planned_resume else None
        except Exception:
            planned_resume_dt = None

        if idle_since_dt is None:
            # Оценка по расписанию: последний прошедший departure
            last_dep = None
            for sched in ship.get("schedule", []):
                dep = sched.get("departure")
                if dep:
                    try:
                        dep_dt = pd.to_datetime(dep)
                        if dep_dt <= now and (last_dep is None or dep_dt > last_dep):
                            last_dep = dep_dt
                    except Exception:
                        continue
            idle_since_dt = last_dep

        # Конец интервала
        end_dt = planned_resume_dt if (planned_resume_dt is not None and planned_resume_dt > now) else now

        idle_hours = 0.0
        if idle_since_dt is not None and end_dt is not None and end_dt > idle_since_dt:
            idle_hours = (end_dt - idle_since_dt).total_seconds() / 3600.0

        total_idle_hours += idle_hours
        details.append({
            "vessel": ship.get("name", "Unknown"),
            "idle_since": str(idle_since_dt) if idle_since_dt is not None else None,
            "planned_resume": str(planned_resume_dt) if planned_resume_dt is not None else None,
            "idle_hours": round(idle_hours, 2),
        })

    return {
        "total_idle_hours": round(total_idle_hours, 2),
        "vessels": details
    }