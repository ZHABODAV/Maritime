import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List
import streamlit as st

def load_maritime_data() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load maritime data from Excel/CSV files with extended support"""
    try:
        # Try to load regions
        try:
            regions_df = pd.read_excel("attached_assets/regions.xlsx")
            print(f"Loaded regions: {len(regions_df)} rows")
        except:
            try:
                regions_df = pd.read_excel("regions.xlsx")
                print(f"Loaded regions: {len(regions_df)} rows")
            except:
                regions_df = pd.DataFrame(columns=["ID региона", "Наименование региона"])
                print("No regions file, using empty DataFrame")

        # Ports
        try:
            ports_path = 'attached_assets/ports'
            if Path(ports_path + '.csv').exists():
                ports_df = pd.read_csv(ports_path + '.csv')
            else:
                ports_df = pd.read_excel(ports_path + '.xlsx')
            print(f"Loaded ports: {len(ports_df)} rows from {ports_path}")
        except:
            ports_path = 'ports'
            if Path(ports_path + '.csv').exists():
                ports_df = pd.read_csv(ports_path + '.csv')
            else:
                ports_df = pd.read_excel(ports_path + '.xlsx')
            print(f"Loaded ports: {len(ports_df)} rows from {ports_path}")

        # Ships
        try:
            vessels_path = 'attached_assets/vessels'
            if Path(vessels_path + '.csv').exists():
                vessels_df = pd.read_csv(vessels_path + '.csv')
            else:
                vessels_df = pd.read_excel(vessels_path + '.xlsx')
            print(f"Loaded vessels: {len(vessels_df)} rows from {vessels_path}")
        except:
            vessels_path = 'vessels'
            if Path(vessels_path + '.csv').exists():
                vessels_df = pd.read_csv(vessels_path + '.csv')
            else:
                vessels_df = pd.read_excel(vessels_path + '.xlsx')
            print(f"Loaded vessels: {len(vessels_df)} rows from {vessels_path}")

        # Voyages
        try:
            voyages_path = 'attached_assets/voyages'
            if Path(voyages_path + '.csv').exists():
                voyages_df = pd.read_csv(voyages_path + '.csv')
            else:
                voyages_df = pd.read_excel(voyages_path + '.xlsx')
            print(f"Loaded voyages: {len(voyages_df)} rows from {voyages_path}")
        except:
            voyages_path = 'voyages'
            if Path(voyages_path + '.csv').exists():
                voyages_df = pd.read_csv(voyages_path + '.csv')
            else:
                voyages_df = pd.read_excel(voyages_path + '.xlsx')
            print(f"Loaded voyages: {len(voyages_df)} rows from {voyages_path}")

        # Voyage legs
        try:
            legs_path = 'attached_assets/voyage_legs'
            if Path(legs_path + '.csv').exists():
                voyage_legs_df = pd.read_csv(legs_path + '.csv')
            else:
                voyage_legs_df = pd.read_excel(legs_path + '.xlsx')
            print(f"Loaded voyage legs: {len(voyage_legs_df)} rows from {legs_path}")
        except:
            legs_path = 'voyage_legs'
            if Path(legs_path + '.csv').exists():
                voyage_legs_df = pd.read_csv(legs_path + '.csv')
            else:
                voyage_legs_df = pd.read_excel(legs_path + '.xlsx')
            print(f"Loaded voyage legs: {len(voyage_legs_df)} rows from {legs_path}")

        regions = regions_df.to_dict("records")
        ports = ports_df.to_dict('records')
        vessels = vessels_df.to_dict('records')
        voyages = voyages_df.to_dict("records")

        # Process voyage legs with proper date handling
        processed_voyages = []
        for _, leg in voyage_legs_df.iterrows():
            processed_leg = {
                'VOY_ID': leg['VOY_ID'],
                'LEG_ID': leg.get('LEG_ID', f"LEG_{_}"),
                'OPS_GROUP': leg.get('OPS_GROUP', ""),
                'OPS_RELATE': leg.get('OPS_RELATE', ""),
                'LEG_TYPE': leg.get('LEG_TYPE', ""),
                'Vessel_ID': leg.get('Vessel_ID', ""),
                'PortStart': leg.get('PortStart', ""),
                'PortEnd': leg.get('PortEnd', ""),
                'Berth': leg.get('Berth', ''),
                'CargoType': leg.get('CargoType', ''),
                'Quantity': leg.get('Quantity', 0),
                'CharterType': leg.get('CharterType', ''),
                'StartTimePlan': convert_excel_date(leg.get('StartTimePlan')),
                'EndTimePlan': convert_excel_date(leg.get('EndTimePlan')),
                'StartTimeAct': convert_excel_date(leg.get('StartTimeAct')),
                'EndTimeAct': convert_excel_date(leg.get('EndTimeAct')),
                'Status': leg.get('Status', 'Unknown'),
                'Region': leg.get('Region', 'Unknown')
            }
            processed_voyages.append(processed_leg)

        return {
            "regions": regions,
            "ports": ports,
            "vessels": vessels,
            "voyages": voyages,
            "voyage_legs": processed_voyages
        }, {}, {}
    except Exception as e:
        print(f"Error loading data from Excel/CSV files: {e}")
        return create_sample_data()

def convert_excel_date(excel_date):
    """Convert Excel serial date to ISO format string"""
    if pd.isna(excel_date) or excel_date is None:
        return None
    
    try:
        if isinstance(excel_date, (int, float)):
            # Excel serial date
            base_date = datetime(1900, 1, 1)
            # Excel incorrectly considers 1900 a leap year
            if excel_date > 59:
                excel_date -= 1
            converted_date = base_date + timedelta(days=excel_date - 2)
            return converted_date.isoformat()
        elif isinstance(excel_date, str):
            return excel_date
        elif hasattr(excel_date, 'isoformat'):
            return excel_date.isoformat()
        else:
            return str(excel_date)
    except:
        return None

def create_sample_data() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Create sample maritime data for demonstration"""
    
    ports = [
        {"PortCode": "RUSPB", "PortName": "St Petersburg", "Lat": 59.93, "Lon": 30.25, "Region": "Region 2"},
        {"PortCode": "EGPSD", "PortName": "Port Said", "Lat": 31.25, "Lon": 32.3, "Region": "Region 2"},
        {"PortCode": "RUOYA", "PortName": "Olya", "Lat": 45.3, "Lon": 47.9, "Region": "Region 1"},
        {"PortCode": "RUBKO", "PortName": "Balakovo", "Lat": 52.05, "Lon": 47.85, "Region": "Region 1"},
        {"PortCode": "NLRTM", "PortName": "Rotterdam", "Lat": 51.9225, "Lon": 4.47917, "Region": "Region 3"},
        {"PortCode": "DEHAM", "PortName": "Hamburg", "Lat": 53.5511, "Lon": 9.9937, "Region": "Region 3"},
    ]
    
    vessels = [
        {
            "Vessel_ID": "VESSEL_001",
            "VesselName": "ATLANTIC STAR",
            "Status": "EnRoute",
            "CurrentPort": "",
            "CurrentLeg": "LEG_001",
            "CurrentLat": 60.0017,
            "CurrentLon": 26.8671,
            "CurrentOperation": "TRANSIT",
            "NextOperation": "Loading",
            "ETA_Next": (datetime.now() + timedelta(hours=12)).timestamp()
        },
        {
            "Vessel_ID": "VESSEL_002",
            "VesselName": "PACIFIC GLORY",
            "Status": "Inport",
            "CurrentPort": "RUOYA",
            "CurrentLeg": "LEG_002",
            "CurrentLat": 46.463507,
            "CurrentLon": 47.970837,
            "CurrentOperation": "LOADING",
            "NextOperation": "Discharge",
            "ETA_Next": (datetime.now() + timedelta(days=2)).timestamp()
        },
        {
            "Vessel_ID": "VESSEL_003",
            "VesselName": "NORDIC WIND",
            "Status": "EnRoute",
            "CurrentPort": "",
            "CurrentLeg": "LEG_003",
            "CurrentLat": 55.7558,
            "CurrentLon": 37.6176,
            "CurrentOperation": "TRANSIT",
            "NextOperation": "Loading",
            "ETA_Next": (datetime.now() + timedelta(hours=6)).timestamp()
        }
    ]
    
    voyage_legs = []
    base_date = datetime.now()
    
    for i, vessel in enumerate(vessels):
        for j in range(3):  # 3 legs per vessel
            voyage_legs.append({
                "VOY_ID": f"VOY_{i+1:03d}",
                "LEG_ID": f"LEG_{i+1:03d}_{j+1}",
                "OPS_GROUP": "Cargo Operation" if j % 2 == 0 else "Transit",
                "OPS_RELATE": "Loading" if j % 3 == 0 else "Discharge" if j % 3 == 1 else "Transit",
                "LEG_TYPE": "Port" if j % 2 == 0 else "Sea",
                "Vessel_ID": vessel["Vessel_ID"],
                "PortStart": ports[j % len(ports)]["PortCode"],
                "PortEnd": ports[(j+1) % len(ports)]["PortCode"],
                "Berth": f"BERTH_{j+1}" if j % 2 == 0 else "",
                "CargoType": "Containers" if i % 2 == 0 else "Bulk Cargo",
                "Quantity": np.random.randint(5000, 25000),
                "CharterType": "Time" if i % 2 == 0 else "Voyage",
                "StartTimePlan": (base_date + timedelta(days=j*2)).isoformat(),
                "EndTimePlan": (base_date + timedelta(days=j*2+1)).isoformat(),
                "StartTimeAct": (base_date + timedelta(days=j*2, hours=2)).isoformat(),
                "EndTimeAct": (base_date + timedelta(days=j*2+1, hours=1)).isoformat(),
                "Status": "Complete" if j == 0 else "In Progress" if j == 1 else "Planned",
                "Region": ports[j % len(ports)]["Region"]
            })
    
    return {
        "ports": ports,
        "vessels": vessels,
        "voyage_legs": voyage_legs
    }, {}, {}

def transform_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw data into application format and guarantee required sections exist"""
    # Ensure presence of all sections for downstream merging/UI
    transformed: Dict[str, Any] = {
        "regions": [],
        "ports": [],
        "ships": [],
        "voyages": [],
        "voy_route_legs": [],
        "cargo_dict": [],
        "berths": [],
    }

    # Regions (pass-through/normalize)
    for idx, reg in enumerate(raw_data.get("regions", [])):
        transformed["regions"].append({
            "id": str(reg.get("ID региона", reg.get("id", f"REG_{idx:03d}"))),
            "name": reg.get("Наименование региона", reg.get("name", f"Регион {idx+1}")),
        })

    # Ports
    for port in raw_data.get("ports", []):
        transformed["ports"].append({
            "id": port.get("PortCode", f"PORT_{len(transformed['ports'])}"),
            "name": port.get("PortName", port.get("Порт", "Unknown Port")),
            "region": port.get("Region", port.get("Регион", "Unknown Region")),
            "lat": port.get("Lat", 0.0),
            "lon": port.get("Lon", 0.0),
            "berths": port.get("Берths", port.get("Берths".lower(), port.get("Берths".upper(), 0))),
            "capacity": port.get("Capacity", port.get("Накопление максимум", 0)),
            "locode": port.get("LOCODE", port.get("Locode", "")),
        })

    # Vessels -> Ships
    vessel_dict: Dict[str, Any] = {}
    for vessel in raw_data.get("vessels", []):
        vessel_id = str(vessel.get("Vessel_ID", vessel.get("ИМО №", f"VESSEL_{len(vessel_dict)}")))
        vessel_name = vessel.get("VesselName", vessel.get("Судно", "Unknown Vessel"))

        # Determine vessel type based on name or default assignment
        if isinstance(vessel_name, str) and ("container" in vessel_name.lower() or "cont" in vessel_name.lower()):
            vessel_type = "Container Ship"
        elif isinstance(vessel_name, str) and ("bulk" in vessel_name.lower() or "carrier" in vessel_name.lower()):
            vessel_type = "Bulk Carrier"
        elif isinstance(vessel_name, str) and ("tanker" in vessel_name.lower()):
            vessel_type = "Tanker"
        else:
            vessel_type = ["Container Ship", "Bulk Carrier", "Tanker"][len(vessel_dict) % 3]

        ship = {
            "id": vessel_id,
            "name": vessel_name,
            "type": vessel.get("Type", vessel_type),
            "capacity": vessel.get("Capacity", np.random.randint(10000, 30000)),
            "current_port": vessel.get("CurrentPort", ""),
            "status": map_vessel_status(vessel.get("Status", "Unknown")),
            "cargo": get_cargo_type(vessel_type),
            "schedule": [],
            "lat": vessel.get("CurrentLat", 0.0),
            "lon": vessel.get("CurrentLon", 0.0),
            "speed": vessel.get("Speed", np.random.uniform(12, 18)),
            "fuel_consumption": vessel.get("FuelConsumption", np.random.uniform(15, 35)),
            "crew_size": vessel.get("CrewSize", np.random.randint(15, 25)),
            "charter_type": vessel.get("CharterType", vessel.get("Тип контракта", "")),
            "idle_since": vessel.get("idle_since"),
            "planned_resume": vessel.get("planned_resume"),
        }
        vessel_dict[vessel_id] = ship
        transformed["ships"].append(ship)

    # Voyages (pass-through minimal normalization)
    for v in raw_data.get("voyages", []):
        transformed["voyages"].append({
            "id": str(v.get("id", v.get("VOY_ID", ""))) or None,
            "VOY_ID": str(v.get("VOY_ID", v.get("id", ""))),
            "Судно": v.get("Судно"),
            "Тип контракта": v.get("Тип контракта"),
            "Порты погрузки": v.get("Порты погрузки"),
            "Порты назначения": v.get("Порты назначения"),
            "Груз": v.get("Груз"),
            "Тоннаж": v.get("Тоннаж"),
            "Дата погрузки": v.get("Дата погрузки"),
            "Планируемая дата выгрузки": v.get("Планируемая дата выгрузки"),
            "STATUS": v.get("STATUS", v.get("Статус", "")),
        })

    # Voyage legs -> schedules + keep raw as voy_route_legs for merging
    for leg in raw_data.get("voyage_legs", []):
        # Keep a copy for merging as 'voy_route_legs'
        transformed["voy_route_legs"].append(dict(leg))

        vessel_id = leg.get("Vessel_ID")
        if vessel_id not in vessel_dict:
            continue

        port_start = next((p for p in transformed["ports"] if p["id"] == leg.get("PortStart")), None)
        if port_start:
            vessel_dict[vessel_id]["schedule"].append({
                "port": port_start["name"],
                "port_code": port_start["id"],
                "arrival": leg.get("StartTimePlan"),
                "departure": leg.get("EndTimePlan"),
                "operation": leg.get("OPS_RELATE", "Unknown"),
                "cargo_type": leg.get("CargoType", ""),
                "quantity": leg.get("Quantity", 0),
                "status": leg.get("Status", "Unknown"),
                "berth": leg.get("Berth", "")
            })

    # Generate berths for each port
    berth_counter = 0
    for port in transformed["ports"]:
        if isinstance(port.get("berths"), list):
            for berth_data in port["berths"]:
                berth_counter += 1
                transformed["berths"].append({
                    "id": f"BERTH_{berth_counter:03d}",
                    "name": berth_data.get("Причал", f"Berth {berth_counter}"),
                    "port": port["name"],
                    "port_id": port["id"],
                    "status": berth_data.get("Статус", "Unknown"),
                    "capacity": berth_data.get("Вместимость", 0),
                    "depth": berth_data.get("Глубина", 0),
                    "length": berth_data.get("Длина", 0)
                })
        else:
            try:
                count = int(port.get("berths", 0))
            except:
                count = 0
            for i in range(count):
                berth_counter += 1
                transformed["berths"].append({
                    "id": f"BERTH_{berth_counter:03d}",
                    "name": f"Berth {i+1}",
                    "port": port["name"],
                    "port_id": port["id"],
                    "status": "Available",
                    "capacity": port.get("Capacity", 0),
                    "depth": 0,
                    "length": 0
                })

    # Ensure keys exist for merging even if empty
    for key in ["regions", "ports", "ships", "voyages", "voy_route_legs", "cargo_dict", "berths"]:
        transformed.setdefault(key, [])

    # Merge uploaded files from session_state (if any) into transformed data
    try:
        uploads = st.session_state.get("uploaded_files", {})
    except Exception:
        uploads = {}

    def _df(name: str):
        v = uploads.get(name)
        if v is None:
            return None
        return v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)

    # Regions
    df = _df("regions")
    if df is not None and not df.empty:
        existing_ids = {r.get("id") for r in transformed["regions"]}
        for _, row in df.iterrows():
            reg_id = str(row.get("ID региона", row.get("id", f"REG_{len(existing_ids)+1}")))
            if reg_id in existing_ids:
                continue
            transformed["regions"].append({
                "id": reg_id,
                "name": row.get("Наименование региона", row.get("name", "")),
            })
            existing_ids.add(reg_id)

    # Ports
    df = _df("ports")
    if df is not None and not df.empty:
        existing_keys = {(p.get("name"), p.get("locode")) for p in transformed["ports"]}
        for _, row in df.iterrows():
            name = row.get("Порт") or row.get("PortName") or row.get("Наименование порта", "")
            locode = row.get("LOCODE", "")
            key = (name, locode)
            if key in existing_keys:
                continue
            transformed["ports"].append({
                "id": row.get("LOCODE", row.get("PortCode", f"PORT_{len(transformed['ports'])}")),
                "name": name,
                "region": row.get("Регион") or row.get("Наименование региона") or row.get("Region", "Unknown Region"),
                "lat": row.get("Lat", 0.0),
                "lon": row.get("Lon", 0.0),
                "berths": row.get("Причал", 0),
                "capacity": row.get("Накопление максимум", row.get("Capacity", 0)),
                "locode": locode,
            })
            existing_keys.add(key)

    # Ships
    df = _df("ships")
    if df is not None and not df.empty:
        existing_ids = {s.get("id") for s in transformed["ships"]}
        for _, row in df.iterrows():
            imo = str(row.get("ИМО №", "")).strip()
            ship_id = imo or f"VESSEL_{len(existing_ids)+1}"
            if ship_id in existing_ids:
                continue
            ship = {
                "id": ship_id,
                "name": row.get("Судно", row.get("VesselName", "")),
                "type": row.get("Тип судна", row.get("type", "Unknown")),
                "capacity": row.get("Capacity", np.random.randint(10000, 30000)),
                "current_port": row.get("CurrentPort", ""),
                "status": map_vessel_status(row.get("Status", "Unknown")),
                "cargo": get_cargo_type(row.get("Тип судна", "Bulk Carrier")),
                "schedule": [],
                "lat": row.get("CurrentLat", 0.0),
                "lon": row.get("CurrentLon", 0.0),
                "speed": row.get("Speed", np.random.uniform(12, 18)),
                "fuel_consumption": row.get("FuelConsumption", np.random.uniform(15, 35)),
                "crew_size": row.get("CrewSize", np.random.randint(15, 25)),
                "charter_type": row.get("Тип контракта", row.get("CharterType", "")),
                "idle_since": row.get("idle_since"),
                "planned_resume": row.get("planned_resume"),
            }
            transformed["ships"].append(ship)
            vessel_dict[ship_id] = ship
            existing_ids.add(ship_id)

    # Voyages
    df = _df("voyages")
    if df is not None and not df.empty:
        existing_ids = {v.get("VOY_ID") for v in transformed["voyages"] if v.get("VOY_ID")}
        for _, row in df.iterrows():
            voy_id = str(row.get("VOY_ID", "")).strip()
            if not voy_id or voy_id in existing_ids:
                continue
            transformed["voyages"].append({
                "id": voy_id,
                "VOY_ID": voy_id,
                "Судно": row.get("Судно"),
                "Тип контракта": row.get("Тип контракта"),
                "Порты погрузки": row.get("Порты погрузки"),
                "Порты назначения": row.get("Порты назначения"),
                "Груз": row.get("Груз"),
                "Тоннаж": row.get("Тоннаж"),
                "Дата погрузки": row.get("Дата погрузки"),
                "Планируемая дата выгрузки": row.get("Планируемая дата выгрузки"),
                "STATUS": row.get("STATUS", row.get("Статус", "")),
            })
            existing_ids.add(voy_id)

    # Voyage route legs
    df = _df("voy_route_legs")
    if df is not None and not df.empty:
        existing_pairs = {(l.get("VOY_ID"), l.get("ROUTE.ID")) for l in transformed["voy_route_legs"]}
        for _, row in df.iterrows():
            pair = (row.get("VOY_ID"), row.get("ROUTE.ID"))
            if pair in existing_pairs:
                continue
            transformed["voy_route_legs"].append(row.to_dict())
            existing_pairs.add(pair)

    # Cargo dictionary
    df = _df("cargo_dict")
    if df is not None and not df.empty:
        existing_keys = {(c.get("Тип груза"), c.get("Группа груза"), c.get("Наименование груза"))
                         for c in transformed["cargo_dict"]}
        for _, row in df.iterrows():
            key = (row.get("Тип груза"), row.get("Группа груза"), row.get("Наименование груза"))
            if key in existing_keys:
                continue
            transformed["cargo_dict"].append(row.to_dict())
            existing_keys.add(key)

    # Rebuild schedules from merged voyage route legs (session uploads) mapped via voyages
    try:
        # Map VOY_ID to ship name from transformed voyages
        voy_to_ship = {}
        for v in transformed.get("voyages", []):
            voy_id = str(v.get("VOY_ID") or v.get("id") or "").strip()
            ship_name = v.get("Судно")
            if voy_id and ship_name:
                voy_to_ship[voy_id] = ship_name

        # Helper port lookup
        locode_to_port = {}
        name_to_port = {}
        id_to_port = {}
        for p in transformed.get("ports", []):
            if p.get("locode"):
                locode_to_port[p["locode"]] = p
            if p.get("name"):
                name_to_port[p["name"]] = p
            if p.get("id"):
                id_to_port[p["id"]] = p

        # Ship lookup by name
        name_to_ship = {s.get("name"): s for s in transformed.get("ships", [])}

        # Build a set of existing schedule entries to avoid duplicates
        existing = set()
        for s in transformed.get("ships", []):
            for sch in s.get("schedule", []):
                existing.add((
                    s.get("id"),
                    sch.get("port_code") or sch.get("port"),
                    sch.get("arrival"),
                    sch.get("departure"),
                    sch.get("operation")
                ))

        def _to_dt_str(val):
            try:
                dt = pd.to_datetime(val)
                return dt.isoformat()
            except Exception:
                return None

        for row in transformed.get("voy_route_legs", []):
            voy_id = str(row.get("VOY_ID") or "").strip()
            ship_name = voy_to_ship.get(voy_id)
            ship = name_to_ship.get(ship_name) if ship_name else None
            if not ship:
                # create placeholder ship so schedules and visuals can render even without ships upload
                placeholder_id = f"PL_{(ship_name or voy_id)}"
                ship = {
                    "id": placeholder_id,
                    "name": ship_name or placeholder_id,
                    "type": "Unknown",
                    "capacity": 0,
                    "current_port": "",
                    "status": "Transit",
                    "cargo": "",
                    "schedule": [],
                    "lat": 0.0,
                    "lon": 0.0,
                    "speed": 0,
                    "fuel_consumption": 0,
                    "crew_size": 0,
                    "charter_type": "",
                    "idle_since": None,
                    "planned_resume": None,
                }
                transformed["ships"].append(ship)
                name_to_ship[ship["name"]] = ship

            leg_type = str(row.get("ROUTE.LEG.TYPE", "")).lower()
            op = row.get("OPS.RELATE") or row.get("OPS RELATE") or ""

            beg = _to_dt_str(row.get("DT.TIME-BEG") or row.get("DT.TIME BEG"))
            fin = _to_dt_str(row.get("DT.TIME FIN") or row.get("DT.TIME-FIN") or row.get("DT.TIME_FIN"))
            if not (beg and fin):
                continue

            port_name = ""
            port_code = ""
            if leg_type == "node":
                rp = row.get("ROUTE POINT DESCRIPTION") or row.get("ROUTE POINT DECSRIPTION") or row.get("ROUTE POINT")
                if rp:
                    port = locode_to_port.get(rp) or name_to_port.get(rp) or id_to_port.get(rp)
                    if port:
                        port_name = port.get("name", "")
                        port_code = port.get("id", "")

            key = (ship.get("id"), port_code or port_name, beg, fin, op or ("Transit" if leg_type == "edge" else "Operation"))
            if key in existing:
                continue

            ship.setdefault("schedule", []).append({
                "port": port_name,
                "port_code": port_code,
                "arrival": beg,
                "departure": fin,
                "operation": op or ("Transit" if leg_type == "edge" else "Operation"),
                "status": row.get("STATUS", ""),
                "berth": ""
            })
            existing.add(key)
    except Exception:
        # Fail-safe: do not break transform if uploads are partial
        pass

    return transformed

def map_vessel_status(status: str) -> str:
    """Map various status strings to standardized statuses"""
    status_mapping = {
        "EnRoute": "Transit",
        "Inport": "At Port", 
        "Loading": "Loading",
        "Discharging": "Discharge",
        "Anchored": "At Anchor",
        "Maintenance": "Maintenance"
    }
    return status_mapping.get(status, "Unknown")

def get_cargo_type(vessel_type: str) -> str:
    """Get typical cargo type for vessel"""
    cargo_mapping = {
        "Container Ship": "Containers",
        "Bulk Carrier": "Bulk Cargo", 
        "Tanker": "Liquid Cargo"
    }
    return cargo_mapping.get(vessel_type, "General Cargo")
