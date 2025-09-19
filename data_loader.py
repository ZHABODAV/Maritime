import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List

def load_maritime_data() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load maritime data from Excel files"""
    try:
        # Load data from Excel files - check attached_assets folder first
        try:
            ports_df = pd.read_excel('attached_assets/ports.xlsx')
        except:
            ports_df = pd.read_excel('ports.xlsx')

        try:
            vessels_df = pd.read_excel('attached_assets/vessels.xlsx')
        except:
            vessels_df = pd.read_excel('vessels.xlsx')

        try:
            voyage_legs_df = pd.read_excel('attached_assets/voyage_legs.xlsx')
        except:
            voyage_legs_df = pd.read_excel('voyage_legs.xlsx')
        
        ports = ports_df.to_dict('records')
        vessels = vessels_df.to_dict('records')
        
        # Process voyage legs with proper date handling
        processed_voyages = []
        for _, leg in voyage_legs_df.iterrows():
            processed_leg = {
                'VOY_ID': leg['VOY_ID'],
                'LEG_ID': leg['LEG_ID'],
                'OPS_GROUP': leg['OPS_GROUP'],
                'OPS_RELATE': leg['OPS_RELATE'],
                'LEG_TYPE': leg['LEG_TYPE'],
                'Vessel_ID': leg['Vessel_ID'],
                'PortStart': leg['PortStart'],
                'PortEnd': leg['PortEnd'],
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
            "ports": ports,
            "vessels": vessels,
            "voyage_legs": processed_voyages
        }, {}, {}
        
    except Exception as e:
        print(f"Error loading data from Excel files: {e}")
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
    """Transform raw data into application format"""
    transformed = {
        "ships": [],
        "ports": [],
        "berths": []
    }
    
    # Transform ports
    for port in raw_data["ports"]:
        transformed["ports"].append({
            "id": port.get("PortCode", f"PORT_{len(transformed['ports'])}"),
            "name": port.get("PortName", "Unknown Port"),
            "region": port.get("Region", "Unknown Region"),
            "lat": port.get("Lat", 0.0),
            "lon": port.get("Lon", 0.0),
            "berths": np.random.randint(3, 8),  # Random berths between 3-8
            "capacity": np.random.randint(25000, 75000)  # Random capacity
        })
    
    # Transform vessels
    vessel_dict = {}
    for vessel in raw_data["vessels"]:
        vessel_id = vessel.get("Vessel_ID", f"VESSEL_{len(vessel_dict)}")
        
        # Determine vessel type based on name or default assignment
        vessel_name = vessel.get("VesselName", "Unknown Vessel")
        if "container" in vessel_name.lower() or "cont" in vessel_name.lower():
            vessel_type = "Container Ship"
        elif "bulk" in vessel_name.lower() or "carrier" in vessel_name.lower():
            vessel_type = "Bulk Carrier"
        elif "tanker" in vessel_name.lower():
            vessel_type = "Tanker"
        else:
            vessel_type = ["Container Ship", "Bulk Carrier", "Tanker"][len(vessel_dict) % 3]
        
        vessel_dict[vessel_id] = {
            "id": vessel_id,
            "name": vessel_name,
            "type": vessel_type,
            "capacity": np.random.randint(10000, 30000),
            "current_port": vessel.get("CurrentPort", ""),
            "status": map_vessel_status(vessel.get("Status", "Unknown")),
            "cargo": get_cargo_type(vessel_type),
            "schedule": [],
            "lat": vessel.get("CurrentLat", 0.0),
            "lon": vessel.get("CurrentLon", 0.0),
            "speed": np.random.uniform(12, 18),  # knots
            "fuel_consumption": np.random.uniform(15, 35),  # tons/day
            "crew_size": np.random.randint(15, 25)
        }
        transformed["ships"].append(vessel_dict[vessel_id])
    
    # Transform voyage legs into schedules
    for leg in raw_data["voyage_legs"]:
        vessel_id = leg.get("Vessel_ID")
        if vessel_id not in vessel_dict:
            continue
            
        port_start = next((p for p in transformed["ports"] if p["id"] == leg.get("PortStart")), None)
        port_end = next((p for p in transformed["ports"] if p["id"] == leg.get("PortEnd")), None)
        
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
        for i in range(port["berths"]):
            berth_counter += 1
            status = np.random.choice(["Available", "Occupied", "Maintenance", "Reserved"], 
                                    p=[0.4, 0.3, 0.1, 0.2])
            
            berth = {
                "id": f"BERTH_{berth_counter:03d}",
                "name": f"Berth {i+1}",
                "port": port["name"],
                "port_id": port["id"],
                "status": status,
                "capacity": np.random.randint(15000, 45000),
                "depth": np.random.uniform(10, 18),  # meters
                "length": np.random.randint(200, 400)  # meters
            }
            
            if status == "Occupied":
                # Assign a random vessel
                occupied_vessel = np.random.choice(transformed["ships"])
                berth.update({
                    "ship": occupied_vessel["name"],
                    "ship_id": occupied_vessel["id"],
                    "occupied_since": (datetime.now() - timedelta(hours=np.random.randint(1, 48))).isoformat(),
                    "estimated_departure": (datetime.now() + timedelta(hours=np.random.randint(6, 72))).isoformat()
                })
            
            transformed["berths"].append(berth)
    
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
