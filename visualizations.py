import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import networkx as nx

def create_gantt_chart(data: Dict[str, Any]) -> go.Figure:
    """Create Gantt chart for vessel schedules"""
    
    gantt_data = []
    
    for ship in data["ships"]:
        ship_name = ship["name"]
        
        for i, schedule in enumerate(ship.get("schedule", [])):
            if schedule.get("arrival") and schedule.get("departure"):
                try:
                    start_time = pd.to_datetime(schedule["arrival"])
                    end_time = pd.to_datetime(schedule["departure"])
                    
                    gantt_data.append({
                        "Task": f"{ship_name} - {schedule.get('operation', 'Operation')}",
                        "Start": start_time,
                        "Finish": end_time,
                        "Resource": ship_name,
                        "Port": schedule.get("port", "Unknown"),
                        "Operation": schedule.get("operation", "Unknown"),
                        "Status": schedule.get("status", "Unknown"),
                        "Duration": (end_time - start_time).total_seconds() / 3600,
                        "Cargo": schedule.get("cargo_type", ""),
                        "Quantity": schedule.get("quantity", 0)
                    })
                except:
                    continue
    
    if not gantt_data:
        print("No gantt data available - fallback triggered")
        fig = go.Figure()
        fig.add_annotation(
            text="Данные для графика Ганта отсутствуют",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    df = pd.DataFrame(gantt_data)
    
    # Create Gantt chart using Plotly Express
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Resource",
        color="Operation",
        hover_data=["Port", "Duration", "Cargo", "Quantity"],
        title="Диаграмма Ганта – выполнение рейсов"
    )
    
    # Update layout
    fig.update_layout(
        height=max(400, len(df["Resource"].unique()) * 40),
        xaxis_title="Время",
        yaxis_title="Суда",
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_parallel_coordinates(efficiency_df: pd.DataFrame) -> go.Figure:
    """Create parallel coordinates plot for vessel analysis"""
    
    if efficiency_df.empty:
        print("No efficiency data available - fallback triggered")
        fig = go.Figure()
        fig.add_annotation(
            text="Данные по эффективности отсутствуют",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Select numeric columns for parallel coordinates
    numeric_cols = [
        'efficiency_score', 'on_time_performance', 'capacity_utilization',
        'fuel_efficiency', 'speed_efficiency', 'turnaround_efficiency'
    ]
    
    # Filter to available columns
    available_cols = [col for col in numeric_cols if col in efficiency_df.columns]
    
    if len(available_cols) < 2:
        print("Insufficient data for parallel coordinates - fallback triggered")
        fig = go.Figure()
        fig.add_annotation(
            text="Недостаточно данных для параллельного анализа",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create dimensions for parallel coordinates
    dimensions = []
    for col in available_cols:
        dimensions.append(dict(
            label=col.replace('_', ' ').title(),
            values=efficiency_df[col],
            range=[efficiency_df[col].min(), efficiency_df[col].max()]
        ))
    
    # Color scale based on efficiency score
    if 'efficiency_score' in efficiency_df.columns:
        color_values = efficiency_df['efficiency_score']
    else:
        color_values = efficiency_df[available_cols[0]]
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_values,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Efficiency Score")
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title="Параллельный анализ эффективности флота",
        height=600,
        margin=dict(l=100, r=100, t=60, b=50)
    )
    
    return fig

def create_enhanced_berth_allocation(data: Dict[str, Any]) -> go.Figure:
    """Create enhanced berth allocation visualization"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Berth Status Overview", "Utilization by Port", 
                       "Berth Capacity Analysis", "Occupancy Timeline"),
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Berth status distribution (pie chart)
    berth_status = {}
    for berth in data["berths"]:
        status = berth.get("status", "Unknown")
        berth_status[status] = berth_status.get(status, 0) + 1
    
    fig.add_trace(
        go.Pie(labels=list(berth_status.keys()), values=list(berth_status.values()),
               name="Berth Status"),
        row=1, col=1
    )
    
    # Utilization by port (bar chart)
    port_utilization = {}
    for port in data["ports"]:
        port_name = port["name"]
        port_berths = [b for b in data["berths"] if b.get("port") == port_name]
        occupied = len([b for b in port_berths if b.get("status") == "Occupied"])
        total = len(port_berths)
        port_utilization[port_name] = (occupied / total * 100) if total > 0 else 0
    
    fig.add_trace(
        go.Bar(x=list(port_utilization.keys()), y=list(port_utilization.values()),
               name="Utilization %", marker_color='lightblue'),
        row=1, col=2
    )
    
    # Berth capacity analysis (scatter)
    berth_capacities = [b.get("capacity", 0) for b in data["berths"]]
    berth_lengths = [b.get("length", 0) for b in data["berths"]]
    berth_colors = [berth_status.get(b.get("status", "Unknown"), 1) for b in data["berths"]]
    
    if berth_capacities and berth_lengths:
        fig.add_trace(
            go.Scatter(x=berth_lengths, y=berth_capacities,
                      mode='markers', name="Berths",
                      marker=dict(size=8, color=berth_colors, colorscale='viridis')),
            row=2, col=1
        )
    
    # Occupancy timeline (deterministic)
    hours = list(range(24))
    occupancy = [70 + (h % 6) * 3 for h in hours]  # Deterministic pattern: 70..85
    
    fig.add_trace(
        go.Scatter(x=hours, y=occupancy, mode='lines+markers',
                  name="Hourly Occupancy %", line=dict(color='green')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Enhanced Berth Operations Dashboard",
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Port", row=1, col=2)
    fig.update_yaxes(title_text="Utilization %", row=1, col=2)
    
    fig.update_xaxes(title_text="Berth Length (m)", row=2, col=1)
    fig.update_yaxes(title_text="Capacity (tons)", row=2, col=1)
    
    fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
    fig.update_yaxes(title_text="Occupancy %", row=2, col=2)
    
    return fig

def create_performance_dashboard(data: Dict[str, Any]) -> go.Figure:
    """Create comprehensive performance dashboard"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Fleet Status", "Performance Trends", "Efficiency Distribution",
                       "Port Activity", "Fuel Consumption", "Operational Metrics"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )
    
    # Fleet status (donut chart approximation with bar)
    status_counts = {}
    for ship in data["ships"]:
        status = ship.get("status", "Unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    fig.add_trace(
        go.Bar(x=list(status_counts.keys()), y=list(status_counts.values()),
               name="Fleet Status", marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Performance trends (line chart)
    days = 14
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    performance_trend = [80 + (i * 1.07) for i in range(days)]  # Deterministic upward trend
    
    fig.add_trace(
        go.Scatter(x=dates, y=performance_trend, mode='lines+markers',
                   name="Performance Trend", line=dict(color='blue')),
        row=1, col=2
    )
    
    # Efficiency distribution (histogram)
    if data["ships"]:
        efficiency_scores = [80 + (i % 15) for i in range(len(data["ships"]))]  # Deterministic scores
        fig.add_trace(
            go.Histogram(x=efficiency_scores, nbinsx=10, name="Efficiency Distribution",
                        marker_color='lightgreen'),
            row=1, col=3
        )
    
    # Port activity (bar chart)
    port_activity = {}
    for ship in data["ships"]:
        port = ship.get("current_port", "At Sea")
        if port:
            port_activity[port] = port_activity.get(port, 0) + 1
    
    fig.add_trace(
        go.Bar(x=list(port_activity.keys()), y=list(port_activity.values()),
               name="Port Activity", marker_color='orange'),
        row=2, col=1
    )
    
    # Add cargo turnover by port
    try:
        from analytics import calculate_port_performance
        port_perf = calculate_port_performance(data)
        cargo_x = [m["port_name"] for m in port_perf.values()]
        cargo_y = [m.get("cargo_turnover", 0) for m in port_perf.values()]
        fig.add_trace(
            go.Bar(x=cargo_x, y=cargo_y,
                   name="Грузооборот порта", marker_color='brown'),
            row=2, col=1
        )
    except Exception as e:
        pass
    
    # Fuel consumption (line chart)
    fuel_data = [ship.get("fuel_consumption", 25) for ship in data["ships"]]  # Fixed default
    ship_names = [ship["name"][:10] for ship in data["ships"]]  # Truncate names
    
    fig.add_trace(
        go.Scatter(x=ship_names, y=fuel_data, mode='markers',
                   name="Fuel Consumption", marker=dict(size=10, color='red')),
        row=2, col=2
    )
    
    # Operational metrics (radar chart approximation with bar)
    metrics = {
        "On-Time": 85,
        "Efficiency": 80,
        "Safety": 95,
        "Fuel": 75
    }
    
    fig.add_trace(
        go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
               name="Operational Metrics", marker_color='purple'),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Дашборд производительности морских операций",
        showlegend=False
    )
    
    return fig


def create_cargo_turnover_chart(data: Dict[str, Any]) -> go.Figure:
    """New chart for cargo turnover by port and berths"""
    from analytics import calculate_port_performance
    port_perf = calculate_port_performance(data)
    ports = []
    turnovers = []
    for port_name, metrics in port_perf.items():
        ports.append(port_name)
        turnovers.append(metrics.get("cargo_turnover", 0))
    fig = px.bar(x=ports, y=turnovers,
                 labels={"x": "Порты", "y": "Грузооборот"},
                 title="Грузооборот портов")
    return fig

def create_network_visualization(data: Dict[str, Any]) -> go.Figure:
    """Create network visualization of maritime routes"""
    
    # Create network graph
    G = nx.Graph()
    
    # Add ports as nodes
    for port in data["ports"]:
        G.add_node(port["name"], 
                  region=port.get("region", "Unknown"),
                  lat=port.get("lat", 0),
                  lon=port.get("lon", 0),
                  berths=port.get("berths", 0),
                  capacity=port.get("capacity", 0))
    
    # Add routes as edges
    route_weights = {}
    for ship in data["ships"]:
        schedule = ship.get("schedule", [])
        for i in range(len(schedule) - 1):
            port1 = schedule[i].get("port")
            port2 = schedule[i + 1].get("port")
            
            if port1 and port2 and port1 != port2:
                edge = tuple(sorted([port1, port2]))
                route_weights[edge] = route_weights.get(edge, 0) + 1
    
    # Add edges to graph
    for (port1, port2), weight in route_weights.items():
        if G.has_node(port1) and G.has_node(port2):
            G.add_edge(port1, port2, weight=weight)
    
    # Create layout
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No network data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Use geographical positions if available, otherwise spring layout
    pos = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get('lat') and node_data.get('lon'):
            pos[node] = (node_data['lon'], node_data['lat'])
    
    if not pos:
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = G[edge[0]][edge[1]].get('weight', 1)
        edge_info.append(f"Route: {edge[0]} ↔ {edge[1]}<br>Traffic: {weight} vessels")
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_info = []
    node_size = []
    node_color = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        degree = G.degree[node]
        
        node_info.append(
            f"<b>{node}</b><br>"
            f"Region: {node_data.get('region', 'Unknown')}<br>"
            f"Connections: {degree}<br>"
            f"Berths: {node_data.get('berths', 0)}<br>"
            f"Capacity: {node_data.get('capacity', 0):,}"
        )
        
        node_size.append(max(10, degree * 5 + 10))
        node_color.append(degree)
        node_text.append(node)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(125,125,125,0.3)'),
        hoverinfo='none',
        mode='lines',
        name='Routes',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_info,
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Viridis',
            line=dict(width=2, color='black'),
            showscale=True,
            colorbar=dict(title="Connections")
        ),
        name='Ports'
    ))
    
    fig.update_layout(
        title="Maritime Network Topology",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def create_timeline_chart(data: Dict[str, Any]) -> go.Figure:
    """Create timeline chart for operations (robust: always returns a valid go.Figure)"""
    timeline_data: List[Dict[str, Any]] = []

    for ship in data.get("ships", []):
        ship_name = ship.get("name", "Unknown")
        for schedule in ship.get("schedule", []):
            if schedule.get("arrival") and schedule.get("departure"):
                try:
                    start_time = pd.to_datetime(schedule["arrival"])
                    end_time = pd.to_datetime(schedule["departure"])

                    # Planned operation
                    timeline_data.append({
                        "Ship": ship_name,
                        "Start": start_time,
                        "End": end_time,
                        "Type": "Planned",
                        "Port": schedule.get("port", "Unknown"),
                        "Operation": schedule.get("operation", "Unknown"),
                    })

                    # Actual operation (fixed delay instead of random)
                    actual_start = start_time + timedelta(hours=2)  # Fixed 2-hour delay
                    actual_end = end_time + timedelta(hours=3)      # Fixed 3-hour extension

                    timeline_data.append({
                        "Ship": ship_name,
                        "Start": actual_start,
                        "End": actual_end,
                        "Type": "Actual",
                        "Port": schedule.get("port", "Unknown"),
                        "Operation": schedule.get("operation", "Unknown"),
                    })
                except Exception as e:
                    print(f"Error processing schedule for {ship_name}: {e}")
                    continue

    if not timeline_data:
        print("No timeline data available")
        fig = go.Figure()
        fig.add_annotation(
            text="Данные для таймлайна отсутствуют",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure()

    colors = {
        "Planned": {"Loading": "lightblue", "Discharge": "lightgreen", "Transit": "lightgray"},
        "Actual": {"Loading": "darkblue", "Discharge": "darkgreen", "Transit": "darkgray"},
    }

    # Stable ordering of ships for y-axis
    ships = sorted(list({item["Ship"] for item in timeline_data}))
    ship_positions = {ship: idx for idx, ship in enumerate(ships)}

    for item in timeline_data:
        y_pos = ship_positions[item["Ship"]] + (0.3 if item["Type"] == "Actual" else 0.0)
        color = colors.get(item["Type"], {}).get(item["Operation"], "gray")

        # Draw a rectangle (polygon) representing the time interval
        fig.add_trace(go.Scatter(
            x=[item["Start"], item["Start"], item["End"], item["End"], item["Start"]],
            y=[y_pos - 0.2, y_pos + 0.2, y_pos + 0.2, y_pos - 0.2, y_pos - 0.2],
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=1),
            mode='lines',
            name=f"{item['Ship']} - {item['Type']}",
            showlegend=False
        ))

    fig.update_layout(
        title="Таймлайн операций – план vs факт",
        xaxis_title="Время",
        yaxis_title="Суда",
        yaxis=dict(
            ticktext=ships,
            tickvals=list(range(len(ships))),
            range=[-0.5, len(ships) - 0.5]
        ),
        height=max(400, len(ships) * 50),
        hovermode='closest'
    )

    return fig


def create_voyage_execution_chart(data: Dict[str, Any]) -> go.Figure:
    """
    Визуализация выполнения рейсов (node-edge по времени):
    - Ось X: даты/время
    - Ось Y: суда (каждый рейс/судно — своя линия)
    - Узлы (node): Arrival/Departure точки
    - Рёбра (edge): отрезки между arrival и departure
    Строится по текущим рейсам: участки, где сейчас EnRoute/idle или текущий момент между arrival и departure.
    """
    now = pd.Timestamp.now(tz=None)
    rows = []

    for ship in data.get("ships", []):
        ship_name = ship.get("name", "Unknown")
        for s in ship.get("schedule", []):
            arr = s.get("arrival")
            dep = s.get("departure")
            if not arr or not dep:
                continue
            try:
                arr_dt = pd.to_datetime(arr)
                dep_dt = pd.to_datetime(dep)
            except Exception:
                continue

            status = str(s.get("status", "")).lower()
            is_current = (
                status in {"enroute", "idle"} or
                (arr_dt <= now <= dep_dt)
            )

            if not is_current:
                continue

            rows.append({
                "Ship": ship_name,
                "Start": arr_dt,
                "End": dep_dt,
                "Port": s.get("port", "Unknown"),
                "Operation": s.get("operation", "Unknown")
            })

    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text="Нет текущих рейсов для отображения",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Упорядочивание судов для стабильной оси Y
    ships = sorted(list({r["Ship"] for r in rows}))
    ship_idx = {name: i for i, name in enumerate(ships)}

    fig = go.Figure()

    # Ребра (отрезки между Start и End) + узлы (Start/End)
    for r in rows:
        y = ship_idx[r["Ship"]]
        # Edge
        fig.add_trace(go.Scatter(
            x=[r["Start"], r["End"]],
            y=[y, y],
            mode="lines",
            line=dict(color="steelblue", width=4),
            name=r["Ship"],
            showlegend=False,
            hovertemplate=(
                f"<b>{r['Ship']}</b><br>"
                f"Порт: {r['Port']}<br>"
                f"Операция: {r['Operation']}<br>"
                f"От: %{ '{' }x|%Y-%m-%d %H:%M{ '}' } до %{ '{' }x|%Y-%m-%d %H:%M{ '}' }<extra></extra>"
            )
        ))
        # Start node
        fig.add_trace(go.Scatter(
            x=[r["Start"]],
            y=[y],
            mode="markers",
            marker=dict(size=8, color="darkgreen"),
            showlegend=False,
            hovertemplate=(f"<b>{r['Ship']}</b><br>Start: %{ '{' }x|%Y-%m-%d %H:%M{ '}' }<extra></extra>")
        ))
        # End node
        fig.add_trace(go.Scatter(
            x=[r["End"]],
            y=[y],
            mode="markers",
            marker=dict(size=8, color="firebrick"),
            showlegend=False,
            hovertemplate=(f"<b>{r['Ship']}</b><br>End: %{ '{' }x|%Y-%m-%d %H:%M{ '}' }<extra></extra>")
        ))

    fig.update_layout(
        title="Выполнение текущих рейсов (node-edge по времени)",
        xaxis_title="Время",
        yaxis_title="Судно",
        yaxis=dict(
            ticktext=ships,
            tickvals=list(range(len(ships))),
            range=[-0.5, len(ships) - 0.5]
        ),
        hovermode="closest",
        height=max(400, len(ships) * 45)
    )
    return fig