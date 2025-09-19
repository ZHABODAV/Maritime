
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List

# Import our custom modules
from data_loader import load_maritime_data, transform_data
from analytics import calculate_kpis, calculate_vessel_efficiency, get_performance_trends, calculate_port_performance, analyze_schedule_conflicts
from visualizations import (
    create_gantt_chart, create_parallel_coordinates, create_enhanced_berth_allocation,
    create_performance_dashboard, create_network_visualization, create_timeline_chart
)
from predictive_models import (
    predict_arrival_times, predict_berth_availability, predict_port_congestion,
    calculate_eta_accuracy, optimize_vessel_routing
)
from neo4j_integration import Neo4jIntegration

# Configure Streamlit page
st.set_page_config(
    page_title="Дашборд Морских Операций",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .status-excellent { color: #28a745; }
    .status-good { color: #17a2b8; }
    .status-fair { color: #ffc107; }
    .status-poor { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_transform_data():
    """Load and transform maritime data with caching"""
    raw_data, _, _ = load_maritime_data()
    transformed_data = transform_data(raw_data)
    return transformed_data, raw_data

@st.cache_data(ttl=600)  # Cache for 10 minutes
def calculate_analytics(data):
    """Calculate analytics with caching"""
    kpis = calculate_kpis(data)
    vessel_efficiency = calculate_vessel_efficiency(data)
    performance_trends = get_performance_trends(data)
    port_performance = calculate_port_performance(data)
    schedule_conflicts = analyze_schedule_conflicts(data)
    return kpis, vessel_efficiency, performance_trends, port_performance, schedule_conflicts

@st.cache_data(ttl=900)  # Cache for 15 minutes
def generate_predictions(data):
    """Generate predictions with caching"""
    arrival_predictions = predict_arrival_times(data)
    berth_predictions = predict_berth_availability(data)
    congestion_predictions = predict_port_congestion(data)
    routing_recommendations = optimize_vessel_routing(data)
    accuracy_metrics = calculate_eta_accuracy(arrival_predictions, [])
    return arrival_predictions, berth_predictions, congestion_predictions, routing_recommendations, accuracy_metrics

def main():
    # Header
    st.markdown('<h1 class="main-header">⚓ Дашборд Морских Операций</h1>', unsafe_allow_html=True)
    st.markdown("**Расширенная аналитика • Прогнозное моделирование • Мониторинг в реальном времени**")
    
    # Load data
    with st.spinner("Загрузка морских данных..."):
        data, raw_data = load_and_transform_data()
    
    # Sidebar filters
    st.sidebar.header("⚙️ Настройки панели")
    
    # Refresh button
    if st.sidebar.button("🔄 Обновить данные", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Filters
    st.sidebar.subheader("📊 Фильтры данных")
    
    # Region filter fixed to main three
    regions = ["Мировой регион №1", "Волго-Каспийский регион №2", "Азово-Черноморский и Средиземноморский"]
    selected_regions = st.sidebar.multiselect(
        "Выберите регион",
        regions,
        default=regions
    )

    # Contract type filter (ТЧ и Спот)
    contract_types = ["Тайм-чартер", "Спот"]
    selected_contract_types = st.sidebar.multiselect(
        "Тип контракта",
        contract_types,
        default=contract_types
    )

    # Vessel categories filter
    vessel_types = ["Барже-буксирные составы", "Танкеры река-море", "Сухогрузы река-море", "Deep Sea сухогрузы", "Deep Sea танкеры"]
    selected_vessel_types = st.sidebar.multiselect(
        "Типы судов",
        vessel_types,
        default=vessel_types
    )

    # Data range filter (loading/discharge operations)
    date_range = st.sidebar.date_input(
        "Диапазон дат рейсов",
        value=(datetime.now().date() - timedelta(days=7), datetime.now().date() + timedelta(days=30)),
        format="DD/MM/YYYY"
    )
    
    # Apply filters
    filtered_data = apply_filters(data, selected_regions, selected_vessel_types, date_range)
    
    # Analytics sidebar
    st.sidebar.subheader("📈 Быстрая аналитика")
    with st.sidebar:
        quick_stats = calculate_kpis(filtered_data)
        st.metric("Всего судов", quick_stats["total_vessels"], delta=quick_stats.get("vessel_change", 0))
        st.metric("Активные операции", quick_stats["active_operations"], delta=quick_stats.get("operations_change", 0))
        st.metric("Загруженность портов", f"{quick_stats['port_utilization']:.1f}%",
                 delta=f"{quick_stats.get('utilization_change', 0):.1f}%")
        st.metric("Эффективность флота", f"{quick_stats['fleet_efficiency']:.1f}%",
                 delta=f"{quick_stats.get('efficiency_change', 0):.1f}%")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Обзор", "📅 Планирование (Гант)", "🎯 Параллельный анализ",
        "🏗️ Портовые операции", "🔮 Прогнозная аналитика", "📈 Метрики производительности"
    ])
    
    with tab1:
        display_overview_tab(filtered_data)
    
    # Download/upload templates section
    st.subheader("📥 Шаблоны данных")
    from data_templates import download_template_ui, upload_data_ui
    download_template_ui()
    uploaded_files = upload_data_ui()
    if uploaded_files:
        from data_templates import TEMPLATES
        for name, df in uploaded_files.items():
            st.markdown(f"**Загружен шаблон для {name}**")
            st.dataframe(df)
            print(f"Uploaded {name} with {len(df)} rows")
            
            # Validate columns
            expected_cols = TEMPLATES.get(name, {}).get("columns", [])
            actual_cols = list(df.columns)
            if set(expected_cols) != set(actual_cols):
                print(f"Validation failed for {name}: Expected {expected_cols}, got {actual_cols}")
                st.error(f"Неверные столбцы в {name}: Ожидалось {expected_cols}, получено {actual_cols}")
                continue
            
            # Basic type coercion (example for numeric columns)
            for col in df.columns:
                if "quantity" in col.lower() or "capacity" in col.lower():
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    print(f"Coerced {col} to numeric in {name}")
            
            # Append merge instead of overwrite
            if name in data:
                existing_ids = {item.get("id") for item in data[name] if "id" in item}
                new_records = df.to_dict("records")
                added = 0
                for rec in new_records:
                    rec_id = rec.get("id")
                    if rec_id and rec_id not in existing_ids:
                        data[name].append(rec)
                        added += 1
                    else:
                        print(f"Skipped duplicate or invalid record in {name} with ID {rec_id}")
                print(f"Appended {added} new records to {name}, total now: {len(data[name])}")
            else:
                print(f"Warning: Uploaded {name} but no matching key in data")
        st.cache_data.clear()  # Clear all caches to ensure fresh data
        st.experimental_rerun()  # Force refresh after upload
    
    with tab2:
        display_gantt_tab(filtered_data)
    
    with tab3:
        display_parallel_analysis_tab(filtered_data)
    
    with tab4:
        display_berth_operations_tab(filtered_data)
    
    with tab5:
        display_predictive_analytics_tab(filtered_data)
    
    with tab6:
        display_performance_metrics_tab(filtered_data)

    # Neo4j section
    st.subheader("🌐 Граф морских маршрутов (Neo4j)")
    try:
        from neo4j_integration import neo4j_connection, build_and_query_graph
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        conn = neo4j_connection(uri, user, password)
        if conn:
            graph_fig = build_and_query_graph(data)
            st.plotly_chart(graph_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Neo4j visualization недоступна: {e}")

def apply_filters(data, regions, vessel_types, date_range):
    """Apply filters to the data"""
    filtered_data = data.copy()
    
    # Filter ports by region
    if regions:
        filtered_data["ports"] = [p for p in filtered_data["ports"] if p.get("region") in regions]
    
    # Filter ships by type
    if vessel_types:
        filtered_data["ships"] = [s for s in filtered_data["ships"] if s.get("type") in vessel_types]
    
    # Filter berths based on filtered ports
    port_names = [p["name"] for p in filtered_data["ports"]]
    filtered_data["berths"] = [b for b in filtered_data["berths"] if b.get("port") in port_names]
    
    return filtered_data

def display_overview_tab(data):
    """Display overview dashboard"""
    st.header("📊 Operations Overview")
    
    # KPI Cards
    kpis = calculate_kpis(data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Vessels", kpis["total_vessels"], delta=kpis.get("vessel_change", 0))
    with col2:
        st.metric("Active Operations", kpis["active_operations"], delta=kpis.get("operations_change", 0))
    with col3:
        st.metric("Port Utilization", f"{kpis['port_utilization']:.1f}%", 
                 delta=f"{kpis.get('utilization_change', 0):.1f}%")
    with col4:
        st.metric("Fleet Efficiency", f"{kpis['fleet_efficiency']:.1f}%",
                 delta=f"{kpis.get('efficiency_change', 0):.1f}%")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 Network Topology")
        network_fig = create_network_visualization(data)
        st.plotly_chart(network_fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Performance Dashboard")
        performance_fig = create_performance_dashboard(data)
        st.plotly_chart(performance_fig, use_container_width=True)
    
    # Timeline chart
    st.subheader("⏱️ Operations Timeline")
    timeline_fig = create_timeline_chart(data)
    st.plotly_chart(timeline_fig, use_container_width=True)

def display_gantt_tab(data):
    """Display Gantt chart and scheduling analysis"""
    st.header("📅 Vessel Scheduling & Gantt Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🗓️ Interactive Gantt Chart")
        gantt_fig = create_gantt_chart(data)
        st.plotly_chart(gantt_fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Schedule Conflicts")
        conflicts = analyze_schedule_conflicts(data)
        
        if conflicts:
            for conflict in conflicts[:5]:  # Show top 5 conflicts
                severity_color = {
                    "High": "🔴",
                    "Medium": "🟡", 
                    "Low": "🟢"
                }.get(conflict["severity"], "⚪")
                
                st.markdown(f"""
                **{severity_color} {conflict['port']}**
                - {conflict['vessel1']} ↔ {conflict['vessel2']}
                - Overlap: {conflict['overlap_hours']:.1f}h
                """)
        else:
            st.success("✅ No schedule conflicts detected")
    
    # Schedule statistics
    st.subheader("📊 Schedule Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    total_operations = sum(len(ship.get("schedule", [])) for ship in data["ships"])
    completed_operations = sum(len([s for s in ship.get("schedule", []) if s.get("status") == "Complete"]) 
                             for ship in data["ships"])
    
    with col1:
        st.metric("Total Operations", total_operations)
    with col2:
        st.metric("Completed", completed_operations)
    with col3:
        completion_rate = (completed_operations / total_operations * 100) if total_operations > 0 else 0
        st.metric("Completion Rate", f"{completion_rate:.1f}%")

def display_parallel_analysis_tab(data):
    """Display parallel coordinates analysis"""
    st.header("🎯 Multi-dimensional Performance Analysis")
    
    # Calculate vessel efficiency metrics
    efficiency_data = calculate_vessel_efficiency(data)
    efficiency_df = pd.DataFrame(efficiency_data)
    
    if not efficiency_df.empty:
        st.subheader("📊 Parallel Coordinates Plot")
        st.markdown("**Analyze vessel performance across multiple dimensions simultaneously**")
        
        parallel_fig = create_parallel_coordinates(efficiency_df)
        st.plotly_chart(parallel_fig, use_container_width=True)
        
        # Performance insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performers")
            top_performers = efficiency_df.nlargest(5, 'efficiency_score')[['vessel_name', 'efficiency_score', 'performance_rating']]
            for _, vessel in top_performers.iterrows():
                rating_color = {
                    "Excellent": "status-excellent",
                    "Good": "status-good", 
                    "Fair": "status-fair",
                    "Needs Improvement": "status-poor"
                }.get(vessel['performance_rating'], "")
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{vessel['vessel_name']}</strong><br>
                    Score: {vessel['efficiency_score']:.1f}<br>
                    <span class="{rating_color}">{vessel['performance_rating']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📈 Performance Distribution")
            
            # Performance rating distribution
            rating_counts = efficiency_df['performance_rating'].value_counts()
            fig = px.pie(values=rating_counts.values, names=rating_counts.index,
                        title="Performance Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Performance Metrics")
        display_columns = [
            'vessel_name', 'vessel_type', 'efficiency_score', 'on_time_performance',
            'capacity_utilization', 'fuel_efficiency', 'performance_rating'
        ]
        
        available_columns = [col for col in display_columns if col in efficiency_df.columns]
        st.dataframe(
            efficiency_df[available_columns].round(2),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("⚠️ No efficiency data available for parallel analysis")

def display_berth_operations_tab(data):
    """Display enhanced berth allocation and operations"""
    st.header("🏗️ Enhanced Berth Operations Dashboard")
    
    # Enhanced berth allocation visualization
    berth_fig = create_enhanced_berth_allocation(data)
    st.plotly_chart(berth_fig, use_container_width=True)
    
    # Port performance analysis
    st.subheader("🏭 Port Performance Analysis")
    port_performance = calculate_port_performance(data)
    
    if port_performance:
        # Convert to DataFrame for display
        port_df = pd.DataFrame(port_performance).T
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Utilization by Port")
            fig = px.bar(
                x=port_df.index,
                y=port_df['berth_utilization'],
                title="Berth Utilization Rates",
                labels={'x': 'Port', 'y': 'Utilization %'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Processing Efficiency")
            # Convert data to regular lists for plotly compatibility
            wait_times = port_df['avg_waiting_time'].tolist()
            processing_times = port_df['avg_processing_time'].tolist()
            berth_sizes = port_df['total_berths'].tolist()
            efficiency = port_df['throughput_efficiency'].tolist()
            port_names = port_df.index.tolist()
            
            fig = px.scatter(
                x=wait_times,
                y=processing_times,
                size=berth_sizes,
                color=efficiency,
                hover_name=port_names,
                title="Wait Time vs Processing Time",
                labels={'x': 'Average Waiting Time (hours)', 'y': 'Average Processing Time (hours)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Port details table
        st.subheader("📋 Port Details")
        display_columns = [
            'total_berths', 'occupied_berths', 'berth_utilization',
            'avg_waiting_time', 'avg_processing_time', 'throughput_efficiency'
        ]
        
        available_columns = [col for col in display_columns if col in port_df.columns]
        st.dataframe(
            port_df[available_columns].round(2),
            use_container_width=True
        )

def display_predictive_analytics_tab(data):
    """Display predictive analytics and machine learning insights"""
    st.header("🔮 Predictive Analytics & Machine Learning")
    
    # Generate predictions
    with st.spinner("Generating ML predictions..."):
        arrival_predictions, berth_predictions, congestion_predictions, routing_recommendations, accuracy_metrics = generate_predictions(data)
    
    # Accuracy metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction Accuracy", f"{accuracy_metrics['accuracy_percentage']:.1f}%")
    with col2:
        st.metric("Mean Absolute Error", f"{accuracy_metrics['mean_absolute_error']:.1f}h")
    with col3:
        st.metric("Model Confidence", f"{accuracy_metrics.get('avg_confidence', 85.0):.1f}%")
    
    # Predictions tabs
    pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs([
        "🚢 Arrival Predictions", "⚓ Berth Availability", "🚦 Port Congestion", "🗺️ Route Optimization"
    ])
    
    with pred_tab1:
        display_arrival_predictions(arrival_predictions)
    
    with pred_tab2:
        display_berth_predictions(berth_predictions)
    
    with pred_tab3:
        display_congestion_predictions(congestion_predictions)
    
    with pred_tab4:
        display_routing_recommendations(routing_recommendations)

def display_arrival_predictions(predictions):
    """Display vessel arrival predictions"""
    st.subheader("🚢 Vessel Arrival Predictions")
    
    if predictions:
        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # ETA visualization
        if 'predicted_arrival' in pred_df.columns:
            pred_df['predicted_arrival_dt'] = pd.to_datetime(pred_df['predicted_arrival'])
            pred_df['hours_to_arrival'] = (pred_df['predicted_arrival_dt'] - datetime.now()).dt.total_seconds() / 3600
            
            fig = px.scatter(
                pred_df,
                x='hours_to_arrival',
                y='vessel_name',
                size=pred_df['confidence'].values if 'confidence' in pred_df.columns else None,
                color='delay_estimate_hours',
                title="Predicted Arrival Times",
                labels={'hours_to_arrival': 'Hours to Arrival', 'vessel_name': 'Vessel'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Predictions table
        display_columns = ['vessel_name', 'predicted_arrival', 'confidence', 'delay_estimate_hours', 'distance_remaining']
        available_columns = [col for col in display_columns if col in pred_df.columns]
        
        st.dataframe(
            pred_df[available_columns].round(2),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("ℹ️ No arrival predictions available")

def display_berth_predictions(predictions):
    """Display berth availability predictions"""
    st.subheader("⚓ Berth Availability Forecast")
    
    if predictions:
        pred_df = pd.DataFrame(predictions)
        
        # Availability timeline
        if 'available_from' in pred_df.columns and 'available_until' in pred_df.columns:
            pred_df['available_from_dt'] = pd.to_datetime(pred_df['available_from'])
            pred_df['available_until_dt'] = pd.to_datetime(pred_df['available_until'])
            
            fig = px.timeline(
                pred_df,
                x_start='available_from_dt',
                x_end='available_until_dt',
                y='berth_name',
                color='current_status',
                title="Berth Availability Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # High-priority berths
        st.subheader("🎯 High-Priority Berths")
        top_berths = pred_df.nlargest(5, 'utilization_score')
        for _, berth in top_berths.iterrows():
            st.markdown(f"""
            **{berth['berth_name']}** ({berth['port']})
            - Status: {berth['current_status']}
            - Utilization Score: {berth['utilization_score']:.1f}
            - Confidence: {berth['confidence']:.1f}%
            """)
    else:
        st.info("ℹ️ No berth predictions available")

def display_congestion_predictions(predictions):
    """Display port congestion predictions"""
    st.subheader("🚦 Port Congestion Forecast")
    
    if predictions:
        # Create congestion overview
        congestion_summary = []
        for port_name, port_data in predictions.items():
            congestion_summary.append({
                'port': port_name,
                'avg_congestion': port_data['avg_congestion'],
                'peak_congestion': port_data['peak_congestion'],
                'recommendation': port_data['recommended_action']
            })
        
        if congestion_summary:
            summary_df = pd.DataFrame(congestion_summary)
            
            # Congestion levels chart
            fig = px.bar(
                summary_df,
                x='port',
                y=['avg_congestion', 'peak_congestion'],
                title="Port Congestion Levels",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            for _, row in summary_df.iterrows():
                congestion_level = row['avg_congestion']
                color = "🔴" if congestion_level > 80 else "🟡" if congestion_level > 60 else "🟢"
                
                st.markdown(f"""
                **{color} {row['port']}** (Avg: {congestion_level:.1f}%)
                - {row['recommendation']}
                """)
    else:
        st.info("ℹ️ No congestion predictions available")

def display_routing_recommendations(recommendations):
    """Display route optimization recommendations"""
    st.subheader("🗺️ Route Optimization Recommendations")
    
    if recommendations:
        for rec in recommendations:
            savings_color = "🟢" if rec.get('time_savings_hours', 0) > 0 else "🟡"
            
            st.markdown(f"""
            **{savings_color} {rec['vessel_name']}**
            - Current: {rec['current_destination']}
            - Recommended: {rec['recommended_alternative']}
            - Time Savings: {rec.get('time_savings_hours', 0):.1f} hours
            - Fuel Savings: {rec.get('fuel_savings_estimate', 0):.1f}%
            - Confidence: {rec.get('recommendation_confidence', 0):.1f}%
            - Reason: {rec.get('reasoning', 'Optimization based on current conditions')}
            """)
    else:
        st.info("ℹ️ No routing recommendations available")

def display_performance_metrics_tab(data):
    """Display advanced performance metrics and trends"""
    st.header("📈 Advanced Performance Metrics")
    
    # Performance trends
    trends = get_performance_trends(data)
    
    if trends and trends.get('dates'):
        # Create trends DataFrame
        trends_df = pd.DataFrame(trends)
        trends_df['date'] = pd.to_datetime(trends_df['dates'])
        
        # Performance trends chart
        fig = go.Figure()
        
        metrics = ['vessel_utilization', 'port_efficiency', 'on_time_performance']
        colors = ['blue', 'green', 'orange']
        
        for metric, color in zip(metrics, colors):
            if metric in trends_df.columns:
                fig.add_trace(go.Scatter(
                    x=trends_df['date'],
                    y=trends_df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color)
                ))
        
        fig.update_layout(
            title="Performance Trends (30 Days)",
            xaxis_title="Date",
            yaxis_title="Performance (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_util = trends_df['vessel_utilization'].iloc[-1] if 'vessel_utilization' in trends_df.columns else 0
            trend_util = trends_df['vessel_utilization'].iloc[-1] - trends_df['vessel_utilization'].iloc[0] if 'vessel_utilization' in trends_df.columns else 0
            st.metric("Current Vessel Utilization", f"{current_util:.1f}%", delta=f"{trend_util:.1f}%")
        
        with col2:
            current_eff = trends_df['port_efficiency'].iloc[-1] if 'port_efficiency' in trends_df.columns else 0
            trend_eff = trends_df['port_efficiency'].iloc[-1] - trends_df['port_efficiency'].iloc[0] if 'port_efficiency' in trends_df.columns else 0
            st.metric("Current Port Efficiency", f"{current_eff:.1f}%", delta=f"{trend_eff:.1f}%")
        
        with col3:
            current_otp = trends_df['on_time_performance'].iloc[-1] if 'on_time_performance' in trends_df.columns else 0
            trend_otp = trends_df['on_time_performance'].iloc[-1] - trends_df['on_time_performance'].iloc[0] if 'on_time_performance' in trends_df.columns else 0
            st.metric("On-Time Performance", f"{current_otp:.1f}%", delta=f"{trend_otp:.1f}%")
    
    else:
        st.warning("⚠️ No performance trend data available")

if __name__ == "__main__":
    main()
