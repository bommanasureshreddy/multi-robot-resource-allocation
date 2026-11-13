from pathlib import Path
import sys, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
DATA, EXP, MODELS = ROOT/"data"/"dataset", ROOT/"experiments", ROOT/"models"
sys.path.insert(0, str(ROOT/"src"))

st.set_page_config(page_title="Multi-Robot Resource Allocation", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.4rem}
[data-testid="stMetric"]{background:#F4F7FA;border:1px solid #E4EAF0;padding:12px;border-radius:10px}
h1,h2,h3{color:#243B53}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def csv(path):
    p=DATA/path
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def expcsv(name):
    p=EXP/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_resource
def project():
    from robots import RobotFleet
    from tasks import TaskManager
    from environment import Environment
    return RobotFleet(dataset_root=str(DATA)), TaskManager(dataset_root=str(DATA)), Environment(dataset_root=str(DATA))

@st.cache_resource
def flow_result():
    from network_flow import build_flow_graph, edmonds_karp_max_flow, flow_to_assignment
    f,t,_=project()
    g=build_flow_graph(list(f.robots.values()),t.active_tasks())
    value,flow=edmonds_karp_max_flow(g)
    return value,flow_to_assignment(flow),g.number_of_nodes(),g.number_of_edges()

def overview():
    st.title("Multi-Robot Resource Allocation")
    st.write("Battery-aware, priority-aware task allocation for heterogeneous UGV, USV and UAV fleets.")
    f,t,e=project()
    r,ts=f.as_dataframe(),t.as_dataframe()
    a,b,c,d=st.columns(4)
    a.metric("Robots",len(r)); b.metric("Tasks",len(ts))
    c.metric("Emergency tasks",int(ts.emergency.sum()))
    d.metric("Environment zones",len(e.zones))
    left,right=st.columns(2)
    with left:
        st.subheader("Fleet composition")
        counts=r.robot_type.value_counts().rename_axis("Robot type").reset_index(name="Count")
        st.plotly_chart(px.bar(counts,x="Robot type",y="Count",color="Robot type",
            color_discrete_map={"UGV":"#287D6B","USV":"#3976A8","UAV":"#9B6BB3"}),use_container_width=True)
    with right:
        st.subheader("Task categories")
        counts=ts.task_type.value_counts().rename_axis("Task type").reset_index(name="Count")
        st.plotly_chart(px.bar(counts,x="Task type",y="Count",color_discrete_sequence=["#6C91BF"]),use_container_width=True)
    st.subheader("System components")
    st.markdown("""
    - Edmonds–Karp maximum-flow allocation with robot/task feasibility constraints
    - A* path planning and battery-aware auction allocation
    - Gradient Boosting predictions for energy, completion time and task risk
    - Event-driven simulation, dynamic reallocation and Jain's fairness index
    - Baseline, ablation, scenario and scalability experiments
    """)

def environment_map():
    st.title("Environment & Mission Map")
    from environment import Environment
    e=Environment(dataset_root=str(DATA))
    robots=csv("robots/robots.csv"); tasks=csv("tasks/tasks.csv")
    stations=csv("environment/charging_stations.csv"); obs=csv("environment/obstacles.csv")
    fig=go.Figure()
    colors={"LAND":"#DCEFE6","WATER":"#DCEAF7","SAFE":"#E7F4E8","RESTRICTED":"#F9EBCB",
            "FORBIDDEN":"#F4DADA","EMERGENCY":"#F8E2D8","CHARGING":"#E8E3F7"}
    for z in e.zones:
        if z.zone_type=="AIR": continue
        fig.add_shape(type="rect",x0=z.x_min,y0=z.y_min,x1=z.x_max,y1=z.y_max,
            fillcolor=colors.get(z.zone_type,"#EDF2F7"),opacity=.4,
            line=dict(color="#AAB7C4",width=.7,dash="dot"),layer="below")
    if not obs.empty:
        for _,o in obs.iterrows():
            fig.add_shape(type="rect",x0=o.x,y0=o.y,x1=o.x+o.width,y1=o.y+o.height,
                fillcolor="#8795A1",opacity=.8,line=dict(color="#52606D",width=1))
    if not stations.empty:
        fig.add_trace(go.Scatter(x=stations.x,y=stations.y,mode="markers",name="Charging stations",
            text=stations.station_id,marker=dict(symbol="star",size=12,color="#8B75B5")))
    if not tasks.empty:
        fig.add_trace(go.Scatter(x=tasks.x,y=tasks.y,mode="markers",name="Tasks",text=tasks.task_id,
            customdata=tasks[["task_type","priority","emergency"]],
            marker=dict(symbol="diamond",size=9,color="#D97757"),
            hovertemplate="%{text}<br>%{customdata[0]}<br>Priority %{customdata[1]}<br>Emergency %{customdata[2]}<extra></extra>"))
    if not robots.empty:
        for typ,g in robots.groupby("robot_type"):
            fig.add_trace(go.Scatter(x=g.x,y=g.y,mode="markers+text",name=typ,text=g.robot_id,
                textposition="top center",marker=dict(size=12,color={"UGV":"#287D6B","USV":"#3976A8","UAV":"#9B6BB3"}[typ])))
    fig.update_layout(height=620,plot_bgcolor="white",xaxis=dict(range=[0,100],title="X",dtick=10),
        yaxis=dict(range=[0,100],title="Y",dtick=10,scaleanchor="x",scaleratio=1),
        legend=dict(orientation="h",y=1.02),margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Static MAP01 dataset view; not live robot telemetry.")

def allocation():
    st.title("Network Flow Allocation")
    value,assignment,nodes,edges=flow_result()
    a,b,c=st.columns(3); a.metric("Maximum flow",value); b.metric("Assignments",len(assignment)); c.metric("Graph edges",edges)
    df=pd.DataFrame([{"Robot":r,"Task":t} for r,t in sorted(assignment.items())])
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.caption(f"Flow graph contains {nodes} nodes. Each robot and task has capacity 1.")
    st.info("This is feasibility-based maximum flow. Distance is only an edge annotation in this implementation; Edmonds–Karp does not optimize distance cost.")

def robot_page():
    st.title("Robot Fleet Monitoring")
    r=csv("robots/robots.csv"); b=csv("robots/robot_battery.csv")
    if r.empty: st.warning("Robot dataset not found."); return
    if not b.empty:
        r=r.merge(b[["robot_id","initial_soc","initial_soh","initial_dod","min_soc"]],on="robot_id",how="left")
    typ=st.selectbox("Robot type",["All"]+sorted(r.robot_type.unique()))
    v=r if typ=="All" else r[r.robot_type==typ]
    st.dataframe(v,use_container_width=True,hide_index=True)
    if "initial_soc" in v:
        st.plotly_chart(px.bar(v,x="robot_id",y="initial_soc",color="robot_type",
            color_discrete_map={"UGV":"#287D6B","USV":"#3976A8","UAV":"#9B6BB3"},
            labels={"initial_soc":"Initial SOC (%)"}),use_container_width=True)
    st.caption("Battery values shown here are initial dataset values, not live readings.")

def predictions():
    st.title("Machine Learning Predictions")
    paths={k:MODELS/f"{k}_gradient_boosting.joblib" for k in ["energy_consumption","completion_time","task_risk"]}
    missing=[p.name for p in paths.values() if not p.exists()]
    if missing: st.error("Missing models: "+", ".join(missing)); return
    st.write("Predict with the saved Gradient Boosting models.")
    with st.form("predict"):
        x1,x2,x3=st.columns(3)
        with x1:
            rt=st.selectbox("Robot type",["UGV","USV","UAV"])
            soc=st.slider("SOC (%)",10.0,100.0,80.0); dod=st.slider("DOD (%)",0.0,93.5,20.0)
            soh=st.slider("SOH",.613,.997,.90); speed=st.slider("Speed",1.0,10.0,5.0)
        with x2:
            dist=st.slider("Distance",1.0,90.0,20.0); plen=st.slider("Path length",1.0,125.0,25.0)
            workload=st.slider("Task workload",5.0,40.0,20.0); priority=st.slider("Priority",1,5,3)
        with x3:
            zone=st.selectbox("Zone type",["LAND","WATER","AIR","SAFE","RESTRICTED"])
            risk=st.slider("Zone risk",.5,4.96,2.0); load=st.slider("Robot load",.1,1.0,.5)
            temp=st.slider("Environment factor",2.51,46.89,25.0)
        go=st.form_submit_button("Predict outcomes",type="primary")
    if go:
        cols=["soc","dod","soh","distance","path_length","speed","task_workload","task_priority",
              "zone_risk","robot_load","temperature_or_environment_factor","robot_type","zone_type"]
        row=pd.DataFrame([[soc,dod,soh,dist,plen,speed,workload,priority,risk,load,temp,rt,zone]],columns=cols)
        try:
            out={k:float(joblib.load(p).predict(row)[0]) for k,p in paths.items()}
            a,b,c=st.columns(3); a.metric("Predicted energy",f"{out['energy_consumption']:.2f}")
            b.metric("Predicted completion time",f"{out['completion_time']:.2f}")
            c.metric("Predicted task risk",f"{out['task_risk']:.2f}")
        except Exception as ex: st.error(f"Prediction failed: {ex}")
    st.caption("Estimates are based on the project's synthetic training dataset and are not guaranteed real-world measurements.")

def dynamic():
    st.title("Dynamic Simulation History")
    files={n:csv("simulation/"+n+".csv") for n in ["robot_state_history","task_state_history","graph_history","allocation_history"]}
    ids=set()
    for df in files.values():
        if not df.empty and "run_id" in df: ids.update(df.run_id.dropna().astype(str))
    if not ids: st.info("No simulation history with run IDs is available."); return
    run=st.selectbox("Simulation run",sorted(ids))
    selected={}
    for name,df in files.items():
        selected[name]=df[df.run_id.astype(str)==run].copy() if not df.empty and "run_id" in df else pd.DataFrame()
    rh=selected["robot_state_history"]
    if not rh.empty and {"t","soc","robot_id"}.issubset(rh.columns):
        st.plotly_chart(px.line(rh,x="t",y="soc",color="robot_id",labels={"t":"Simulation time","soc":"SOC (%)"}),use_container_width=True)
    for name,label in [("task_state_history","Task states"),("graph_history","Event history"),("allocation_history","Allocation explanations")]:
        if not selected[name].empty:
            st.subheader(label); st.dataframe(selected[name],use_container_width=True,hide_index=True)

def analytics():
    st.title("Experiments & Performance Analytics")
    baseline=expcsv("baseline_comparison.csv"); ablation=expcsv("ablation_study.csv")
    scenarios=expcsv("scenarios.csv"); scale=expcsv("scalability_full_metrics.csv")
    if baseline.empty: st.warning("Experiment CSVs not found."); return
    metrics={"Task completion rate":"task_completion_rate","Successful allocation rate":"successful_allocation_rate",
        "Jain fairness index":"jains_fairness_index","Average completion time":"avg_completion_time",
        "Total energy consumption":"total_energy_consumption","Deadline satisfaction":"deadline_satisfaction",
        "Unsafe allocations":"unsafe_allocation_count"}
    label=st.selectbox("Metric",list(metrics)); m=metrics[label]
    if m in baseline:
        st.plotly_chart(px.bar(baseline,x="method",y=m,color_discrete_sequence=["#6C91BF"]),use_container_width=True)
        st.dataframe(baseline[["method",m]],use_container_width=True,hide_index=True)
    t1,t2,t3=st.tabs(["Ablation","Scenarios","Scalability"])
    with t1:
        if not ablation.empty and m in ablation:
            st.plotly_chart(px.bar(ablation,x="method",y=m,color_discrete_sequence=["#76A99A"]),use_container_width=True)
            st.dataframe(ablation,use_container_width=True,hide_index=True)
    with t2:
        if not scenarios.empty and m in scenarios:
            st.plotly_chart(px.bar(scenarios,x="scenario",y=m,hover_data=["scenario_description"] if "scenario_description" in scenarios else None,
                color_discrete_sequence=["#D59A72"]),use_container_width=True)
            st.dataframe(scenarios,use_container_width=True,hide_index=True)
    with t3:
        if not scale.empty:
            st.plotly_chart(px.line(scale,x="n_robots",y="allocation_computation_time_sec",color="n_tasks",markers=True),use_container_width=True)
            st.dataframe(scale,use_container_width=True,hide_index=True)
    p=EXP/"gat_ppo_summary.json"
    if p.exists():
        with open(p,encoding="utf-8") as f: rl=json.load(f)
        st.subheader("Separate Graph RL evaluation")
        a,b,c=st.columns(3); a.metric("GAT + PPO mean reward",f"{rl['gatppo_mean']:.3f}")
        b.metric("Auction baseline reward",f"{rl['auction_baseline_mean']:.3f}")
        c.metric("Evaluation seeds",rl["n_eval_seeds"])
        st.caption("This reward comparison is separate and is not directly comparable with task-completion-rate metrics.")

st.sidebar.title("Navigation")
page=st.sidebar.radio("Dashboard sections",["Overview","Environment Map","Network Flow Allocation",
    "Robot Monitoring","ML Predictions","Dynamic Simulation","Performance Analytics"])
st.sidebar.caption("Research prototype • MAP01")
if page=="Overview": overview()
elif page=="Environment Map": environment_map()
elif page=="Network Flow Allocation": allocation()
elif page=="Robot Monitoring": robot_page()
elif page=="ML Predictions": predictions()
elif page=="Dynamic Simulation": dynamic()
else: analytics()
st.divider()
st.caption("Research prototype. Data and results are loaded from the supplied project datasets and saved experiment outputs. No live robot connection.")
