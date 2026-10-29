from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dynamic_network import NODES, create_network, default_link_table, find_best_route, route_metrics

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
st.set_page_config(page_title="PredictiveRoute | Network Intelligence", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:radial-gradient(ellipse at 10% 0%,#102544 0%,transparent 38%),#070D18;color:#E5EEFF}
[data-testid="stSidebar"]{background:#0B1424;border-right:1px solid #1B304A}[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.4rem;max-width:1600px}.hero{padding:1.4rem 1.7rem;border:1px solid #203956;border-radius:18px;background:linear-gradient(115deg,#101F36,#0B1424 70%);margin-bottom:1.2rem}
.eyebrow{color:#39D9FF;letter-spacing:2px;font-size:.72rem;font-weight:700}.hero h1{color:#F4F8FF;font-size:2.35rem;margin:.35rem 0;font-weight:800}.hero p{color:#9EB2CC;margin:0}
.metric-card{background:linear-gradient(145deg,#111F33,#0C1524);border:1px solid #213750;border-radius:15px;padding:1rem;min-height:110px}.metric-label{color:#91A7C3;font-size:.82rem}.metric-value{color:#49D9FF;font-size:1.65rem;font-weight:800;margin-top:.35rem}.metric-note{color:#7D94B1;font-size:.75rem;margin-top:.2rem}
div.stButton>button,div.stFormSubmitButton>button{background:linear-gradient(100deg,#087DF5,#16B9E8);color:white;border:0;border-radius:10px;font-weight:700;min-height:2.7rem}h2,h3{color:#EDF5FF!important}
</style>""", unsafe_allow_html=True)

POS = {"Source":(0,2),"R1":(1,3),"R2":(1,1),"R3":(2,3.6),"R4":(2,2.5),"R5":(2,.5),"R6":(3,3.2),"R7":(3,2),"R8":(3,.8),"R9":(4,2.8),"R10":(4,1.2),"Destination":(5,2)}

def draw_graph(graph, route):
    fig=go.Figure(); chosen={frozenset((a,b)) for a,b in zip(route,route[1:])}
    for a,b,d in graph.edges(data=True):
        x0,y0=POS[a];x1,y1=POS[b];sel=frozenset((a,b)) in chosen
        color="#25E6B8" if sel else ("#FF647C" if d["congested"] else "#354A67")
        fig.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",line=dict(color=color,width=4 if sel else 1.7,dash="solid" if sel else "dash"),hovertemplate=f"{a} ↔ {b}<br>Capacity: {d['capacity_mbps']} Mbps<br>Latency: {d['base_latency_ms']} ms<br>Background: {d['background_mbps']} Mbps<br>Utilization: {d['utilization_pct']:.1f}%<extra></extra>",showlegend=False))
        fig.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2+.08,text=f"{d['utilization_pct']:.0f}%",showarrow=False,font=dict(size=9,color=color))
    for n,(x,y) in POS.items():
        color="#168BFF" if n in ("Source","Destination") else ("#FF9B54" if any(graph[n][v]["congested"] for v in graph.neighbors(n)) else "#765CFF")
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[n],textposition="bottom center",textfont=dict(color="#DDEBFF",size=10),marker=dict(size=25 if n in ("Source","Destination") else 19,color=color,line=dict(color="#BCEBFF",width=1.2)),showlegend=False,hovertemplate=f"{n}<extra>Node</extra>"))
    fig.update_layout(height=460,margin=dict(l=5,r=5,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(visible=False,range=[-.3,5.4]),yaxis=dict(visible=False,range=[0,4.1]))
    return fig

def card(label,value,note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

if "links" not in st.session_state: st.session_state.links=pd.DataFrame(default_link_table())
if "result" not in st.session_state: st.session_state.result=None
if "traffic" not in st.session_state: st.session_state.traffic=30.0
with st.sidebar:
    st.markdown("## 🌐 PredictiveRoute");st.caption("NETWORK INTELLIGENCE PLATFORM");st.markdown("---")
    page=st.radio("NAVIGATION",["Overview","Run Simulation","Network Topology","Performance Analysis","Dataset & Files","About"])
    st.markdown("---");st.markdown("**SYSTEM STATUS**");st.success("Simulation engine ready");st.caption("10 routers · Multiple paths · Synthetic environment")
st.markdown('<div class="hero"><div class="eyebrow">AI • NETWORK OPTIMIZATION • SMART ROUTING</div><h1>Predictive<span style="color:#37CFFF">Route</span></h1><p>Predict congestion. Evaluate alternate paths. Understand network performance.</p></div>',unsafe_allow_html=True)

if page in ["Overview","Run Simulation","Network Topology"]:
    st.markdown("### Simulation Configuration")
    with st.form("config"):
        c1,c2=st.columns(2)
        with c1: source=st.selectbox("Source",["Source"])
        with c2: destination=st.selectbox("Destination",["Destination"])
        traffic=st.number_input("Traffic load to send (Mbps)",min_value=0.0,max_value=10000.0,value=float(st.session_state.traffic),step=5.0)
        st.caption("Edit each link directly: capacity, base latency, background traffic, and whether the link is enabled.")
        edited=st.data_editor(st.session_state.links,use_container_width=True,hide_index=True,num_rows="fixed",key="link_table_editor",column_config={
            "source":st.column_config.SelectboxColumn("From",options=NODES,required=True),"target":st.column_config.SelectboxColumn("To",options=NODES,required=True),
            "capacity_mbps":st.column_config.NumberColumn("Capacity (Mbps)",min_value=.1,step=5.0,format="%.1f"),
            "base_latency_ms":st.column_config.NumberColumn("Base latency (ms)",min_value=0.0,step=1.0,format="%.1f"),
            "background_mbps":st.column_config.NumberColumn("Background traffic (Mbps)",min_value=0.0,step=5.0,format="%.1f"),
            "enabled":st.column_config.CheckboxColumn("Enabled")})
        submit=st.form_submit_button("▶ Apply Values & Run Simulation",use_container_width=True)
    if submit:
        try:
            if (edited["capacity_mbps"]<=0).any(): raise ValueError("Capacity must be greater than zero.")
            if (edited["base_latency_ms"]<0).any() or (edited["background_mbps"]<0).any(): raise ValueError("Latency and background traffic cannot be negative.")
            graph=create_network(edited.to_dict("records"),traffic);route=find_best_route(graph,source,destination);metrics=route_metrics(graph,route)
            st.session_state.links=edited.copy();st.session_state.traffic=float(traffic);st.session_state.result={"graph":graph,"route":route,"metrics":metrics};st.success("Simulation updated using your input values.")
        except Exception as e: st.error(str(e))
    if st.session_state.result is None:
        try:
            graph=create_network(st.session_state.links.to_dict("records"),st.session_state.traffic);route=find_best_route(graph);metrics=route_metrics(graph,route);st.session_state.result={"graph":graph,"route":route,"metrics":metrics}
        except Exception as e: st.error(str(e));st.stop()
    result=st.session_state.result;graph,route,metrics=result["graph"],result["route"],result["metrics"]
    if page=="Overview":
        st.markdown("### Network Overview");a,b,c,d=st.columns(4)
        with a: card("Traffic Load",f"{st.session_state.traffic:g} Mbps","Entered traffic")
        with b: card("Estimated Path Latency",f"{metrics['estimated_latency_ms']:.1f} ms","Sum of base link latency")
        with c: card("Mean Path Utilization",f"{metrics['mean_utilization_pct']:.1f}%","Selected route")
        with d: card("Congested Links",str(metrics["congested_links"]),"On selected route")
        left,right=st.columns([1.65,1])
        with left: st.markdown("### Network Topology");st.plotly_chart(draw_graph(graph,route),use_container_width=True)
        with right: st.markdown("### Routing Decision");st.success(" → ".join(route));st.metric("Total route cost",metrics["route_cost"]);st.metric("Illustrative loss estimate",f"{metrics['estimated_loss_pct']:.2f}%")
        st.info("Synthetic simulation only. Dijkstra selects the lowest-cost path. The Random Forest model from main.py is not yet connected to this topology simulation.")
    elif page=="Network Topology":
        st.markdown("### Network Topology Explorer");st.plotly_chart(draw_graph(graph,route),use_container_width=True)
        rows=[]
        for a,b,d in graph.edges(data=True): rows.append({"Link":f"{a} — {b}","Capacity Mbps":d["capacity_mbps"],"Latency ms":d["base_latency_ms"],"Background Mbps":d["background_mbps"],"Total load Mbps":round(d["total_load_mbps"],1),"Utilization %":round(d["utilization_pct"],1),"Status":"Congested" if d["congested"] else "Normal","Selected path":any({a,b}=={x,y} for x,y in zip(route,route[1:]))})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else:
        st.markdown("### Simulation Result");st.write("Selected path:"," → ".join(route));st.write("Route cost:",metrics["route_cost"]);st.plotly_chart(draw_graph(graph,route),use_container_width=True)
        st.caption("This currently computes one route for the entered conditions; time-series route-change history can be added next.")
elif page=="Performance Analysis":
    st.markdown("### Performance Analysis");p=RESULTS/"comparison_results.csv"
    if p.exists():
        df=pd.read_csv(p);st.dataframe(df,use_container_width=True,hide_index=True);metric=st.selectbox("Metric",["avg_latency_ms","packet_loss_pct","route_changes"]);st.bar_chart(df.pivot(index="scenario",columns="strategy",values=metric));st.caption("These are existing results from main.py, separate from the editable topology simulation.")
    else: st.warning("Run main.py first to generate results/comparison_results.csv.")
    for name in ["latency_comparison.png","packet_loss_comparison.png","route_changes_comparison.png"]:
        if (RESULTS/name).exists(): st.image(str(RESULTS/name),caption=name,use_container_width=True)
elif page=="Dataset & Files":
    st.markdown("### Dataset & Generated Files")
    for name in ["synthetic_dataset.csv","comparison_results.csv","latency_comparison.png","packet_loss_comparison.png","route_changes_comparison.png"]:
        p=RESULTS/name
        if not p.exists(): st.caption(f"{name} — not generated yet");continue
        st.markdown(f"**{name}**")
        if p.suffix==".csv":
            df=pd.read_csv(p);st.caption(f"{len(df)} rows · {len(df.columns)} columns");st.dataframe(df.head(10),use_container_width=True);st.download_button(f"Download {name}",p.read_bytes(),file_name=name,key=f"download_{name}")
        else: st.image(str(p),use_container_width=True);st.download_button(f"Download {name}",p.read_bytes(),file_name=name,key=f"download_{name}")
elif page=="About":
    st.markdown("### About PredictiveRoute");st.write("Academic prototype for synthetic network congestion and congestion-aware routing.")
    st.markdown("- Exactly 10 routers plus Source and Destination.\n- Multiple alternative paths.\n- Editable per-link capacity, latency, background traffic and enabled state.\n- Typed traffic-load input.\n- Dijkstra route selection based on congestion-aware costs.\n\n**Limitations:** Synthetic traffic only; no live network; Random Forest is not yet integrated into this new topology; packet-loss estimate is illustrative, not measured.")
