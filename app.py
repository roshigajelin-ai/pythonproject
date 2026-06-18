import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #F7F9FC; }

    .stApp { background-color: #F7F9FC; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A3C5E 0%, #0D2137 100%);
    }
    section[data-testid="stSidebar"] * { color: #E8F0F8 !important; }
    section[data-testid="stSidebar"] .stRadio label { color: #E8F0F8 !important; }

    /* Header cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(26,60,94,0.08);
        border-left: 4px solid #2E75B6;
        margin-bottom: 12px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #6B7A8D; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card h2 { margin: 4px 0 0; font-size: 28px; color: #1A3C5E; font-weight: 700; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1A3C5E, #2E75B6);
        color: white !important;
        padding: 14px 20px;
        border-radius: 10px;
        margin: 24px 0 16px;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Insight boxes */
    .insight-box {
        background: #EBF4FF;
        border-left: 4px solid #2E75B6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 14px;
        color: #1A3C5E;
    }

    /* Title */
    .hero-title {
        font-size: 38px;
        font-weight: 700;
        color: #1A3C5E;
        margin-bottom: 4px;
    }
    .hero-sub {
        font-size: 16px;
        color: #5A7A9A;
        margin-bottom: 24px;
    }

    /* Plot containers */
    .plot-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(26,60,94,0.06);
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-left: 3px solid #2E75B6;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        color: #1A3C5E;
        border: 1px solid #D5E8F0;
    }
    .stTabs [aria-selected="true"] {
        background: #1A3C5E !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Try common paths
    for path in [
        "airlines_flights_data.csv",
        "data/airlines_flights_data.csv",
        os.path.join(os.path.dirname(__file__), "airlines_flights_data.csv"),
        os.path.join(os.path.dirname(__file__), "data", "airlines_flights_data.csv"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'index' in df.columns:
                df.drop(columns='index', inplace=True)
            return df
    return None

@st.cache_data
def add_delay_features(df):
    np.random.seed(42)
    df = df.copy()
    df['delay_minutes'] = (
        np.where(df['stops'] == 'zero',        np.random.randint(0, 20, len(df)),   0)
      + np.where(df['stops'] == 'one',         np.random.randint(10, 60, len(df)),  0)
      + np.where(df['stops'] == 'two_or_more', np.random.randint(30, 120, len(df)), 0)
      + (df['duration'] * 2)
    ).astype(int)
    df['is_delayed'] = np.where(df['delay_minutes'] > 30, 1, 0)
    def delay_risk(x):
        if x <= 15:   return 'Low'
        elif x <= 45: return 'Medium'
        else:         return 'High'
    df['delay_risk'] = df['delay_minutes'].apply(delay_risk)
    return df

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Flight Analysis")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 EDA — Q1 to Q9",
        "🔥 Correlation Detective",
        "📖 Data Storytelling",
        "📈 Interactive Plotly",
        "🗺️ Route Heatmap",
        "🎯 ML Model",
    ])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Airlines Flights Data")
    st.markdown("300,153 records · 11 features")
    st.markdown("---")
    st.markdown("**Project by**")
    st.markdown("Big Data Analytics · Python")

# ── LOAD ──────────────────────────────────────────────────────────────────────
data_raw = load_data()

if data_raw is None:
    st.error("⚠️  **airlines_flights_data.csv not found.**")
    st.info("Place `airlines_flights_data.csv` in the same folder as `app.py` and restart.")
    st.markdown("""
    **Expected location:**
    ```
    flight_app/
        app.py
        airlines_flights_data.csv
    ```
    """)
    st.stop()

data = add_delay_features(data_raw)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="hero-title">✈️ Flight Delay Detection & Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Big Data Analytics on 300,153 Indian domestic flight records</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(data):,}")
    c2.metric("Airlines", data['airline'].nunique())
    c3.metric("Avg Price (INR)", f"₹{data['price'].mean():,.0f}")
    c4.metric("Delayed Flights", f"{data['is_delayed'].sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Dataset Sample</div>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True, height=300)

    with col2:
        st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(data[['duration','days_left','price','delay_minutes']].describe().round(2), use_container_width=True)

    st.markdown('<div class="section-header">Conclusion</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(26,60,94,0.06);line-height:1.8;color:#2C3E50;font-size:15px;">
    This project presents a comprehensive analysis of Indian domestic airline flight data using Python-based data science tools.
    Beginning with exploratory data analysis on 300,153 flight records, the study uncovered key pricing patterns — Vistara and Air India
    command the highest average fares, while AirAsia and SpiceJet cater to budget travelers. The analysis revealed that ticket prices rise
    sharply when booked within 1–2 days of departure and gradually stabilize beyond 15 days, confirming that early booking is a financially
    sound strategy. The correlation heatmap demonstrated that flight duration holds the strongest positive relationship with price among
    numeric features, while days left shows a mild negative correlation. Through data storytelling, four purposeful charts were used to
    answer the central question of what drives airline ticket prices. Interactive Plotly visualizations further enhanced the exploratory
    experience by enabling drill-down analysis across airlines, routes, and seat classes. A route-level heatmap identified the most and
    least expensive city pairs, and a boxplot clearly highlighted the significant price gap between Economy and Business class. Finally,
    a synthetic delay feature was engineered using domain logic around stops and flight duration, and a Random Forest classifier was trained
    to predict flight delays with strong accuracy. Feature importance analysis confirmed that duration and number of stops are the most
    influential predictors of delay. Overall, this project demonstrates the full data science pipeline — from raw data ingestion and cleaning,
    through visual storytelling, to machine learning — making it a well-rounded and GitHub-ready portfolio project.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA Q1–Q9
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA — Q1 to Q9":
    st.markdown('<div class="hero-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("Q1 through Q9 — all questions from the original notebook")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Q1 Airlines", "Q2 Time", "Q3 Cities", "Q4 Price/Airline",
        "Q5 Price/Time", "Q6 Price/City", "Q7 Days Left", "Q8 Class", "Q9 Vistara"
    ])

    # ── Q1 ──
    with tab1:
        st.markdown('<div class="section-header">Q.1 What are the airlines and their frequencies?</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            vc = data['airline'].value_counts()
            st.dataframe(vc.rename("Count").reset_index().rename(columns={'index':'Airline'}), use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            data['airline'].value_counts(ascending=True).plot.barh(
                color=['lightgreen', 'lightblue', 'lightgreen', 'lightblue', 'lightgreen', 'lightblue'], ax=ax)
            ax.set_title("Airlines with frequencies", fontsize=14, fontweight='bold')
            ax.set_xlabel("Number of flights")
            ax.set_ylabel("Airlines")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Q2 ──
    with tab2:
        st.markdown('<div class="section-header">Q.2 Departure Time & Arrival Time</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Departure Time counts:**")
            st.dataframe(data['departure_time'].value_counts().rename("Count"), use_container_width=True)
        with col2:
            st.write("**Arrival Time counts:**")
            st.dataframe(data['arrival_time'].value_counts().rename("Count"), use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        axes[0].bar(data['departure_time'].value_counts().index,
                    data['departure_time'].value_counts().values, color=['r', 'b', 'r', 'b', 'r', 'b'])
        axes[0].set_title("Departure Time"); axes[0].set_xlabel("D. Time"); axes[0].set_ylabel("D. Freq")
        axes[0].tick_params(axis='x', rotation=30)

        axes[1].bar(data['arrival_time'].value_counts().index,
                    data['arrival_time'].value_counts().values, color=['g', 'y', 'g', 'y', 'g', 'y'])
        axes[1].set_title("Arrival Time"); axes[1].set_xlabel("A. Time"); axes[1].set_ylabel("A. Freq")
        axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Q3 ──
    with tab3:
        st.markdown('<div class="section-header">Q.3 Source City & Destination City</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data['source_city'].value_counts().rename("Count"), use_container_width=True)
        with col2:
            st.dataframe(data['destination_city'].value_counts().rename("Count"), use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        axes[0].barh(data['source_city'].value_counts().index,
                     data['source_city'].value_counts().values, color=['m', 'g', 'm', 'g', 'm', 'g'])
        axes[0].set_title("Source Cities with No. of flights"); axes[0].set_ylabel("Cities"); axes[0].set_xlabel("No. of flights")

        axes[1].barh(data['destination_city'].value_counts().index,
                     data['destination_city'].value_counts().values, color=['y', 'r', 'y', 'r', 'y', 'r'])
        axes[1].set_title("Destination Cities with No. of flights"); axes[1].set_ylabel("Cities"); axes[1].set_xlabel("No. of flights")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Q4 ──
    with tab4:
        st.markdown('<div class="section-header">Q.4 Does price vary with airlines?</div>', unsafe_allow_html=True)
        st.dataframe(data.groupby('airline')['price'].mean().round(2).rename("Mean Price"), use_container_width=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        mean_prices = data.groupby(['airline', 'class'])['price'].mean().unstack()
        mean_prices.plot(kind='bar', ax=ax, colormap='rocket')
        ax.set_title("Mean Ticket Price per Airline by Class", fontsize=14, fontweight='bold')
        ax.set_xlabel("Airline"); ax.set_ylabel("Average Price")
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title='Class')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">💡 Vistara and Air India have significantly higher average fares than budget carriers like AirAsia and SpiceJet.</div>', unsafe_allow_html=True)

    # ── Q5 ──
    with tab5:
        st.markdown('<div class="section-header">Q.5 Does price change with departure & arrival time?</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Mean price by Departure Time:**")
            st.dataframe(data.groupby('departure_time')['price'].mean().round(2).rename("Mean Price"), use_container_width=True)
        with col2:
            st.write("**Mean price by Arrival Time:**")
            st.dataframe(data.groupby('arrival_time')['price'].mean().round(2).rename("Mean Price"), use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        dep_mean = data.groupby('departure_time')['price'].mean()
        axes[0].bar(dep_mean.index, dep_mean.values, color='steelblue')
        axes[0].set_title("Price vs Departure Time"); axes[0].tick_params(axis='x', rotation=30)
        axes[0].set_ylabel("Average Price")

        arr_mean = data.groupby('arrival_time')['price'].mean()
        axes[1].bar(arr_mean.index, arr_mean.values, color='coral')
        axes[1].set_title("Price vs Arrival Time"); axes[1].tick_params(axis='x', rotation=30)
        axes[1].set_ylabel("Average Price")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.write("**Arrival Time vs Price by Departure Time (relplot-style):**")
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        for dep in data['departure_time'].unique():
            subset = data[data['departure_time'] == dep].groupby('arrival_time')['price'].mean()
            ax2.plot(subset.index, subset.values, marker='o', label=dep)
        ax2.set_title("Arrival Time vs Price (grouped by Departure Time)")
        ax2.set_xlabel("Arrival Time"); ax2.set_ylabel("Average Price")
        ax2.legend(title='Departure Time', bbox_to_anchor=(1.01, 1))
        ax2.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    # ── Q6 ──
    with tab6:
        st.markdown('<div class="section-header">Q.6 How does price change with Source & Destination?</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data.groupby('source_city')['price'].mean().round(2).rename("Mean Price"), use_container_width=True)
        with col2:
            st.dataframe(data.groupby('destination_city')['price'].mean().round(2).rename("Mean Price"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(14, 5))
        for src in data['source_city'].unique():
            subset = data[data['source_city'] == src].groupby('destination_city')['price'].mean()
            ax.plot(subset.index, subset.values, marker='o', label=src)
        ax.set_title("Destination City vs Price (grouped by Source City)")
        ax.set_xlabel("Destination City"); ax.set_ylabel("Average Price")
        ax.legend(title='Source City', bbox_to_anchor=(1.01, 1))
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Q7 ──
    with tab7:
        st.markdown('<div class="section-header">Q.7 How is price affected by days left before departure?</div>', unsafe_allow_html=True)
        days_price = data.groupby('days_left')['price'].mean()
        st.dataframe(days_price.round(2).rename("Mean Price"), use_container_width=True, height=200)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(days_price.index, days_price.values, color='#2E75B6', linewidth=2)
        ax.fill_between(days_price.index, days_price.values, alpha=0.1, color='#2E75B6')
        ax.set_title("Price vs Days Left Before Departure", fontsize=14, fontweight='bold')
        ax.set_xlabel("Days Left"); ax.set_ylabel("Average Price")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">💡 Prices spike sharply 1–2 days before departure. Book at least 15 days ahead for stable, lower fares.</div>', unsafe_allow_html=True)

    # ── Q8 ──
    with tab8:
        st.markdown('<div class="section-header">Q.8 Ticket price — Economy vs Business?</div>', unsafe_allow_html=True)
        eco = data[data['class'] == 'Economy']
        biz = data[data['class'] == 'Business']
        col1, col2 = st.columns(2)
        col1.metric("Economy Mean Price", f"₹{eco['price'].mean():,.2f}", f"{len(eco):,} records")
        col2.metric("Business Mean Price", f"₹{biz['price'].mean():,.2f}", f"{len(biz):,} records")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='class', y='price', data=data, ax=ax, palette=['#4A90D9', '#E8704A'])
        ax.set_title("Economy vs Business Price Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Class"); ax.set_ylabel("Price")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Q9 ──
    with tab9:
        st.markdown('<div class="section-header">Q.9 Average Price — Vistara, Delhi → Hyderabad, Business</div>', unsafe_allow_html=True)
        new_data = data[
            (data['airline'] == 'Vistara') &
            (data['source_city'] == 'Delhi') &
            (data['destination_city'] == 'Hyderabad') &
            (data['class'] == 'Business')
        ]
        st.metric("Average Price", f"₹{new_data['price'].mean():,.2f}", f"{len(new_data):,} matching records")
        st.dataframe(new_data[['airline','flight','source_city','destination_city','stops','duration','days_left','price']].head(20), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CORRELATION DETECTIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔥 Correlation Detective":
    st.markdown('<div class="hero-title">Correlation Detective — Heatmap</div>', unsafe_allow_html=True)
    st.markdown("Since only `duration`, `days_left`, and `price` are numeric, we create a correlation heatmap.")

    corr = data[['duration', 'days_left', 'price']].corr()

    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, ax=ax,
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.heatmap(data[['duration', 'days_left', 'price']].corr(),
                    annot=True, cmap='YlGnBu', ax=ax2,
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        ax2.set_title("Feature Relationship Heatmap (YlGnBu)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("### Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="insight-box">📈 <b>Duration vs Price → Positive correlation</b><br>Longer flights generally cost more.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">📉 <b>Days Left vs Price → Negative correlation</b><br>Early bookings are usually cheaper.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="insight-box">✈️ Flight duration has a <b>stronger effect on price</b> than days left before departure.</div>', unsafe_allow_html=True)
        st.dataframe(corr.round(4), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA STORYTELLING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 Data Storytelling":
    st.markdown('<div class="hero-title">Data Storytelling in 4 Charts</div>', unsafe_allow_html=True)
    st.markdown('**Story:** *"What affects airline ticket prices?"*')

    # Chart 1
    st.markdown('<div class="section-header">Chart 1: Airline vs Average Price (Bar Chart)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_price = data.groupby('airline')['price'].mean().sort_values()
    avg_price.plot(kind='bar', ax=ax, color='#2E75B6')
    ax.set_title("Average Ticket Price by Airline", fontsize=14, fontweight='bold')
    ax.set_xlabel("Airline"); ax.set_ylabel("Average Price")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">📖 <b>Story:</b> Vistara and Air India have the highest average fares.</div>', unsafe_allow_html=True)

    # Chart 2
    st.markdown('<div class="section-header">Chart 2: Days Left vs Price (Line Chart)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    daily_price = data.groupby('days_left')['price'].mean()
    ax.plot(daily_price.index, daily_price.values, color='#2E75B6', linewidth=2)
    ax.set_title("Booking Days Left vs Ticket Price", fontsize=14, fontweight='bold')
    ax.set_xlabel("Days Left"); ax.set_ylabel("Average Price")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">📖 <b>Story:</b> Last-minute bookings are significantly more expensive.</div>', unsafe_allow_html=True)

    # Chart 3
    st.markdown('<div class="section-header">Chart 3: Ticket Price Distribution (Histogram)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data['price'], bins=50, color='#2E75B6', edgecolor='white', alpha=0.85)
    ax.set_title("Distribution of Flight Prices", fontsize=14, fontweight='bold')
    ax.set_xlabel("Price"); ax.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">📖 <b>Story:</b> Most tickets are concentrated in lower price ranges with a few very expensive business-class tickets.</div>', unsafe_allow_html=True)

    # Chart 4
    st.markdown('<div class="section-header">Chart 4: Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data[['duration', 'days_left', 'price']].corr(),
                annot=True, cmap='YlGnBu', ax=ax, annot_kws={'size': 13})
    ax.set_title("Feature Relationship Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown('<div class="insight-box">📖 <b>Story:</b> Flight duration has a stronger effect on price than days left.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INTERACTIVE PLOTLY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Interactive Plotly":
    st.markdown('<div class="hero-title">Interactive Graphs using Plotly</div>', unsafe_allow_html=True)
    st.markdown("Hover, zoom, and click to explore the data interactively.")

    # Bar
    st.markdown('<div class="section-header">Interactive Airline Price Chart</div>', unsafe_allow_html=True)
    avg_price = data.groupby('airline')['price'].mean().reset_index()
    avg_price.columns = ['airline', 'price']
    avg_price['price'] = avg_price['price'].round(0)
    fig = px.bar(avg_price, x='airline', y='price',
                 title='Average Ticket Price by Airline',
                 text='price', color='airline',
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter
    st.markdown('<div class="section-header">Interactive Scatter — Duration vs Price</div>', unsafe_allow_html=True)
    sample = data.sample(5000, random_state=42)
    fig = px.scatter(sample, x='duration', y='price', color='class',
                     hover_data=['airline'],
                     title='Duration vs Price',
                     color_discrete_map={'Economy': '#2E75B6', 'Business': '#E8704A'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Sunburst
    st.markdown('<div class="section-header">Interactive Sunburst — Class → Airline Revenue</div>', unsafe_allow_html=True)
    fig = px.sunburst(data, path=['class', 'airline'], values='price',
                      title='Class → Airline Revenue Distribution',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Treemap
    st.markdown('<div class="section-header">Interactive Treemap — Route Price Distribution</div>', unsafe_allow_html=True)
    fig = px.treemap(data, path=['source_city', 'destination_city'], values='price',
                     title='Route Price Distribution',
                     color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ROUTE HEATMAP + BOXPLOT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Route Heatmap":
    st.markdown('<div class="hero-title">Route Analysis</div>', unsafe_allow_html=True)

    # Route heatmap
    st.markdown('<div class="section-header">Average Ticket Price by Route</div>', unsafe_allow_html=True)
    route_price = data.pivot_table(values='price', index='source_city',
                                   columns='destination_city', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(route_price, annot=True, cmap='Reds', fmt='.0f', ax=ax,
                annot_kws={'size': 11, 'fontweight': 'bold'},
                linewidths=0.5)
    ax.set_title("Average Ticket Price by Route", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Boxplot
    st.markdown('<div class="section-header">Business vs Economy Price Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='class', y='price', data=data, ax=ax,
                palette={'Economy': '#4A90D9', 'Business': '#E8704A'})
    ax.set_title("Economy vs Business Price Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Class"); ax.set_ylabel("Price")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Delay risk
    st.markdown('<div class="section-header">Flight Delay Risk Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ['Low', 'Medium', 'High']
    palette = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
    sns.countplot(x='delay_risk', data=data, order=order, palette=palette, ax=ax)
    ax.set_title("Flight Delay Risk Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Delay Risk"); ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ML MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 ML Model":
    st.markdown('<div class="hero-title">Machine Learning — Delay Prediction</div>', unsafe_allow_html=True)
    st.markdown("Random Forest Classifier trained on synthetic delay labels.")

    if st.button("▶  Train Model", type="primary"):
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        with st.spinner("Training Random Forest... please wait"):
            le = LabelEncoder()
            df_ml = data.copy()
            categorical_cols = ['airline','source_city','destination_city',
                                 'departure_time','arrival_time','stops','class']
            for col in categorical_cols:
                df_ml[col] = le.fit_transform(df_ml[col])

            X = df_ml[['airline','source_city','destination_city','departure_time',
                        'arrival_time','stops','class','duration','days_left']]
            y = df_ml['is_delayed']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, output_dict=True)

        st.markdown("### Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc*100:.2f}%")
        col2.metric("Precision (delayed)", f"{report['1']['precision']:.2f}")
        col3.metric("Recall (delayed)", f"{report['1']['recall']:.2f}")

        st.markdown("**Classification Report:**")
        st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)

        # Feature importance
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2E75B6' if v > importance['Importance'].median() else '#90BCE3'
                  for v in importance['Importance']]
        bars = ax.barh(importance['Feature'], importance['Importance'], color=colors)
        for bar, val in zip(bars, importance['Importance']):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)
        ax.set_title("Flight Delay Prediction — Feature Importance", fontsize=14, fontweight='bold')
        ax.set_xlabel("Importance Score")
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    else:
        st.info("Click **▶ Train Model** to run the Random Forest classifier.")
        st.markdown("""
        **What happens when you train:**
        - Categorical features get label-encoded
        - 80/20 train/test split
        - Random Forest with 100 trees
        - Accuracy, precision, recall displayed
        - Feature importance chart generated
        """)
