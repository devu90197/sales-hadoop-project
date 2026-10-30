"""Shop Sales Analyzer website. Run with: python -m streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline as pl

BLUE = "#2a78d6"
INK2 = "#52514e"
GRID = "#e1e0d9"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Shop Sales Analyzer", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning the dataset...")
def load_all():
    raw = pl.load_raw()
    cleaned, steps = pl.clean(raw)
    return raw, cleaned, steps


@st.cache_data(show_spinner="Building customer data...")
def cached_rfm(df):
    return pl.build_rfm(df)


@st.cache_data(show_spinner="Running MapReduce...")
def cached_mapreduce(df, job, n_splits, use_combiner):
    return pl.run_mapreduce(df, job, n_splits, use_combiner)


@st.cache_data(show_spinner="Testing cluster counts...")
def cached_evaluate(rfm):
    return pl.evaluate_k(rfm)


@st.cache_data(show_spinner="Clustering customers...")
def cached_segments(rfm, k):
    return pl.segment_customers(rfm, k)


def style(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=16, b=8), paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(family="system-ui, Segoe UI, sans-serif", color=INK2, size=13), showlegend=False,
        hoverlabel=dict(bgcolor="#fcfcfb", font_color="#0b0b0b"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def hbar(data, x, y, x_label, y_label, height=420, fmt=",.0f"):
    fig = px.bar(data, x=x, y=y, orientation="h", labels={x: x_label, y: y_label},
                 color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate=f"%{{y}}<br>{x_label}: %{{x:{fmt}}}<extra></extra>", marker_line_width=0)
    fig.update_yaxes(categoryorder="total ascending", showgrid=False)
    return style(fig, height)


raw, df, steps = load_all()
rfm = cached_rfm(df)

st.title("Shop Sales Analyzer")
st.caption("End-to-End Big Data Analytics Pipeline: from MapReduce to Machine Learning Insights")
st.write("**Question:** which products, countries and customers bring in the money, "
         "and what should the shop do next? All amounts are in British pounds (£).")

tabs = st.tabs(["1. Dataset & Cleaning", "2. MapReduce", "3. Analytics", "4. Machine Learning", "5. Insights"])

# ---------- 1. Dataset ----------
with tabs[0]:
    st.subheader("Dataset")
    st.write("**Online Retail** from the [UCI Machine Learning Repository]"
             "(https://archive.ics.uci.edu/dataset/352/online+retail): real transactions of a UK online "
             "gift shop, 1 Dec 2010 to 9 Dec 2011. Many of its customers are wholesalers.")
    a, b, c, d = st.columns(4)
    a.metric("Raw rows", f"{len(raw):,}")
    b.metric("Clean rows", f"{len(df):,}")
    c.metric("Rows removed", f"{len(raw) - len(df):,}")
    d.metric("Customers", f"{df['CustomerID'].nunique():,}")

    st.subheader("Cleaning steps")
    st.dataframe(steps, hide_index=True, width="stretch",
                 column_config={"Rows removed": st.column_config.NumberColumn(format="localized"),
                                "Rows left": st.column_config.NumberColumn(format="localized")})
    no_id = raw[raw["CustomerID"].isna() & (raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)]
    st.info(f"**Trade-off:** rows with no customer ID are real sales worth about "
            f"£{(no_id['Quantity'] * no_id['UnitPrice']).sum():,.0f}. They are removed, so revenue "
            "figures here cover identified customers only, not the shop's full revenue.")

    st.subheader("Look at the data")
    view = st.radio("Show", ["Clean data", "Raw data", "Only cancelled invoices (removed)"], horizontal=True)
    if view == "Clean data":
        st.dataframe(df.head(200), width="stretch", hide_index=True)
    elif view == "Raw data":
        st.dataframe(raw.head(200), width="stretch", hide_index=True)
    else:
        st.dataframe(raw[raw["InvoiceNo"].str.startswith("C")].head(200), width="stretch", hide_index=True)

# ---------- 2. MapReduce ----------
with tabs[1]:
    st.subheader("MapReduce, step by step")
    st.write("Mapper, Combiner, Shuffle & Sort and Reducer are written as Python functions in `pipeline.py`. "
             "This simulates the MapReduce flow on one computer, not on a Hadoop cluster.")
    a, b, c = st.columns(3)
    job = a.selectbox("Job", list(pl.JOBS))
    n_splits = b.slider("Input splits (mappers)", 1, 8, 4)
    use_combiner = c.toggle("Use Combiner", value=True)

    key_col, value_col = pl.JOBS[job]
    run = cached_mapreduce(df, job, n_splits, use_combiner)
    results = run["results"].sort_values(value_col, ascending=False)

    a, b, c, d = st.columns(4)
    a.metric("Records per split", f"{run['split_sizes'][0]:,}")
    b.metric("Map output pairs", f"{run['map_output']:,}")
    c.metric("Pairs sent to shuffle", f"{run['shuffled']:,}",
             delta=f"{run['shuffled'] / run['map_output'] - 1:.2%}" if use_combiner else None,
             delta_color="off")
    d.metric("Final results", f"{run['reduced']:,}")

    st.markdown("#### Follow 6 real rows through the steps")
    sample = df[[key_col, value_col]].head(6)
    records = list(sample.itertuples(index=False, name=None))
    pairs = [pl.mapper(r) for r in records]
    groups = pl.shuffle_sort(pairs)
    stage = st.select_slider("Step", ["Input", "Map", "Shuffle & Sort", "Reduce"])
    if stage == "Input":
        st.dataframe(df[["InvoiceNo", "Description", "Quantity", "UnitPrice", "Country", "Revenue"]].head(6),
                     hide_index=True, width="stretch")
        st.caption("Each row is one product on an invoice.")
    elif stage == "Map":
        st.code("\n".join(f"({k}, {round(v, 2)})" for k, v in pairs))
        st.caption("Each row becomes one (key, value) pair.")
    elif stage == "Shuffle & Sort":
        st.code("\n".join(f"{k} -> {[round(x, 2) for x in v]}" for k, v in groups.items()))
        st.caption("Values with the same key are put together.")
    else:
        st.code("\n".join(f"{k} -> {pl.reducer(k, v)[1]}" for k, v in groups.items()))
        st.caption("The Reducer adds each list to give one total.")

    st.markdown("#### Output")
    top = results.head(10)
    label = "Revenue (£)" if value_col == "Revenue" else "Units sold"
    if job == "Revenue by month":
        monthly = run["results"]
        fig = px.bar(monthly, x="Month", y="Revenue", labels={"Revenue": label}, color_discrete_sequence=[BLUE])
        fig.update_traces(hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>")
        show(style(fig))
    else:
        show(hbar(top, value_col, key_col, label, key_col))

    check = df.groupby(key_col)[value_col].sum().round(2)
    merged = run["results"].set_index(key_col)[value_col]
    matches = (merged - check.reindex(merged.index)).abs().max() < 0.01 and len(merged) == len(check)
    if matches:
        st.success(f"Checked: all {len(merged):,} results match a separate pandas calculation.")
    else:
        st.error("The MapReduce results do not match the pandas calculation.")

# ---------- 3. Analytics ----------
with tabs[2]:
    st.subheader("Trends and patterns")
    countries = ["All countries"] + df["Country"].value_counts().index.tolist()
    chosen = st.selectbox("Country", countries)
    part = df if chosen == "All countries" else df[df["Country"] == chosen]

    a, b, c, d = st.columns(4)
    a.metric("Revenue", f"£{part['Revenue'].sum():,.0f}")
    b.metric("Orders", f"{part['InvoiceNo'].nunique():,}")
    c.metric("Customers", f"{part['CustomerID'].nunique():,}")
    d.metric("Average order", f"£{part.groupby('InvoiceNo')['Revenue'].sum().mean():,.0f}")

    st.markdown("#### Monthly revenue")
    monthly = part.groupby("Month", as_index=False)["Revenue"].sum()
    fig = px.line(monthly, x="Month", y="Revenue", markers=True, labels={"Revenue": "Revenue (£)"},
                  color_discrete_sequence=[BLUE])
    fig.update_traces(line_width=2, marker_size=8, hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>")
    show(style(fig, 320))
    st.caption("December 2011 looks low because the data stops on 9 December.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top 10 products by revenue")
        products = part.groupby("Description", as_index=False)["Revenue"].sum().nlargest(10, "Revenue")
        show(hbar(products, "Revenue", "Description", "Revenue (£)", "Product"))
    with right:
        st.markdown("#### Top 10 countries by revenue")
        if chosen == "All countries":
            st.caption("The UK is left out here because it would hide every other country.")
            country_totals = df[df["Country"] != "United Kingdom"].groupby("Country", as_index=False)["Revenue"].sum()
            show(hbar(country_totals.nlargest(10, "Revenue"), "Revenue", "Country", "Revenue (£)", "Country"))
        else:
            st.write("Choose **All countries** to compare countries.")

    st.markdown("#### When do customers order?")
    orders = part.drop_duplicates("InvoiceNo")
    grid = orders.pivot_table(index="Weekday", columns="Hour", values="InvoiceNo", aggfunc="count", fill_value=0)
    grid = grid.reindex([d for d in WEEKDAYS if d in grid.index])
    fig = px.imshow(grid, aspect="auto", color_continuous_scale=["#cde2fb", "#3987e5", "#0d366b"],
                    labels=dict(x="Hour of day", y="Day", color="Orders"))
    fig.update_traces(hovertemplate="%{y}, %{x}:00<br>%{z} orders<extra></extra>", xgap=2, ygap=2)
    fig.update_xaxes(dtick=1, showgrid=False)
    fig.update_yaxes(showgrid=False)
    show(style(fig, 300))
    if "Saturday" not in grid.index:
        st.caption("Saturday does not appear because the data has no Saturday orders.")

# ---------- 4. Machine learning ----------
with tabs[3]:
    st.subheader("Customer segmentation with K-Means")
    st.write("K-Means puts similar customers into groups, without being told the groups first. "
             "Each customer is described by three **RFM** numbers:")
    st.markdown("- **Recency:** days since the last order\n- **Frequency:** number of orders\n"
                "- **Monetary:** total money spent")
    st.write("The values are log-transformed and scaled first, because a few very large buyers would "
             "otherwise control the groups.")

    scores = cached_evaluate(rfm)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Elbow: lower is tighter")
        fig = px.line(scores, x="k", y="Inertia", markers=True, color_discrete_sequence=[BLUE],
                      labels={"k": "Number of clusters"})
        fig.update_traces(line_width=2, marker_size=8)
        show(style(fig, 260))
    with right:
        st.markdown("#### Silhouette: higher is cleaner")
        fig = px.line(scores, x="k", y="Silhouette", markers=True, color_discrete_sequence=[BLUE],
                      labels={"k": "Number of clusters", "Silhouette": "Silhouette score"})
        fig.update_traces(line_width=2, marker_size=8)
        show(style(fig, 260))

    best_k = int(scores.loc[scores["Silhouette"].idxmax(), "k"])
    best_score = scores["Silhouette"].max()
    four_score = scores.loc[scores["k"] == 4, "Silhouette"].iloc[0]
    st.write(f"The best silhouette score is **{best_score:.2f} at k = {best_k}**, but two groups are too simple to act on. "
             f"k = 4 scores lower, at {four_score:.2f}, about the same as k = 3, and gives the shop more useful "
             "groups, so it is the default. A score this size means the groups overlap: customers do not fall "
             "into perfectly separate types.")
    k = st.slider("Number of customer groups (k)", 2, 6, 4)
    segmented, order = cached_segments(rfm, k)
    summary = pl.segment_summary(segmented, order)
    k_score = scores.loc[scores["k"] == k, "Silhouette"].iloc[0]
    st.metric(f"Silhouette score at k = {k}", f"{k_score:.2f}")

    st.markdown("#### What each group looks like")
    st.dataframe(summary.reset_index().rename(columns={
        "Median_recency_days": "Days since last order", "Median_orders": "Orders", "Median_spend": "Spend (£)",
        "Total_revenue": "Total revenue (£)", "Revenue_share": "Share of revenue"}),
        hide_index=True, width="stretch", column_config={
            "Spend (£)": st.column_config.NumberColumn(format="%.0f"),
            "Total revenue (£)": st.column_config.NumberColumn(format="%.0f"),
            "Share of revenue": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption("Days, orders and spend are medians: the middle customer in each group.")

    st.markdown("#### Customers in each group")
    plot_data = segmented.reset_index()
    fig = px.scatter(plot_data, x="Recency", y="Monetary", facet_col="Segment", category_orders={"Segment": order},
                     log_y=True, opacity=0.35, color_discrete_sequence=[BLUE], hover_name="CustomerID",
                     labels={"Recency": "Days since last order", "Monetary": "Spend (£)"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(marker_size=5)
    show(style(fig, 340))

    st.markdown("#### Look up a customer")
    customer = st.selectbox("Customer ID", segmented.index.tolist())
    row = segmented.loc[customer]
    st.write(f"Customer **{customer}** is in **{row['Segment']}**: last order {row['Recency']} days ago, "
             f"orders placed: {row['Frequency']}, £{row['Monetary']:,.0f} spent.")

# ---------- 5. Insights ----------
with tabs[4]:
    st.subheader("Findings and recommendations")
    st.caption("Calculated from the data, using 4 customer groups.")

    segmented4, order4 = cached_segments(rfm, 4)
    s4 = pl.segment_summary(segmented4, order4)
    total = df["Revenue"].sum()
    uk_share = df.loc[df["Country"] == "United Kingdom", "Revenue"].sum() / total
    full_months = df[df["Month"] != "2011-12"].groupby("Month")["Revenue"].sum()
    peak_month = full_months.idxmax()
    autumn = full_months.loc["2011-09":"2011-11"].sum() / full_months.sum()
    non_uk = df[df["Country"] != "United Kingdom"].groupby("Country")["Revenue"].sum().nlargest(3)
    product_totals = df.groupby("Description")["Revenue"].sum().sort_values(ascending=False)
    top_20_share = product_totals.head(int(len(product_totals) * 0.2)).sum() / total
    order_grid = df.drop_duplicates("InvoiceNo").groupby(["Weekday", "Hour"]).size()
    busy_day, busy_hour = order_grid.idxmax()

    insights = [
        ("Prepare early for autumn",
         f"September to November brought {autumn:.0%} of revenue across the 12 full months, peaking in "
         f"{peak_month}.",
         "Build stock by August and start seasonal campaigns in September."),
        ("Protect the best customers",
         f"The Best group is {s4.loc['Best', 'Customers'] / s4['Customers'].sum():.0%} of customers but "
         f"brings {s4.loc['Best', 'Revenue_share']:.0%} of revenue.",
         "Offer them loyalty rewards, early access and a personal account contact."),
        ("Win back at-risk customers",
         f"The At risk group has {s4.loc['At risk', 'Customers']:,.0f} customers. The typical one last ordered "
         f"{s4.loc['At risk', 'Median_recency_days']:.0f} days ago.",
         "Send a low-cost win-back offer and measure who returns before spending more."),
        ("Grow beyond the UK",
         f"The UK brings {uk_share:.0%} of revenue. The largest other markets are {', '.join(non_uk.index)}.",
         "Test targeted marketing and shipping offers in those markets first."),
        ("Focus stock on top products",
         f"The top 20% of products bring {top_20_share:.0%} of revenue.",
         "Keep these well stocked, and review the slowest products."),
        ("Time campaigns to buying habits",
         f"Orders peak on {busy_day} around {busy_hour}:00.",
         "Send emails and promotions shortly before this time."),
    ]
    for title, finding, action in insights:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(finding)
            st.write(f"**Recommendation:** {action}")

    st.markdown("#### Limits of this analysis")
    st.markdown("- It covers one year, so the autumn peak is not proven to repeat every year.\n"
                "- Rows with no customer ID were removed, so totals are lower than actual shop revenue.\n"
                "- Many customers are wholesalers, so patterns may differ for ordinary shoppers.\n"
                "- These results show patterns, not causes. Test each recommendation before rolling it out.")
