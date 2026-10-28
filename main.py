# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "joblib==1.6.0",
#     "marimo>=0.23.3",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "pandas==3.0.6",
#     "plotly==7.1.0",
#     "polars[pyarrow]==2.0.0",
#     "pyarrow>=26.0.0",
#     "python-lsp-ruff==2.3.4",
#     "scikit-learn>=1.9.1",
#     "scipy==1.18.1",
#     "shap==0.54.1",
#     "sinica==0.1.6",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
#     "xgboost==3.4.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="columns", layout_file="layouts/main.slides.json")

with app.setup:
    import joblib
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as p
    import polars.selectors as cs
    import shap
    from plotly.subplots import make_subplots
    from scipy.stats import chi2_contingency
    from sinica import ProfileReport
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import RocCurveDisplay, classification_report
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
    )
    from xgboost import XGBClassifier


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #  End-to-End BFSI Marketing Campaign Strategy & Target Audience Optimization Pipeline

    ## Primary Objective

    Analyze historical banking customer contact data to evaluate previous marketing campaign ROI, build predictive models to target customers likely to subscribe to financial products, perform behavioral segmentation, and execute statistical hypothesis testing to optimize future campaign acquisition strategies.

    ---

    ## Problem Statement (Key Deliverables)

    1. **Customer Segmentation & Profiling:**
    * Apply unsupervised clustering algorithms (e.g., K-Means or Hierarchical Clustering) and Decision Trees to segment the bank's customer base into distinct behavioral personas based on demographic, financial, and past response attributes.




    2. **Propensity / Conversion Modeling (Targeting Optimization):**
    * Build and validate supervised machine learning models (e.g., XGBoost, Logistic Regression, Random Forest) in Python to predict product adoption (campaign response).


    * Focus on ROI optimization and class imbalance, evaluating models using Precision-Recall curves, ROC-AUC, and Lift/Gain charts.




    3. **Statistical A/B Testing & Campaign Evaluation:**
    * Design and execute hypothesis testing ($Z$-test / Chi-Square test) to assess the statistical significance of conversion rate differences across different contact channels (e.g., cellular vs. telephone) and campaign timing.



    4. **Executive Tableau Dashboard & Insight Reporting:**
    *  Build an interactive Tableau dashboard tailored for non-technical leadership showcasing campaign KPIs (Conversion Rate, Cost per Acquisition, Audience Reach) and visual audience breakdowns.

    ---

    # About the data
    | # | Variable | Description | Values |
    |---|---|---|---|
    | 1 | `age` | Age | Numeric |
    | 2 | `job` | Type of job | `admin.`, `unknown`, `unemployed`, `management`, `housemaid`, `entrepreneur`, `student`, `blue-collar`, `self-employed`, `retired`, `technician`, `services` |
    | 3 | `marital` | Marital status (`divorced` includes divorced or widowed) | `married`, `divorced`, `single` |
    | 4 | `education` | Education level | `unknown`, `secondary`, `primary`, `tertiary` |
    | 5 | `default` | Has credit in default? | `yes`, `no` |
    | 6 | `balance` | Average yearly balance, in euros | Numeric |
    | 7 | `housing` | Has a housing loan? | `yes`, `no` |
    | 8 | `loan` | Has a personal loan? | `yes`, `no` |

    ### Last contact of the current campaign

    | # | Variable | Description | Values |
    |---|---|---|---|
    | 9 | `contact` | Contact communication type | `unknown`, `telephone`, `cellular` |
    | 10 | `day` | Day of the month of the last contact | Numeric |
    | 11 | `month` | Month of the last contact | `jan`, `feb`, `mar`, …, `nov`, `dec` |
    | 12 | `duration` | Duration of the last contact, in seconds | Numeric |

    ### Other attributes

    | # | Variable | Description | Values |
    |---|---|---|---|
    | 13 | `campaign` | Number of contacts during this campaign for this client, including the last contact | Numeric |
    | 14 | `pdays` | Days since the client was last contacted in a previous campaign (`-1` means the client was not previously contacted) | Numeric |
    | 15 | `previous` | Number of contacts before this campaign for this client | Numeric |
    | 16 | `poutcome` | Outcome of the previous marketing campaign | `unknown`, `other`, `failure`, `success` |

    ### Target variable

    | # | Variable | Description | Values |
    |---|---|---|---|
    | 17 | `y` | Has the client subscribed to a term deposit? | `yes`, `no` |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exploratory Data Analysis
    """)
    return


@app.cell
def _():
    d = p.read_csv("data/bank-sample.csv")
    lf = p.scan_csv("data/bank-full.csv")
    return d, lf


@app.cell
def _(d):
    d
    return


@app.cell
def _(d):
    d.glimpse()
    return


@app.cell
def _(d):
    d.describe()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Data Type | Alias / Syntax | Bit-Width | Range / Description |
    | --- | --- | --- | --- |
    | **Signed Integer** | `pl.Int8` | 8-bit | `-128` to `127` |
    | **Signed Integer** | `pl.Int16` | 16-bit | `-32,768` to `32,767` |
    | **Signed Integer** | `pl.Int32` | 32-bit | `-2,147,483,648` to `2,147,483,647` |
    | **Signed Integer** | `pl.Int64` (default) | 64-bit | `-9,223,372,036,854,775,808` to `9,223,372,036,854,775,807` |
    | **Unsigned Integer** | `pl.UInt8` | 8-bit | `0` to `255` |
    | **Unsigned Integer** | `pl.UInt16` | 16-bit | `0` to `65,535` |
    | **Unsigned Integer** | `pl.UInt32` | 32-bit | `0` to `4,294,967,295` |
    | **Unsigned Integer** | `pl.UInt64` | 64-bit | `0` to `18,446,744,073,709,551,615` |
    | **Float** | `pl.Float32` | 32-bit | Approx. $\pm 1.5 \times 10^{-45}$ to $\pm 3.4 \times 10^{38}$ (7 decimal digits precision) |
    | **Float** | `pl.Float64` (default) | 64-bit | Approx. $\pm 5.0 \times 10^{-324}$ to $\pm 1.7 \times 10^{308}$ (15-17 decimal digits precision) |

    ---
    """)
    return


@app.function
def correct_types(df) -> p.DataFrame:

    df_corrected = df.with_columns(
        # Numeric columns
        p.col("age", "campaign", "previous").cast(p.UInt16),
        p.col("duration").cast(p.UInt16),
        p.col("balance", "pdays").cast(p.Int32),
        p.col("day").cast(p.UInt8),
        # String columns
        p.col(df.select(~cs.numeric()).columns).cast(p.Categorical),
    )

    # size_before = df.estimated_size("kb")
    # size_after = df_corrected.estimated_size("kb")
    # red_pct = abs(size_after - size_before) / size_before * 100
    # print(f"Size BEFORE: {size_before:.2f} KB")
    # print(f"Size AFTER: {size_after:.2f} KB")
    # print(f"Reduction: {red_pct:.2f}%")
    return df_corrected


@app.function
def binary_encode(df):
    return df.with_columns(
        p.col("loan", "housing", "default").replace_strict(
            {"yes": 1, "no": 0}, return_dtype=p.UInt8
        ),
    )


@app.cell
def _(d):
    pr = ProfileReport(correct_types(d))
    return (pr,)


@app.cell
def _(pr):
    pr.show_overview()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Univariate Analysis**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Numeric Columns
    """)
    return


@app.cell
def _(d):
    numeric_cols = cs.expand_selector(d, cs.numeric())
    numeric_cols
    return


@app.cell
def _(pr):
    pr.univ("age", "numeric")
    return


@app.cell
def _(pr):
    pr.univ("balance", "numeric")
    return


@app.cell
def _(pr):
    pr.univ("day", "numeric")
    return


@app.cell
def _(pr):
    pr.univ("duration", "numeric")
    return


@app.cell
def _(pr):
    pr.univ("campaign", "numeric")
    return


@app.cell
def _(pr):
    pr.univ("pdays", "numeric")
    return


@app.cell
def _(pr):
    pr.univ("previous", "numeric")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Categorical Columns
    """)
    return


@app.cell
def _(d):
    cat_cols = cs.expand_selector(d, cs.categorical())
    cat_cols
    return


@app.cell
def _(pr):
    pr.univ("job", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("marital", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("education", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("default", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("housing", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("loan", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("contact", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("month", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("poutcome", "categorical")
    return


@app.cell
def _(pr):
    pr.univ("y", "categorical")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Number of contacts have 6x than previous year
    """)
    return


@app.cell
def _(d):

    mo.ui.plotly(
        go.Figure()
        .add_trace(
            go.Histogram(
                x=d["campaign"],
                name="Current Campaign Contacts",
                xbins=dict(start=0, end=20, size=1),  # Force sensible bin sizes
            )
        )
        .add_trace(
            go.Histogram(
                x=d["previous"],
                name="Previous Campaign Contacts",
                xbins=dict(start=0, end=20, size=1),
            )
        )
        .update_layout(
            title="Distribution of Contact Frequency in previous campaigns vs now",
            xaxis_title="Number of Times Contacted",
            yaxis_title="Count of Customers",
            barmode="group",
            legend_title_text="Campaign Phase",
            template="plotly_white",
            xaxis=dict(range=[0, 15]),  # Zoom in on where the actual data lives
        )
        .update_traces(opacity=0.75, marker_line_width=1, marker_line_color="black")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - High catch all marketing effort does not really translate to more subscriptions
    """)
    return


@app.cell
def _(d):
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    subs_by_month = (
        d.group_by("month")
        .agg(
            p.len().alias("Total Contacts"),
            (p.col("y").replace_strict({'yes':1 , 'no':0})).sum().alias("Total Subscriptions"),
        )
        .with_columns(
            conversion_rate=(p.col("Total Subscriptions") / p.col("Total Contacts"))
            * 100
        )
        .sort(
            p.col("month").cast(p.String).str.to_lowercase().replace_strict(month_map)
        )
    )
    # 1. Initialize the figure with a secondary y-axis enabled
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 2. Add the Volume (Total Contacts) as a background bar chart
    fig.add_trace(
        go.Bar(
            x=subs_by_month["month"],
            y=subs_by_month["Total Contacts"],
            name="Total Contacts (Volume)",
            marker_color="lightgray",
            opacity=0.6,
        ),
        secondary_y=False,
    )

    # 3. Add the Efficiency (Conversion Rate) as a sharp line over the bars
    fig.add_trace(
        go.Scatter(
            x=subs_by_month["month"],
            y=subs_by_month["conversion_rate"],
            name="Conversion Rate (%)",
            mode="lines+markers",
            line=dict(color="crimson", width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )

    # 4. Manager-Proof the Layout
    fig.update_layout(
        title="Monthly Campaign Volume vs. Conversion Efficiency",
        xaxis_title="Month",
        template="plotly_white",
        hovermode="x unified",  # This makes a single hover box show both numbers simultaneously
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Number of Contacts", secondary_y=False)
    fig.update_yaxes(title_text="Conversion Rate (%)", secondary_y=True, showgrid=False)

    mo.ui.plotly(fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Historical Baseline Conversion: 11.69% (approx. 12 subscriptions per 100 random cold calls).
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    # Customer segmentation

    **Problem Statement: Customer Segmentation and Propensity Optimization**

    **Background:**
    The institution currently executes broad, undifferentiated marketing campaigns across its customer base. This generalized approach results in highly inefficient resource allocation, escalating customer friction due to redundant outreach, and missed conversion opportunities among high-net-worth and high-propensity segments.

    **The Challenge:**
    There is a critical business need to transition from mass-market outreach to precision-targeted interventions. The organization lacks a systematic, data-driven framework to identify distinct customer personas, define the boundaries between low-value and high-value accounts.

    **The Objective:**
    To engineer a robust, scalable machine learning pipeline utilizing K-Means clustering to partition the customer base into distinct, actionable demographic and behavioral cohorts.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data Split
    """)
    return


@app.cell
def _(lf):
    df = lf.pipe(correct_types).pipe(binary_encode)

    X = df.drop("y").collect()

    y = df.select(
        p.col("y").replace_strict({"yes": 1, "no": 0}, return_dtype=p.UInt8)
    ).collect()
    return X, y


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Demographic segmentation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Processing Data
    """)
    return


@app.cell
def _(X_train):
    import sklearn

    sklearn.set_config(transform_output="polars")

    edu_categories = [["unknown", "primary", "secondary", "tertiary"]]

    edu_transformer = Pipeline(
        steps=[
            ("ordinal", OrdinalEncoder(categories=edu_categories)),
            ("scaler", StandardScaler()),
        ]
    )

    age_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    cat_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
    )

    demo_preprocessor = ColumnTransformer(
        transformers=[
            ("age", age_transformer, ["age"]),
            ("edu", edu_transformer, ["education"]),
            ("nom", cat_transformer, ["job", "marital"]),
        ],
        remainder="drop",
    )

    demographic_data_encoded_scaled = demo_preprocessor.fit_transform(X_train)
    return demo_preprocessor, demographic_data_encoded_scaled


@app.cell
def _(X_test, demo_preprocessor):
    demographic_data_encoded_scaled_test = demo_preprocessor.transform(X_test)
    return (demographic_data_encoded_scaled_test,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Finding most apt number of clusters
    """)
    return


@app.function
def plot_inertia_vs_clusters(df):
    inertia_list = []

    for i in range(2, 10):
        kmeans = KMeans(i, random_state=42, n_init=3)
        kmeans.fit(df)
        inertia_list.append(kmeans.inertia_)

    plot_df = p.DataFrame(
        {"n_clusters": list(range(2, 10)), "inertia": inertia_list}
    )

    return px.line(
        plot_df,
        x="n_clusters",
        y="inertia",
        title="Inertia vs n_clusters",
    )


@app.cell
def _(demographic_data_encoded_scaled):
    mo.ui.plotly(plot_inertia_vs_clusters(demographic_data_encoded_scaled))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Training and Evaluation
    """)
    return


@app.cell
def _(demographic_data_encoded_scaled, demographic_data_encoded_scaled_test):
    kmeans = KMeans(n_init=5, random_state=42, n_clusters=5)
    demographic_labels = kmeans.fit_predict(demographic_data_encoded_scaled)
    demo_test_labels = kmeans.predict(demographic_data_encoded_scaled_test)
    return demo_test_labels, demographic_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    -  Choosing kmeans with 5 clusters as the best model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Understanding demographic clusters
    """)
    return


@app.cell
def _(X_train, demographic_labels):
    cluster_summary = (
        X_train.with_columns(demographic_group=demographic_labels)
        .group_by("demographic_group", "marital")
        .agg(
            p.len().alias("count"),
            # Age
            p.col("age").min().alias("min_age"),
            p.col("age").median().alias("median_age"),
            p.col("age").max().alias("max_age"),
            # Education
            p.col("education").mode().first().alias("dominant_education"),
            p.col("education").n_unique().alias("unique_education"),
            # Job
            p.col("job").mode().first().alias("dominant_job"),
            p.col("job").n_unique().alias("unique_job_types"),
        ).sort('demographic_group')
    )
    cluster_summary
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Group 0 ("Mature"):** The oldest cohort with a median age of 53 and maximum ages reaching 95. Predominantly hold secondary education and work in management or administrative roles.
    * **Group 1 ("Mid Corporate"):** Exclusively tertiary education and strictly management roles. Median ages range from 32 to 39.
    * **Group 2 ("Young Tech"):** The youngest bracket. Median ages of 28 and 31, predominantly single, with a hard age cap at 47.
    * **Group 3 ("Mid Blue-Collar"):** Characterized by primary education and blue-collar jobs, with maximum ages stretching into the 90s.
    * **Group 4 ("Young Family"):** Mid-30s median age, predominantly married (7,591 count), with secondary education.
    """)
    return


@app.cell
def _():
    demographic_mapping = {
        0: "Mature",  # Median age ~53, ranges up to 95.
        1: "Mid Corporate",  # Highly educated (tertiary), management roles, median age 30s.
        2: "Young Tech",  # The youngest group (median ~31), overwhelmingly single.
        3: "Mid Blue-Collar",  # Primary education, blue-collar roles, older median age (43-51).
        4: "Young Family",  # Heavily married (7,591 count), median age 36-37, secondary education.
    }
    return (demographic_mapping,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Behavioral Segmentation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Preparing Data
    """)
    return


@app.cell
def _(X_test, X_train):
    poutcome_cats = [["success", "failure"]]

    beh_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), ["balance", "previous", "pdays", "campaign"]),
            (
                "poutcome",
                OneHotEncoder(
                    categories=poutcome_cats,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["poutcome"],
            ),
            ("passthrough", "passthrough", ["housing", "loan"]),
        ],
        remainder="drop",
    )

    behavioral_data_encoded_scaled = beh_preprocessor.fit_transform(X_train)
    behavioral_data_encoded_scaled_test = beh_preprocessor.transform(X_test)
    return behavioral_data_encoded_scaled, behavioral_data_encoded_scaled_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Finding the most apt no of clusters
    """)
    return


@app.cell
def _(behavioral_data_encoded_scaled):
    mo.ui.plotly(plot_inertia_vs_clusters(behavioral_data_encoded_scaled))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Clusters chosen: 4
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Training and Evaluation
    """)
    return


@app.cell
def _(behavioral_data_encoded_scaled, behavioral_data_encoded_scaled_test):
    kmeans_cluster = KMeans(n_init=5, random_state=0, n_clusters=4)
    behavioral_labels = kmeans_cluster.fit_predict(behavioral_data_encoded_scaled)
    beh_test_labels = kmeans_cluster.predict(behavioral_data_encoded_scaled_test)
    return beh_test_labels, behavioral_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Understanding the behavioral clusters
    """)
    return


@app.cell
def _(X_train, behavioral_labels):
    (
        X_train.with_columns(behavioral_group=behavioral_labels)
        .group_by("behavioral_group")
        .agg(
            p.len().alias("count"),
            p.col("balance", "previous", "pdays", "campaign")
            .median()
            .name.suffix("_median"),
            p.col("housing", "loan").mode().first().name.suffix("_mode"),
            (p.col("poutcome") == "success").sum().alias("poutcome_success_count"),
        ).sort('behavioral_group')
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Group 3 ("Over-Contacted"):** The `campaign_median` of 12 stands out immediately. The sales team has hammered this group with calls, resulting in a near-zero success rate.
    * **Group 2 ("High-Net-Worth"):** The `balance_median` of 10,855 isolates the wealthiest individuals. This group also happens to be your true "debt-free" cohort, as both `housing_mode` and `loan_mode` are 0.
    * **Group 1 ("Historical"):** The only group with a history of being called in previous campaigns (`previous_median` of 2 and a `pdays_median` of 248). They also hold the highest absolute number of past successes (755).
    * **Group 0 ("Standard Mortgaged"):** This is the bulk of your dataset (26,454 leads). They have not been contacted previously (`pdays_median` = -1), carry a normal balance, and actively hold a housing mortgage.
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    beh_test_labels,
    behavioral_labels,
    demo_test_labels,
    demographic_labels,
    demographic_mapping,
):

    behavioral_mapping = {
        0: "Standard Mortgaged Leads",  # The baseline pool (26k+ count, normal balance, has housing loan).
        1: "Historical Lead Pool",  # Previously contacted leads (median pdays = 248, previous contacts = 2).
        2: "High-Net-Worth Savers",  # Extremely high liquidity (median balance = 10,855) and debt-free (no loans).
        3: "Over-Contacted Friction Group",  # Heavily spammed this campaign (median campaign calls = 12).
    }

    X_train_with_clusters = X_train.with_columns(
        p.Series("demographic_group", demographic_labels).replace_strict(
            demographic_mapping
        ),
        p.Series("behavioral_group", behavioral_labels).replace_strict(
            behavioral_mapping
        ),
    )

    X_test_with_clusters = X_test.with_columns(
        p.Series("demographic_group", demo_test_labels).replace_strict(
            demographic_mapping
        ),
        p.Series("behavioral_group", beh_test_labels).replace_strict(
            behavioral_mapping
        ),
    )
    return X_test_with_clusters, X_train_with_clusters


@app.cell
def _(X_train_with_clusters, y_train):
    X_train_with_clusters_y = X_train_with_clusters.with_columns(y=p.Series(y_train))
    return (X_train_with_clusters_y,)


@app.cell
def _(X_train_with_clusters_y):
    mo.ui.plotly(
        px.bar(
            X_train_with_clusters_y.group_by("demographic_group")
            .agg(
                p.len().alias("count"), p.col("y").sum(),
                sub_rate = p.col('y').sum() / p.len()
                )
            .sort("count"),
            x="demographic_group",
            y="count",
            color = 'sub_rate',
            title = 'Demographic distribution and subscription rate'
        )
    )
    return


@app.cell
def _(X_train_with_clusters_y):
    mo.ui.plotly(
        px.bar(
            X_train_with_clusters_y.group_by("behavioral_group")
            .agg(p.len().alias("count"), p.col("y").sum(),
                            sub_rate = p.col('y').sum() / p.len())
            .sort("count"),
            x="behavioral_group",
            y="count",
            color = 'sub_rate',
            title = 'Behavioral distribution and subscription rate'
        )
    )
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    # Hypothesis Testing

    **Problem Statement: Statistical Significance of Conversion Rate Differences**

    **Background:**
    The segmentation phase produced persona-level conversion rates that distinguish between groups with high, moderate, and low propensity to convert. Before the marketing team reallocates acquisition budget toward specific personas, channels, or campaign timings, these observed differences must be validated for statistical significance.

    **The Challenge:**
    Conversion rate differences can arise purely from random sampling variability. Acting on insignificant gaps wastes campaign budget and may cause the team to undervalue genuinely high-performing segments.

    **The Objective:**
    Design and execute appropriate hypothesis tests (Z-test / Chi-Square test) to assess whether conversion rate differences across contact channels, personas, or campaign timings are statistically significant.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test 1
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Null Hypothesis ($H_0$):**
    There is no statistically significant difference in subscription conversion rates across different contact channels (e.g., cellular vs. telephone). The communication medium is independent of the conversion outcome, and any observed variance is purely due to random sampling variability.

    **Alternative Hypothesis ($H_a$):**
    There is a statistically significant difference in subscription conversion rates based on the contact channel used. The communication medium directly influences the likelihood of a customer subscribing.

    **The Decision Rule (Chi-Square Test of Independence):**

    * **If $p \ge 0.05$ (Fail to reject $H_0$)**
    * **If $p < 0.05$ (Reject $H_0$):**
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    htest1_data = X_train_with_clusters_y.select("contact", "y")
    return (htest1_data,)


@app.cell
def _(htest1_data):
    contin1 = htest1_data.group_by("contact").agg(
        converted=p.col("y").sum(), not_converted=p.len() - p.col("y").sum()
    )
    contin1
    return (contin1,)


@app.function
def chi2test(contingency_df):

    obs = contingency_df.select(["converted", "not_converted"]).to_numpy()

    chi2, p_value, dof, expected = chi2_contingency(obs)


    if p_value < 0.05:
        return mo.md( f"""
        P-Value: {(p_value)}

        
        Reject null Hypothesis
        
        """)
    else:
                return mo.md( f"""
        P-Value: {(p_value)}

        
        Failed to reject null Hypothesis
        
        """)


@app.cell
def _(contin1):
    chi2test(contin1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Conclusion: Rejection of the Null Hypothesis**

    **Statistical Finding:**
    The hypothesis test yielded a $p$-value below the strict $\alpha = 0.05$ significance threshold. Consequently, we reject the null hypothesis ($H_0$) in favor of the alternative hypothesis ($H_a$).

    **Business Interpretation:**
    There is a statistically significant relationship between the contact channel utilized and the customer's conversion probability. The variance in subscription rates across cellular and telephone mediums is a confirmed, deterministic behavioral pattern, not a byproduct of random sampling variability.

    **Actionable Insight:**
    The communication medium mathematically influences campaign ROI. The marketing and sales departments must cease channel-agnostic outreach immediately and reallocate acquisition budgets and labor hours to prioritize the highest-converting contact channels.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test 2
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Null Hypothesis ($H_0$):**
    There is no statistically significant difference in subscription conversion rates across the five demographic clusters (Mid Corporate, Mid Blue-Collar, Mature, Young Family, Young Tech). A customer's assigned demographic persona is completely independent of their likelihood to subscribe, and any observed variance in conversion rates is due to random sampling variability.

    **Alternative Hypothesis ($H_a$):**
    There is a statistically significant difference in subscription conversion rates based on a customer's demographic cluster. A customer's persona mathematically influences their likelihood to convert.

    **The Decision Rule (Chi-Square Test of Independence, $\alpha = 0.05$):**

    * **If $p \ge 0.05$ (Fail to reject $H_0$):**
    * **If $p < 0.05$ (Reject $H_0$):**
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    htest2_data = X_train_with_clusters_y.select("demographic_group", "y")
    return (htest2_data,)


@app.cell
def _(htest2_data):
    contin2 = htest2_data.group_by("demographic_group").agg(
        converted=p.col("y").sum(), not_converted=p.len() - p.col("y").sum()
    )

    contin2
    return (contin2,)


@app.cell
def _(contin2):
    chi2test(contin2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Conclusion: Rejection of the Null Hypothesis**

    **Statistical Finding:**
    The hypothesis test yielded a $p$-value strictly below the $\alpha = 0.05$ threshold. We definitively reject the null hypothesis ($H_0$) in favor of the alternative hypothesis ($H_a$).

    **Business Interpretation:**
    There is a confirmed, statistically significant relationship between a customer's assigned demographic persona and their likelihood to subscribe. The variance in conversion rates across groups like "Mid Corporate" and "Mature" is mathematically real, proving that these segments possess fundamentally distinct purchasing behaviors rather than random fluctuations.

    **Actionable Insight:**
    The marketing department is now mathematically obligated to abandon generalized mass-market campaigns; they must tailor their ad spend, messaging, and outreach strategies to target the specific, high-propensity personas you just validated.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test 3
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Null Hypothesis (H0):**
    There is no statistically significant difference in subscription conversion rates across the four behavioral clusters (Historical Lead Pool, Debt-Free Unmortgaged Savers, Over-Contacted Friction Group, High-Net-Worth Savers). A customer's historical engagement and financial baggage are completely independent of their likelihood to subscribe.

    **Alternative Hypothesis (Ha):**
    There is a statistically significant difference in subscription conversion rates based on a customer's behavioral cluster. A customer's past interaction history and account balances mathematically dictate their future propensity to convert.

    **The Decision Rule (Chi-Square Test of Independence, α = 0.05):**

    * **If p ≥ 0.05 (Fail to reject H0**
    * **If p < 0.05 (Reject H0):**
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    htest3_data = X_train_with_clusters_y.select("behavioral_group", "y")
    return (htest3_data,)


@app.cell
def _(htest3_data):
    contin3 = htest3_data.group_by("behavioral_group").agg(
        converted=p.col("y").sum(), not_converted=p.len() - p.col("y").sum()
    )

    contin3
    return (contin3,)


@app.cell
def _(contin3):
    chi2test(contin3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Conclusion: Rejection of the Null Hypothesis**

    **Statistical Finding:**
    The hypothesis test yielded a $p$-value safely below the $\alpha = 0.05$ threshold. We definitively reject the null hypothesis ($H_0$) in favor of the alternative hypothesis ($H_a$).

    **Business Interpretation:**
    There is a mathematically undeniable relationship between a customer's historical engagement profile and their current likelihood to subscribe.

    **Actionable Insight:**
    The marketing team must immediately pivot their acquisition budget away from the friction group and toward the high-propensity whales just identified.
    """)
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    # Propensity Modelling

    **Problem Statement: Subscription Propensity Classification**

    **Background:**
    Following the successful segmentation of the customer base, the institution now possesses mathematically rigorous profiles of customer behavior and demographics. However, while unsupervised clustering effectively identifies *who* the customers are and *how* they interact with the bank, it does not deterministically predict *what* they will do next.

    **The Challenge:**
    Historical marketing campaigns have suffered from inefficient conversion rates due to a generalized outreach strategy. Without a probabilistic model to score individual customer intent, the sales team expends critical labor hours engaging low-likelihood prospects—such as the massive baseline of unengaged accounts—while potentially missing latent conversion opportunities in higher-value segments. The business lacks a dynamic, predictive mechanism to rank new and existing customers based on their exact likelihood to purchase a subscription.

    **The Objective:**
    To engineer and deploy a robust supervised classification model that predicts the binary probability of a customer subscribing to a financial product. This propensity pipeline will integrate raw transactional data with the categorical labels. The ultimate goal is to output a strictly ranked, high-precision lead queue that minimizes Customer Acquisition Cost (CAC), eliminates outreach to high-friction/low-intent targets, and directly maximizes campaign ROI.
    """)
    return


@app.cell
def _(X_train_with_clusters):
    X_train_with_clusters
    return


@app.cell
def _(X_test_with_clusters, X_train_with_clusters):
    def engineer_features(df):
        return df.with_columns(
            p.col("pdays").replace({-1: 0}),
            is_in_debt=p.col("balance") <= 0,
            never_contacted_before=p.col("pdays") == -1,
            customer_segment=p.concat_str(
                [p.col("demographic_group"), p.lit(" "), p.col("behavioral_group")]
            ),
            new_ratio=p.col("campaign") / (p.col("previous") + 1),
            wealth_index=p.col("balance") / p.col("age"),
            financially_trapped=(p.col("housing") == 1) & (p.col("loan") == 1),
        ).drop(
            "behavioral_group",
            "demographic_group",
            "day",
            "duration",
        )

    X_train_with_features = engineer_features(X_train_with_clusters)
    X_test_with_features = engineer_features(X_test_with_clusters)
    return X_test_with_features, X_train_with_features


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Constructing pipeline
    """)
    return


@app.cell
def _():
    nominal_cols = [
        "job",
        "marital",
        "contact",
        "poutcome",
        "month",
        "customer_segment",
    ]
    ordinal_cols = ["education"]

    # 1. Fixed syntax: Added the missing comma between the lists
    # and renamed it to match both columns accurately
    ordinal_categories = [
        ["unknown", "primary", "secondary", "tertiary"],  # Categories for 'education'
        # Categories for 'balance_tier' (make sure these match your tiers!)
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal_cols,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
        ],
        remainder="passthrough",
    )  # <-- Added the missing closing parenthesis and bracket here!
    return (preprocessor,)


@app.cell
def _(preprocessor):

    # 1. Define your base XGBoost model
    xgb_model = XGBClassifier(
        random_state=42,
        scale_pos_weight=7.548118737001324,
        n_jobs=-1,
    )

    # 2. Define the hyperparameters you want to search and tune
    param_grid = {
        "max_depth": [15],
        "learning_rate": [0.01],
        "n_estimators": [155],
        "subsample": [0.22],
    }

    # 3. Set up GridSearchCV with n_jobs=-1 to use all CPU cores
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="f1",  # Optimizes directly for your F1-score (or use 'average_precision' for PR-AUC)
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Uses all available CPU cores simultaneously
        verbose=1,  # Prints out progress updates during fitting
    )

    # 4. Plug GridSearchCV directly into your Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", grid_search),
        ]
    )
    return (pipeline,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(pipeline):
    pipeline
    return


@app.cell
def _(X_train_with_features, pipeline, y_train):
    pipeline.fit(X_train_with_features, y_train)
    return


@app.cell
def _(pipeline):
    best_params = pipeline.named_steps["classifier"].best_params_
    best_params
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell
def _(X_test_with_features, pipeline):
    predictions = pipeline.predict(X_test_with_features)
    return (predictions,)


@app.cell
def _(predictions, y_test):
    from sklearn.metrics import recall_score, precision_score, f1_score

    mo.md(f'''
    Precision: {precision_score(y_test, predictions)} (4x than baseline)

    Recall: {recall_score(y_test, predictions)} 


    F1: {f1_score(y_test, predictions)}
    ''')

    return


@app.cell
def _(pipeline):
    # 1. Extract the best XGBoost model from the GridSearchCV object
    best_model = pipeline.named_steps["classifier"].best_estimator_

    # 2. Get feature importances from the best estimator
    importances = best_model.feature_importances_
    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()

    # 3. Create the Polars DataFrame and sort descending
    feature_importance_df = p.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort("Importance", descending=True)
    return best_model, feature_importance_df, feature_names


@app.cell
def _(feature_importance_df):
    mo.ui.plotly(
        px.bar(
            feature_importance_df[:10],
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Drivers of Customer Subscription",
            color="Importance",
        ).update_layout(coloraxis_showscale=False)
    )
    return


@app.cell
def _(X_test_with_features, pipeline, y_test):
    y_scores = pipeline.predict_proba(X_test_with_features)[:, 1]
    RocCurveDisplay.from_predictions(
        y_test,
        y_scores,
        name="Bank Default Classifier",
    )

    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    mo.ui.matplotlib(plt.gca())
    return


@app.cell
def _(X_test_with_features, best_model, feature_names, preprocessor):
    X_test_transformed = preprocessor.transform(X_test_with_features)
    if hasattr(X_test_transformed, "to_pandas"):
        X_test_df = X_test_transformed.to_pandas()
    else:
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test_df)

    shap.summary_plot(shap_values, X_test_df, show=False)
    mo.ui.matplotlib(plt.gca())

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Positive impact features:
     - Mature Age
     - Previous outcome success
     - Contacts made in month of June
    ### Negative impact features:
     - Unknown Contact
     - High number of contacts
     - Loan of any kind
     - Low Balance
     - No cellular
     - Blue Collar Job
    """)
    return


@app.cell
def _(X_test_with_features, pipeline, y_test):
    subscription_probs = pipeline.predict_proba(X_test_with_features)[:, 1]

    top_100_leads = (
        X_test_with_features.with_columns(
            propensity_score=p.Series(subscription_probs), subscribed=p.Series(y_test)
        )
        .sort("propensity_score", descending=True)
        .head(100)  # Give the sales reps exactly 100 phone numbers
    )
    return (top_100_leads,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hit list
     - People to contact first and foremost, High chance of conversion (78%)
    """)
    return


@app.cell
def _(top_100_leads):
    top_100_leads
    return


@app.cell
def _(y):
    baseline = y.sum() / len(y) * 100
    baseline.item()
    return


@app.cell
def _(top_100_leads):
    top_100_leads["subscribed"].sum()
    return


@app.cell
def _(X_test_with_features, pipeline, y_test):
    # 1. Generate probabilities ONLY on the unseen test set
    probs = pipeline.predict_proba(X_test_with_features)[:, 1]

    # 2. Stitch those probabilities directly to the test features and the TEST targets
    X_test_with_features_y = X_test_with_features.with_columns(
        subscribed=p.Series(y_test),  # MUST be the exact y_test array from your split
        propensity_score=p.Series(probs),
    )

    # 3. Calculate the deciles
    lift_df = (
        X_test_with_features_y.sort("propensity_score", descending=True)
        .with_row_index("rank")
        .with_columns(decile=(p.col("rank") / (p.len() / 10)).cast(p.Int32) + 1)
        .group_by("decile")
        .agg(actual_conversion_rate=p.col("subscribed").mean() * 100)
        .sort("decile")
    )

    # 4. Plot the reality
    fig2 = px.bar(
        lift_df,
        x="decile",
        y="actual_conversion_rate",
        title="Test Set Conversion Rate by Decile (1 = Best Leads, 10 = Worst)",
        labels={
            "decile": "Customer Decile",
            "actual_conversion_rate": "Actual Conversion Rate (%)",
        },
    )
    fig2.add_hline(
        y=11.7,
        line_dash="dash",
        line_color="red",
        annotation_text="Bank Baseline (11.7%)",
    )

    mo.ui.plotly(fig2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Propensity Modeling: Business Impact and Lead Optimization

    ### Executive Summary
    The objective of this modeling phase was to transition the marketing strategy from a traditional mass-outreach approach to a highly targeted, propensity-based campaign. By predicting the likelihood of customer subscription prior to any sales contact, the pipeline significantly optimizes resource allocation and increases overall conversion efficiency.

    ### Methodology and Data Integrity
    A critical component of this pipeline involved ensuring strict production viability. Historical models on this dataset often suffer from data leakage by incorporating the `duration` of a call—a metric entirely unknown prior to the marketing action. This feature was explicitly removed to ensure the model evaluates leads solely on pre-existing demographic, behavioral, and financial profiles.

    To counteract the dataset's inherent class imbalance, the XGBoost classification engine utilized dynamic class-weight balancing (`scale_pos_weight`). This forced the algorithm to mathematically prioritize the minority class of actual subscribers during training.

    ### Key Achievements and ROI
    Rather than utilizing a rigid binary classification threshold (which yielded a 0.43 precision score on the imbalanced validation set), the final deployment leverages probability scoring (`predict_proba`) to rank the customer base from highest to lowest propensity to convert.

    - **Historical Baseline Conversion:** 11.69% (approx. 12 subscriptions per 100 random cold calls).
    - **Optimized Model Conversion (Top 100 Leads):** 78.00% (78 subscriptions per 100 targeted calls).
    - **Business Lift:** By concentrating sales efforts strictly on the model's top-ranked leads, the acquisition efficiency improved by approximately **6.6x** compared to the historical baseline.
    """)
    return


@app.cell
def _():
    comparison_df = p.DataFrame(
        {
            "Strategy": ["Historical Spray & Pray", "Top 100 AI Hit-List"],
            "Conversion Rate (%)": [11.7, 78.0],
        }
    )

    mo.ui.plotly(
        px.bar(
            comparison_df,
            x="Strategy",
            y="Conversion Rate (%)",
            color="Strategy",
            text="Conversion Rate (%)",  # Puts the number directly on the bar so they can't miss it
            color_discrete_sequence=[
                "#d3d3d3",
                "#2ca02c",
            ],  # Gray for the old way, green for money
            title="Sales Efficiency: Historical Cold Calling vs. Propensity Targeting",
        )
        .update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        .update_layout(showlegend=False)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conclusion
    This predictive pipeline successfully minimizes customer acquisition costs and reduces time wasted on low-probability prospects. By systematically ranking the dataset, the marketing and sales teams can prioritize high-value outreach, drastically improving the return on investment for future telephone campaigns.
    """)
    return


@app.cell
def _(pipeline):
    joblib.dump(pipeline, "model/classification_pipeline.pkl")
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    ### I. Targeted Marketing Directives

    To improve campaign efficiency and conversion rates, future marketing efforts will be guided by the newly developed customer segmentation and propensity models. Please align campaign workflows with the following data-driven parameters:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Exclude the Over-Contacted Segment:** Limit outreach to the "Over-Contacted Friction Group." Analysis indicates that excessive contact (e.g., exceeding a median of 12 touchpoints) yields diminishing returns and negatively impacts the customer experience.
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    mo.ui.plotly(
        px.line(
            (
        X_train_with_clusters_y.filter(p.col("campaign") < 10)
        .group_by("campaign")
        .agg((p.col("y").sum() / p.len()).alias("sub_rate"))
        .sort("campaign")
    ),
            x="campaign",
            y="sub_rate",
            markers=True,
            labels={"campaign": "Number of contacts", "y": "Subscription Rate"},
            title="Number of contacts VS Subscription Rate",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Prioritize High-Value Personas:** Reallocate resources toward targeted, personalized outreach for our highest-converting segments, specifically the "High-Net-Worth Savers" and the "Historical Lead Pool." These should be cross-referenced with the "Mid Corporate" and "Mature" demographic profiles, which have demonstrated a statistically significant higher likelihood to subscribe.
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    mo.ui.plotly(
        px.bar(
            (
        X_train_with_clusters_y.group_by("behavioral_group")
        .agg(
            p.col("campaign").mean().alias("contacts_mean"),
            (p.col("y").sum() / p.len()).alias("conversion_rate"),
        )
        .sort("contacts_mean", descending=True)
    ),
            x="behavioral_group",
            y="contacts_mean",
            color="conversion_rate",
            title="Average contacts by customer segment",
            labels={
                "behavioral_group": "Customer segment",
                "contacts_mean": "Average no of contacts",
            },
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Focus on Validated Channels:** Direct communication efforts through channels that have been validated by our statistical hypothesis testing. Budgets should be reallocated away from mediums that did not demonstrate a significant impact on historical conversion rates.
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    mo.ui.plotly(
        px.bar(
            X_train_with_clusters_y.group_by('contact').agg( sub_rate = (p.col('y').sum() / p.len()) ),
            x = 'contact',
            y = 'sub_rate',
            title = 'Subscription rate by channel',
            labels = {'contact':'Contact Type' , 'sub_rate':'Subscription Rate'}
        )
    )


    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### II. Realized Business Benefits

    The implementation of this end-to-end segmentation and predictive propensity pipeline provides the following strategic advantages to the organization:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Increased Campaign ROI:** By focusing the marketing budget on statistically significant, high-propensity customer profiles, the project ensures that capital is deployed where it is most likely to generate a return, ultimately maximizing the revenue impact of future campaigns.
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    mo.vstack([
        mo.md('Most valuable segments'),
        X_train_with_clusters_y.group_by("demographic_group", "behavioral_group").agg(
        p.len().alias("count"),
        p.col("y").sum().alias("total_subs"),
    ).with_columns(sub_percentage=p.col("total_subs") / p.col("count") * 100).sort(
        "sub_percentage", descending=True
    )[:7]
    
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Reduced Customer Acquisition Cost (CAC):** Transitioning from mass-market outreach to targeted, probability-based marketing reduces unnecessary campaign expenditures, lowering the overall cost required to acquire a new subscription.
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    mo.ui.plotly(
        px.bar(
            (
        X_train_with_clusters_y.group_by("demographic_group", "behavioral_group")
        .agg(
            p.len().alias("count"),
            p.col("y").sum().alias("total_subs"),
        )
        .with_columns(
            conversion_rate=p.col("total_subs") / p.col("count") * 100,
            group_name=p.concat_str(
                [p.col("demographic_group"), p.lit(" "), p.col("behavioral_group")]
            ),
        )
        .sort("total_subs", descending=True)
    ),
            x="group_name",
            y="total_subs",
            color="conversion_rate",
            labels={
                "group_name": "Customer Segment",
                "total_subs": "Total Subscriptions",
            },
            title="Total Subscriptions by customer segment",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    * **Improved Resource Allocation:** By identifying segments with low conversion probability, sales and marketing teams can redirect their time and effort toward high-potential leads, significantly increasing operational efficiency.
    """)
    return


@app.cell
def _(X_train_with_clusters_y):
    mo.vstack([
        mo.md('Least valuable rate groups'),
        X_train_with_clusters_y.group_by("demographic_group", "behavioral_group").agg(
        p.len().alias("count"),
        p.col("y").sum().alias("total_subs"),
    ).with_columns(
        conversion_rate=p.col("total_subs") / p.col("count") * 100,
    ).sort("conversion_rate", descending=False)[:7]
    
    ])


    return


@app.cell
def _(X_train_with_clusters_y):
    suppression_list = X_train_with_clusters_y.filter(
        p.col("balance") < 0, p.col("housing") == 1, p.col("loan") == 1, p.col('contact') == 'unknown'
    ).sort("balance")
    return (suppression_list,)


@app.cell
def _(suppression_list):
    mo.vstack([
        mo.md('Suppression List (Lowest probability of a  subscription)'),
        suppression_list
    
    ])


    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Thank you
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
