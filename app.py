# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------- Theme Selector ----------
theme_choice = st.sidebar.radio("🎨 Select Theme", ["Ocean Blue", "Steel Grey"])

# ---------- Ocean Blue Theme ----------
ocean_blue = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #bbdefb, #90caf9);
    color: #0d47a1; font-weight: 500;
}
h1,h2,h3,h4,h5,h6 { color:#0d47a1; font-weight:700; }
.stButton>button {
    background: linear-gradient(90deg, #1565c0, #1e88e5); color:white;
    border-radius:8px; border:none; padding:8px 14px; font-weight:600;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1e88e5, #42a5f5); transform: scale(1.03);
}
</style>
"""

# ---------- Steel Grey Theme ----------
steel_grey = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0e0e0, #bdbdbd);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #9e9e9e, #757575);
    color: white; font-weight: 500;
}
h1,h2,h3,h4,h5,h6 { color:#212121; font-weight:700; }
.stButton>button {
    background: linear-gradient(90deg, #424242, #616161); color:white;
    border-radius:8px; border:none; padding:8px 14px; font-weight:600;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #616161, #9e9e9e); transform: scale(1.03);
}
/* Specific Style for "Created by" Text */
.created-by-text {
    color: black;
}
</style>
"""

# ---------- Apply Selected Theme ----------
if theme_choice == "Ocean Blue":
    st.markdown(ocean_blue, unsafe_allow_html=True)
else:
    st.markdown(steel_grey, unsafe_allow_html=True)

# ---------- File Paths ----------
DATA_DIR = "Data"
USERS_FILE = os.path.join(DATA_DIR, "users_db.xlsx")
VENDORS_FILE = os.path.join(DATA_DIR, "vendors_db.xlsx")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products_db.xlsx")
COMPLAINTS_FILE = os.path.join(DATA_DIR, "complaints_db.xlsx")
REVIEWS_FILE = os.path.join(DATA_DIR, "rexiews_db.xlsx")
# ---------- Helper Functions ----------

def load_data(file_path):
    return pd.read_excel(file_path)

def save_data(df, file_path):
    df.to_excel(file_path, index=False)

def generate_new_id(df, column_name, prefix):
    if df.empty:
        return f"{prefix}001"
    else:
        nums = df[column_name].str.replace(prefix, "").astype(int)
        new_num = nums.max() + 1
        return f"{prefix}{new_num:03d}"

# Load all data
users_df = load_data(USERS_FILE)
vendors_df = load_data(VENDORS_FILE)
products_df = load_data(PRODUCTS_FILE)
complaints_df = load_data(COMPLAINTS_FILE)
rexiews_df = load_data(REVIEWS_FILE)

# ---------- Pages ----------



def page_home():
    st.title("🌟 Customer Feedback & Analytics Hub")
    st.markdown("""
    <h3 style='text-align:center; color:#16A085; font-family:Arial;'>
    Submit complaints • Track status • Vendor insights • Analytics
    </h3>
    """, unsafe_allow_html=True)

    # ----------------- About Section -----------------
    st.markdown(
        """
        <div style="
            background-color:#f8f9fc;
            padding:20px;
            border-radius:10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            text-align:center;
            font-family: 'Trebuchet MS', sans-serif;
        ">
            <h2 style='color:#4e73df;'>About This Website</h2>
            <p style='font-size:16px; color:#5a5c69;'>
                Welcome to our Customer Feedback Platform! This platform allows Indian customers to:
            </p>
            <ul style='text-align:left; display:inline-block; color:#5a5c69;'>
                <li>Submit complaints about products and services</li>
                <li>Submit reviews and ratings</li>
                <li>Track complaint status in real-time</li>
                <li>View vendor performance and analytics</li>
                <li>Get insights through KPIs and charts</li>
            </ul>
            <p style='font-style:italic; color:#858796;'>Empowering customers and vendors through transparency and data-driven insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------- Compute KPIs -----------------
    total_complaints = len(complaints_df)
    resolved_complaints = complaints_df[complaints_df['complaint_status'].str.lower() == 'resolved'].shape[0]
    pending_complaints = complaints_df[complaints_df['complaint_status'].str.lower() == 'pending'].shape[0]
    avg_rating = rexiews_df['rating'].mean() if not rexiews_df.empty else 0
    total_users = len(users_df)
    total_vendors = len(vendors_df)

    # Top 5 most insightful KPIs
    kpi_data = [
        ("Total Complaints", total_complaints, "#f6c23e"),
        ("Resolved Complaints", resolved_complaints, "#1cc88a"),
        ("Pending Complaints", pending_complaints, "#e74a3b"),
        ("Average Rating", round(avg_rating, 2), "#36b9cc"),
        ("Total Users", total_users, "#858796"),
    ]
    # ----------------- Quick Stats (smaller KPI cards, equal size) -----------------
    st.markdown("### Quick Stats")

    # Create equal-width columns
    cols = st.columns([1] * len(kpi_data))

    for col, (title, value, color) in zip(cols, kpi_data):
        col.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:6px;
                border-radius:6px;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.15);
                text-align:center;
                height:80px;
                display:flex;
                flex-direction:column;
                justify-content:center;
            ">
                <h6 style='color:white; margin:2px 0; font-size:13px;'>{title}</h6>
                <p style='color:white; margin:0; font-size:18px; font-weight:bold;'>{value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------- Recent Complaints -----------------
    st.markdown("### 📝 Recent Complaints")
    if not complaints_df.empty:
        recent_complaints = complaints_df.sort_values("complaint_date", ascending=False).head(5)
        # Replace IDs with names
        recent_complaints_display = recent_complaints.copy()
        recent_complaints_display["user_id"] = recent_complaints_display["user_id"].map(
            dict(zip(users_df["user_id"], users_df["name"]))
        )
        recent_complaints_display["product_id"] = recent_complaints_display["product_id"].map(
            dict(zip(products_df["product_id"], products_df["product_name"]))
        )
        recent_complaints_display["vendor_id"] = recent_complaints_display["vendor_id"].map(
            dict(zip(vendors_df["vendor_id"], vendors_df["vendor_name"]))
        )
        st.dataframe(recent_complaints_display[["complaint_id","user_id","product_id","vendor_id","complaint_status"]])
    else:
        st.info("No complaints yet.")

# ---------- Submit Complaint ----------
def page_submit_complaint():
    st.subheader("📝 Submit a Complaint")

    # Option to select existing user or enter new one
    user_choice = st.radio("Select User Option", ["Existing User", "New User"])
    if user_choice == "Existing User":
        user_name = st.selectbox("Select User", users_df['name'])
        user_id = users_df[users_df['name'] == user_name]['user_id'].values[0]
    else:
        user_name = st.text_input("Enter Your Name")
        if user_name.strip() != "":
            # Generate new user_id
            new_user_id = generate_new_id(users_df, "user_id", "U")
            users_df.loc[len(users_df)] = {
                "user_id": new_user_id,
                "name": user_name,
                "state": "Unknown"  # default
            }
            save_data(users_df, USERS_FILE)
            user_id = new_user_id
        else:
            st.warning("Please enter a valid name")
            return

    # Select product by name
    product_name = st.selectbox("Select Product", products_df['product_name'])
    product_row = products_df[products_df['product_name'] == product_name].iloc[0]
    product_id = product_row['product_id']
    vendor_id = product_row['vendor_id']
    fssai_code = product_row.get('fssai_code', 'N/A')   # <-- fetch FSSAI Code
    is_verified = product_row.get('is_verified', False)  # <-- fetch verification status

    # Display vendor + FSSAI info + product verification
    vendor_name = vendors_df[vendors_df['vendor_id'] == vendor_id]['vendor_name'].values[0]
    st.text(f"Vendor: {vendor_name}")
    st.text(f"FSSAI Code: {fssai_code}")   # <-- show FSSAI code
    st.text(f"Product Verified: {'✅ Yes' if is_verified else '❌ No'}")  # <-- show verification

    complaint_text = st.text_area("Your Complaint")
    complaint_priority = st.selectbox("Priority", ["Low", "Medium", "High"])
    complaint_status = "Pending"

    if st.button("Submit Complaint"):
        new_id = generate_new_id(complaints_df, "complaint_id", "C")
        new_row = {
            "complaint_id": new_id,
            "user_id": user_id,
            "product_id": product_id,
            "vendor_id": vendor_id,
            "fssai_code": fssai_code,
            "is_verified": is_verified,   # <-- save verification with complaint
            "complaint_text": complaint_text,
            "complaint_status": complaint_status,
            "complaint_priority": complaint_priority,
            "complaint_date": datetime.now(),
            "complaint_image_url": ""
        }
        complaints_df.loc[len(complaints_df)] = new_row
        save_data(complaints_df, COMPLAINTS_FILE)
        st.success("Complaint submitted successfully!")

# ---------- Submit Review ----------
def page_submit_review():
    st.subheader("📝 Submit a Review")

    # Option to select existing user or enter new one
    user_choice = st.radio("Select User Option", ["Existing User", "New User"])
    if user_choice == "Existing User":
        user_name = st.selectbox("Select User", users_df['name'])
        user_id = users_df[users_df['name'] == user_name]['user_id'].values[0]
    else:
        user_name = st.text_input("Enter Your Name")
        if user_name.strip() != "":
            # Generate new user_id
            new_user_id = generate_new_id(users_df, "user_id", "U")
            users_df.loc[len(users_df)] = {
                "user_id": new_user_id,
                "name": user_name,
                "state": "Unknown"  # default, can be extended later
            }
            save_data(users_df, USERS_FILE)
            user_id = new_user_id
        else:
            st.warning("Please enter a valid name")
            return

    # Select product by name
    product_name = st.selectbox("Select Product", products_df['product_name'])
    product_row = products_df[products_df['product_name'] == product_name].iloc[0]
    product_id = product_row['product_id']
    vendor_id = product_row['vendor_id']
    fssai_code = product_row.get('fssai_code', 'N/A')      # <-- fetch FSSAI Code
    is_verified = product_row.get('is_verified', False)    # <-- fetch verification status

    # Display vendor info + FSSAI code + verification
    vendor_name = vendors_df[vendors_df['vendor_id'] == vendor_id]['vendor_name'].values[0]
    st.text(f"Vendor: {vendor_name}")
    st.text(f"FSSAI Code: {fssai_code}")
    st.text(f"Product Verified: {'✅ Yes' if is_verified else '❌ No'}")

    rating = st.slider("Rating (1-5)", 1, 5, 5)
    review_text = st.text_area("Your Review")
    review_sentiment = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])

    if st.button("Submit Review"):
        new_id = generate_new_id(rexiews_df, "review_id", "R")
        new_row = {
            "review_id": new_id,
            "user_id": user_id,
            "product_id": product_id,
            "vendor_id": vendor_id,
            "fssai_code": fssai_code,       # <-- save FSSAI code
            "is_verified": is_verified,     # <-- save verification status
            "rating": rating,
            "review_text": review_text,
            "review_date": datetime.now(),
            "review_sentiment": review_sentiment
        }
        rexiews_df.loc[len(rexiews_df)] = new_row
        save_data(rexiews_df, REVIEWS_FILE)
        st.success("Review submitted successfully!")

# ---------- Track Complaints ----------
def page_track_complaints():
    st.subheader("📊 Track Complaints")

    # Display complaints with user, product, and vendor names instead of IDs
    display_df = complaints_df.copy()
    display_df['user_name'] = display_df['user_id'].map(dict(zip(users_df['user_id'], users_df['name'])))
    display_df['product_name'] = display_df['product_id'].map(
        dict(zip(products_df['product_id'], products_df['product_name'])))
    display_df['vendor_name'] = display_df['vendor_id'].map(
        dict(zip(vendors_df['vendor_id'], vendors_df['vendor_name'])))

    display_df = display_df[['complaint_id', 'user_name', 'product_name', 'vendor_name',
                             'complaint_text', 'complaint_status', 'complaint_priority', 'complaint_date']]

    # Show the complaints table
    st.dataframe(display_df)

    # Select a complaint to update
    selected_id = st.selectbox("Select Complaint to Update Status", display_df['complaint_id'])
    current_status = complaints_df.loc[complaints_df['complaint_id'] == selected_id, 'complaint_status'].values[0]

    new_status = st.selectbox("Update Status", ["Pending", "Resolved"], index=0 if current_status == "Pending" else 1)

    if st.button("Update Status"):
        complaints_df.loc[complaints_df['complaint_id'] == selected_id, 'complaint_status'] = new_status
        save_data(complaints_df, COMPLAINTS_FILE)
        st.success(f"Complaint {selected_id} status updated to {new_status}!")


# ---------- Vendor Dashboard ----------
def page_vendor_dashboard():
    st.subheader("🏭 Vendor Dashboard")

    # -------- Existing Vendor Insights --------
    vendor_name = st.selectbox("Select Vendor", vendors_df['vendor_name'])
    vendor_row = vendors_df[vendors_df['vendor_name'] == vendor_name].iloc[0]
    vendor_id = vendor_row['vendor_id']

    vendor_complaints = complaints_df[complaints_df['vendor_id'] == vendor_id]
    vendor_reviews = rexiews_df[rexiews_df['vendor_id'] == vendor_id]

    total_complaints = vendor_complaints.shape[0]
    resolved_complaints = vendor_complaints[vendor_complaints['complaint_status'] == 'Resolved'].shape[0]
    pending_complaints = total_complaints - resolved_complaints
    avg_rating = round(vendor_reviews['rating'].dropna().mean(), 2) if not vendor_reviews.empty else 0

    # -------- KPI Cards --------
    kpi_html = f"""
    <div style="display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;">
        <div style="flex:1; min-width:150px; background:#3498db; color:white; padding:10px; border-radius:10px; text-align:center;">
            <h5>Total Complaints</h5><h2>{total_complaints}</h2>
        </div>
        <div style="flex:1; min-width:150px; background:#2ecc71; color:white; padding:10px; border-radius:10px; text-align:center;">
            <h5>Resolved</h5><h2>{resolved_complaints}</h2>
        </div>
        <div style="flex:1; min-width:150px; background:#e67e22; color:white; padding:10px; border-radius:10px; text-align:center;">
            <h5>Pending</h5><h2>{pending_complaints}</h2>
        </div>
        <div style="flex:1; min-width:150px; background:#9b59b6; color:white; padding:10px; border-radius:10px; text-align:center;">
            <h5>Avg. Rating</h5><h2>{avg_rating}</h2>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown("---")

    # -------- Complaints & Reviews --------
    st.subheader("Complaints")
    if not vendor_complaints.empty:
        st.dataframe(vendor_complaints[['complaint_id', 'user_id', 'product_id', 'complaint_text',
                                        'complaint_status', 'complaint_priority', 'complaint_date']])
    else:
        st.write("No complaints for this vendor.")

    st.subheader("Reviews")
    if not vendor_reviews.empty:
        st.dataframe(vendor_reviews[['review_id', 'user_id', 'product_id', 'rating',
                                     'review_text', 'review_sentiment', 'review_date']])
    else:
        st.write("No reviews for this vendor.")

    st.markdown("---")

    # -------- Add New Vendor --------
    st.subheader("➕ Add New Vendor")
    with st.form("add_vendor_form", clear_on_submit=True):
        new_vendor_name = st.text_input("Vendor Name")
        new_vendor_state = st.text_input("State")
        submit_vendor = st.form_submit_button("Add Vendor")

    if submit_vendor:
        # ✅ Validation checks
        if not new_vendor_name.strip():
            st.error("⚠️ Vendor Name is required.")
        elif not new_vendor_state.strip():
            st.error("⚠️ State is required.")
        else:
            # All validations passed
            new_vendor_id = generate_new_id(vendors_df, "vendor_id", "V")
            vendors_df.loc[len(vendors_df)] = {
                "vendor_id": new_vendor_id,
                "vendor_name": new_vendor_name.strip(),
                "state": new_vendor_state.strip()
            }
            save_data(vendors_df, VENDORS_FILE)
            st.success(f"✅ Vendor '{new_vendor_name}' added successfully!")

    # -------- Add New Product --------
    st.subheader("➕ Add New Product")
    with st.form("add_product_form", clear_on_submit=True):
        new_product_name = st.text_input("Product Name")
        product_vendor = st.selectbox("Select Vendor for Product", vendors_df['vendor_name'])
        product_vendor_id = vendors_df[vendors_df['vendor_name'] == product_vendor]['vendor_id'].values[0]
        new_product_fssai = st.text_input("Product FSSAI Code (10 digits)")
        is_verified = st.checkbox("Verified Product?", value=False)
        submit_product = st.form_submit_button("Add Product")

    if submit_product:
        # ✅ Validation checks
        if not new_product_name.strip():
            st.error("⚠️ Product Name is required.")
        elif not new_product_fssai.strip().isdigit() or len(new_product_fssai.strip()) != 10:
            st.error("⚠️ Product FSSAI Code must be exactly 10 digits.")
        else:
            # All validations passed
            new_product_id = generate_new_id(products_df, "product_id", "P")
            products_df.loc[len(products_df)] = {
                "product_id": new_product_id,
                "product_name": new_product_name.strip(),
                "vendor_id": product_vendor_id,
                "fssai_code": new_product_fssai.strip(),
                "is_verified": is_verified
            }
            save_data(products_df, PRODUCTS_FILE)
            st.success(f"✅ Product '{new_product_name}' added successfully!")


# ---------- Analytics Page ----------
import plotly.express as px
# ---------- Analytics Page ----------
def page_analytics():
    st.subheader("📈 Analytics Dashboard")

    # --- Dynamic KPI values ---
    total_complaints = len(complaints_df)
    resolved_complaints = complaints_df[complaints_df['complaint_status'].str.lower() == 'resolved'].shape[0]
    pending_complaints = complaints_df[complaints_df['complaint_status'].str.lower() == 'pending'].shape[0]
    avg_rating = round(rexiews_df['rating'].mean(), 2) if not rexiews_df.empty else 0
    total_users = len(users_df)

    # --- Prepare KPI data ---
    kpis = [
        {"label": "Total Complaints", "value": total_complaints, "color": "#FF6B6B"},
        {"label": "Resolved Complaints", "value": resolved_complaints, "color": "#1DD1A1"},
        {"label": "Pending Complaints", "value": pending_complaints, "color": "#FDCB6E"},
        {"label": "Average Rating", "value": avg_rating, "color": "#5DADE2"},
        {"label": "Total Users", "value": total_users, "color": "#A29BFE"}
    ]

    # --- Display KPI cards ---
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        col.markdown(f"""
            <div style="
                background-color:{kpi['color']};
                padding:20px;
                border-radius:12px;
                text-align:center;
                min-height:120px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                margin:5px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
            ">
                <h6 style='margin:0; color:white; font-size:14px;'>{kpi['label']}</h6>
                <h3 style='margin:0; color:white; font-size:28px;'>{kpi['value']}</h3>
            </div>
        """, unsafe_allow_html=True)

    # --- Top 5 products by rating ---
    if not rexiews_df.empty:
        top_rated = rexiews_df.groupby("product_id")['rating'].mean().sort_values(ascending=False).head(5)
        top_rated = top_rated.reset_index().merge(products_df[['product_id', 'product_name']], on='product_id')
        fig3 = px.bar(top_rated, x='product_name', y='rating', color='product_name', title="Top 5 Products by Rating")
        st.plotly_chart(fig3, use_container_width=True)

    # ==============================
    # --- Predictive Analytics ---
    # ==============================
    st.markdown("## 🔮 Predictive & Progressive Analytics")

    # 1️⃣ Predict Complaints by State (Top 6)
    if "state" in vendors_df.columns:
        state_counts = vendors_df["state"].value_counts().reset_index()
        state_counts.columns = ["state", "complaint_count"]

        fig1 = px.bar(state_counts.head(6), x="state", y="complaint_count",
                      title="Top 6 States by Complaints (Predictive Trend)")
        st.plotly_chart(fig1, use_container_width=True)

    # 2️⃣ Predict Complaint Resolution Likelihood by Vendor Rating
    if "rating" in rexiews_df.columns and "vendor_id" in complaints_df.columns:
        merged_df = complaints_df.merge(vendors_df, on="vendor_id", how="left")
        merged_df = merged_df.merge(rexiews_df, on="vendor_id", how="left")

        if not merged_df.empty:
            vendor_resolution = merged_df.groupby("vendor_name")["rating"].mean().reset_index()
            vendor_resolution = vendor_resolution.sort_values("rating", ascending=False).head(6)

            fig2 = px.bar(vendor_resolution, x="vendor_name", y="rating",
                          title="Top 6 Vendors Likely to Resolve Complaints")
            st.plotly_chart(fig2, use_container_width=True)

    # --- Complaint Categorization (LDA) ---
    st.subheader("📂 Complaint Categorization (NLP)")
    if not complaints_df.empty:
        texts = complaints_df["complaint_text"].dropna().astype(str).tolist()

        if len(texts) > 2:  # Need at least 3 complaints
            vectorizer = CountVectorizer(stop_words="english", max_features=1000)
            X = vectorizer.fit_transform(texts)

            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            lda.fit(X)

            topic_assignments = lda.transform(X).argmax(axis=1)
            topic_counts = pd.Series(topic_assignments).value_counts(normalize=True) * 100

            topic_labels = []
            for idx, comp in enumerate(lda.components_):
                top_words = [vectorizer.get_feature_names_out()[i] for i in comp.argsort()[-5:]]
                topic_labels.append(", ".join(top_words))

            cat_df = pd.DataFrame({
                "Category": [f"Topic {i+1}: {topic_labels[i]}" for i in range(len(topic_labels))],
                "Percent": topic_counts.values
            })

            fig_cat = px.pie(cat_df, names="Category", values="Percent", title="Complaint Topics (%)")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Not enough complaints for topic modeling.")
    else:
        st.info("No complaints available for categorization.")

# ---------- Chatbot Page ----------

# ---------- Chatbot FAQs ----------
faq = {
    # English
    "en": {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! How can I assist you?",
        "help": "You can submit complaints, reviews, track complaints, view analytics, or get vendor info.",
        "submit complaint": "Go to 'Submit Complaint' page to lodge a new complaint.",
        "submit review": "Go to 'Submit Review' page to add a product review.",
        "track complaints": "Go to 'Track Complaints' page to see complaint status.",
        "vendor dashboard": "Go to 'Vendor Dashboard' page to see vendor stats.",
        "analytics": "Go to 'Analytics' page to view KPIs and charts.",
        "quick stats": "Total complaints, resolved complaints, pending complaints, avg. ratings, total users, total vendors.",
        "thank you": "You're welcome!",
        "bye": "Goodbye! Have a great day!"
    },

    # Hindi
    "hi": {
        "hello": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
        "hi": "हाय! मैं आपकी कैसे सहायता करूँ?",
        "help": "आप शिकायत दर्ज कर सकते हैं, समीक्षा जोड़ सकते हैं, शिकायतें ट्रैक कर सकते हैं, एनालिटिक्स देख सकते हैं, या विक्रेता जानकारी प्राप्त कर सकते हैं।",
        "submit complaint": "नया शिकायत दर्ज करने के लिए 'Submit Complaint' पेज पर जाएँ।",
        "submit review": "उत्पाद समीक्षा जोड़ने के लिए 'Submit Review' पेज पर जाएँ।",
        "track complaints": "'Track Complaints' पेज पर जाकर शिकायत स्थिति देखें।",
        "vendor dashboard": "'Vendor Dashboard' पेज पर जाकर विक्रेता आँकड़े देखें।",
        "analytics": "KPIs और चार्ट देखने के लिए 'Analytics' पेज पर जाएँ।",
        "quick stats": "कुल शिकायतें, हल की गई शिकायतें, लंबित शिकायतें, औसत रेटिंग, कुल उपयोगकर्ता, कुल विक्रेता।",
        "thank you": "आपका स्वागत है!",
        "bye": "अलविदा! आपका दिन शुभ हो!"
    }
}


def chatbot_response(user_input, lang="en"):
    user_input = user_input.lower()

    # Match keywords in FAQ
    for key in faq[lang].keys():
        if key in user_input:
            return faq[lang][key]

    # Default response if no match
    if lang == "hi":
        return "माफ़ कीजिये, मैं इसे समझ नहीं पाया। कृपया पुनः प्रयास करें।"
    return "Sorry, I didn't understand that. Please try again."


def page_chatbot():
    st.subheader("💬 Chatbot Support")

    # Language selection
    lang = st.radio("Select Language / भाषा चुनें", ["English", "Hindi"], horizontal=True)
    lang_code = "en" if lang == "English" else "hi"

    st.markdown("**Example Questions / उदाहरण प्रश्न:**")

    # FAQ section using expander (scrollable & compact)
    with st.expander("Click to see example questions / उदाहरण प्रश्न देखें"):
        example_questions = list(faq[lang_code].keys())[:8]  # top 8 FAQ
        for question in example_questions:
            if st.button(question.capitalize(), key=f"faq_{question}"):
                response = faq[lang_code][question]
                st.text_area("Chatbot Response / चैटबॉट जवाब", value=response, height=100)

    # User input for custom questions
    user_input = st.text_input("Type your question here / यहाँ अपना प्रश्न लिखें")
    if st.button("Send", key="send_custom"):
        if user_input.strip() != "":
            response = chatbot_response(user_input, lang_code)
            st.text_area("Chatbot Response / चैटबॉट जवाब", value=response, height=100)
        else:
            st.warning("Please type a question / कृपया एक प्रश्न टाइप करें")


def page_powerbi():
    st.subheader("📊 Power BI Dashboard")

    st.markdown("Here is the Power BI report screenshot:")

    # Build safe path dynamically
    image_path = os.path.join(os.getcwd(), "Images", "bgg.JPG")

    if os.path.exists(image_path):
        st.image(image_path, caption="Power BI Screenshot", use_container_width=True)
    else:
        st.error(f"❌ Could not find image at: {image_path}")
# ---------- Page Routing ----------
PAGES = {
    "🏠 Home": page_home,
    "📝 Submit Complaint": page_submit_complaint,
    "⭐ Submit Review": page_submit_review,
    "📊 Track Complaints": page_track_complaints,
    "🏭 Vendor Dashboard": page_vendor_dashboard,
    "📈 Analytics": page_analytics,
    "💬 Chatbot": page_chatbot,
    "📊 Power BI": page_powerbi
}

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)  # Home default

# --- Created By Section (AFTER navigation) ---
st.sidebar.markdown(
    """
    <div style="background-color: #e9f2fa; padding: 10px; border-radius: 8px; 
                box-shadow: 0px 2px 5px rgba(0,0,0,0.1); font-size: 12px;">
        <strong>👩‍💻 Created By:</strong><br>
        Nikita Singh – <a href="mailto:nikiita307@gmail.com">nikiita307@gmail.com</a><br>
        Nikki Yadav – <a href="mailto:nikkiyadav8828@gmail.com">nikkiyadav8828@gmail.com</a><br>
        Aditi Singh – <a href="mailto:1aditisingh2121@gmail.com">1aditisingh2121@gmail.com</a><br>
        Simi Dubey – <a href="mailto:simidubey028@gmail.com">simidubey028@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Load selected page
page = PAGES[selection]
page()
