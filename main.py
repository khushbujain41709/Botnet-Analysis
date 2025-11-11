import streamlit as st
import pandas as pd
import joblib
import numpy as np

# PAGE CONFIGURATION
# PAGE CONFIGURATION
st.set_page_config(
    page_title="Bot vs Human Detection",
    page_icon="👻"
)

# LOAD MODEL
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model_xgb.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
model, vectorizer = load_model()

label_map = {
    0: "human",
    1: "content_bot", 
    2: "follower_bot",
    3: "spam_bot",
    4: "customer_service_bot"
}

# PREDICTION FUNCTION
def model_prediction(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    result_index = prediction[0]
    confidence = np.max(probabilities[0])
    return result_index, confidence, probabilities[0]

# SIDEBAR NAVIGATION
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", 
    ["Home", "About Project", "Bot Detection", "Test Examples"])
st.markdown("""
<style>
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        text-align: left !important;
        padding-right: 16px !important;
    }
</style>
""",  unsafe_allow_html=True)

# HOME PAGE
if app_mode == "Home":
    st.markdown("""
        <style>
        /* GLOBAL BACKGROUND */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0D1117 40%, #1C1C22 100%);
            color: #E6EDF3;
        }
        /* TITLE & HEADINGS */
        h1, h2, h3, h4 {
            color: #58A6FF;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
        }
        /* NORMAL TEXT */
        p, li {
            color: #E6EDF3;
            font-size: 16px;
            line-height: 1.6;
        }
        /* IMAGE STYLING */
        img {
            border-radius: 20px;
            box-shadow: 0 0 25px rgba(88,166,255,0.3);
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        /* SECTION HEADERS */
        .section-header {
            color: #79C0FF;
            font-size: 22px;
            margin-top: 30px;
            font-weight: 600;
            border-left: 4px solid #58A6FF;
            padding-left: 10px;
        }
        /* EMPHASIS TEXT */
        strong {
            color: #FFB86C;
        }
        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: #161B22;
            color: #E6EDF3;
            box-shadow: 8px 0 20px rgba(88, 166, 255, 0.15);
        }
        /* LINKS */
        a {
            color: #58A6FF;
            text-decoration: none;
        }
        a:hover {
            color: #FFB86C;
        }
        img:hover {
            transform: scale(1.02);
            box-shadow: 0 0 40px rgba(255, 120, 90, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)
    st.header("Bot vs Human Detection System")
    st.image("botImageHome.png", use_container_width=False)
    st.markdown("""
    Welcome to the **Bot vs Human Detection System!**  
    Our mission is to help identify automated bot accounts and distinguish them from genuine human users.  
    Upload text content or paste messages, and our system will analyze them to detect bot-like patterns.  
    Together, let's create safer digital spaces!  
    """)
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Input Text:** Go to the **Bot Detection** page and paste or type text content.  
    2. **Analysis:** Our system will process the text using advanced character-level machine learning algorithms.  
    3. **Results:** View the classification results with confidence scores and detailed insights.
    """)
    st.markdown('<div class="section-header">Detection Categories</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Human:** Genuine human conversations and expressions  
    - **Content Bot:** Automated news feeds and content aggregators  
    - **Follower Bot:** Social media growth automation services  
    - **Spam Bot:** Promotional and scam content  
    - **Customer Service Bot:** Automated support and helpdesk systems
    """)
    st.markdown('<div class="section-header">Why Choose Our System?</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Accuracy:** 84% accurate character-level pattern recognition  
    - **Real-time Analysis:** Get instant results for quick decision making  
    - **Comprehensive Detection:** Identifies multiple types of automated behavior  
    - **User-Friendly:** Simple interface requiring no technical expertise
    """)
    st.markdown('<div class="section-header">Get Started</div>', unsafe_allow_html=True)
    st.markdown("""
    Click on the **Bot Detection** page in the sidebar to start analyzing text content!
    """)
    st.markdown('<div class="section-header">Learn More</div>', unsafe_allow_html=True)
    st.markdown("""
    Visit the **About Project** page to understand our methodology and technology.
    """)

# ABOUT PAGE
elif app_mode == "About Project":
    st.markdown("""
    <style>
    /* MAIN PAGE BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(150deg, #4B1C1C 0%, #2B1313 50%, #1A0A0A 100%);
        color: #F8EDED;
    }
    /* HEADER STYLES */
    h1, h2 {
        color: #FFB870;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        text-align: center;
        text-shadow: 0 0 20px rgba(255, 184, 112, 0.25);
    }
    h3, h4 {
        color: #FFD8AA;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        margin-top: 30px;
        border-left: 4px solid #FFB870;
        padding-left: 10px;
    }
    /* SECTION HEADERS */
    .section-header {
        color: #FFCFA5;
        font-size: 22px;
        font-weight: 600;
        margin-top: 35px;
        border-left: 4px solid #FFB870;
        padding-left: 12px;
        text-shadow: 0 0 8px rgba(255, 184, 112, 0.25);
    }
    /* BODY TEXT */
    p, li {
        color: #F3E4E4;
        font-size: 16px;
        line-height: 1.7;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.3px;
    }
    /* IMAGE STYLING */
    img {
        border-radius: 18px;
        margin-top: 15px;
        box-shadow: 0 0 30px rgba(255, 150, 100, 0.25);
        display: block;
        margin-left: auto;
        margin-right: auto;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 0 45px rgba(255, 150, 100, 0.45);
    }
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3A1616 0%, #1A0A0A 100%);
        color: #F8EDEA;
        border-right: 1px solid rgba(255, 184, 112, 0.2);
        box-shadow: 4px 0 20px rgba(255, 184, 112, 0.1);
    }
    /* EMPHASIS TEXT */
    strong {
        color: #FFB870;
    }
    /* LINKS */
    a {
        color: #FFB870;
        text-decoration: none;
        font-weight: 500;
    }
    a:hover {
        color: #FFD9A0;
        text-shadow: 0 0 10px rgba(255, 184, 112, 0.3);
    }
    /* DIVIDER */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(255,184,112,0.3), transparent);
        margin: 25px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    st.header("About Our Project")
    st.image("retroRobots.jpg", use_container_width=True)
    st.markdown('<div class="section-header">About the Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    Our training dataset consists of **250 carefully curated text samples** evenly distributed across 5 categories.  
    Each category represents distinct behavioral and linguistic patterns found in human or automated communication.
    """)
    st.markdown('<div class="section-header">Dataset Composition</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Human Samples:** 50 authentic human conversations and social media posts  
    - **Content Bot Samples:** 50 automated content aggregation messages  
    - **Follower Bot Samples:** 50 social media growth service promotions  
    - **Spam Bot Samples:** 50 scam and promotional messages  
    - **Customer Service Bot Samples:** 50 automated support responses  
    """)
    st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Algorithm:** XGBoost Classifier  
    - **Feature Extraction:** Character-level 3–5 grams  
    - **Vectorization:** CountVectorizer with character analysis  
    - **Accuracy:** 84% on validation data  
    - **Classes:** 5 distinct categories  
    """)
    st.markdown('<div class="section-header">Technical Approach</div>', unsafe_allow_html=True)
    st.markdown("""
    We use **character-level n-gram analysis** instead of traditional word-based approaches because it:  
    - Captures bot-specific text structures like underscores and repetitive tokens  
    - Is **language-agnostic** and works across writing styles  
    - Is robust against vocabulary variations  
    - Detects subtle **automation fingerprints** missed by word-based methods  
    """)
    st.markdown('<div class="section-header">Training Process</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Data Collection:** Curated dataset of 250 samples  
    2. **Feature Engineering:** Character 3–5 gram extraction  
    3. **Model Training:** XGBoost with multi-class classification  
    4. **Validation:** 20% stratified split for testing  
    5. **Deployment:** Streamlit-based web application for interactive analysis  
    """)

# BOT DETECTION PAGE
elif app_mode == "Bot Detection":
    st.header("Bot Detection Analysis")
    st.markdown("""
    <style>
    /* GLOBAL BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #2B2520 0%, #1A1510 100%);
        color: #E7D9C1;
    }
    /* HEADER STYLING */
    h1, h2, h3 {
        color: #D4B27B;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
        text-shadow: 0 0 10px rgba(212, 178, 123, 0.3);
        font-weight: 700;
    }
    /* TEXTAREA STYLING */
    textarea {
        background: linear-gradient(160deg, #3A322A 0%, #241E18 100%) !important;
        color: #F5E9D0 !important;
        border: 1px solid rgba(212, 178, 123, 0.35) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 12px rgba(212, 178, 123, 0.25);
    }
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(120deg, #6E4B2F 0%, #3F2A1C 100%);
        color: #FFF1D9;
        font-weight: 600;
        border-radius: 10px;
        border: 1px solid rgba(212, 178, 123, 0.3);
        padding: 0.5em 1.2em;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(212, 178, 123, 0.25);
    }
    div.stButton > button:hover {
        background: linear-gradient(120deg, #8A603C 0%, #533523 100%);
        transform: scale(1.03);
        box-shadow: 0 0 18px rgba(212, 178, 123, 0.5);
    }
    /* METRIC CARDS */
    div[data-testid="stMetricValue"] {
        color: #E7D9C1 !important;
        text-shadow: 0 0 6px rgba(212, 178, 123, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #C8B69B !important;
    }
    /* IMAGE */
    img {
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0 0 30px rgba(180, 140, 90, 0.2);
        display: block;
        margin-left: auto;
        margin-right: auto;
        filter: sepia(0.3) contrast(1.05);
    }
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2B231C 0%, #1B1510 100%);
        color: #E7D9C1;
        border-right: 1px solid rgba(212, 178, 123, 0.25);
        box-shadow: 4px 0 12px rgba(212, 178, 123, 0.15);
    }
    /* ALERTS */
    div.stAlert {
        border-radius: 10px;
        background-color: rgba(50, 40, 32, 0.8);
        box-shadow: 0 0 15px rgba(212, 178, 123, 0.15);
        border: 1px solid rgba(212, 178, 123, 0.3);
        font-weight: 500;
        color: #E7D9C1;
    }
    /* LINKS */
    a {
        color: #D4B27B;
        text-decoration: none;
    }
    a:hover {
        color: #E7C690;
    }

    /* IMAGE HOVER */
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 0 35px rgba(212, 178, 123, 0.35);
    }
    </style>
    """, unsafe_allow_html=True)
    st.image("Loki_TVA.jpg", use_container_width=True)
    input_text = st.text_area(
        "Enter text to analyze:", 
        placeholder="Paste or type any text content here (social media posts, messages, profiles, etc.)...",
        height=150
    )
    col1, col2 = st.columns(2)
    with col1:
        show_text = st.button("Preview Text")
    with col2:
        predict_btn = st.button("Analyze Text")
    if show_text and input_text:
        st.subheader("Text Preview")
        st.info(input_text)
    if predict_btn and input_text:
        if model is None or vectorizer is None:
            st.error("Model not loaded properly. Please check if model files exist.")
        else:
            st.write("### Analysis Results")
            with st.spinner("Analyzing text patterns..."):
                result_index, confidence, all_probabilities = model_prediction(input_text)
                class_name = label_map[result_index]
                st.success("**Prediction:** {}".format(class_name.upper()))
                st.metric("Confidence Score", f"{confidence:.2%}")
                st.subheader("Detailed Confidence Scores")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Human", f"{all_probabilities[0]:.2%}")
                with col2:
                    st.metric("Content Bot", f"{all_probabilities[1]:.2%}")
                with col3:
                    st.metric("Follower Bot", f"{all_probabilities[2]:.2%}")
                with col4:
                    st.metric("Spam Bot", f"{all_probabilities[3]:.2%}")
                with col5:
                    st.metric("Service Bot", f"{all_probabilities[4]:.2%}")
                st.subheader("Behavioral Insights")
                insights = {
                    "human": """**Human Profile Detected**
                    - Natural language with emotional expression  
                    - Personal experiences and authentic storytelling  
                    - Varied sentence structure and natural flow  
                    - Uses emojis and expressive punctuation""",
                    "content_bot": """**Content Bot Detected**
                    - Automated news or information feeds  
                    - Formal, information-dense content  
                    - Often uses underscores and specific terminology  
                    - Repetitive structural patterns""",
                    "follower_bot": """**Follower Bot Detected**
                    - Social media growth automation services  
                    - Keywords: follow, boost, growth, likes, subscribers  
                    - Promotional and engagement-focused tone""",
                    "spam_bot": """**Spam Bot Detected**
                    - Promotional or scam content  
                    - Urgent or exaggerated language  
                    - Free offers, prize claims, financial opportunities""",
                    "customer_service_bot": """**Customer Service Bot Detected**
                    - Automated support and helpdesk responses  
                    - Transaction-related or structured replies  
                    - Polite, standardized phrasing"""
                }
                st.info(insights[class_name])
    elif predict_btn and not input_text:
        st.warning("Please enter some text to analyze.")
        
# TEST EXAMPLES PAGE
elif app_mode == "Test Examples":
    st.header("Test Examples for Bot Detection")
    st.image("RedRobo.png", use_container_width=True)
    st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #2A3B3B 0%, #1C2B2B 40%, #3A1F1F 100%); color: #EAEAEA;
    }
    /* HEADER STYLING */
    h1, h2, h3 {
        color: #FFB870;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
        text-shadow: 0 0 15px rgba(255, 184, 112, 0.3);
        font-weight: 700;
    }
    /*SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2A1A1A 0%, #1A0F0F 100%); color: #F2E5E5;
        border-right: 1px solid rgba(255, 184, 112, 0.25);
        box-shadow: 4px 0 20px rgba(255, 184, 112, 0.15);
    }
    /* TEXTAREA */
    textarea {
        background: linear-gradient(160deg, #2E1B1B 0%, #1A0F0F 100%) !important;
        color: #FFECDC !important;
        border: 1px solid rgba(255, 184, 112, 0.4) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 10px rgba(255, 184, 112, 0.08);
        transition: all 0.3s ease-in-out !important;
        font-size: 15px !important;
        padding: 10px !important;
    }
    textarea:focus {
        border-color: #D8A47F !important;
        box-shadow: 0 0 18px rgba(216, 164, 127, 0.4) !important;
        background-color: #3B2020 !important;
    }
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(120deg, #A34C4C 0%, #6B2D2D 100%);
        color: #FFF6F0;
        font-weight: 600;
        border-radius: 10px;
        border: 1px solid rgba(255, 184, 112, 0.4);
        padding: 0.5em 1.2em;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(255, 184, 112, 0.15);
    }
    div.stButton > button:hover {
        background: linear-gradient(120deg, #C06161 0%, #8C3A3A 100%);
        transform: scale(1.04);
        box-shadow: 0 0 25px rgba(255, 184, 112, 0.4);
    }
    /* METRICS */
    div[data-testid="stMetricValue"] {
        color: #FFD8A8 !important;
        text-shadow: 0 0 6px rgba(255, 184, 112, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #F8E6E0 !important;
    }
    /* IMAGES */
    img {
        border-radius: 15px;
        margin-top: 10px;
        box-shadow: 0 0 25px rgba(255, 120, 90, 0.15);
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    /* INFO & ALERT BOXES */
    div.stAlert {
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(255, 184, 112, 0.25);
        font-weight: 500;
        background: rgba(40, 20, 20, 0.4);
        border: 1px solid rgba(255, 184, 112, 0.25);
    }
    /* SPINNER COLOR */
    .stSpinner > div > div {
        border-top-color: #FFB870 !important;
    }
    /* LINKS */
    a {
        color: #FFB870;
        text-decoration: none;
    }
    a:hover {
        color: #FFDCA5;
    }
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(255, 120, 90, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    Use these examples to test the bot detection system. 
    Copy and paste any of these into the **Bot Detection** page.
    """)
    # Human Examples
    st.subheader("Human Examples")
    human_examples = [
        "Just got back from an amazing hike with friends! The view was absolutely breathtaking 🏞️",
        "Celebrating my birthday today with family - so grateful for all the love and wishes! ❤️",
        "Finally finished that book I've been reading for weeks. Highly recommend it to everyone!",
        "Learning to bake sourdough bread and my first attempt actually turned out edible! 🍞",
        "Spent the whole day cleaning and organizing my apartment - feels so refreshing now!"
    ]
    for i, example in enumerate(human_examples, 1):
        st.text_area(f"Human Example {i}", value=example, height=80, key=f"human_{i}")
    # Content Bot Examples
    st.subheader("Content Bot Examples")
    content_examples = [
        "breaking_news tech_update ai_developments market_trends",
        "weather_alert storm_warning safety_updates emergency_info", 
        "sports_scores game_results player_stats team_news",
        "crypto_prices blockchain_news defi_updates nft_trends",
        "entertainment_news celebrity_updates movie_releases tv_highlights"
    ]
    for i, example in enumerate(content_examples, 1):
        st.text_area(f"Content Bot Example {i}", value=example, height=70, key=f"content_{i}", max_chars=200)
    # Follower Bot Examples  
    st.subheader("Follower Bot Examples")
    follower_examples = [
        "get_more_followers instant_growth social_media_boost",
        "follow4follow back gain_followers_fast viral_content",
        "instagram_growth twitter_followers tiktok_views boost", 
        "social_media_marketing audience_expansion profile_optimization",
        "auto_follow engagement_boost follower_increase trending"
    ]
    for i, example in enumerate(follower_examples, 1):
        st.text_area(f"Follower Bot Example {i}", value=example, height=70, key=f"follower_{i}", max_chars=200)
    # Spam Bot Examples
    st.subheader("Spam Bot Examples")
    spam_examples = [
        "win_free_iphone click_here_now limited_time_offer",
        "make_money_fast work_from_home easy_cash_guaranteed",
        "miracle_weight_loss secret_formula instant_results",
        "free_bitcoin_bonus crypto_investment_opportunity", 
        "urgent_alert important_message immediate_action_required"
    ]
    for i, example in enumerate(spam_examples, 1):
        st.text_area(f"Spam Bot Example {i}", value=example, height=70, key=f"spam_{i}", max_chars=200)
    # Service Bot Examples
    st.subheader("Customer Service Bot Examples")
    service_examples = [
        "automated_support help_desk customer_service_assistance", 
        "order_tracking shipping_update delivery_confirmation",
        "account_verification security_check password_reset",
        "technical_support system_diagnosis issue_resolution",
        "billing_inquiry payment_processing refund_request"
    ]
    for i, example in enumerate(service_examples, 1):
        st.text_area(f"Service Bot Example {i}", value=example, height=70, key=f"service_{i}", max_chars=200)
        
# FOOTER
st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.markdown("""
    <style>
    .sidebar-footer {
        font-size: 14px;                     /* keeps same text size */
        color: #F5E6D0;                      /* matches sidebar text color */
        text-align: center;                  /* centers all text */
        line-height: 1.5;                    /* adds clean spacing */
        font-family: 'Segoe UI', sans-serif;
    }
    .sidebar-footer strong {
        font-weight: 600;                    /* subtle bold, not oversized */
        color: #F5E6D0;                      /* highlight color */
    }
    </style>
    <div class="sidebar-footer">
        <strong>Developed by:</strong><br>
        Khushbu Jain • 23115047<br>
        Kruti Dadriwal • 23115050<br>
        Mukka Manideep • 23115058<br>
        P.Sai Madhuri Bai • 23115067<br>
    </div>
""", unsafe_allow_html=True)
st.sidebar.image("rustyRobot.jpeg", use_container_width=True)
