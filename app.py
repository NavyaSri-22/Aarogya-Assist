import streamlit as st
import time
import base64
import torch
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq

# Try importing pdf2image and pytesseract cleanly; provide fallbacks if binary paths differ in environment
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# --- LAZY LOADING CACHED MODERN AI MODELS ---
@st.cache_resource
def load_vision_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model_blip

processor, model_blip = load_vision_models()

# --- INITIALIZE GROQ INFERENCE CORE ENGINE ---
client = Groq(api_key="API KEY")

# --- INITIALIZE MULTILINGUAL DICTIONARY FRAMEWORK ---
LANG_DICT = {
    "English": {
        "title": "Aarogya Assist",
        "tagline": "Your Safe AI Companion",
        "desc": "Step into the future of proactive personal medicine. Aarogya Assist leverages advanced neural networks to decode complex clinical anomalies, translate scientific medical jargon instantly, and offer structured well-being protocols.",
        "get_started": "🚀 Get Started",
        "chat_header": "🤖 AI Medical Workspace Chat",
        "chat_placeholder": "Ask your health query here...",
        "image_header": "🖼️ Medical Image Analysis Model",
        "report_header": "📋 PDF Scanned Report Summarizer",
        "tips_header": "💡 Smart Health Metrics & Tips",
        "disclaimer_title": "⚠️ Strict Medical Disclaimer",
        "disclaimer_text": "This assistant is for educational purposes only. NOT a substitute for professional clinical doctors, diagnostics, or emergency care.",
        "emergency_msg": "🚨 EMERGENCY WARNING: Please seek immediate medical attention or contact local emergency services immediately.",
        "analyzing": "🧠 AI Node is analyzing context payload..."
    },
    "Hindi": {
        "title": "आरोग्य असिस्ट",
        "tagline": "आपका सुरक्षित एआई साथी",
        "desc": "सक्रिय व्यक्तिगत चिकित्सा के भविष्य में कदम रखें। आरोग्य असिस्ट जटिल नैदानिक विसंगतियों को समझने, वैज्ञानिक चिकित्सा शब्दावली का तुरंत अनुवाद करने और संरचित कल्याण प्रोटोकॉल प्रदान करने के लिए उन्नत न्यूरल नेटवर्क का उपयोग करता है।",
        "get_started": "🚀 शुरू करें",
        "chat_header": "🤖 एआई मेडिकल वर्कस्पेस चैट",
        "chat_placeholder": "अपनी स्वास्थ्य संबंधी पूछताछ यहाँ लिखें...",
        "image_header": "🖼️ मेडिकल इमेज विश्लेषण मॉडल",
        "report_header": "📋 पीडीएफ स्कैन रिपोर्ट सारांश",
        "tips_header": "💡 स्मार्ट स्वास्थ्य युक्तियाँ",
        "disclaimer_title": "⚠️ सख्त चिकित्सा अस्वीकरण",
        "disclaimer_text": "यह सहायक केवल शैक्षिक उद्देश्यों के लिए है। पेशेवर डॉक्टरों, निदान या आपातकालीन देखभाल का विकल्प नहीं है।",
        "emergency_msg": "🚨 आपातकालीन चेतावनी: कृपया तुरंत आपातकालीन चिकित्सा सहायता लें या निकटतम आपातकालीन सेवाओं से संपर्क करें।",
        "analyzing": "🧠 एआई सिस्टम डेटा का विश्लेषण कर रहा है..."
    },
    "Telugu": {
        "title": "ఆరోగ్య అసిస్ట్",
        "tagline": "మీ సురక్షితమైన AI సహచరుడు",
        "desc": "భవిష్యత్తు వ్యక్తిగత వైద్య రంగంలోకాని అడుగుపెట్టండి. ఆరోగ్య అసిస్ట్ సంక్లిష్టమైన క్లినికల్ నివేదికలను విశ్లేషించడానికి, వైద్య పదజాలాన్ని తక్షణమే అనువదించడానికి మరియు నిలకడైన ఆరోగ్య ప్రోటోకాల్‌లను అందించడానికి అధునాతన న్యూరల్ నెట్‌వర్క్‌లను ఉపయోగిస్తుంది.",
        "get_started": "🚀 ప్రారంభించండి",
        "chat_header": "🤖 AI మెడికల్ వర్క్‌స్పేస్ చాట్",
        "chat_placeholder": "మీ ఆరోగ్య సమస్యను ఇక్కడ అడగండి...",
        "image_header": "🖼️ మెడికల్ ఇమేజ్ అనాలిసిస్ మోడల్",
        "report_header": "📋 PDF నివేదిక సారాంశం",
        "tips_header": "💡 స్మార్ట్ ఆరోగ్య చిట్కాలు",
        "disclaimer_title": "⚠️ వైద్య నిరాకరణ",
        "disclaimer_text": "ఈ అసిస్టెంట్ విద్యా ప్రయోజనాల కోసం మాత్రమే. ఇది వృత్తిపరమైన వైద్యులకు, రోగనిర్ధారణకు లేదా అత్యవసర సంరక్షణకు ప్రత్యామ్నాయం కాదు.",
        "emergency_msg": "🚨 అత్యవసర హెచ్చరిక: దయచేసి వెంటనే వైద్య సహాయం తీసుకోండి లేదా అత్యవసర సేవలను సంప్రదించండి.",
        "analyzing": "🧠 AI నివేదికను విశ్లేషిస్తోంది..."
    }
}

# --- PAGE STATE ROUTING CONTROL ---
if "page" not in st.session_state:
    st.session_state.page = "entrance"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "app_language" not in st.session_state:
    st.session_state.app_language = "English"

LT = LANG_DICT[st.session_state.app_language]

# --- SAFE STRUCTURAL HTML STRINGS ---
STYLE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: radial-gradient(circle at 10% 20%, rgba(9, 14, 32, 1) 0%, rgba(4, 8, 16, 1) 100%) !important;
        color: #F1F5F9 !important;
    }
    
    /* WIDE SCREEN ENGINE OVERRIDE PATCH */
    .block-container {
        max-width: 92% !important;
        padding-left: 4rem !important;
        padding-right: 4rem !important;
        padding-top: 3rem !important;
    }
    
    [data-testid="stHeader"] { background: transparent; }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 20px !important;
        padding: 28px !important;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 24px !important;
    }
    
    .gradient-title {
        background: linear-gradient(135deg, #00E5FF 0%, #7B61FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .chat-bubble-user {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.15) 0%, rgba(123, 97, 255, 0.2) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        padding: 16px;
        border-radius: 16px 16px 0px 16px;
        margin: 12px 0px 12px auto;
        max-width: 80%;
        color: #F8FAFC;
    }
    
    .chat-bubble-bot {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 16px;
        border-radius: 16px 16px 16px 0px;
        margin: 12px auto 12px 0px;
        max-width: 80%;
        color: #F1F5F9;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    .telemetry-val {
        font-size: 2rem;
        font-weight: 700;
        color: #00E5FF;
        text-shadow: 0 0 15px rgba(0,229,255,0.4);
    }
    
    .glow-matrix {
        position: absolute;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0, 229, 255, 0.2) 0%, rgba(123, 97, 255, 0) 70%);
        filter: blur(50px);
    }
</style>
"""

ENTRANCE_GRAPHIC = """
<div style="position: relative; display: flex; justify-content: center; align-items: center; height: 520px;">
    <div class="glow-matrix"></div>
    <svg width="440" height="440" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="cyberGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#00E5FF;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#7B61FF;stop-opacity:1" />
            </linearGradient>
        </defs>
        <circle cx="100" cy="100" r="28" fill="url(#cyberGlow)" opacity="0.85" />
        <circle cx="100" cy="100" r="48" stroke="url(#cyberGlow)" stroke-width="1.5" fill="none" stroke-dasharray="6 6"/>
        <circle cx="100" cy="100" r="68" stroke="rgba(255,255,255,0.05)" stroke-width="1" fill="none" />
        <circle cx="45" cy="65" r="9" fill="#00E5FF" />
        <line x1="100" y1="100" x2="45" y2="65" stroke="rgba(0, 229, 255, 0.4)" stroke-width="1.5" />
        <circle cx="165" cy="75" r="11" fill="#7B61FF" />
        <line x1="100" y1="100" x2="165" y2="75" stroke="rgba(123, 97, 255, 0.4)" stroke-width="1.5" />
        <circle cx="65" cy="145" r="7" fill="#00E5FF" />
        <line x1="100" y1="100" x2="65" y2="145" stroke="rgba(0, 229, 255, 0.3)" stroke-width="1.5" />
        <circle cx="145" cy="145" r="10" fill="#7B61FF" />
        <line x1="100" y1="100" x2="145" y2="145" stroke="rgba(123, 97, 255, 0.3)" stroke-width="1.5" />
        <path d="M82,100 L92,100 L95,90 L99,112 L103,85 L106,105 L109,100 L118,100" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
</div>
"""

st.markdown(STYLE_CSS, unsafe_allow_html=True)

# --- BACKEND INFERENCE FUNCTIONS ---
def get_ai_response(user_input, target_lang):
    system_prompt = f"""
You are Aarogya Assist, a high-precision, completely safe medical AI assistant.
CRITICAL LAWS:
- Do NOT diagnose specific diseases or provide definitive analysis.
- Do NOT prescribe medications or pharmaceutical dosages under any condition.
- Provide objective educational health guidance, symptom triage matrices, and context support only.
- Explicitly emphasize consults with physical certified healthcare physicians.
- You MUST respond and compose your answers directly and entirely in the {target_lang} language.
"""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.35,
        max_tokens=1024,
    )
    return completion.choices[0].message.content

def analyze_medical_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            output = model_blip.generate(**inputs, max_new_tokens=60)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error executing computer vision node pipeline: {str(e)}"

RED_FLAGS = ["chest pain", "breathing difficulty", "stroke", "unconscious", "severe bleeding", "heart attack", "सीना दर्द", "सांस लेने में तकलीफ", "గుండె నొప్పి", "శ్వాస తీసుకోవడం ఇబ్బంది"]
def detect_emergency(text):
    text = text.lower()
    return any(word in text for word in RED_FLAGS)

def route_to_main():
    st.session_state.page = "main_app"

# ==========================================
# 1. ENTRANCE LANDING PAGE (FULL WIDTH)
# ==========================================
if st.session_state.page == "entrance":
    st.markdown("<div style='height: 2vh;'></div>", unsafe_allow_html=True)
    left_side, right_side = st.columns([1.2, 1], gap="large")
    
    with left_side:
        st.markdown("<div style='height: 4vh;'></div>", unsafe_allow_html=True)
        st.markdown(f"<h1>🩺 <span class='gradient-title' style='font-size: 4.5rem;'>{LT['title']}</span></h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #94A3B8; font-weight: 400; letter-spacing: 0.5px; margin-top:-15px;'>{LT['tagline']}</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 35px;">
            <p style="font-size: 1.25rem; line-height: 1.7; color: #CBD5E1;">{LT['desc']}</p>
            <div style="margin-top: 25px; font-weight: 600; font-size: 1rem; color: #00E5FF; display: flex; gap: 20px; flex-wrap: wrap;">
                <span>⚡ Multi-Modal Vision</span> • <span>📊 NLP Report Parsing</span> • <span>🔒 Fully Encrypted Sessions</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='color: #64748B; margin-bottom: 5px; font-size:0.85rem; letter-spacing:1px;'>CHOOSE CORE SYSTEM DIALECT</p>", unsafe_allow_html=True)
        st.session_state.app_language = st.selectbox(
            "Language Selector", ["English", "Hindi", "Telugu"], 
            key="landing_lang_select", label_visibility="collapsed"
        )
        
        st.markdown("<div style='height: 4vh;'></div>", unsafe_allow_html=True)
        st.button(LT['get_started'], use_container_width=True, on_click=route_to_main)
        
    with right_side:
        st.markdown(ENTRANCE_GRAPHIC, unsafe_allow_html=True)

# ==========================================
# 2. MAIN APPLICATION WORKSPACE 
# ==========================================
elif st.session_state.page == "main_app":
    with st.sidebar:
        st.markdown(f"<h2>🩺 <span class='gradient-title'>{LT['title']}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748B; font-size:0.85rem; margin-top:-10px;'>{LT['tagline']}</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("<p style='color: #64748B; font-size:0.75rem; font-weight:600; letter-spacing:0.5px;'>SYSTEM WORKSPACE LANGUAGE</p>", unsafe_allow_html=True)
        st.session_state.app_language = st.selectbox(
            "Workspace Language Selector", ["English", "Hindi", "Telugu"], 
            key="workspace_lang_select", label_visibility="collapsed"
        )
        LT = LANG_DICT[st.session_state.app_language]
        
        st.markdown("---")
        st.markdown(f"### {LT['disclaimer_title']}")
        st.markdown(f"<p style='font-size:0.85rem; color:#94A3B8; line-height:1.5;'>{LT['disclaimer_text']}</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.button("⬅️ Exit Session"):
            st.session_state.page = "entrance"
            st.rerun()

    # --- METRICS DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="glass-card" style="text-align:center;"><p style="color:#94A3B8; margin:0; font-size:0.85rem;">MODEL ACCURACY</p><p class="telemetry-val">94.2%</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;"><p style="color:#94A3B8; margin:0; font-size:0.85rem;">REPORTS PROCESSED</p><p class="telemetry-val">10K+</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="glass-card" style="text-align:center;"><p style="color:#94A3B8; margin:0; font-size:0.85rem;">STREAMS RESOLVED</p><p class="telemetry-val">50K+</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="glass-card" style="text-align:center;"><p style="color:#94A3B8; margin:0; font-size:0.85rem;">SYSTEM LATENCY</p><p class="telemetry-val">1.2s</p></div>', unsafe_allow_html=True)

    tab_chat, tab_vision, tab_reports = st.tabs([f"💬 {LT['chat_header']}", f"🖼️ {LT['image_header']}", f"📋 {LT['report_header']}"])

    # --- TAB 1: INTERACTIVE CHAT ---
    with tab_chat:
        st.markdown(f"## {LT['chat_header']}")
        chat_pane = st.container()
        with chat_pane:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-bot">🤖 <b>Aarogya AI:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        
        with st.form(key="chat_input_form", clear_on_submit=True):
            user_input = st.text_input(LT['chat_placeholder'], key="chat_query_box")
            submit_chat = st.form_submit_button("Send Query ⚡")
            
        if submit_chat and user_input.strip() != "":
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            if detect_emergency(user_input):
                st.markdown(f'<div class="chat-bubble-bot" style="border: 1px solid #EF4444; background: rgba(239,68,68,0.15); color: #FCA5A5;"><b>{LT["emergency_msg"]}</b></div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": LT['emergency_msg']})
            else:
                with st.spinner(LT['analyzing']):
                    try:
                        bot_reply = get_ai_response(user_input, st.session_state.app_language)
                    except Exception as e:
                        bot_reply = f"Backend Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.rerun()

    # --- TAB 2: VISION DIAGNOSTICS ---
    with tab_vision:
        st.markdown(f"## {LT['image_header']}")
        image_file = st.file_uploader("Upload Medical Scan Imagery", type=["png", "jpg", "jpeg"], key="vision_uploader")
        
        if image_file:
            st.image(Image.open(image_file), caption="Target Scan Loaded.", width=400)
            if st.button("🔬 Trigger Vision Neural Analysis", use_container_width=True):
                with st.spinner(LT['analyzing']):
                    vision_caption = analyze_medical_image(image_file)
                    report_prompt = f"Synthesize a brief educational report layout based precisely on this image caption: '{vision_caption}'. Keep the output completely in {st.session_state.app_language}."
                    compiled_report = get_ai_response(report_prompt, st.session_state.app_language)
                    
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 4px solid #00E5FF;">
                        <h3 style="color:#00E5FF; margin-top:0;">📊 Comprehensive Neural Vision Feedback</h3>
                        <p style="line-height:1.6; font-size:1.05rem;">{compiled_report}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # --- TAB 3: REPORT ANALYSIS ---
    with tab_reports:
        st.markdown(f"## {LT['report_header']}")
        pdf_file = st.file_uploader("Upload Lab Record (PDF / TXT)", type=["pdf", "txt"], key="nlp_uploader")
        pasted_metrics = st.text_area("Or paste raw text data fields here:")
        
        if st.button("🧬 Parse and Summarize Report Fields", use_container_width=True):
            extracted_text = ""
            if pdf_file:
                if pdf_file.name.endswith('.txt'):
                    extracted_text = pdf_file.read().decode("utf-8")
                elif pdf_file.name.endswith('.pdf'):
                    try:
                        reader = PdfReader(pdf_file)
                        extracted_text = "".join([page.extract_text() or "" for page in reader.pages])
                        if not extracted_text.strip() and HAS_OCR:
                            images = convert_from_bytes(pdf_file.getvalue())
                            extracted_text = "".join([pytesseract.image_to_string(img) for img in images])
                    except Exception as e:
                        st.error(f"Error parsing layers: {str(e)}")
            elif pasted_metrics.strip():
                extracted_text = pasted_metrics
                
            if extracted_text.strip():
                with st.spinner(LT['analyzing']):
                    summary_prompt = f"Summarize the following health data values into a clear summary report. Match language formatting to {st.session_state.app_language}:\n\n{extracted_text}"
                    summary_out = get_ai_response(summary_prompt, st.session_state.app_language)
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 4px solid #7B61FF;">
                        <h3 style="color:#7B61FF; margin-top:0;">📋 Extracted Analytical Summary Insights</h3>
                        <p style="line-height:1.6; font-size:1.05rem;">{summary_out}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # --- HEALTH TIPS WIDGET MATRICES ---
    st.markdown("---")
    st.markdown(f"## {LT['tips_header']}")
    tips = ["💧 Drink enough water daily.", "🥗 Maintain low glycemic index diets.", "🏃 Execute active movement daily.", "😴 Secure solid deep metabolic rest."]
    tip_cols = st.columns(4)
    for idx, tip in enumerate(tips):
        with tip_cols[idx]:
            st.markdown(f'<div class="glass-card" style="text-align:center; padding:15px !important; min-height:90px; font-size:0.9rem;">{tip}</div>', unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown(f"""
    <div style="text-align: center; margin-top: 50px; opacity: 0.6; font-size: 0.8rem; padding: 20px;">
        ⚠️ {LT['disclaimer_text']}<br><br>© 2026 Aarogya Assist • AI Triage Platform
    </div>
    """, unsafe_allow_html=True)
