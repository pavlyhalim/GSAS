import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import pandas as pd
from typing import List, Dict, Any
import json
import os
import re
from datetime import datetime, timedelta
import logging
from pathlib import Path
import asyncio
from collections import deque
import time
import docx

# Utility Functions
def read_file_content(file):
    """Read content from either TXT or DOCX file"""
    try:
        file_extension = file.name.lower().split('.')[-1]
        if file_extension == 'txt':
            return file.read().decode('utf-8')
        elif file_extension == 'docx':
            doc_bytes = file.read()
            with open("temp.docx", "wb") as temp_file:
                temp_file.write(doc_bytes)
            doc = docx.Document("temp.docx")
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            os.remove("temp.docx")
            return content
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise Exception(f"Error reading file {file.name}: {str(e)}")

# Rate Limiter Class
class RateLimiter:
    def __init__(self, rpm_limit=2, tpm_limit=32000, rpd_limit=50):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.requests = deque()
        self.tokens = deque()
        self.daily_requests = deque()

    def can_make_request(self, token_count=0):
        now = datetime.now()
        while self.requests and (now - self.requests[0]) > timedelta(minutes=1):
            self.requests.popleft()
        while self.tokens and (now - self.tokens[0][0]) > timedelta(minutes=1):
            self.tokens.popleft()
        while self.daily_requests and (now - self.daily_requests[0]) > timedelta(days=1):
            self.daily_requests.popleft()
        if (len(self.requests) >= self.rpm_limit or 
            sum(tokens for _, tokens in self.tokens) + token_count > self.tpm_limit or 
            len(self.daily_requests) >= self.rpd_limit):
            return False
        return True

    def add_request(self, token_count=0):
        now = datetime.now()
        self.requests.append(now)
        self.tokens.append((now, token_count))
        self.daily_requests.append(now)

    def wait_time(self):
        if not self.requests:
            return 0
        oldest_request = self.requests[0]
        return max(0, 60 - (datetime.now() - oldest_request).total_seconds())

# Interview Analyzer Class
class InterviewAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the analyzer with API key and setup logging"""
        self.setup_logging()
        self.setup_gemini(api_key)
        self.rate_limiter = RateLimiter()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_gemini(self, api_key: str):
        """Setup Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            self.logger.info("Gemini API initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise

    def extract_interview_data(self, content: str, filename: str) -> Dict[str, str]:
        """Extract relevant information from the document content"""
        # Extract name from content using regex
        name_match = re.search(r'Interview\s*[-–—]\s*([^\n]+)', content)
        name = name_match.group(1).strip() if name_match else "Unknown"
        # Clean the content
        cleaned_content = self.clean_text(content)
        return {
            'name': name,
            'content': cleaned_content,
            'source': filename
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove special characters and normalize whitespace
        text = re.sub(r'\\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\*\*', '', text) # Remove markdown-style formatting
        return text.strip()

    def clean_gemini_response(self, response_text: str) -> str:
        """Clean and format Gemini API response to ensure valid JSON"""
        try:
            # Use regex to find the first JSON object
            match = re.search(r'({.*})', response_text, re.DOTALL)
            if not match:
                self.logger.error("No JSON structure found in response")
                return None

            json_str = match.group(1)

            # Remove any Markdown or code block indicators
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)

            # Validate JSON structure
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON structure after cleaning")
            return None
        except Exception as e:
            self.logger.error(f"Error cleaning response: {str(e)}")
            return None

    def create_analysis_prompt(self, interview_data: Dict[str, str]) -> str:
        """Create a structured prompt for analysis"""
        return f"""
You are a career analysis expert. Analyze this interview transcript for {interview_data['name']}
regarding their career path prediction (Academia vs Industry).
Provide your analysis in the following JSON format ONLY, without any additional text or markdown:

{{
"interviewee": {{
    "name": "{interview_data['name']}",
    "source_document": "{interview_data['source']}"
}},
"sentiment_analysis": {{
    "academia": {{
        "research": {{"score": 0.0, "quotes": []}},
        "teaching": {{"score": 0.0, "quotes": []}},
        "publication": {{"score": 0.0, "quotes": []}},
        "grant_writing": {{"score": 0.0, "quotes": []}},
        "mentoring": {{"score": 0.0, "quotes": []}},
        "work_life_balance": {{"score": 0.0, "quotes": []}},
        "collaboration": {{"score": 0.0, "quotes": []}}
    }},
    "industry": {{
        "product_development": {{"score": 0.0, "quotes": []}},
        "business_strategy": {{"score": 0.0, "quotes": []}},
        "management": {{"score": 0.0, "quotes": []}},
        "work_life_balance": {{"score": 0.0, "quotes": []}},
        "financial_rewards": {{"score": 0.0, "quotes": []}}
    }}
}},
"keyword_analysis": {{
    "academia": [],
    "industry": []
}},
"themes": [],
"motivations": {{
    "primary": [],
    "intrinsic": [],
    "extrinsic": [],
    "evidence_quotes": []
}},
"risk_assessment": {{
    "level": "",
    "description": "",
    "supporting_quotes": []
}},
"long_term_goals": {{
    "vision": "",
    "alignment": "",
    "supporting_quotes": []
}},
"career_prediction": {{
    "path": "",
    "confidence": 0,
    "rationale": ""
}}
}}

Here is the interview transcript to analyze:

{interview_data['content']}

Analyze the transcript and fill in the JSON structure with your findings. Ensure all scores are between -1.0 and 1.0,
confidence is between 1 and 10, and include relevant quotes from the text to support your analysis.
Return ONLY the JSON structure without any additional text or formatting.
"""

    def analyze_transcript(self, interview_data: Dict[str, str]) -> Dict:
        """Analyze a single transcript using Gemini API with rate limiting"""
        try:
            prompt = self.create_analysis_prompt(interview_data)
            estimated_tokens = len(prompt.split())  # rough estimation

            # Check rate limits
            while not self.rate_limiter.can_make_request(estimated_tokens):
                wait_time = self.rate_limiter.wait_time()
                st.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)  # Add 1 second buffer

            # Make the API call
            st.info("Making API request...")
            response = self.model.generate_content(prompt)
            self.rate_limiter.add_request(estimated_tokens)

            if not response or not response.text:
                st.error("Received empty response from API")
                return None

            # Clean and parse the response
            cleaned_response = self.clean_gemini_response(response.text)
            if not cleaned_response:
                st.error("Failed to extract valid JSON from API response")
                return None

            # Parse the cleaned JSON
            analysis = json.loads(cleaned_response)
            analysis['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'source_document': interview_data['source']
            }
            return analysis
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            self.logger.error(f"Error analyzing transcript for {interview_data['name']}: {str(e)}")
            return None

    def _validate_analysis_structure(self, analysis: Dict, interview_data: Dict) -> Dict:
        """Validate and fix analysis structure if needed"""
        template = {
            "interviewee": {
                "name": interview_data['name'],
                "source_document": interview_data['source']
            },
            "sentiment_analysis": {
                "academia": {},
                "industry": {}
            },
            "keyword_analysis": {
                "academia": [],
                "industry": []
            },
            "themes": [],
            "motivations": {
                "primary": [],
                "intrinsic": [],
                "extrinsic": [],
                "evidence_quotes": []
            },
            "risk_assessment": {
                "level": "",
                "description": "",
                "supporting_quotes": []
            },
            "long_term_goals": {
                "vision": "",
                "alignment": "",
                "supporting_quotes": []
            },
            "career_prediction": {
                "path": "",
                "confidence": 0,
                "rationale": ""
            },
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "source_document": interview_data['source']
            }
        }

        for key, value in template.items():
            if key not in analysis:
                analysis[key] = value
        return analysis

    def display_analysis_results(self, analysis: Dict):
        """Display analysis results in Streamlit with enhanced visualizations"""
        try:
            st.subheader(f"Analysis for {analysis.get('interviewee', {}).get('name', 'Unknown')}")
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Career Prediction")
                prediction = analysis.get('career_prediction', {})
                st.write(f"**Predicted Path:** {prediction.get('path', 'Unknown')}")
                st.write(f"**Confidence:** {prediction.get('confidence', 0)}/10")
                st.write("**Rationale:**", prediction.get('rationale', 'Not available'))

            with col2:
                st.write("#### Sentiment Analysis")
                sentiment = analysis.get('sentiment_analysis', {})
                st.write(f"**Academia:** {self._safe_calculate_sentiment(sentiment.get('academia', {}))}")
                st.write(f"**Industry:** {self._safe_calculate_sentiment(sentiment.get('industry', {}))}")

            # Radar Chart
            st.write("### Sentiment Analysis Radar Chart")
            radar_fig = self.create_radar_chart(analysis)
            st.plotly_chart(radar_fig)

            # Bar Chart
            st.write("### Sentiment Scores Bar Chart")
            bar_fig = self.create_bar_chart(analysis)
            st.plotly_chart(bar_fig)

            # Pie Chart for Motivations
            st.write("### Motivations Breakdown")
            motivations = analysis.get('motivations', {})
            pie_fig = self.create_pie_chart({
                'Primary': motivations.get('primary', []),
                'Intrinsic': motivations.get('intrinsic', []),
                'Extrinsic': motivations.get('extrinsic', [])
            }, "Motivations Distribution")
            st.plotly_chart(pie_fig)

            # Themes
            st.write("### Key Themes")
            themes = analysis.get('themes', [])
            if isinstance(themes, list):
                for theme in themes:
                    if isinstance(theme, dict):
                        st.write(f"- **{theme.get('name', '')}:** {theme.get('description', '')}")

            # Motivations Details
            st.write("### Motivations")
            motivations = analysis.get('motivations', {})
            st.write(f"**Primary:** {', '.join(motivations.get('primary', []))}")
            st.write(f"**Intrinsic:** {', '.join(motivations.get('intrinsic', []))}")
            st.write(f"**Extrinsic:** {', '.join(motivations.get('extrinsic', []))}")

            # Risk Assessment
            st.write("### Risk Assessment")
            risk = analysis.get('risk_assessment', {})
            st.write(f"**Level:** {risk.get('level', 'Unknown')}")
            st.write(risk.get('description', 'Not available'))

            # Long Term Goals
            st.write("### Long Term Goals")
            goals = analysis.get('long_term_goals', {})
            st.write(f"**Vision:** {goals.get('vision', '')}")
            st.write(f"**Alignment:** {goals.get('alignment', '')}")

        except Exception as e:
            st.error(f"Error displaying analysis: {str(e)}")

    def generate_summary_table(self, analyses: List[Dict]) -> pd.DataFrame:
        """Generate a summary table from all analyses"""
        summary_data = []
        for analysis in analyses:
            if not analysis:
                continue
            try:
                themes = []
                if isinstance(analysis.get('themes', []), list):
                    themes = [theme.get('name', '') for theme in analysis['themes'] if isinstance(theme, dict)]
                motivations = []
                if isinstance(analysis.get('motivations', {}), dict):
                    motivations = analysis['motivations'].get('primary', [])
                summary_row = {
                    'Name': analysis.get('interviewee', {}).get('name', 'Unknown'),
                    'Predicted Career Path': analysis.get('career_prediction', {}).get('path', 'Unknown'),
                    'Confidence Score': analysis.get('career_prediction', {}).get('confidence', 0),
                    'Primary Motivations': ', '.join(motivations) if isinstance(motivations, list) else '',
                    'Risk Level': analysis.get('risk_assessment', {}).get('level', 'Unknown'),
                    'Key Themes': ', '.join(themes),
                    'Academia Sentiment': self._safe_calculate_sentiment(analysis.get('sentiment_analysis', {}).get('academia', {})),
                    'Industry Sentiment': self._safe_calculate_sentiment(analysis.get('sentiment_analysis', {}).get('industry', {})),
                    'Source Document': analysis.get('metadata', {}).get('source_document', 'Unknown')
                }
                summary_data.append(summary_row)
            except Exception as e:
                self.logger.error(f"Error processing analysis for summary: {str(e)}")
                st.error(f"Error processing analysis: {str(e)}")
                continue
        return pd.DataFrame(summary_data)

    def _safe_calculate_sentiment(self, sentiment_dict: Dict) -> float:
        """Safely calculate average sentiment score from a sentiment dictionary"""
        try:
            if not isinstance(sentiment_dict, dict):
                return 0.0
            scores = []
            for item in sentiment_dict.values():
                if isinstance(item, dict) and 'score' in item:
                    try:
                        scores.append(float(item['score']))
                    except (ValueError, TypeError):
                        continue
            return round(sum(scores) / len(scores), 2) if scores else 0.0
        except Exception:
            return 0.0

    # Visualization Methods
    def create_radar_chart(self, analysis: Dict) -> go.Figure:
        """Create a radar chart for sentiment analysis"""
        try:
            academia_scores = analysis.get('sentiment_analysis', {}).get('academia', {})
            industry_scores = analysis.get('sentiment_analysis', {}).get('industry', {})

            # Define separate categories for academia and industry
            academia_categories = list(academia_scores.keys())
            industry_categories = list(industry_scores.keys())

            # Combine categories for consistent radar chart
            categories = sorted(list(set(academia_categories + industry_categories)))

            # Prepare values, aligning categories
            academia_values = [academia_scores.get(cat, {}).get('score', 0.0) for cat in categories]
            industry_values = [industry_scores.get(cat, {}).get('score', 0.0) for cat in categories]

            # Create radar chart
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=academia_values,
                theta=categories,
                fill='toself',
                name='Academia'
            ))
            fig.add_trace(go.Scatterpolar(
                r=industry_values,
                theta=categories,
                fill='toself',
                name='Industry'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-1, 1]
                    )
                ),
                showlegend=True,
                title=f"Sentiment Analysis Radar for {analysis['interviewee']['name']}"
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating radar chart: {str(e)}")
            return go.Figure()

    def create_bar_chart(self, analysis: Dict) -> go.Figure:
        """Create a bar chart for sentiment scores"""
        try:
            academia_scores = analysis.get('sentiment_analysis', {}).get('academia', {})
            industry_scores = analysis.get('sentiment_analysis', {}).get('industry', {})

            # Define categories for academia and industry
            academia_categories = list(academia_scores.keys())
            industry_categories = list(industry_scores.keys())

            # Create separate dataframes
            df_academia = pd.DataFrame({
                'Category': academia_categories,
                'Sentiment': [academia_scores[cat]['score'] for cat in academia_categories],
                'Sector': ['Academia'] * len(academia_categories)
            })

            df_industry = pd.DataFrame({
                'Category': industry_categories,
                'Sentiment': [industry_scores[cat]['score'] for cat in industry_categories],
                'Sector': ['Industry'] * len(industry_categories)
            })

            # Combine dataframes
            df = pd.concat([df_academia, df_industry], ignore_index=True)

            fig = px.bar(df, x='Category', y='Sentiment', color='Sector', barmode='group',
                         title=f"Sentiment Scores for {analysis['interviewee']['name']}",
                         range_y=[-1, 1])

            return fig
        except Exception as e:
            self.logger.error(f"Error creating bar chart: {str(e)}")
            return go.Figure()

    def create_pie_chart(self, motivations: Dict[str, List[str]], title: str) -> go.Figure:
        """Create a pie chart for motivations breakdown"""
        try:
            labels = list(motivations.keys())
            values = [len(v) for v in motivations.values()]
            fig = px.pie(names=labels, values=values, title=title)
            return fig
        except Exception as e:
            self.logger.error(f"Error creating pie chart: {str(e)}")
            return go.Figure()

    # Additional Features
    def comparative_dashboard(self):
        """Comparative Analysis Dashboard"""
        st.header("Comparative Analysis")
        selected = st.multiselect("Select Interviewees to Compare", 
                                  options=[a['interviewee']['name'] for a in st.session_state.analyses])

        if len(selected) >= 2:
            selected_analyses = [a for a in st.session_state.analyses if a['interviewee']['name'] in selected]
            fig = go.Figure()

            for analysis in selected_analyses:
                academia_scores = analysis.get('sentiment_analysis', {}).get('academia', {})
                industry_scores = analysis.get('sentiment_analysis', {}).get('industry', {})

                academia_categories = list(academia_scores.keys())
                industry_categories = list(industry_scores.keys())
                categories = sorted(list(set(academia_categories + industry_categories)))

                academia_values = [academia_scores.get(cat, {}).get('score', 0.0) for cat in categories]
                industry_values = [industry_scores.get(cat, {}).get('score', 0.0) for cat in categories]

                # Academia Trace
                fig.add_trace(go.Scatterpolar(
                    r=academia_values,
                    theta=categories,
                    fill='toself',
                    name=f"{analysis['interviewee']['name']} - Academia"
                ))

                # Industry Trace
                fig.add_trace(go.Scatterpolar(
                    r=industry_values,
                    theta=categories,
                    fill='toself',
                    name=f"{analysis['interviewee']['name']} - Industry"
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-1, 1]
                    )
                ),
                showlegend=True,
                title="Comparative Sentiment Analysis Radar Chart"
            )
            st.plotly_chart(fig)
        elif len(selected) == 1:
            st.info("Select at least two interviewees to compare sentiment analyses.")

    def clear_chat(self):
        """Clear the chat history"""
        st.session_state.chat_history = []

    def chat_interface(self):
        """Chat section to ask questions about uploaded files"""
        st.header("📢 Chat with Analysis")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for chat in st.session_state.chat_history:
            if chat['role'] == 'user':
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**Assistant:** {chat['content']}")

        # User input
        user_input = st.text_input("Ask a question about your uploaded files:")

        if st.button("Send") and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = self.handle_chat(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    def handle_chat(self, question: str) -> str:
        """Handle user questions by sending them to the Gemini API with context"""
        try:
            # Compile all analyses into a summary
            summary = ""
            for analysis in st.session_state.analyses:
                summary += f"Interviewee: {analysis.get('interviewee', {}).get('name', 'Unknown')}\n"
                summary += json.dumps(analysis, indent=2) + "\n\n"

            # Create prompt for Gemini
            prompt = f"""
            You are an assistant specialized in analyzing interview data. Based on the following analysis summaries, answer the user's question concisely. Provide your answer in JSON format only, without any additional text or explanations.

            Analysis Summaries:
            {summary}

            User Question: {question}

            JSON Response:
            """

            # Rate limiting
            estimated_tokens = len(prompt.split())
            while not self.rate_limiter.can_make_request(estimated_tokens):
                wait_time = self.rate_limiter.wait_time()
                st.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds before answering your question...")
                time.sleep(wait_time + 1)  # Add 1 second buffer

            # Make API call
            response = self.model.generate_content(prompt)
            self.rate_limiter.add_request(estimated_tokens)

            if not response or not response.text:
                return "I'm sorry, I couldn't retrieve an answer at this time."

            # Clean and parse the response
            cleaned_response = self.clean_gemini_response(response.text)
            if not cleaned_response:
                self.logger.debug(f"Raw API response: {response.text}")
                return "I'm sorry, I couldn't understand the response from the analysis."

            return cleaned_response
        except Exception as e:
            self.logger.error(f"Chat Error: {str(e)}")
            return "An error occurred while processing your request."
# Main Application
def main():
    st.set_page_config(
        page_title="Interview Analysis App",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📝 Interview Analysis App")
    st.markdown("""
    This app analyzes interview transcripts to predict career paths and provide detailed insights.
    Upload your interview transcripts to get started.
    """)

    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Your Gemini API key is required for analysis"
        )

        st.subheader("⚙️ Analysis Settings")
        show_details = st.checkbox("Show detailed analysis", value=True)
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=1,
            max_value=10,
            value=5,
            help="Minimum confidence score for predictions"
        )

        st.markdown("""
        ### 📈 API Rate Limits
        - **2 requests per minute**
        - **32,000 tokens per minute**
        - **50 requests per day**
        """)

        if not api_key:
            st.warning("⚠️ Please enter your Gemini API Key in the sidebar to continue")
            st.stop()

    # Initialize session state
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    try:
        analyzer = InterviewAnalyzer(api_key)
        st.header("📤 Upload Interview Transcripts")
        uploaded_files = st.file_uploader(
            "Upload your interview transcripts",
            accept_multiple_files=True,
            type=['txt', 'docx'],
            help="Supported formats: TXT, DOCX"
        )

        if uploaded_files:
            for file in uploaded_files:
                file_id = f"{file.name}_{file.size}"
                if file_id in st.session_state.processed_files:
                    st.info(f"✅ {file.name} has already been processed.")
                    continue
                with st.expander(f"Processing {file.name}", expanded=True):
                    try:
                        st.info(f"📄 Reading file: {file.name}")
                        content = read_file_content(file)
                        interview_data = analyzer.extract_interview_data(content, file.name)
                        st.write(f"🔍 Analyzing interview for: {interview_data['name']}")
                        with st.spinner('Analyzing interview...'):
                            analysis = analyzer.analyze_transcript(interview_data)
                            if analysis:
                                analyzer._validate_analysis_structure(analysis, interview_data)
                                st.session_state.analyses.append(analysis)
                                st.session_state.processed_files.add(file_id)
                                st.success(f"✅ Analysis complete for {interview_data['name']}")
                            else:
                                st.error(f"❌ Failed to analyze {file.name}")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                        continue

        if st.session_state.analyses:
            st.header("📊 Analysis Results")
            try:
                summary_df = analyzer.generate_summary_table(st.session_state.analyses)
                if confidence_threshold > 1:
                    summary_df = summary_df[summary_df['Confidence Score'] >= confidence_threshold]
                st.subheader("📋 Summary of Analyses")
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Download Options
                st.subheader("💾 Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    json_str = json.dumps(st.session_state.analyses, indent=2)
                    st.download_button(
                        label="📥 Download Detailed Analysis (JSON)",
                        data=json_str,
                        file_name=f"interview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Download the complete analysis results in JSON format"
                    )
                with col2:
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv,
                        file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the summary table in CSV format"
                    )

                # Detailed Analysis with Visualizations
                if show_details:
                    st.header("🔍 Detailed Analysis")
                    for idx, analysis in enumerate(st.session_state.analyses, 1):
                        with st.expander(
                            f"📊 Analysis {idx}: {analysis.get('interviewee', {}).get('name', 'Unknown')}",
                            expanded=False
                        ):
                            analyzer.display_analysis_results(analysis)

                # Comparative Analysis Dashboard
                analyzer.comparative_dashboard()

                # Chat Interface
                analyzer.chat_interface()

                # Clear Results Button
                if st.button("🗑️ Clear All Results and Chat History"):
                    st.session_state.analyses = []
                    st.session_state.processed_files = set()
                    st.session_state.rate_limiter = RateLimiter()
                    st.session_state.chat_history = []
                    st.experimental_rerun()

                # Instructions
                st.header("ℹ️ Instructions")
                st.markdown("""
                ### How to Use:
                
                1. **Enter your Gemini API key** in the sidebar.
                2. **Upload** one or more interview transcript files (`.txt` or `.docx`).
                3. **Wait** for the analysis to complete.
                4. **View** results, explore detailed analyses, and download reports.
                5. **Use the chat section** to ask specific questions about your uploaded files.
                
                **Supported file formats:** `.txt`, `.docx`
                """)

            except Exception as e:
                st.error("Error displaying results")
                st.exception(e)
        else:
            st.info("👆 Upload your interview transcripts to begin analysis")

    except Exception as e:
        st.error("Critical Error")
        st.exception(e)
        st.warning("Please refresh the page and try again")

if __name__ == "__main__":
    main()
