import streamlit as st
import google.generativeai as genai
import pandas as pd
from typing import List, Dict, Any
import json
import os
import re
from datetime import datetime
import logging
from pathlib import Path
import asyncio
import io
from collections import deque
import time
from datetime import timedelta
import docx

def read_file_content(file):
    """Read content from either TXT or DOCX file"""
    try:
        # Get file extension
        file_extension = file.name.lower().split('.')[-1]
        
        if file_extension == 'txt':
            return file.read().decode('utf-8')
        elif file_extension == 'docx':
            # Save the uploaded file temporarily
            doc_bytes = file.read()
            with open("temp.docx", "wb") as temp_file:
                temp_file.write(doc_bytes)
            
            # Read the DOCX file
            doc = docx.Document("temp.docx")
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Clean up temporary file
            import os
            os.remove("temp.docx")
            
            return content
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        raise Exception(f"Error reading file {file.name}: {str(e)}")

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
        
        # Clean up old requests
        while self.requests and (now - self.requests[0]) > timedelta(minutes=1):
            self.requests.popleft()
        while self.tokens and (now - self.tokens[0][0]) > timedelta(minutes=1):
            self.tokens.popleft()
        while self.daily_requests and (now - self.daily_requests[0]) > timedelta(days=1):
            self.daily_requests.popleft()
            
        # Check limits
        if (len(self.requests) >= self.rpm_limit or  # RPM limit
            sum(tokens for _, tokens in self.tokens) + token_count > self.tpm_limit or  # TPM limit
            len(self.daily_requests) >= self.rpd_limit):  # RPD limit
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
        name_match = re.search(r'Interview\s*[-–—]\s*([^\\n]+)', content)
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
        text = re.sub(r'\*\*', '', text)  # Remove markdown-style formatting
        return text.strip()

    def clean_gemini_response(self, response_text: str) -> str:
        """Clean and format Gemini API response to ensure valid JSON"""
        try:
            # Find the first occurrence of a JSON-like structure
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start == -1 or json_end == -1:
                self.logger.error("No JSON structure found in response")
                return None
                
            # Extract the JSON part
            json_str = response_text[json_start:json_end + 1]
            
            # Clean up common formatting issues
            json_str = re.sub(r'```json\s*', '', json_str)  # Remove JSON code block markers
            json_str = re.sub(r'```\s*', '', json_str)      # Remove any remaining code block markers
            
            # Validate JSON structure
            try:
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
            estimated_tokens = len(prompt.split()) # rough estimation
            
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
            
            # Add metadata
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
        
        # Ensure all required keys exist
        for key, value in template.items():
            if key not in analysis:
                analysis[key] = value
        
        return analysis

    def display_analysis_results(self, analysis: Dict):
        """Display analysis results in Streamlit with error handling"""
        try:
            st.subheader(f"Analysis for {analysis.get('interviewee', {}).get('name', 'Unknown')}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Career Prediction")
                prediction = analysis.get('career_prediction', {})
                st.write(f"Predicted Path: {prediction.get('path', 'Unknown')}")
                st.write(f"Confidence: {prediction.get('confidence', 0)}/10")
                st.write("Rationale:", prediction.get('rationale', 'Not available'))
            
            with col2:
                st.write("Sentiment Analysis")
                sentiment = analysis.get('sentiment_analysis', {})
                st.write(f"Academia: {self._safe_calculate_sentiment(sentiment.get('academia', {}))}")
                st.write(f"Industry: {self._safe_calculate_sentiment(sentiment.get('industry', {}))}")
            
            st.write("Key Themes")
            themes = analysis.get('themes', [])
            if isinstance(themes, list):
                for theme in themes:
                    if isinstance(theme, dict):
                        st.write(f"- {theme.get('name', '')}: {theme.get('description', '')}")
            
            st.write("Motivations")
            motivations = analysis.get('motivations', {})
            st.write("Primary:", ', '.join(motivations.get('primary', [])))
            st.write("Intrinsic:", ', '.join(motivations.get('intrinsic', [])))
            st.write("Extrinsic:", ', '.join(motivations.get('extrinsic', [])))
            
            st.write("Risk Assessment")
            risk = analysis.get('risk_assessment', {})
            st.write(f"Level: {risk.get('level', 'Unknown')}")
            st.write(risk.get('description', 'Not available'))
            
        except Exception as e:
            st.error(f"Error displaying analysis: {str(e)}")


    def generate_summary_table(self, analyses: List[Dict]) -> pd.DataFrame:
        """Generate a summary table from all analyses"""
        summary_data = []
        
        for analysis in analyses:
            if not analysis:
                continue
                
            try:
                # Safely extract data with default values
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

def main():
    st.set_page_config(
        page_title="Interview Analysis App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        .upload-text {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .rate-limit-warning {
            color: #ff9800;
            padding: 1rem;
            border: 1px solid #ff9800;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.title("Interview Analysis App")
    st.markdown("""
        This app analyzes interview transcripts to predict career paths and provide detailed insights.
        Upload your interview transcripts to get started.
    """)
    
    # Rate limit info
    st.sidebar.markdown("""
        ### API Rate Limits
        - 2 requests per minute
        - 32,000 tokens per minute
        - 50 requests per day
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Your Gemini API key is required for analysis"
        )
        
        # Analysis settings
        st.subheader("Analysis Settings")
        show_details = st.checkbox("Show detailed analysis", value=True)
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=1,
            max_value=10,
            value=5,
            help="Minimum confidence score for predictions"
        )

    if not api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar to continue")
        st.stop()
    
    # Initialize session state if needed
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()

    try:
        # Initialize analyzer
        analyzer = InterviewAnalyzer(api_key)
        
        # File upload section
        st.header("Upload Interview Transcripts")
        uploaded_files = st.file_uploader(
            "Upload your interview transcripts",
            accept_multiple_files=True,
            type=['txt', 'docx'],
            help="Supported formats: TXT, DOCX"
        )
        
        if uploaded_files:
            # Process new files
            for file in uploaded_files:
                file_id = f"{file.name}_{file.size}"
                
                # Skip if already processed
                if file_id in st.session_state.processed_files:
                    continue
                
                with st.expander(f"Processing {file.name}", expanded=True):
                    try:
                        st.info(f"📄 Reading file: {file.name}")
                        
                        # Check rate limits
                        while not st.session_state.rate_limiter.can_make_request():
                            wait_time = st.session_state.rate_limiter.wait_time()
                            st.warning(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                            time.sleep(wait_time + 1)
                        
                        # Read file content
                        content = read_file_content(file)
                        # content = file.read().decode('utf-8')
                        
                        # Extract and analyze data
                        interview_data = analyzer.extract_interview_data(content, file.name)
                        st.write(f"🔍 Analyzing interview for: {interview_data['name']}")
                        
                        # Show progress during analysis
                        with st.spinner('Analyzing interview...'):
                            analysis = analyzer.analyze_transcript(interview_data)
                            st.session_state.rate_limiter.add_request()
                        
                        if analysis:
                            st.session_state.analyses.append(analysis)
                            st.session_state.processed_files.add(file_id)
                            st.success(f"✅ Analysis complete for {interview_data['name']}")
                        else:
                            st.error(f"❌ Failed to analyze {file.name}")
                            
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                        continue
            
            # Display results if we have analyses
            if st.session_state.analyses:
                st.header("Analysis Results")
                
                try:
                    # Create summary table
                    summary_df = analyzer.generate_summary_table(st.session_state.analyses)
                    
                    # Filter by confidence threshold
                    if confidence_threshold > 1:
                        summary_df = summary_df[summary_df['Confidence Score'] >= confidence_threshold]
                    
                    # Display summary
                    st.subheader("Summary of Analyses")
                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download section
                    st.subheader("Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON download
                        json_str = json.dumps(st.session_state.analyses, indent=2)
                        st.download_button(
                            label="📥 Download Detailed Analysis (JSON)",
                            data=json_str,
                            file_name=f"interview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download the complete analysis results in JSON format"
                        )
                    
                    with col2:
                        # CSV download
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary (CSV)",
                            data=csv,
                            file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download the summary table in CSV format"
                        )
                    
                    # Detailed analysis section
                    if show_details:
                        st.header("Detailed Analysis")
                        for idx, analysis in enumerate(st.session_state.analyses, 1):
                            with st.expander(
                                f"📊 Analysis {idx}: {analysis.get('interviewee', {}).get('name', 'Unknown')}",
                                expanded=False
                            ):
                                analyzer.display_analysis_results(analysis)
                    
                    # Clear results button
                    if st.button("🗑️ Clear All Results"):
                        st.session_state.analyses = []
                        st.session_state.processed_files = set()
                        st.session_state.rate_limiter = RateLimiter()  # Reset rate limiter
                        st.experimental_rerun()
                        
                except Exception as e:
                    st.error("Error displaying results")
                    st.exception(e)
        
        else:
            # Show instructions when no files are uploaded
            st.info("👆 Upload your interview transcripts to begin analysis")
            st.markdown("""
                ### Instructions:
                1. Enter your Gemini API key in the sidebar
                2. Upload one or more interview transcript files
                3. Wait for the analysis to complete
                4. View results and download reports
                
                Supported file formats: `.txt`, `.docx`
            """)
    
    except Exception as e:
        st.error("Critical Error")
        st.exception(e)
        st.warning("Please refresh the page and try again")

if __name__ == "__main__":
    main()