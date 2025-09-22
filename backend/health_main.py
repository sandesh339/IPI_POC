from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from openai import OpenAI
from openai import OpenAIError
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from health_utils import (
    get_district_health_data,
    get_state_wise_indicator_extremes,
    get_border_districts,
    get_districts_within_radius,
    get_districts_by_constraints,
    get_top_bottom_districts,
    get_indicator_change_analysis,
    get_district_performance_comparison,
    get_multi_indicator_performance,
    get_state_multi_indicator_performance,
    get_district_similarity_analysis,
    get_district_classification,
    get_district_classification_change,
    get_db_connection,
    match_indicator_name_to_database,
    get_all_indicators,
    parse_constraint_text
)
import tiktoken
from datetime import datetime
import os

# Constants for conversation management
MAX_HISTORY_MESSAGES = 7  
SYSTEM_MESSAGE = {
    "role": "system", 
    "content": """You are an advanced Health Indicators Analysis Assistant for India Policy Insights data, specializing in district-level health data from 2016 and 2021 surveys.



**DATA CONTEXT:**
- 722 districts across Indian states
- 122 health indicators covering various health outcomes
- 2016 and 2021 data with trend analysis (prevalence_change)
- Population headcount data for 2021
- Indicator directions: "higher_is_better" (like vaccination coverage) vs "lower_is_better" (like disease prevalence)

**FUNCTION USAGE GUIDELINES:**

**For Individual District Queries:**
- Use get_individual_district_data() for specific district analysis
- Handles fuzzy district name matching automatically
- Provides comprehensive indicator data with trend analysis

**For State-wise Comparisons:**
- Use get_state_wise_indicator_extremes() to find best/worst districts per state
- Helps identify intra-state disparities and exemplary districts
- Requires specific indicator name

**For Geographic Analysis:**
- Use get_border_districts() for cross-border health patterns
- Use get_districts_within_radius() for spatial health analysis with raw data
- Both support coordinate-based and district-based center points

**For Top/Bottom Performance Ranking:**
- Use get_top_bottom_districts() to find best/worst performing districts across all states
- Supports single or multiple indicators with proper direction handling
- Includes state filtering and configurable number of districts
- Ideal for identifying exemplary districts and areas needing intervention

**For Indicator Change Analysis:**
- Use get_indicator_change_analysis() to investigate health indicator changes from 2016 to 2021
- Supports country, state, and district level analysis
- Country level: National averages with 10 example districts
- State level: State averages with 5 example districts from that state
- District level: Specific district trend with detailed analysis
- Automatically handles data availability validation and direction interpretation

**For District Similarity/Contrasting Analysis:**
- Use get_district_similarity_analysis() for finding districts with similar or different patterns
- Supports multiple indicators or indicator categories
- "similar" analysis finds districts with comparable performance patterns using statistical clustering
- "different" analysis finds districts with contrasting/diverse patterns using percentile distribution
- **CRITICAL**: Each indicator gets its own SEPARATE set of most relevant districts
- **IMPORTANT**: The function returns detailed analysis text with indicator-specific district lists
- **NEVER summarize**: Always include the full analysis text which contains detailed district breakdowns
- **ALWAYS show**: All indicator-specific districts and their values from the analysis field
- Supports state filtering and customizable number of districts
- Perfect for queries like "districts with contrasting nutrition patterns" or "similar health performance"
- **Response includes**: Complete analysis with each indicator's selected districts, values, and trends

**For Radius-based Analysis:**
- get_districts_within_radius() returns structured raw data, not pre-formatted analysis
- Data includes: district names, health indicators, prevalence values (2016/2021), changes, headcounts
- Use this raw data to provide natural, conversational responses about regional health patterns
- Sort districts by geographic distance from center point
- Interpret changes based on indicator direction (higher_is_better vs lower_is_better)

**RESPONSE STRATEGY:**
1. **Understanding**: Analyze user intent (district focus, indicator focus, geographic scope)
2. **Function Selection**: Choose appropriate function based on query type
3. **Data Analysis**: Process raw results focusing on health trends, changes, and geographic patterns
4. **Natural Interpretation**: For radius queries, use raw data to provide conversational insights about:
   - Which districts are within the specified radius
   - Current health indicator values for each district
   - How indicators have changed from 2016 to 2021 (considering direction)
   - Population-level impacts based on headcount data
   - Geographic patterns and regional health variations
5. **Similarity Analysis Results**: For district similarity analysis, ALWAYS:
   - Present the complete analysis text from the 'analysis' field
   - Show ALL indicator-specific district lists with their values
   - Explain the enhanced algorithm used (statistical vs percentile-based)
   - Highlight how different indicators have different district selections
   - Include the statistical measures (range, diversity, spread)
6. **Contextual Explanation**: Explain health outcomes, improvement directions, and policy relevance
7. **Visualization**: Support mapping and chart visualization for spatial health data

**KEY PRINCIPLES:**
- Work with raw data to provide natural, conversational responses
- Interpret prevalence changes considering indicator direction (higher_is_better vs lower_is_better)
- Use headcount data to explain population-level impacts
- Provide actionable insights for health policy and planning
- Explain data trends and their significance for public health
- Support both technical analysis and accessible explanations
- Focus on geographic patterns and regional variations in radius-based queries

You excel at transforming complex health data into actionable insights for policymakers, health professionals, and researchers."""
}

MAX_TOKENS = 128000  
SAFE_TOKEN_LIMIT = int(MAX_TOKENS * 0.4)  

def count_tokens(messages, model="gpt-4o"):
    """Count tokens in messages with explicit encoding handling"""
    try:
        # Try to get encoding for the specific model
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding for GPT-4 models
        enc = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        for key, value in message.items():
            num_tokens += len(enc.encode(str(value)))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <|start|>assistant
    return num_tokens

def manage_conversation_history(history: list, new_message: dict, model="gpt-4o") -> list:
    """
    Manage conversation history by token count.
    Always keeps the system message and as many recent messages as possible under the token limit.
    """
    # Ensure system message is always present
    if not history or history[0].get("role") != "system":
        history = [SYSTEM_MESSAGE] + history

    # Add the new message
    history.append(new_message)

    # Aggressive trimming to stay under token limit
    while count_tokens(history, model=model) > SAFE_TOKEN_LIMIT and len(history) > 3:
        history.pop(1)
    
    # If still too large, keep only system message and last 2 messages
    if count_tokens(history, model=model) > SAFE_TOKEN_LIMIT and len(history) > 3:
        history = [history[0]] + history[-2:]
    
    # Final safety check - truncate last message if needed
    if count_tokens(history, model=model) > SAFE_TOKEN_LIMIT and len(history) > 1:
        last_message = history[-1]
        if len(last_message.get("content", "")) > 1000:
            last_message["content"] = last_message["content"][:800] + "... [Message truncated]"

    return history

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(
    api_key=os.getenv("OPEN_API_KEY")

)

# Session storage
session_store = {
    "history": []
}

class ChatbotRequest(BaseModel):
    query: str
    history: list[dict] | None = None
    timestamp: int | None = None  # Accept timestamp for cache busting
    
    class Config:
        # Allow extra fields to be ignored instead of causing validation errors
        extra = "ignore"

class SaveVisualizationRequest(BaseModel):
    visualization_data: dict
    metadata: dict | None = None
    
    class Config:
        extra = "ignore"

class GetSavedVisualizationsRequest(BaseModel):
    limit: int = 50
    offset: int = 0
    filter_by: dict | None = None
    
    class Config:
        extra = "ignore"

class ReactionRequest(BaseModel):
    message_id: int | None = None
    reaction: str
    user_query: str | None = None
    assistant_response: str | None = None
    timestamp: str | None = None  # ISO string preferred

    class Config:
        extra = "ignore"

def init_visualization_db():
    """Initialize the PostgreSQL database table for storing visualizations"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create table for saved visualizations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_visualizations (
                id SERIAL PRIMARY KEY,
                visualization_data TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating visualization table: {e}")
        return False

# Initialize the database on startup
try:
    init_visualization_db()
    print("Visualization database initialized successfully")
except Exception as e:
    print(f"Error initializing visualization database: {e}")

@app.get("/")
def home():
    return {"message": "Health Indicators Chatbot Backend is running!"}

@app.post("/chatbot/")
async def chatbot(request: ChatbotRequest):
    try:
        # Test database connection
        try:
            conn = get_db_connection()
            conn.close()
        except psycopg2.Error as e:
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

        user_input = request.query
        print("User query:", user_input)
        session = session_store

        if "history" not in session:
            session["history"] = [SYSTEM_MESSAGE]

        # Add user message to history
        user_message = {"role": "user", "content": user_input}
        session["history"] = manage_conversation_history(session["history"], user_message)

        def execute_function_call(function_name, arguments):
            """Execute a single function call and return the result"""

            # Handle indicator_name parameter - match it to database indicators
            indicator_ids_to_use = None

            # If indicator_name is provided, match it to database indicator
            if arguments.get("indicator_name"):
                matched_indicator = match_indicator_name_to_database(arguments["indicator_name"])
                if matched_indicator:
                    indicator_ids_to_use = [matched_indicator["indicator_id"]]
                    print(f"🎯 Matched '{arguments['indicator_name']}' to '{matched_indicator['indicator_name']}' for {function_name}")
                else:
                    print(f"❌ Could not match indicator name '{arguments['indicator_name']}' for {function_name}")

            if function_name == "get_district_health_data":
                # Handle both single and multiple districts/indicators
                
                # Determine district names - support both single and multiple
                district_names = None
                if arguments.get("district_names"):
                    district_names = arguments["district_names"]
                elif arguments.get("district_name"):
                    district_names = arguments["district_name"]  # Pass as string, function will handle it
                else:
                    return {"error": "Either district_name or district_names must be provided"}
                
                # Handle indicator matching for multiple indicators
                indicator_ids_to_use = None
                if arguments.get("indicator_names"):
                    # Multiple indicators provided
                    indicator_ids_list = []
                    for indicator_name in arguments["indicator_names"]:
                        matched_indicator = match_indicator_name_to_database(indicator_name)
                        if matched_indicator:
                            indicator_ids_list.append(matched_indicator["indicator_id"])
                            print(f"🎯 Matched '{indicator_name}' to '{matched_indicator['indicator_name']}' for district health analysis")
                        else:
                            print(f"❌ Could not match indicator name '{indicator_name}' for district health analysis")
                    if indicator_ids_list:
                        indicator_ids_to_use = indicator_ids_list
                elif arguments.get("indicator_name"):
                    # Single indicator provided (existing logic)
                    matched_indicator = match_indicator_name_to_database(arguments["indicator_name"])
                    if matched_indicator:
                        indicator_ids_to_use = [matched_indicator["indicator_id"]]
                        print(f"🎯 Matched '{arguments['indicator_name']}' to '{matched_indicator['indicator_name']}' for district health analysis")
                    else:
                        print(f"❌ Could not match indicator name '{arguments['indicator_name']}' for district health analysis")
                
                return get_district_health_data(
                    district_names=district_names,
                    indicator_ids=indicator_ids_to_use,
                    year=arguments.get("year", 2021),
                    state_name=arguments.get("state_name")
                )
            
            elif function_name == "get_state_wise_indicator_extremes":
                # Handle both single and multiple indicators with the new function signature
                return get_state_wise_indicator_extremes(
                    indicator_names=arguments.get("indicator_names"),
                    indicator_name=arguments.get("indicator_name"),
                    states=arguments.get("states"),
                    year=arguments.get("year", 2021),
                    include_trend=arguments.get("include_trend", True),
                    min_districts_per_state=arguments.get("min_districts_per_state", 3)
                )
            
            elif function_name == "get_border_districts":
                # Handle both single indicator and multiple indicators
                indicator_ids_to_use = None
                if arguments.get("indicator_names"):
                    # Multiple indicators provided
                    indicator_ids_list = []
                    for indicator_name in arguments["indicator_names"]:
                        matched_indicator = match_indicator_name_to_database(indicator_name)
                        if matched_indicator:
                            indicator_ids_list.append(matched_indicator["indicator_id"])
                            print(f"🎯 Matched '{indicator_name}' to '{matched_indicator['indicator_name']}' for border districts analysis")
                        else:
                            print(f"❌ Could not match indicator name '{indicator_name}' for border districts analysis")
                    if indicator_ids_list:
                        indicator_ids_to_use = indicator_ids_list
                elif arguments.get("indicator_name"):
                    # Single indicator provided (backward compatibility)
                    matched_indicator = match_indicator_name_to_database(arguments["indicator_name"])
                    if matched_indicator:
                        indicator_ids_to_use = [matched_indicator["indicator_id"]]
                        print(f"🎯 Matched '{arguments['indicator_name']}' to '{matched_indicator['indicator_name']}' for border districts analysis")
                    else:
                        print(f"❌ Could not match indicator name '{arguments['indicator_name']}' for border districts analysis")

                return get_border_districts(
                    state1=arguments["state1"],
                    state2=arguments.get("state2"),
                    indicator_ids=indicator_ids_to_use,
                    year=arguments.get("year", 2021),
                    include_boundary_data=arguments.get("include_boundary_data", True),
                    include_state_comparison=arguments.get("include_state_comparison", True)
                )

            elif function_name == "get_districts_within_radius":
                print(f"🔍 get_districts_within_radius called with:")
                print(f"  center_point: {arguments['center_point']}")
                print(f"  radius_km: {arguments['radius_km']}")
                print(f"  indicator_ids_to_use: {indicator_ids_to_use}")
                print(f"  max_districts: {arguments.get('max_districts', 50)}")
                print(f"  arguments: {arguments}")
                
                try:
                    result = get_districts_within_radius(
                        center_point=arguments["center_point"],
                        radius_km=arguments["radius_km"],
                        indicator_ids=indicator_ids_to_use,
                        max_districts=arguments.get("max_districts", 50),
                        include_boundary_data=arguments.get("include_boundary_data", True)
                    )
                    
                    print(f"📊 get_districts_within_radius result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    print(f"  result keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                    print(f"  districts count: {len(result.get('districts', []))}")
                    
                    if result.get('districts') and len(result['districts']) > 0:
                        sample = result['districts'][0]
                        print(f"  sample district: {sample['district_name']}")
                        print(f"  sample district type: {type(sample)}")
                        print(f"  sample has indicators: {'indicators' in sample}")
                        print(f"  sample keys: {list(sample.keys())}")
                        if 'indicators' in sample and sample['indicators']:
                            print(f"  sample indicators count: {len(sample['indicators'])}")
                            print(f"  first indicator: {sample['indicators'][0] if sample['indicators'] else 'None'}")
                        else:
                            print(f"  sample indicators: {sample.get('indicators', 'NOT_FOUND')}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_districts_within_radius:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}
                
            elif function_name == "get_districts_by_constraints":
                print(f"🎯 get_districts_by_constraints called with:")
                print(f"  constraints: {arguments.get('constraints', [])}")
                print(f"  constraint_text: {arguments.get('constraint_text', 'None')}")
                print(f"  year: {arguments.get('year', 2021)}")
                print(f"  states: {arguments.get('states', 'None')}")
                print(f"  max_districts: {arguments.get('max_districts', 100)}")
                
                try:
                    # Handle both structured constraints and natural language text
                    constraints = arguments.get("constraints")
                    
                    # If no structured constraints but constraint_text is provided, parse it
                    if not constraints and arguments.get("constraint_text"):
                        constraints = parse_constraint_text(arguments["constraint_text"])
                        print(f"🔍 Parsed constraint_text into {len(constraints)} constraints")
                    
                    if not constraints:
                        return {"error": "Either 'constraints' array or 'constraint_text' must be provided"}
                    
                    result = get_districts_by_constraints(
                        constraints=constraints,
                        year=arguments.get("year", 2021),
                        states=arguments.get("states"),
                        max_districts=arguments.get("max_districts", 100),
                        include_boundary_data=arguments.get("include_boundary_data", True)
                    )
                    
                    print(f"📊 get_districts_by_constraints result:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  districts found: {result.get('total_districts_found', 0)}")
                        print(f"  constraints applied: {len(result.get('constraints_applied', []))}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_districts_by_constraints:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_top_bottom_districts":
                print(f"🏆 get_top_bottom_districts called with:")
                print(f"  indicator_names: {arguments.get('indicator_names', 'None')}")
                print(f"  indicator_name: {arguments.get('indicator_name', 'None')}")
                print(f"  n_districts: {arguments.get('n_districts', 10)}")
                print(f"  performance_type: {arguments.get('performance_type', 'top')}")
                print(f"  states: {arguments.get('states', 'None')}")
                print(f"  year: {arguments.get('year', 2021)}")
                
                try:
                    result = get_top_bottom_districts(
                        indicator_names=arguments.get("indicator_names"),
                        indicator_name=arguments.get("indicator_name"),
                        n_districts=arguments.get("n_districts", 10),
                        performance_type=arguments.get("performance_type", "top"),
                        states=arguments.get("states"),
                        year=arguments.get("year", 2021),
                        include_boundary_data=arguments.get("include_boundary_data", True)
                    )
                    
                    print(f"📊 get_top_bottom_districts result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  districts found: {result.get('total_districts_found', 0)}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_top_bottom_districts:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_indicator_change_analysis":
                print(f"📈 get_indicator_change_analysis called with:")
                print(f"  indicator_name: {arguments.get('indicator_name', 'None')}")
                print(f"  analysis_level: {arguments.get('analysis_level', 'country')}")
                print(f"  location_name: {arguments.get('location_name', 'None')}")
                
                try:
                    result = get_indicator_change_analysis(
                        indicator_name=arguments.get("indicator_name"),
                        analysis_level=arguments.get("analysis_level", "country"),
                        location_name=arguments.get("location_name"),
                        include_boundary_data=arguments.get("include_boundary_data", True)
                    )
                    
                    print(f"📊 get_indicator_change_analysis result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        print(f"  analysis_level: {result.get('analysis_level', 'unknown')}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_indicator_change_analysis:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_district_performance_comparison":
                print(f"🆚 get_district_performance_comparison called with:")
                print(f"  district_names: {arguments.get('district_names', 'None')}")
                print(f"  indicator_names: {arguments.get('indicator_names', 'None')}")
                print(f"  comparison_type: {arguments.get('comparison_type', 'national')}")
                print(f"  year: {arguments.get('year', 2021)}")
                
                try:
                    result = get_district_performance_comparison(
                        district_names=arguments.get("district_names", []),
                        indicator_names=arguments.get("indicator_names", []),
                        comparison_type=arguments.get("comparison_type", "national"),
                        year=arguments.get("year", 2021),
                        include_boundary_data=arguments.get("include_boundary_data", True)
                    )
                    
                    print(f"📊 get_district_performance_comparison result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        print(f"  total_districts: {result.get('total_districts', 0)}")
                        print(f"  total_indicators: {result.get('total_indicators', 0)}")
                        print(f"  comparison_type: {result.get('comparison_type', 'unknown')}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_district_performance_comparison:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_multi_indicator_performance":
                print(f"📊 get_multi_indicator_performance called with:")
                print(f"  district_names: {arguments.get('district_names', 'None')}")
                print(f"  category_name: {arguments.get('category_name', 'None')}")
                print(f"  indicator_names: {arguments.get('indicator_names', 'None')}")
                print(f"  performance_type: {arguments.get('performance_type', 'specific')}")
                print(f"  n_districts: {arguments.get('n_districts', 10)}")
                print(f"  year: {arguments.get('year', 2021)}")
                
                try:
                    result = get_multi_indicator_performance(
                        district_names=arguments.get("district_names"),
                        category_name=arguments.get("category_name"),
                        indicator_names=arguments.get("indicator_names"),
                        performance_type=arguments.get("performance_type", "specific"),
                        n_districts=arguments.get("n_districts", 10),
                        year=arguments.get("year", 2021),
                        include_boundary_data=arguments.get("include_boundary_data", True)
                    )
                    
                    print(f"📊 get_multi_indicator_performance result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        print(f"  total_districts: {result.get('total_districts', 0)}")
                        print(f"  total_indicators: {result.get('total_indicators', 0)}")
                        print(f"  performance_type: {result.get('performance_type', 'unknown')}")
                        print(f"  category_name: {result.get('category_name', 'None')}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_multi_indicator_performance:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_state_multi_indicator_performance":
                print(f"🏛️ get_state_multi_indicator_performance called with:")
                print(f"  state_names: {arguments.get('state_names', 'None')}")
                print(f"  category_name: {arguments.get('category_name', 'None')}")
                print(f"  indicator_names: {arguments.get('indicator_names', 'None')}")
                print(f"  performance_type: {arguments.get('performance_type', 'top')}")
                print(f"  n_districts: {arguments.get('n_districts', 5)}")
                print(f"  year: {arguments.get('year', 2021)}")
                
                try:
                    result = get_state_multi_indicator_performance(
                        state_names=arguments.get("state_names"),
                        category_name=arguments.get("category_name"),
                        indicator_names=arguments.get("indicator_names"),
                        performance_type=arguments.get("performance_type", "top"),
                        n_districts=arguments.get("n_districts", 5),
                        year=arguments.get("year", 2021),
                        include_boundary_data=arguments.get("include_boundary_data", True),
                        query_hint=arguments.get("query_hint")
                    )
                    
                    print(f"📊 get_state_multi_indicator_performance result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        print(f"  total_states: {result.get('total_states', 0)}")
                        print(f"  total_districts: {result.get('total_districts', 0)}")
                        print(f"  total_indicators: {result.get('total_indicators', 0)}")
                        print(f"  performance_type: {result.get('performance_type', 'unknown')}")
                        print(f"  category_name: {result.get('category_name', 'None')}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_state_multi_indicator_performance:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_district_similarity_analysis":
                print(f"🔍 get_district_similarity_analysis called with:")
                print(f"  indicator_names: {arguments.get('indicator_names', 'None')}")
                print(f"  category_name: {arguments.get('category_name', 'None')}")
                print(f"  analysis_type: {arguments.get('analysis_type', 'similar')}")
                print(f"  state_names: {arguments.get('state_names', 'None')}")
                print(f"  n_districts: {arguments.get('n_districts', 20)}")
                print(f"  year: {arguments.get('year', 2021)}")
                
                try:
                    result = get_district_similarity_analysis(
                        indicator_names=arguments.get("indicator_names"),
                        category_name=arguments.get("category_name"),
                        analysis_type=arguments.get("analysis_type", "similar"),
                        state_names=arguments.get("state_names"),
                        n_districts=arguments.get("n_districts", 20),
                        year=arguments.get("year", 2021),
                        include_boundary_data=arguments.get("include_boundary_data", True)
                    )
                    
                    print(f"📊 get_district_similarity_analysis result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        print(f"  analysis_type: {result.get('analysis_type', 'unknown')}")
                        print(f"  total_districts: {result.get('total_districts', 0)}")
                        print(f"  total_indicators: {result.get('total_indicators', 0)}")
                        print(f"  category_name: {result.get('category_name', 'None')}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_district_similarity_analysis:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_district_classification":
                print(f"📊 get_district_classification called with:")
                print(f"  indicator_name: {arguments.get('indicator_name', 'None')}")
                print(f"  state_names: {arguments.get('state_names', 'None')}")
                print(f"  year: {arguments.get('year', 2021)}")
                
                try:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                    
                    # Use ThreadPoolExecutor with timeout for better control
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            get_district_classification,
                            indicator_name=arguments.get("indicator_name"),
                            state_names=arguments.get("state_names"),
                            year=arguments.get("year", 2021),
                            include_boundary_data=arguments.get("include_boundary_data", True)
                        )
                        
                        try:
                            result = future.result(timeout=120)  # 2 minute timeout
                        except FuturesTimeoutError:
                            return {
                                "error": "District classification timed out after 2 minutes. Please try with fewer states or contact support.",
                                "response_type": "error"
                            }
                    
                    print(f"📊 get_district_classification result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        print(f"  total_districts: {result.get('statistics', {}).get('total_districts', 0)}")
                        print(f"  indicator_name: {result.get('indicator_info', {}).get('indicator_name', 'unknown')}")
                        print(f"  class_counts: {result.get('statistics', {}).get('class_counts', {})}")
                        if 'error' in result:
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_district_classification:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            elif function_name == "get_district_classification_change":
                print(f"📈 get_district_classification_change called with:")
                print(f"  indicator_name: {arguments.get('indicator_name', 'None')}")
                print(f"  state_names: {arguments.get('state_names', 'None')}")
                
                try:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                    
                    # Use ThreadPoolExecutor with timeout for better control
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            get_district_classification_change,
                            indicator_name=arguments.get("indicator_name"),
                            state_names=arguments.get("state_names"),
                            include_boundary_data=arguments.get("include_boundary_data", True)
                        )
                        try:
                            result = future.result(timeout=120)  # 2 minute timeout
                        except FuturesTimeoutError:
                            return {
                                "error": "District change classification timed out after 2 minutes. Please try with fewer states or contact support.",
                                "response_type": "error"
                            }
                    
                    print(f"📈 get_district_classification_change result SUCCESS:")
                    print(f"  result type: {type(result)}")
                    if isinstance(result, dict):
                        print(f"  result keys: {list(result.keys())}")
                        print(f"  response_type: {result.get('response_type', 'unknown')}")
                        print(f"  total_districts: {result.get('statistics', {}).get('total_districts', 0)}")
                        
                        if result.get('response_type') == 'error':
                            print(f"  error: {result['error']}")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ ERROR in get_district_classification_change:")
                    print(f"  Error type: {type(e)}")
                    print(f"  Error message: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Function execution failed: {str(e)}"}

            else:
                return {"error": f"Unknown function: {function_name}"}

        # Initial call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=session["history"],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_district_health_data",
                        "description": """Get comprehensive health indicator data for one or multiple districts. ALWAYS use district_names array for multiple districts in a SINGLE function call.
                        
                        IMPORTANT: When user asks about multiple districts, use ONE function call with district_names array - DO NOT make separate calls for each district.
                        
                        Features:
                        - Supports single or multiple districts analysis in ONE call
                        - Supports single or multiple indicators analysis
                        - Automatic district name resolution with fuzzy matching
                        - Complete health indicators data with 2016 and 2021 values
                        - Trend analysis and interpretation
                        - Boundary data for map visualization
                        - Comparative analysis across districts when multiple districts provided
                        
                        Use Cases:
                        1. Single District: Use district_name parameter
                        2. Multiple Districts: Use district_names array parameter in ONE call
                        3. Multiple Indicators: Use indicator_names array parameter
                        
                        CORRECT Examples:
                        - Multiple districts: {"district_names": ["Mumbai", "Delhi"], "indicator_names": ["diabetes", "vaccination"]}
                        - Single district: {"district_name": "Mumbai", "indicator_names": ["diabetes"]}
                        
                        INCORRECT: Making separate calls for each district - always use district_names array for multiple districts.
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "district_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of district names (will be resolved using fuzzy matching). Use this for multiple districts."
                                },
                                "district_name": {
                                    "type": "string",
                                    "description": "Single district name (will be resolved using fuzzy matching). Use this OR district_names, not both."
                                },
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of health indicator names (can be misspelled or described). Use this for multiple indicators."
                                },
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Single health indicator name (can be misspelled or described). Use this OR indicator_names. If neither provided, returns all indicators."
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year for analysis (2016 or 2021)",
                                    "enum": [2016, 2021]
                                },
                                "state_name": {
                                    "type": "string",
                                    "description": "Optional state name for validation (applies to all districts)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_state_wise_indicator_extremes",
                        "description": """Get the best and worst performing districts for single or multiple health indicators across states. Use this for intra-state comparisons and extreme value analysis.
                        
                        Features:
                        - Supports single or multiple indicators analysis
                        - Identifies top and bottom performing districts within each state or specified states
                        - Automatically handles indicator direction (higher_is_better vs lower_is_better) from database
                        - Calculates intra-state disparities and performance gaps
                        - Helps find exemplary districts and those needing support
                        - State filtering: analyze all states or specific states only
                        - Includes boundary data for visualization
                        
                        Indicator Direction Intelligence:
                        - Uses database indicator_direction field to determine if high values are good or bad
                        - For indicators like "diabetes prevalence": lower values = better performance
                        - For indicators like "vaccination coverage": higher values = better performance
                        - Automatically ranks districts accordingly
                        
                        Examples:
                        - "Show me the best and worst districts for vaccination coverage in each state"
                        - "Which districts have highest and lowest malnutrition rates per state?"
                        - "State-wise extremes for diabetes and blood pressure in Maharashtra and Gujarat"
                        - "Best and worst performing districts for multiple indicators across all states"
                        - "Extreme values for respiratory infections in northern states"
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of health indicator names (can be misspelled or described, e.g., ['diabetes', 'blood pressure', 'vaccination']). Use this for multiple indicators."
                                },
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Single health indicator name (can be misspelled or described, e.g., 'diarrhea', 'watery pooping', 'chest infections'). Use this OR indicator_names."
                                },
                                "states": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of specific state names to analyze (e.g., ['Maharashtra', 'Gujarat', 'Tamil Nadu']). If not provided, analyzes all states."
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year for analysis (2016 or 2021)",
                                    "enum": [2016, 2021]
                                },
                                "include_trend": {
                                    "type": "boolean",
                                    "description": "Whether to include trend analysis (default: true)"
                                },
                                "min_districts_per_state": {
                                    "type": "integer",
                                    "description": "Minimum number of districts required per state (default: 3)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_border_districts",
                        "description": """Find districts that share borders with a specific state and analyze their health performance with comprehensive comparison capabilities. Use for cross-border health analysis.

                        Features:
                        - Spatial analysis to identify all border districts
                        - Supports both bilateral (two states) and multilateral (one state vs all neighbors) analysis
                        - Single or multiple health indicator analysis
                        - State-level comparison for benchmarking border district performance
                        - Useful for cross-state health coordination and policy planning

                        Examples:
                        - "Show districts on the border of Maharashtra"
                        - "Border districts between Karnataka and Tamil Nadu with health data"
                        - "Compare vaccination rates in districts bordering Delhi vs Delhi state average"
                        - "Multiple indicators analysis for border districts of Gujarat"
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "state1": {
                                    "type": "string",
                                    "description": "Primary state name (required)"
                                },
                                "state2": {
                                    "type": "string",
                                    "description": "Second state name (optional). If provided, finds districts between these two states."
                                },
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of health indicator names (can be misspelled or described). If not provided, returns all indicators."
                                },
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Single health indicator name (can be misspelled or described). Use this OR indicator_names."
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year for analysis (2016 or 2021)",
                                    "enum": [2016, 2021]
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data (default: true)"
                                },
                                "include_state_comparison": {
                                    "type": "boolean",
                                    "description": "Whether to include comparison with state averages (default: true)"
                                }
                            },
                            "required": ["state1"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_districts_within_radius",
                        "description": """Find all districts within a specified radius from a center point and return their raw health indicator data for natural interpretation. Use when users want detailed health data for districts in a geographic area.

                        Returns structured data including:
                        - District names and states
                        - Health indicator names and directions
                        - 2021 prevalence values
                        - 2016 baseline values
                        - Calculated change (2016-2021)
                        - Headcount data for affected populations
                        - Geographic distance information

                        Use this function when users ask about:
                        - Districts within a radius of a location
                        - Health indicators for geographic areas
                        - Prevalence changes over time in specific regions
                        - Population-level health data for planning

                        The raw data returned allows for natural, conversational responses about health patterns and trends.
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "center_point": {
                                    "type": "string",
                                    "description": "Either a district name (e.g., 'Delhi') or coordinates as 'lat,lng' (e.g., '28.6139,77.2090')"
                                },
                                "radius_km": {
                                    "type": "number",
                                    "description": "Radius in kilometers to search within",
                                    "minimum": 1,
                                    "maximum": 1000
                                },
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Optional name of specific health indicator (can be misspelled or described)"
                                },
                                "max_districts": {
                                    "type": "integer",
                                    "description": "Maximum number of districts to return (default: 50)",
                                    "minimum": 5,
                                    "maximum": 100
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data (default: true)"
                                }
                            },
                            "required": ["center_point", "radius_km"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_districts_by_constraints",
                        "description": """Find districts that meet multiple constraints on health indicator values. This function enables complex multi-criteria analysis for targeted policy interventions and resource allocation.

                        Key Features:
                        - Multiple constraint support with AND logic (all constraints must be met)
                        - Flexible operator support: >, >=, <, <=, =, != (also supports text: gt, gte, lt, lte, eq, neq)
                        - Automatic indicator name matching with fuzzy matching and AI assistance
                        - State filtering for regional analysis
                        - Comprehensive analysis and visualization
                        - Boundary data for mapping

                        Use Cases:
                        - "Find districts with malnutrition > 30 and vaccination < 50"
                        - "Districts where diabetes > 15 and obesity >= 25"
                        - "Show districts with good performance: vaccination >= 90 and diarrhea <= 5"
                        - "High-need districts: multiple indicators above critical thresholds"

                        Examples:
                        1. Multiple thresholds: [{"indicator_name": "diabetes", "operator": ">", "value": 15}, {"indicator_name": "vaccination", "operator": "<", "value": 70}]
                        2. Natural language: {"constraint_text": "diabetes > 15 and vaccination < 70"}
                        3. Mixed criteria: [{"indicator_name": "malnutrition", "operator": ">=", "value": 30}, {"indicator_name": "institutional childbirth", "operator": "<=", "value": 60}]

                        Perfect for:
                        - Identifying districts needing urgent intervention
                        - Finding exemplary districts that meet multiple criteria
                        - Resource allocation based on multiple health outcomes
                        - Policy evaluation across multiple indicators
                        - Comparative analysis of districts with similar challenges
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "constraints": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "indicator_name": {
                                                "type": "string",
                                                "description": "Health indicator name (can be misspelled or described, e.g., 'diabetes', 'vaccination coverage', 'diarrhea')"
                                            },
                                            "operator": {
                                                "type": "string",
                                                "enum": [">", ">=", "<", "<=", "=", "!=", "gt", "gte", "lt", "lte", "eq", "neq"],
                                                "description": "Comparison operator: >, >=, <, <=, = (equal), != (not equal). Also accepts text: gt, gte, lt, lte, eq, neq"
                                            },
                                            "value": {
                                                "type": "number",
                                                "description": "Threshold value to compare against (e.g., 40, 5.5, 90)"
                                            }
                                        },
                                        "required": ["indicator_name", "operator", "value"]
                                    },
                                    "description": "Array of constraint objects. Each constraint specifies an indicator, operator, and threshold value. Use this OR constraint_text."
                                },
                                "constraint_text": {
                                    "type": "string",
                                    "description": "Natural language constraint text (e.g., 'diabetes > 15 and vaccination < 70', 'malnutrition >= 30, diarrhea <= 5'). Use this OR constraints array."
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year for analysis (2016 or 2021)",
                                    "enum": [2016, 2021]
                                },
                                "states": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of state names to filter by (e.g., ['Maharashtra', 'Tamil Nadu'])"
                                },
                                "max_districts": {
                                    "type": "integer",
                                    "description": "Maximum number of districts to return (default: 100)",
                                    "minimum": 5,
                                    "maximum": 500
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping (default: true)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_top_bottom_districts",
                        "description": """Get top/bottom N districts for single or multiple health indicators with state filtering and proper indicator direction handling.

                        This function returns the best or worst performing districts across all states for given health indicators, properly handling indicator direction (higher_is_better vs lower_is_better).

                        Key Features:
                        - Supports single or multiple indicators analysis
                        - Proper handling of indicator direction (higher_is_better vs lower_is_better)
                        - State filtering capability
                        - Configurable number of districts to return
                        - Choice between top, bottom, or both performers
                        - Composite scoring for multiple indicators
                        - Automated indicator name matching with fuzzy matching and AI assistance

                        Use Cases:
                        - "Show me top 10 districts for vaccination coverage"
                        - "Which are the worst 5 districts for malnutrition in Maharashtra?"
                        - "Top performing districts for diabetes and blood pressure"
                        - "Bottom 15 districts across all health indicators"
                        - "Best 20 districts for child health indicators in northern states"

                        Examples:
                        1. Single indicator: {"indicator_name": "vaccination coverage", "n_districts": 10, "performance_type": "top"}
                        2. Multiple indicators: {"indicator_names": ["diabetes", "blood pressure"], "n_districts": 15, "performance_type": "top"}
                        3. State filtering: {"indicator_name": "malnutrition", "n_districts": 5, "performance_type": "bottom", "states": ["Maharashtra", "Gujarat"]}
                        4. Both top and bottom: {"indicator_name": "diarrhea", "n_districts": 10, "performance_type": "both"}

                        Performance Types:
                        - "top": Best performing districts (considers indicator direction)
                        - "bottom": Worst performing districts (considers indicator direction)
                        - "both": Both top and bottom performers

                        For multiple indicators, districts are ranked using a normalized composite score that properly accounts for each indicator's direction.
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of health indicator names (can be misspelled or described). Use this for multiple indicators."
                                },
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Single health indicator name (can be misspelled or described). Use this OR indicator_names."
                                },
                                "n_districts": {
                                    "type": "integer",
                                    "description": "Number of top/bottom districts to return (default: 10)",
                                    "minimum": 1,
                                    "maximum": 50
                                },
                                "performance_type": {
                                    "type": "string",
                                    "enum": ["top", "bottom", "both"],
                                    "description": "Type of performance to show: 'top' for best performing, 'bottom' for worst performing, 'both' for both (default: 'top')"
                                },
                                "states": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of specific state names to filter by (e.g., ['Maharashtra', 'Gujarat']). If not provided, includes all states."
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year for analysis (2016 or 2021, default: 2021)",
                                    "enum": [2016, 2021]
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping (default: true)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_indicator_change_analysis",
                        "description": """Analyze health indicator value changes from 2016 to 2021 at different geographic levels (country, state, or district).

                        This function investigates how health indicators have changed over the 5-year period between 2016 and 2021, providing comprehensive analysis with examples and visualizations.

                        Key Features:
                        - Multi-level Analysis: Country, state, or district level investigation
                        - Change Availability Validation: Checks data availability and quality
                        - Example Districts: Provides representative examples with actual change values
                        - Comprehensive Mapping: Shows all relevant districts with change data in popups
                        - Smart Visualization: Bar charts for examples, trend analysis for individual districts
                        - Proper Direction Handling: Interprets changes based on indicator direction (higher_is_better vs lower_is_better)

                        Use Cases:
                        - "How has diabetes prevalence changed nationally from 2016 to 2021?"
                        - "What's the change in vaccination coverage in Maharashtra state?"
                        - "How has malnutrition changed in Mumbai district?"
                        - "Show me the trend of maternal mortality at country level"
                        - "Has child stunting improved in Tamil Nadu?"

                        Analysis Levels:
                        1. **Country Level**: Uses national averages from overall_india table + 10 random district examples
                        2. **State Level**: Uses state averages from state_indicator table + 5 random district examples from that state  
                        3. **District Level**: Shows specific district trend with detailed analysis

                        Examples:
                        1. Country analysis: {"indicator_name": "diabetes", "analysis_level": "country"}
                        2. State analysis: {"indicator_name": "vaccination coverage", "analysis_level": "state", "location_name": "Maharashtra"}
                        3. District analysis: {"indicator_name": "malnutrition", "analysis_level": "district", "location_name": "Mumbai"}

                        Perfect for:
                        - Trend analysis and temporal health patterns
                        - Understanding health indicator improvements or deteriorations
                        - Policy impact assessment over the 5-year period
                        - Identifying districts with significant changes
                        - Comparative change analysis across geographic levels
                        """,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Name of health indicator to analyze for changes (can be misspelled or described, e.g., 'diabetes', 'vaccination coverage', 'child malnutrition')"
                                },
                                "analysis_level": {
                                    "type": "string",
                                    "enum": ["country", "state", "district"],
                                    "description": "Geographic level of analysis: 'country' for national trends, 'state' for state-level trends, 'district' for specific district trends (default: 'country')"
                                },
                                "location_name": {
                                    "type": "string", 
                                    "description": "Name of specific state (for state-level analysis) or district (for district-level analysis). Required for state and district levels, ignored for country level."
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping visualization (default: true)"
                                }
                            },
                            "required": ["indicator_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_district_performance_comparison",
                        "description": """Compare performance of multiple districts across multiple health indicators with either national averages or state averages, including comprehensive bar chart visualizations.

This function enables detailed benchmarking analysis, allowing you to compare how specific districts perform relative to national benchmarks or their respective state averages across various health indicators.

Key Features:
- Multi-district, multi-indicator comparison in a single analysis
- Choice between national and state-level benchmarking
- Comprehensive bar chart visualizations for each indicator
- Performance gap analysis showing differences from benchmarks
- Overall performance summary across all indicators
- Proper handling of indicator directions (higher_is_better vs lower_is_better)
- Detailed analysis with recommendations and insights

Use Cases:
- "Compare Mumbai, Delhi, and Kolkata's diabetes and vaccination rates with national averages"
- "How do districts in Maharashtra perform against state averages for malnutrition and child health?"
- "Compare 5 districts across multiple health indicators with national benchmarks"
- "Benchmark district performance: nutrition, vaccination, and disease prevalence vs state averages"

Chart Types Generated:
1. **Indicator Comparison Charts**: Individual bar charts for each indicator showing district values vs benchmarks
2. **Performance Gap Analysis**: Bar chart showing percentage point differences from benchmarks (color-coded by performance)
3. **Overall Performance Summary**: Bar chart showing average performance across all indicators for each district

Perfect For:
- Policy makers comparing district performance against benchmarks
- Identifying districts that consistently outperform or underperform expectations
- Resource allocation decisions based on multi-indicator performance gaps
- Best practice identification by comparing high-performing districts
- Targeted intervention planning based on specific indicator gaps
""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "district_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of district names to compare (will be resolved using fuzzy matching, e.g., ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'])",
                                    "minItems": 2,
                                    "maxItems": 10
                                },
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of health indicator names to analyze (can be misspelled or described, e.g., ['diabetes', 'vaccination coverage', 'child malnutrition', 'maternal mortality'])",
                                    "minItems": 1,
                                    "maxItems": 8
                                },
                                "comparison_type": {
                                    "type": "string",
                                    "enum": ["national", "state"],
                                    "description": "Type of benchmark comparison: 'national' for national averages, 'state' for respective state averages (default: 'national')"
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year for analysis (2016 or 2021, default: 2021)",
                                    "enum": [2016, 2021]
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping visualization (default: true)"
                                }
                            },
                            "required": ["district_names", "indicator_names"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_multi_indicator_performance",
                        "description": """Analyze multi-indicator performance using normalized composite index methodology with support for category-based analysis and top/bottom performers identification.

This function implements a comprehensive 4-step methodology:
1. **Min-Max Normalization**: All indicators normalized to [0,1] scale using global min/max values
2. **Direction Alignment**: "Lower is better" indicators inverted (X'' = 1 - X')  
3. **Composite Index**: Simple average of normalized, direction-aligned indicators
4. **Change Analysis**: Absolute and relative performance changes from 2016 to 2021

Key Features:
- Multi-indicator composite performance index calculation
- Category-based indicator selection (healthcare, nutrition, maternal health, etc.)
- OpenAI-powered category matching for flexible user queries
- Top/bottom performers identification across all districts
- Specific district analysis for named districts
- Proper handling of indicator directions (higher_is_better vs lower_is_better)
- Comprehensive visualizations and analysis

Use Cases:
- "Show me the top 10 districts in healthcare performance"
- "Compare Mumbai, Delhi, and Chennai across all nutrition indicators"
- "Which districts perform worst in maternal health indicators?"
- "Analyze multi-indicator performance for healthcare category"
- "Find districts with best overall health performance"

Performance Types:
- **specific**: Analyze specific named districts
- **top**: Find top N performing districts
- **bottom**: Find worst N performing districts  
- **both**: Show both top and bottom performers

Category Examples:
- "healthcare" → Health Care indicators
- "nutrition" → Nutrition indicators (Clinical/Anthropometry or Diet)
- "maternal health" → Maternal Health and Family Planning
- "mortality" → Morbidity and Mortality indicators

Perfect For:
- Comprehensive multi-dimensional health performance assessment
- Resource allocation based on overall health performance
- Identifying exemplary districts for best practice sharing
- Monitoring multi-indicator improvements over time
- Policy evaluation across multiple health domains simultaneously
""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "district_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of specific district names to analyze (for performance_type='specific')",
                                    "maxItems": 15
                                },
                                "category_name": {
                                    "type": "string",
                                    "description": "Category name for indicator selection (e.g., 'healthcare', 'nutrition', 'maternal health', 'mortality'). Uses OpenAI matching if exact match not found."
                                },
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of specific health indicator names to include in analysis (can be misspelled or described)",
                                    "maxItems": 20
                                },
                                "performance_type": {
                                    "type": "string",
                                    "enum": ["specific", "top", "bottom", "both"],
                                    "description": "Type of analysis: 'specific' for named districts, 'top' for best performers, 'bottom' for worst performers, 'both' for top and bottom (default: 'specific')"
                                },
                                "n_districts": {
                                    "type": "integer",
                                    "description": "Number of top/bottom districts to return (for performance_type='top', 'bottom', or 'both', default: 10)",
                                    "minimum": 5,
                                    "maximum": 50
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Primary year for analysis (2016 or 2021, default: 2021)",
                                    "enum": [2016, 2021]
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping visualization (default: true)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_state_multi_indicator_performance",
                        "description": """Analyze state-level multi-indicator performance with comparison to top/bottom districts within each state. This function extends multi-indicator analysis to state level with district benchmarking.

This function implements comprehensive state-to-district analysis:
1. **State-Level Performance**: Extract state averages from state_indicators table and compute composite performance
2. **District Selection**: For each state, identify top/bottom N districts using same multi-indicator methodology
3. **Comparative Analysis**: Compare state performance with selected districts within that state
4. **Multi-State Support**: Supports comparison between multiple states and their respective districts
5. **Intelligent Detection**: Auto-detects "lowest" hints to show bottom-performing districts

Key Features:
- State-level multi-indicator composite performance index calculation
- District performance ranking within each state (top 5 by default, configurable)
- Support for multiple states comparison simultaneously
- Automatic "lowest/worst" keyword detection for bottom districts
- Category-based and specific indicator selection
- Comprehensive bar chart visualizations for states and districts
- Map visualization with both state and district boundaries
- Proper handling of indicator directions (higher_is_better vs lower_is_better)

Use Cases:
- "Compare Maharashtra and Gujarat multi-indicator performance with their top districts"
- "Show state-level healthcare performance with lowest performing districts in each state"
- "Multi-indicator comparison of northern states with their best districts"
- "Which states perform best in nutrition indicators, and their top 5 districts?"
- "State maternal health performance with worst performing districts"

Performance Types:
- **top**: Show best performing districts within each state (default)
- **bottom**: Show worst performing districts within each state
- **both**: Show both top and bottom districts within each state

Auto-Detection Keywords for Bottom Districts:
- "lowest", "worst", "bottom", "poor", "underperform", "weakest", "bad"

Perfect For:
- State-to-state health performance benchmarking with district context
- Identifying exemplary districts within top-performing states
- Resource allocation decisions based on state and intra-state district performance
- Policy evaluation at both state and district levels
- Finding districts needing intervention within each state
- Understanding intra-state health disparities
""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "state_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of state names to analyze (if not provided, analyzes all states)",
                                    "maxItems": 15
                                },
                                "category_name": {
                                    "type": "string",
                                    "description": "Category name for indicator selection (e.g., 'healthcare', 'nutrition', 'maternal health', 'mortality'). Uses OpenAI matching if exact match not found."
                                },
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of specific health indicator names to include in analysis (can be misspelled or described)",
                                    "maxItems": 20
                                },
                                "performance_type": {
                                    "type": "string",
                                    "enum": ["top", "bottom", "both"],
                                    "description": "Type of district selection: 'top' for best districts in each state, 'bottom' for worst districts, 'both' for top and bottom (default: 'top')"
                                },
                                "n_districts": {
                                    "type": "integer",
                                    "description": "Number of districts per state to include (default: 5)",
                                    "minimum": 3,
                                    "maximum": 10
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Primary year for analysis (2016 or 2021, default: 2021)",
                                    "enum": [2016, 2021]
                                },
                                "query_hint": {
                                    "type": "string",
                                    "description": "Original user query to detect keywords like 'lowest', 'worst' for automatic bottom district selection"
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping visualization (default: true)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_district_similarity_analysis",
                        "description": """Analyze districts with similar or different performance patterns across multiple health indicators.

This function helps identify districts that show similar trends (clustering) or contrasting patterns (diversity analysis) across selected health indicators, enabling policy makers to understand regional patterns and effectiveness.

Key Features:
- **Pattern Recognition**: Identifies districts with similar or contrasting health indicator patterns
- **Multi-indicator Analysis**: Analyzes patterns across multiple indicators simultaneously  
- **Category-based Selection**: Automatically selects 4 random indicators from specified categories
- **State Filtering**: Option to limit analysis to specific states
- **Adaptive Selection**: Intelligently selects most representative districts (up to 20)
- **Comprehensive Visualization**: Generates bar charts for each indicator analyzed

Analysis Types:
- **"similar"**: Finds districts with comparable performance patterns across indicators
- **"different"**: Identifies districts with contrasting/diverse performance patterns

Use Cases:
- "Find districts with similar nutrition performance patterns"
- "Which districts have different maternal health trends?"
- "Show me contrasting healthcare performance across states"
- "Compare similar districts in Maharashtra and Karnataka"
- "Find districts with different patterns in mortality indicators"

Perfect For:
- Identifying policy intervention clusters
- Understanding regional health disparities  
- Finding exemplary districts with similar success patterns
- Discovering unique districts with different approaches
- Cross-state pattern comparison and policy transfer opportunities
""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "indicator_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of specific health indicator names to analyze for patterns (can be misspelled or described)",
                                    "maxItems": 10
                                },
                                "category_name": {
                                    "type": "string",
                                    "description": "Health indicator category name (e.g., 'healthcare', 'nutrition', 'maternal health'). Will randomly select 4 indicators from this category"
                                },
                                "analysis_type": {
                                    "type": "string",
                                    "enum": ["similar", "different"],
                                    "description": "Type of pattern analysis: 'similar' for districts with comparable patterns, 'different' for contrasting patterns (default: 'similar')"
                                },
                                "state_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of state names to filter districts (None for all states)",
                                    "maxItems": 10
                                },
                                "n_districts": {
                                    "type": "integer",
                                    "description": "Maximum number of districts to return in results (default: 20)",
                                    "minimum": 5,
                                    "maximum": 20
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Primary year for analysis (2016 or 2021, default: 2021)",
                                    "enum": [2016, 2021]
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping visualization (default: true)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_district_classification",
                        "description": """Classify all districts in India (or filtered by states) into 4 performance classes using Jenks natural breaks optimization based on a specific health indicator.

This function implements data-driven classification that creates meaningful groups by minimizing within-class variance while maximizing between-class variance. Perfect for identifying performance tiers and targeting interventions.

Key Features:
- **Jenks Natural Breaks Algorithm**: Statistical optimization for meaningful class boundaries
- **4-Class Classification**: Very Low, Low, Moderate, High performance groups
- **Indicator Direction Intelligence**: Automatically interprets whether higher or lower values are better
- **State Filtering**: Analyze all India or specific states only
- **Comprehensive Visualization**: Color-coded map with legend + bar chart showing district counts per class
- **Detailed Analysis**: Statistical summary, class breakdowns, and actionable insights

Use Cases:
- "Classify all districts by vaccination coverage performance"
- "Show diabetes prevalence classes for Maharashtra and Gujarat"
- "Classify districts by malnutrition - which states have most high-risk districts?"
- "Performance classes for maternal mortality across India"
- "Classification of child health indicators in northern states"

Perfect For:
- **Policy Planning**: Identify districts needing urgent intervention vs. exemplary performers
- **Resource Allocation**: Target resources to specific performance classes
- **Benchmarking**: Understand where districts stand relative to natural performance tiers
- **Progress Monitoring**: Track how districts move between performance classes over time
- **Regional Analysis**: Compare how different states distribute across performance classes""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Name of health indicator to classify districts by (can be misspelled or described, e.g., 'diabetes', 'vaccination coverage', 'child malnutrition', 'maternal health')"
                                },
                                "state_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of state names to filter analysis (e.g., ['Maharashtra', 'Gujarat', 'Tamil Nadu']). If not provided, analyzes all districts in India."
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year for analysis (2016 or 2021, default: 2021)",
                                    "enum": [2016, 2021]
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping visualization (default: true)"
                                }
                            },
                            "required": ["indicator_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_district_classification_change",
                        "description": """Classify all districts in India (or filtered by states) into 4 performance classes using Jenks natural breaks optimization based on a specific health indicator's prevalence CHANGE (2021 - 2016).

This function analyzes how districts have improved or declined over time by classifying them based on their change in health indicator values. Perfect for identifying districts with improving vs. declining trends.

Key Features:
- **Change Analysis**: Classifies based on prevalence change (2021 - 2016) in percentage points
- **Jenks Natural Breaks Algorithm**: Statistical optimization for meaningful change class boundaries
- **4-Class Classification**: Major Decline, Minor Decline, Minor Improvement, Major Improvement
- **Direction Intelligence**: Automatically interprets whether positive or negative change is better based on indicator type
- **State Filtering**: Analyze all India or specific states only
- **Trend Visualization**: Color-coded map showing change patterns + bar chart showing district counts per change class
- **Detailed Analysis**: Statistical summary of changes, improvement/decline trends, and actionable insights

Use Cases:
- "Show districts with biggest improvement in vaccination coverage over time"
- "Classify districts by diabetes prevalence change - which are getting worse?"
- "Malnutrition change classification for Maharashtra and Gujarat"
- "Which districts show declining maternal health trends?"
- "Child health improvement patterns across northern states"

Perfect For:
- **Trend Monitoring**: Identify districts with improving vs. declining health trends
- **Policy Evaluation**: Assess impact of interventions over 2016-2021 period  
- **Priority Setting**: Target resources to districts showing concerning decline patterns
- **Success Stories**: Highlight districts with exemplary improvement for replication
- **Regional Comparison**: Compare change patterns across different states""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "indicator_name": {
                                    "type": "string",
                                    "description": "Name of health indicator to classify districts by change (can be misspelled or described, e.g., 'diabetes', 'vaccination coverage', 'child malnutrition', 'maternal health')"
                                },
                                "state_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of state names to filter analysis (e.g., ['Maharashtra', 'Gujarat', 'Tamil Nadu']). If not provided, analyzes all districts in India."
                                },
                                "include_boundary_data": {
                                    "type": "boolean",
                                    "description": "Whether to include boundary geometry data for mapping visualization (default: true)"
                                }
                            },
                            "required": ["indicator_name"]
                        }
                    }
                }
            ]
        )

        # Handle the response
        if response.choices[0].message.tool_calls:
            # Function calls were made
            function_results = []
            all_boundaries = []

            # Execute function calls
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"Executing function: {function_name}")
                print(f"Arguments: {arguments}")
                
                result = execute_function_call(function_name, arguments)
                
                # Collect boundaries if present
                if isinstance(result, dict):
                    boundaries = result.get("boundary") or result.get("boundary_data")
                    if boundaries:
                        if isinstance(boundaries, list):
                            all_boundaries.extend(boundaries)
                        else:
                            all_boundaries.append(boundaries)
                
                function_results.append({
                    "function": function_name,
                    "arguments": arguments,
                    "result": result
                })

            # Create function call messages for OpenAI
            messages_with_results = session["history"] + [response.choices[0].message]
            
            # Add function call results
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                function_result = function_results[i]["result"]
                
                # Clean result for OpenAI (remove large fields and analysis to prevent text contamination)
                clean_result = dict(function_result) if isinstance(function_result, dict) else function_result
                if isinstance(clean_result, dict):
                    # Remove large data fields and analysis text to prevent OpenAI from incorporating them into response
                    clean_result.pop("boundary", None)
                    clean_result.pop("boundary_data", None)
                    clean_result.pop("analysis", None)  # Remove detailed analysis to prevent text contamination
                    clean_result.pop("chart_data", None)  # Remove large chart data structure
                
                # Convert to string
                result_str = json.dumps(clean_result, default=str)
                
                # Token limit for results
                if len(result_str) > 4000:
                    result_str = result_str[:3800] + "... [Result truncated for token limit]"
                
                messages_with_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })

            # Get synthesized response from OpenAI
            synthesis_response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages_with_results
            )

            final_response = synthesis_response.choices[0].message.content
            
            # Create truncated version for history
            truncated_response = final_response
            if len(final_response) > 2500:
                truncated_response = final_response[:2000] + "... [Response truncated for conversation history]"
            
            # Add to history with proper token management
            assistant_message = {"role": "assistant", "content": truncated_response}
            session["history"] = manage_conversation_history(session["history"], assistant_message)

            # Determine map type - check all function results for map_type
            # Also detect if multiple individual district calls should be treated as multi-district
            map_type = "health_analysis"
            individual_district_calls = []
            
            for function_result in function_results:
                if isinstance(function_result["result"], dict):
                    result_map_type = function_result["result"].get("map_type")
                    if result_map_type == "individual_district":
                        individual_district_calls.append(function_result)
                    elif result_map_type and result_map_type != "health_analysis":
                        map_type = result_map_type
                        break  # Use the first specific map_type found
            
            # If we have multiple individual district calls, treat as multi-district
            if len(individual_district_calls) > 1:
                map_type = "multi_district_comparison"
                print(f"🔄 Detected {len(individual_district_calls)} individual district calls, converting to multi_district_comparison")

            # Return comprehensive response
            base_response = {
                "response": final_response,
                "map_type": map_type,
                "function_calls": [{"function": fr["function"], "arguments": fr["arguments"]} for fr in function_results],
                "data": function_results,
                "boundary": all_boundaries
            }
            
            # If single function call result, flatten it to top level for easier frontend access
            # Also handle multiple calls of the same function (like get_district_health_data)
            if len(function_results) == 1:
                # Single function call - normal flattening
                single_result = function_results[0]["result"]
                if isinstance(single_result, dict):
                    # Add the function result fields to the top level
                    for key, value in single_result.items():
                        if key not in base_response:  # Don't override existing keys
                            base_response[key] = value
                        elif key == "map_type":  # Always override map_type from function result
                            base_response[key] = value
            elif len(individual_district_calls) > 1:
                # Multiple individual district calls - merge them into multi-district structure
                print(f"🔄 Merging {len(individual_district_calls)} individual district calls into multi-district structure")
                merged_districts = []
                all_indicators = set()
                merged_boundary = []
                
                for call in individual_district_calls:
                    result = call["result"]
                    if isinstance(result, dict):
                        # Convert individual district to districts array format
                        district_data = {
                            "district_name": result.get("district_name"),
                            "state_name": result.get("state_name"),
                            "indicators": result.get("data", [])
                        }
                        merged_districts.append(district_data)
                        
                        # Collect indicators
                        for indicator in result.get("data", []):
                            all_indicators.add(indicator.get("indicator_name"))
                        
                        # Collect boundary data
                        if result.get("boundary"):
                            merged_boundary.extend(result["boundary"])
                
                # Add merged data to base response
                base_response.update({
                    "districts": merged_districts,
                    "total_districts": len(merged_districts),
                    "total_indicators": len(all_indicators),
                    "indicators": list(all_indicators),
                    "boundary": merged_boundary,
                    "map_type": "multi_district_comparison"
                })
            
            # Log final response for get_district_health_data calls
            if any(fc["function"] == "get_district_health_data" for fc in base_response.get("function_calls", [])):
                print("🚀 FINAL BACKEND RESPONSE for get_district_health_data:")
                print(f"  📋 Response keys: {list(base_response.keys())}")
                print(f"  🗺️ map_type: {base_response.get('map_type')}")
                print(f"  📊 has chart_data: {bool(base_response.get('chart_data'))}")
                print(f"  🌍 boundary count: {len(base_response.get('boundary', []))}")
                print(f"  🏙️ districts: {base_response.get('districts', 'N/A')}")
                print(f"  🏙️ district_name: {base_response.get('district_name', 'N/A')}")
                print(f"  📈 total_districts: {base_response.get('total_districts', 'N/A')}")
                print(f"  📈 total_indicators: {base_response.get('total_indicators', 'N/A')}")
                print(f"  🔧 function_calls: {[fc['function'] for fc in base_response.get('function_calls', [])]}")
                
                if base_response.get('districts'):
                    print(f"  📍 Districts data: {len(base_response['districts'])} districts")
                    for i, dist in enumerate(base_response['districts'][:2]):  # Show first 2
                        print(f"    District {i+1}: {dist.get('district_name')} ({dist.get('state_name')}) - {len(dist.get('indicators', []))} indicators")
                
                if base_response.get('chart_data'):
                    chart_data = base_response['chart_data']
                    if isinstance(chart_data, dict):
                        print(f"  📊 Chart data type: {chart_data.get('type', 'unknown')}")
                        print(f"  📊 Chart title: {chart_data.get('title', 'N/A')}")
                        print(f"  📊 Chart datasets: {len(chart_data.get('datasets', []))}")
                    else:
                        print(f"  📊 Chart data: {type(chart_data)}")
            
            return base_response

        else:
            # No function calls - just return conversational response
            final_response = response.choices[0].message.content
            
            # Add to history
            assistant_message = {"role": "assistant", "content": final_response}
            session["history"] = manage_conversation_history(session["history"], assistant_message)
            return {"response": final_response}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/save_visualization/")
async def save_visualization(request: SaveVisualizationRequest):
    """Save a visualization to the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert data to JSON strings
        visualization_data_json = json.dumps(request.visualization_data)
        metadata_json = json.dumps(request.metadata) if request.metadata else None
        
        # Insert into database and get the ID
        cursor.execute('''
            INSERT INTO saved_visualizations (visualization_data, metadata)
            VALUES (%s, %s)
            RETURNING id
        ''', (visualization_data_json, metadata_json))
        
        visualization_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "visualization_id": visualization_id,
            "message": "Health visualization saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving visualization: {str(e)}")

@app.get("/get_saved_visualizations/")
async def get_saved_visualizations():
    """Get all saved visualizations"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, visualization_data, metadata, created_at, updated_at
            FROM saved_visualizations
            ORDER BY created_at DESC
            LIMIT 100
        ''')
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        visualizations = []
        for row in rows:
            visualization_data = json.loads(row[1]) if row[1] else {}
            metadata = json.loads(row[2]) if row[2] else {}
            
            visualizations.append({
                "id": row[0],
                "visualization_data": visualization_data,
                "metadata": metadata,
                "created_at": row[3].isoformat() if row[3] else None,
                "updated_at": row[4].isoformat() if row[4] else None
            })
        
        return {
            "success": True,
            "visualizations": visualizations,
            "count": len(visualizations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving visualizations: {str(e)}")

@app.delete("/delete_visualization/{visualization_id}")
async def delete_visualization(visualization_id: int):
    """Delete a saved visualization"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM saved_visualizations WHERE id = %s', (visualization_id,))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": "Health visualization deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting visualization: {str(e)}")

@app.get("/visualization_stats/")
async def get_visualization_stats():
    """Get statistics about saved visualizations"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM saved_visualizations')
        total_count = cursor.fetchone()[0]
        
        # Get count by date (last 30 days)
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM saved_visualizations
            WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        ''')
        daily_stats = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "total_visualizations": total_count,
            "daily_stats": [{"date": row[0].isoformat() if row[0] else None, "count": row[1]} for row in daily_stats]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting visualization stats: {str(e)}")

# Save user reaction into database
@app.post("/feedback/")
async def save_feedback(request: ReactionRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Prepare timestamp
        created_at = None
        if request.timestamp:
            try:
                # Parse ISO timestamp if provided
                from datetime import datetime
                created_at = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
            except Exception:
                created_at = None

        # Build dynamic insert depending on available columns
        # Prefer inserting into (user_query, bot_response, reaction_type, created_at, message_id)
        insert_sql = None
        params = None

        try:
            # Check table columns
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'botreactions'")
            cols = [row[0] for row in cursor.fetchall()]

            fields = []
            values = []
            placeholders = []

            if 'user_query' in cols:
                fields.append('user_query')
                values.append(request.user_query or '')
                placeholders.append('%s')
            if 'bot_response' in cols:
                fields.append('bot_response')
                values.append(request.assistant_response or '')
                placeholders.append('%s')
            if 'reaction_type' in cols:
                fields.append('reaction_type')
                values.append(request.reaction)
                placeholders.append('%s')
            if 'message_id' in cols and request.message_id is not None:
                fields.append('message_id')
                values.append(request.message_id)
                placeholders.append('%s')
            if 'created_at' in cols:
                fields.append('created_at')
                if created_at is None:
                    # Use NOW() if not provided
                    insert_created_at = None
                else:
                    insert_created_at = created_at
                # We'll handle NOW() by injecting in SQL when value is None
                if insert_created_at is None:
                    # Reserve placeholder but replace later
                    placeholders.append('NOW()')
                else:
                    placeholders.append('%s')
                    values.append(insert_created_at)

            # Fallback: if no columns matched, raise
            if not fields:
                cursor.close()
                conn.close()
                raise HTTPException(status_code=500, detail="botreactions table has no expected columns")

            # Compose SQL safely
            placeholder_str = ', '.join([p for p in placeholders])
            field_str = ', '.join(fields)
            sql = f"INSERT INTO botreactions ({field_str}) VALUES ({placeholder_str}) RETURNING id"
            # Remove literal NOW() from params alignment
            exec_params = tuple(v for v in values)

            cursor.execute(sql, exec_params)
            new_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            return {"success": True, "id": new_id}

        except Exception as e:
            conn.rollback()
            cursor.close()
            conn.close()
            raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
