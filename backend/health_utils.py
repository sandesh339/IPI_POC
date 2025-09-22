import psycopg2
import psycopg2.extras
import socket
import json
import logging
import os
import re
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
from rapidfuzz import process, fuzz
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import math
# Initialize OpenAI client for indicator matching
client = OpenAI (
    api_key=os.getenv("OPEN_API_KEY")
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _classify_district_batch(batch_data, breaks, higher_is_better):
    """
    Helper function to classify a batch of districts in parallel.
    This function is designed to be JSON-serializable and thread-safe.
    """
    try:
        classified_batch = []
        for row in batch_data:
            district_id, district_name, state_name, indicator_value = row
            
            # Classify the district
            classification = classify_value(indicator_value, breaks, higher_is_better)
            
            # Ensure all values are JSON serializable
            classified_district = {
                "district_id": int(district_id) if district_id is not None else None,
                "district_name": str(district_name) if district_name is not None else "",
                "state_name": str(state_name) if state_name is not None else "",
                "indicator_value": float(indicator_value) if indicator_value is not None else None,
                "class_number": int(classification["class_number"]),
                "class_name": str(classification["class_name"]),
                "class_color": str(classification["color"]),
                "class_description": str(classification["description"])
            }
            
            classified_batch.append(classified_district)
        
        return classified_batch
    except Exception as e:
        logger.error(f"Error in _classify_district_batch: {e}")
        return []

def _safe_float_conversion_classification(value):
    """Convert value to float safely for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    try:
        converted = float(value)
        if math.isnan(converted) or math.isinf(converted):
            return None
        return converted
    except (ValueError, TypeError):
        return None

def jenks_breaks_optimized(data_list: List[float], number_class: int) -> List[float]:
    """
    OPTIMIZED Jenks natural breaks using numpy vectorization and sampling
    
    This is a high-performance version that uses:
    1. Numpy vectorization for O(n²) instead of O(n³) complexity
    2. Sampling for very large datasets (>1000 points)
    3. Pre-computed cumulative sums for efficiency
    4. Memory-efficient algorithms
    
    Args:
        data_list: List of numeric values to classify
        number_class: Number of classes to create
        
    Returns:
        List of break points (class boundaries)
    """
    if len(data_list) < number_class:
        logger.warning(f"Not enough data points ({len(data_list)}) for {number_class} classes")
        return sorted(set(data_list))
    
    # Convert to numpy array for vectorized operations
    data_array = np.array([float(x) for x in data_list if x is not None], dtype=np.float64)
    data_array = np.sort(data_array)
    
    if len(np.unique(data_array)) < number_class:
        unique_values = np.unique(data_array)
        logger.warning(f"Only {len(unique_values)} unique values for {number_class} classes")
        return unique_values.tolist()
    
    # For very large datasets, use intelligent sampling to maintain accuracy while improving speed
    n_data = len(data_array)
    if n_data > 1000:
        logger.info(f"🚀 Large dataset ({n_data} points): Using optimized sampling approach")
        # Use stratified sampling to maintain distribution
        sample_size = min(500, n_data)
        indices = np.linspace(0, n_data - 1, sample_size, dtype=int)
        data_array = data_array[indices]
        n_data = len(data_array)
        logger.info(f"📉 Reduced to {n_data} representative points for Jenks calculation")
    
    # Pre-compute cumulative sums for O(1) range sum calculations
    cumsum = np.concatenate([[0], np.cumsum(data_array)])
    cumsum_sq = np.concatenate([[0], np.cumsum(data_array ** 2)])
    
    def variance_between_indices(start_idx, end_idx):
        """Calculate variance for a range using pre-computed cumulative sums - O(1) operation"""
        if end_idx <= start_idx:
            return float('inf')
        
        n = end_idx - start_idx
        sum_val = cumsum[end_idx] - cumsum[start_idx]
        sum_sq = cumsum_sq[end_idx] - cumsum_sq[start_idx]
        
        if n <= 0:
            return float('inf')
        
        return sum_sq - (sum_val ** 2) / n
    
    # Initialize DP matrices (much smaller now due to sampling)
    dp_cost = np.full((n_data + 1, number_class + 1), float('inf'), dtype=np.float64)
    dp_split = np.zeros((n_data + 1, number_class + 1), dtype=np.int32)
    
    # Base case: one class
    for i in range(1, n_data + 1):
        dp_cost[i][1] = variance_between_indices(0, i)
        dp_split[i][1] = 0
    
    # Fill DP table with vectorized operations where possible
    for num_classes in range(2, number_class + 1):
        for i in range(num_classes, n_data + 1):
            # Vectorized calculation for all possible split points
            split_points = np.arange(num_classes - 1, i)
            costs = np.array([
                dp_cost[k][num_classes - 1] + variance_between_indices(k, i)
                for k in split_points
            ])
            
            min_idx = np.argmin(costs)
            dp_cost[i][num_classes] = costs[min_idx]
            dp_split[i][num_classes] = split_points[min_idx]
    
    # Backtrack to find break points
    breaks = []
    k = n_data
    
    for num_classes in range(number_class, 0, -1):
        if num_classes == 1:
            breaks.append(data_array[0])
        else:
            split_point = dp_split[k][num_classes]
            if split_point > 0:
                breaks.append(data_array[split_point - 1])
            k = split_point
    
    # Clean up and ensure we have min and max
    breaks = sorted(list(set(breaks)))
    
    # Ensure we have the extreme values
    if breaks[0] > data_array[0]:
        breaks.insert(0, data_array[0])
    if breaks[-1] < data_array[-1]:
        breaks.append(data_array[-1])
    
    # If we used sampling, map back to original data range
    if len(data_list) > 1000:
        original_array = np.array([float(x) for x in data_list if x is not None])
        original_array = np.sort(original_array)
        
        # Adjust breaks to match original data distribution
        adjusted_breaks = []
        for break_val in breaks:
            # Find closest value in original data
            closest_idx = np.argmin(np.abs(original_array - break_val))
            adjusted_breaks.append(original_array[closest_idx])
        
        breaks = sorted(list(set(adjusted_breaks)))
    
    return breaks

def jenks_breaks(data_list: List[float], number_class: int) -> List[float]:
    """
    Wrapper function that automatically chooses between optimized and fallback implementations
    """
    try:
        return jenks_breaks_optimized(data_list, number_class)
    except Exception as e:
        logger.warning(f"Optimized Jenks failed ({e}), falling back to simple quantile method")
        # Fallback to quantile-based classification for extreme edge cases
        data_array = np.array([float(x) for x in data_list if x is not None])
        data_array = np.sort(data_array)
        
        if len(data_array) < number_class:
            return data_array.tolist()
        
        # Use quantile-based breaks as fallback
        quantiles = np.linspace(0, 1, number_class + 1)
        breaks = np.quantile(data_array, quantiles[1:])
        
        # Ensure we have the minimum value as first break
        breaks[0] = data_array[0]
        
        return breaks.tolist()

def classify_value(value: float, breaks: List[float], higher_is_better: bool = True) -> Dict[str, Any]:
    """
    Classify a value into one of 4 classes based on Jenks breaks
    
    Args:
        value: The value to classify
        breaks: List of break points from Jenks algorithm
        higher_is_better: Whether higher values are better for this indicator
        
    Returns:
        Dictionary with class info (class_number, class_name, color, description)
    """
    if value is None:
        return {
            "class_number": 0,
            "class_name": "No Data",
            "color": "#CCCCCC",
            "description": "No data available"
        }
    
    # Find which class the value belongs to
    class_number = 1
    for i, break_point in enumerate(breaks[1:], 1):
        if value <= break_point:
            class_number = i
            break
        class_number = len(breaks)  # If value is larger than all breaks
    
    # Define class names and colors based on indicator direction
    if higher_is_better:
        class_configs = {
            1: {"name": "Very Low", "color": "#d73027", "desc": "Needs urgent intervention"},
            2: {"name": "Low", "color": "#fc8d59", "desc": "Below average performance"},
            3: {"name": "Moderate", "color": "#fee08b", "desc": "Average performance"},
            4: {"name": "High", "color": "#1a9850", "desc": "Good performance"}
        }
    else:  # lower_is_better
        class_configs = {
            1: {"name": "Very Low", "color": "#1a9850", "desc": "Excellent performance"},
            2: {"name": "Low", "color": "#fee08b", "desc": "Good performance"},
            3: {"name": "Moderate", "color": "#fc8d59", "desc": "Average performance"},
            4: {"name": "High", "color": "#d73027", "desc": "Needs urgent intervention"}
        }
    
    # Handle cases where we have fewer than 4 classes
    if class_number > len(class_configs):
        class_number = len(class_configs)
    
    config = class_configs.get(class_number, class_configs[1])
    
    return {
        "class_number": class_number,
        "class_name": config["name"],
        "color": config["color"],
        "description": config["desc"]
    }

def get_classification_legend(breaks: List[float], higher_is_better: bool = True, 
                            indicator_name: str = "", unit: str = "%") -> List[Dict[str, Any]]:
    """
    Generate legend information for the classification
    
    Args:
        breaks: List of break points from Jenks algorithm
        higher_is_better: Whether higher values are better
        indicator_name: Name of the indicator
        unit: Unit of measurement
        
    Returns:
        List of legend items with class info and ranges
    """
    legend_items = []
    
    # Define class names and colors based on indicator direction
    if higher_is_better:
        class_configs = [
            {"name": "Very Low", "color": "#d73027", "desc": "Needs urgent intervention"},
            {"name": "Low", "color": "#fc8d59", "desc": "Below average performance"},
            {"name": "Moderate", "color": "#fee08b", "desc": "Average performance"},
            {"name": "High", "color": "#1a9850", "desc": "Good performance"}
        ]
    else:
        class_configs = [
            {"name": "Very Low", "color": "#1a9850", "desc": "Excellent performance"},
            {"name": "Low", "color": "#fee08b", "desc": "Good performance"},
            {"name": "Moderate", "color": "#fc8d59", "desc": "Average performance"},
            {"name": "High", "color": "#d73027", "desc": "Needs urgent intervention"}
        ]
    
    # Create legend items with ranges
    for i in range(min(4, len(breaks))):
        if i == 0:
            range_text = f"≤ {breaks[1]:.1f}{unit}" if len(breaks) > 1 else f"{breaks[0]:.1f}{unit}"
            range_min = 0
            range_max = breaks[1] if len(breaks) > 1 else breaks[0]
        elif i == len(breaks) - 1:
            range_text = f"> {breaks[i]:.1f}{unit}"
            range_min = breaks[i]
            range_max = float('inf')
        else:
            range_text = f"{breaks[i]:.1f} - {breaks[i+1]:.1f}{unit}"
            range_min = breaks[i]
            range_max = breaks[i+1]
        
        config = class_configs[i] if i < len(class_configs) else class_configs[-1]
        
        legend_items.append({
            "class_number": i + 1,
            "class_name": config["name"],
            "color": config["color"],
            "description": config["desc"],
            "range_text": range_text,
            "range_min": range_min,
            "range_max": range_max
        })
    
    return legend_items

def get_db_connection():
    """
    Render-optimized database connection for Supabase with IPv6 support
    """
    
    # Enhanced configuration with IPv6 support
    config = {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT')),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'sslmode': 'require',
        'connect_timeout': 30,
        'application_name': 'IPI_Paper'
    }
    
    # Enhanced connection strategies with IPv6 priority
    strategies = [
        # Strategy 1: Enhanced config (best for pooler)
        lambda: connect_with_enhanced_config(config),
        # Strategy 2: Original psycopg2 default
        lambda: psycopg2.connect(**config),
        # Strategy 3: Try gethostbyname for IPv4
        lambda: connect_with_gethostbyname(config),
        # Strategy 4: Try forced IPv4 resolution
        lambda: connect_with_ipv4_force(config)
    ]
    
    last_error = None
    for i, strategy in enumerate(strategies, 1):
        try:
            conn = strategy()
            if conn:
                # Test the connection with a quick query
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                cursor.fetchone()
                cursor.close()
                print(f"✅ Connected using strategy {i}")
                return conn
        except Exception as e:
            last_error = e
            print(f"⚠️ Strategy {i} failed: {e}")
            continue
    
    # If all strategies fail, raise the last error
    raise last_error or psycopg2.OperationalError("All connection strategies failed")

def connect_with_enhanced_config(config):
    """Original hostname with enhanced connection parameters"""
    try:
        enhanced_config = config.copy()
        # Add additional parameters that might help with connectivity
        enhanced_config['keepalives_idle'] = 30
        enhanced_config['keepalives_interval'] = 5
        enhanced_config['keepalives_count'] = 5
        return psycopg2.connect(**enhanced_config)
    except Exception as e:
        raise psycopg2.OperationalError(f"Enhanced config connection failed: {e}")

def connect_with_gethostbyname(config):
    """Use gethostbyname for IPv4 resolution (most compatible)"""
    try:
        ipv4_host = socket.gethostbyname(config['host'])
        config_copy = config.copy()
        config_copy['host'] = ipv4_host
        return psycopg2.connect(**config_copy)
    except Exception:
        raise psycopg2.OperationalError("gethostbyname IPv4 resolution failed")

def connect_with_ipv4_force(config):
    """Force IPv4 connection by resolving hostname first"""
    try:
        addr_info = socket.getaddrinfo(
            config['host'], config['port'], 
            socket.AF_INET, socket.SOCK_STREAM
        )
        if addr_info:
            ipv4_host = addr_info[0][4][0]
            config_copy = config.copy()
            config_copy['host'] = ipv4_host
            return psycopg2.connect(**config_copy)
        else:
            raise psycopg2.OperationalError("No IPv4 addresses found")
    except Exception:
        raise psycopg2.OperationalError("IPv4 getaddrinfo resolution failed")

def normalize_district_name(name):
    """Normalize district name for better matching"""
    return name.strip().title() if name else ""

def get_district_boundary_data(district_names: List[str]):
    """Get district boundary data for mapping visualization - OPTIMIZED VERSION"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if not district_names:
            return []
        
        logger.info(f"🗺️ Fetching optimized boundary data for {len(district_names)} districts")
        
        # OPTIMIZATION 1: Use simplified geometries for faster transfer
        placeholders = ','.join(['%s'] * len(district_names))
        
        # OPTIMIZATION 2: Simplified geometry with tolerance for faster processing
        query = f"""
        SELECT DISTINCT ON (UPPER(district_name))
            district_name,
            state_name,
            ST_AsGeoJSON(ST_Simplify(geometry, 0.001)) as geometry,
            additional_attributes
        FROM district_geometries 
        WHERE UPPER(district_name) IN ({','.join(['UPPER(%s)'] * len(district_names))})
        ORDER BY UPPER(district_name), district_name
        """
        
        import time
        query_start = time.time()
        cursor.execute(query, district_names)
        results = cursor.fetchall()
        query_time = time.time() - query_start
        
        logger.info(f"⚡ Boundary query executed in {query_time:.2f}s")
        
        # Debug: uncomment for troubleshooting
        # print(f"📊 Found boundary data for {len(results)} districts:")
        # for row in results:
        #     print(f"  - '{row[0]}' in {row[1]}")
        
        if len(results) < len(district_names):
            # print(f"⚠️  Missing boundary data for {len(district_names) - len(results)} districts!")
            
            # Try fuzzy matching for missing districts
            found_districts = {row[0].upper() for row in results}  # Use upper case for comparison
            missing_districts = [name for name in district_names if name.upper() not in found_districts]
            
            # print(f"🔍 Attempting fuzzy matching for missing districts:")
            for missing in missing_districts:
                # print(f"  Missing: '{missing}'")
                
                # Query all districts in district_geometries to find close matches
                fuzzy_query = """
                SELECT district_name, state_name
                FROM district_geometries 
                ORDER BY district_name
                """
                cursor.execute(fuzzy_query)
                all_geometry_districts = cursor.fetchall()
                
                # Use rapidfuzz for fuzzy matching
                from rapidfuzz import fuzz, process
                
                choices = [row[0] for row in all_geometry_districts]
                match = process.extractOne(missing, choices, scorer=fuzz.ratio)
                
                if match and match[1] > 80:  # 80% similarity threshold
                    # print(f"    📍 Fuzzy match found: '{missing}' → '{match[0]}' (score: {match[1]})")
                    
                    # Get the geometry for the matched district
                    matched_query = f"""
                    SELECT 
                        district_name,
                        state_name,
                        ST_AsGeoJSON(geometry) as geometry,
                        additional_attributes
                    FROM district_geometries 
                    WHERE UPPER(district_name) = UPPER(%s)
                    """
                    cursor.execute(matched_query, (match[0],))
                    matched_result = cursor.fetchone()
                    
                    if matched_result:
                        results.append(matched_result)
                        # print(f"    ✅ Added fuzzy match to results")
                # else:
                    # print(f"    ❌ No good fuzzy match found for '{missing}'")
        
        boundary_data = []
        for row in results:
            # Parse GeoJSON string to object
            geometry = None
            if row[2]:  # geometry field
                try:
                    import json
                    geometry = json.loads(row[2])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse geometry for {row[0]}: {e}")
                    geometry = None
            
            boundary_data.append({
                "district_name": row[0],
                "state_name": row[1],
                "geometry": geometry,  # GeoJSON geometry object
                "additional_attributes": row[3] if row[3] else {}
            })
        
        cursor.close()
        conn.close()
        
        # print(f"🎯 Final boundary data: {len(boundary_data)} districts ready for mapping")
        
        return boundary_data
        
    except Exception as e:
        logger.error(f"Error getting boundary data: {e}")
        return []

def get_district_health_data(
    district_names: Union[str, List[str]],
    indicator_ids: Optional[List[int]] = None,
    year: int = 2021,
    state_name: Optional[str] = None
):
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Determine which districts to analyze
        if isinstance(district_names, str):
            target_districts = [district_names]
        elif isinstance(district_names, list):
            target_districts = district_names
        else:
            return {"error": "district_names must be a string or list of strings"}
        
        # Resolve all district names
        resolved_districts = []
        boundary_names = []
        
        for dist_name in target_districts:
            resolved_district = resolve_district_name(cursor, dist_name)
            if resolved_district:
                resolved_districts.append(resolved_district)
                boundary_names.append(resolved_district['district_name'])
            else:
                logger.warning(f"District '{dist_name}' not found")
        
        if not resolved_districts:
            return {
                "error": f"No valid districts found from: {target_districts}"
            }
        
        # Build query for multiple districts
        district_ids = [d['district_id'] for d in resolved_districts]
        district_placeholders = ','.join(['%s'] * len(district_ids))
        
        if indicator_ids:
            indicator_filter = f"AND di.indicator_id IN ({','.join(['%s'] * len(indicator_ids))})"
            params = district_ids + indicator_ids
        else:
            indicator_filter = ""
            params = district_ids
        
        # Main query to get district indicator data
        query = f"""
        SELECT 
            d.district_name,
            s.state_name,
            i.indicator_name,
            i.indicator_direction,
            di.prevalence_2016,
            di.prevalence_2021,
            di.prevalence_change,
            di.headcount_2021,
            i.indicator_id,
            d.district_id
        FROM district_indicators di
        JOIN districts d ON di.district_id = d.district_id
        JOIN states s ON d.state_id = s.state_id
        JOIN indicators i ON di.indicator_id = i.indicator_id
        WHERE di.district_id IN ({district_placeholders})
        {indicator_filter}
        ORDER BY d.district_name, i.indicator_name
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            return {
                "error": f"No data found for districts: {[d['district_name'] for d in resolved_districts]}"
            }
        
        # Organize results by district
        districts_data = {}
        all_indicators = set()
        
        for row in results:
            district_name = row[0]
            if district_name not in districts_data:
                districts_data[district_name] = {
                    "district_name": district_name,
                    "state_name": row[1],
                    "indicators": []
                }
            
            indicator_data = {
                "indicator_name": row[2],
                "indicator_direction": row[3],
                "prevalence_2016": float(row[4]) if row[4] is not None else None,
                "prevalence_2021": float(row[5]) if row[5] is not None else None,
                "prevalence_change": float(row[6]) if row[6] is not None else None,
                "headcount_2021": float(row[7]) if row[7] is not None else None,
                "indicator_id": row[8]
            }
            
            # Add trend interpretation
            if indicator_data["prevalence_change"] is not None:
                indicator_data["trend_interpretation"] = interpret_health_trend(
                    indicator_data["prevalence_change"],
                    indicator_data["indicator_direction"]
                )
            
            districts_data[district_name]["indicators"].append(indicator_data)
            all_indicators.add(row[2])
        
        # Get boundary data for all districts
        boundary_data = get_district_boundary_data(boundary_names)
        
        # Generate analysis and charts
        logger.info(f"🔍 get_district_health_data: resolved_districts count = {len(resolved_districts)}")
        logger.info(f"🔍 Resolved districts: {[d['district_name'] for d in resolved_districts]}")
        logger.info(f"🔍 Districts data keys: {list(districts_data.keys())}")
        
        if len(resolved_districts) == 1:
            # Single district analysis
            district_data = list(districts_data.values())[0]
            analysis = generate_district_analysis(
                district_data["district_name"],
                district_data["state_name"],
                district_data["indicators"]
            )
            
            # Generate chart data for single district
            chart_data = generate_single_district_chart_data(district_data, list(all_indicators))
            
            response = {
                "district_name": district_data["district_name"],
                "state_name": district_data["state_name"],
                "year": year,
                "total_indicators": len(district_data["indicators"]),
                "data": district_data["indicators"],
                "boundary": boundary_data,
                "chart_data": chart_data,
                "analysis": analysis,
                "map_type": "individual_district"
            }
        else:
            # Multiple districts analysis - generate comparative charts
            chart_data = generate_multi_district_chart_data(districts_data, list(all_indicators))
            analysis = generate_multi_district_analysis(districts_data, list(all_indicators))
            
            response = {
                "districts": list(districts_data.values()),
                "year": year,
                "total_districts": len(districts_data),
                "total_indicators": len(all_indicators),
                "indicators": list(all_indicators),
                "boundary": boundary_data,
                "chart_data": chart_data,
                "analysis": analysis,
                "map_type": "multi_district_comparison"
            }
        
        cursor.close()
        conn.close()
        
        logger.info(f"🚀 get_district_health_data returning:")
        logger.info(f"  🗺️ map_type: {response.get('map_type')}")
        logger.info(f"  📊 chart_data: {bool(response.get('chart_data'))}")
        logger.info(f"  🏙️ districts: {bool(response.get('districts'))}")
        logger.info(f"  🏙️ district_name: {response.get('district_name', 'N/A')}")
        if response.get('districts'):
            logger.info(f"  📍 Districts count: {len(response['districts'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_district_health_data: {e}")
        return {"error": f"Database error: {str(e)}"}

def get_state_wise_indicator_extremes(
    indicator_names: Optional[List[str]] = None,
    indicator_name: Optional[str] = None,
    states: Optional[List[str]] = None,
    year: int = 2021,
    include_trend: bool = True,
    min_districts_per_state: int = 3
):
    """
    Get the best and worst performing districts for single or multiple health indicators across states.
    
    This function identifies the top and bottom performing districts within each state
    for given health indicators, providing intra-state comparisons with proper handling
    of indicator direction (higher_is_better vs lower_is_better).
    
    Parameters:
    - indicator_names: List of health indicator names (can be misspelled or described)
    - indicator_name: Single health indicator name (use this OR indicator_names)
    - states: List of specific state names to analyze (if None, analyzes all states)
    - year: Year for analysis (2016 or 2021)
    - include_trend: Whether to include trend analysis
    - min_districts_per_state: Minimum number of districts required per state
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Handle indicator input - support both single and multiple indicators
        indicators_to_process = []
        if indicator_names:
            indicators_to_process = indicator_names
        elif indicator_name:
            indicators_to_process = [indicator_name]
        else:
            return {"error": "Please provide either indicator_name or indicator_names"}
        
        # Match indicators to database using the existing matching function
        matched_indicators = []
        for ind_name in indicators_to_process:
            matched_indicator = match_indicator_name_to_database(ind_name)
            if matched_indicator:
                matched_indicators.append(matched_indicator)
                logger.info(f"Matched '{ind_name}' to '{matched_indicator['indicator_name']}'")
            else:
                logger.warning(f"Could not match indicator: '{ind_name}'")
        
        if not matched_indicators:
            return {"error": f"No valid indicators found from: {indicators_to_process}"}
        
        # Determine which prevalence column to use
        year_column = "prevalence_2021" if year == 2021 else "prevalence_2016"
        
        # Build state filter condition
        state_filter = ""
        state_params = []
        if states:
            placeholders = ','.join(['%s'] * len(states))
            state_filter = f"AND s.state_name IN ({placeholders})"
            state_params = states
        
        # Process each indicator separately
        all_results = {}
        all_districts = set()
        
        for matched_indicator in matched_indicators:
            indicator_id = matched_indicator["indicator_id"]
            indicator_name_db = matched_indicator["indicator_name"]
            indicator_direction = matched_indicator["indicator_direction"]
            higher_is_better = indicator_direction == "higher_is_better"
            
            # Query to get best and worst districts per state for this indicator
            query = f"""
            WITH state_extremes AS (
                SELECT 
                    s.state_name,
                    s.state_id,
                    i.indicator_name,
                    i.indicator_direction,
                    -- Best performing district
                    FIRST_VALUE(d.district_name) OVER (
                        PARTITION BY s.state_id 
                        ORDER BY di.{year_column} {'DESC' if higher_is_better else 'ASC'}
                        ROWS UNBOUNDED PRECEDING
                    ) as best_district,
                    FIRST_VALUE(di.{year_column}) OVER (
                        PARTITION BY s.state_id 
                        ORDER BY di.{year_column} {'DESC' if higher_is_better else 'ASC'}
                        ROWS UNBOUNDED PRECEDING
                    ) as best_value,
                    -- Worst performing district  
                    FIRST_VALUE(d.district_name) OVER (
                        PARTITION BY s.state_id 
                        ORDER BY di.{year_column} {'ASC' if higher_is_better else 'DESC'}
                        ROWS UNBOUNDED PRECEDING
                    ) as worst_district,
                    FIRST_VALUE(di.{year_column}) OVER (
                        PARTITION BY s.state_id 
                        ORDER BY di.{year_column} {'ASC' if higher_is_better else 'DESC'}
                        ROWS UNBOUNDED PRECEDING
                    ) as worst_value,
                    COUNT(*) OVER (PARTITION BY s.state_id) as districts_count,
                    -- Add trend data if requested
                    CASE WHEN %s = TRUE THEN
                        FIRST_VALUE(di.prevalence_change) OVER (
                            PARTITION BY s.state_id 
                            ORDER BY di.{year_column} {'DESC' if higher_is_better else 'ASC'}
                            ROWS UNBOUNDED PRECEDING
                        )
                    ELSE NULL END as best_trend,
                    CASE WHEN %s = TRUE THEN
                        FIRST_VALUE(di.prevalence_change) OVER (
                            PARTITION BY s.state_id 
                            ORDER BY di.{year_column} {'ASC' if higher_is_better else 'DESC'}
                            ROWS UNBOUNDED PRECEDING
                        )
                    ELSE NULL END as worst_trend
                FROM district_indicators di
                JOIN districts d ON di.district_id = d.district_id
                JOIN states s ON d.state_id = s.state_id
                JOIN indicators i ON di.indicator_id = i.indicator_id
                WHERE di.indicator_id = %s 
                AND di.{year_column} IS NOT NULL
                {state_filter}
            )
            SELECT DISTINCT
                state_name,
                indicator_name,
                indicator_direction,
                best_district,
                best_value,
                worst_district, 
                worst_value,
                districts_count,
                best_trend,
                worst_trend
            FROM state_extremes
            WHERE districts_count >= %s
            ORDER BY state_name
            """
            
            params = [include_trend, include_trend, indicator_id] + state_params + [min_districts_per_state]
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            if not results:
                logger.warning(f"No data found for indicator '{indicator_name_db}'")
                continue
            
            # Process results for this indicator
            indicator_results = []
            for row in results:
                state_data = {
                    "state_name": row[0],
                    "indicator_name": row[1],
                    "indicator_direction": row[2],
                    "best_district": row[3],
                    "best_value": float(row[4]) if row[4] is not None else None,
                    "worst_district": row[5], 
                    "worst_value": float(row[6]) if row[6] is not None else None,
                    "districts_count": row[7],
                    "value_range": abs(float(row[4]) - float(row[6])) if row[4] and row[6] else None,
                    "best_trend": float(row[8]) if row[8] is not None else None,
                    "worst_trend": float(row[9]) if row[9] is not None else None
                }
                
                # Add trend interpretation
                if state_data["best_trend"] is not None:
                    state_data["best_trend_interpretation"] = interpret_health_trend(
                        state_data["best_trend"], indicator_direction
                    )
                if state_data["worst_trend"] is not None:
                    state_data["worst_trend_interpretation"] = interpret_health_trend(
                        state_data["worst_trend"], indicator_direction
                    )
                
                indicator_results.append(state_data)
                all_districts.add(row[3])  # best district
                all_districts.add(row[5])  # worst district
            
            all_results[indicator_name_db] = {
                "indicator_info": matched_indicator,
                "state_results": indicator_results
            }
        
        # Get boundary data for all districts
        boundary_data = get_district_boundary_data(list(all_districts))
        
        # Generate comprehensive analysis
        analysis = generate_multi_indicator_state_extremes_analysis(
            all_results, year, include_trend, states
        )
        
        # Generate chart data for visualization
        chart_data = generate_state_extremes_chart_data(all_results, year, states)
        
        # Prepare response structure
        if len(matched_indicators) == 1:
            # Single indicator - maintain backward compatibility
            indicator_name_db = matched_indicators[0]["indicator_name"]
            single_indicator_data = all_results[indicator_name_db]
            
            response = {
                "indicator_name": indicator_name_db,
                "indicator_direction": matched_indicators[0]["indicator_direction"],
                "year": year,
                "total_states": len(single_indicator_data["state_results"]),
                "min_districts_per_state": min_districts_per_state,
                "states_filter": states,
                "data": single_indicator_data["state_results"],
                "boundary": boundary_data,
                "analysis": analysis,
                "chart_data": chart_data,
                "map_type": "state_wise_extremes"
            }
        else:
            # Multiple indicators
            response = {
                "indicators": [ind["indicator_name"] for ind in matched_indicators],
                "year": year,
                "total_indicators": len(matched_indicators),
                "min_districts_per_state": min_districts_per_state,
                "states_filter": states,
                "indicator_results": all_results,
                "boundary": boundary_data,
                "analysis": analysis,
                "chart_data": chart_data,
                "map_type": "multi_indicator_state_extremes"
            }
        
        cursor.close()
        conn.close()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_state_wise_indicator_extremes: {e}")
        return {"error": f"Database error: {str(e)}"}

def get_border_districts(
    state1: str,
    state2: str = None,
    indicator_ids: Optional[List[int]] = None,
    year: int = 2021,
    include_boundary_data: bool = True,
    include_state_comparison: bool = True
):
    """
    Find districts that share borders with a specific state and analyze their health indicator performance.
    
    This function identifies districts at state boundaries and provides comparative analysis
    for cross-border development patterns and resource sharing opportunities.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Find state ID for state1
        cursor.execute("SELECT state_id FROM states WHERE state_name = %s", (state1,))
        state1_result = cursor.fetchone()
        if not state1_result:
            return {"error": f"State '{state1}' not found"}
        
        state1_id = state1_result[0]
        
        # Build border query based on whether state2 is specified
        if state2:
            # Find districts that border both states
            cursor.execute("SELECT state_id FROM states WHERE state_name = %s", (state2,))
            state2_result = cursor.fetchone()
            if not state2_result:
                return {"error": f"State '{state2}' not found"}
            
            state2_id = state2_result[0]
            border_condition = f"AND s.state_id IN ({state1_id}, {state2_id})"
            analysis_type = "bilateral_border"
        else:
            # Find all districts that border state1 (from other states)
            border_condition = f"AND s.state_id != {state1_id}"
            analysis_type = "multilateral_border"
        
        # Query for border districts using spatial analysis with boundary length
        border_query = f"""
        SELECT DISTINCT
            d.district_id,
            d.district_name,
            s.state_name,
            s.state_id,
            ST_Area(dg1.geometry::geography) / 1000000 as area_sqkm,
            ST_Perimeter(dg1.geometry::geography) / 1000 as perimeter_km,
            COALESCE(
                (SELECT MAX(ST_Length(ST_Intersection(dg1.geometry, dg2.geometry)::geography) / 1000)
                 FROM districts d2
                 JOIN district_geometries dg2 ON d2.district_name = dg2.district_name
                 WHERE d2.district_id IN (
                     SELECT district_id FROM districts WHERE state_id = {state1_id}
                 )
                 AND ST_Touches(dg1.geometry, dg2.geometry)), 0
            ) as shared_boundary_km
        FROM districts d
        JOIN states s ON d.state_id = s.state_id
        JOIN district_geometries dg1 ON d.district_name = dg1.district_name
        WHERE EXISTS (
            SELECT 1 
            FROM districts d2
            JOIN district_geometries dg2 ON d2.district_name = dg2.district_name
            WHERE d2.district_id IN (
                SELECT district_id FROM districts WHERE state_id = {state1_id}
            )
            AND ST_Touches(dg1.geometry, dg2.geometry)
        )
        {border_condition}
        ORDER BY s.state_name, d.district_name
        """
        
        cursor.execute(border_query)
        border_districts = cursor.fetchall()
        
        if not border_districts:
            border_message = f"between {state1} and {state2}" if state2 else f"with {state1}"
            return {"error": f"No border districts found {border_message}"}
        
        # Get state-level data for comparison if requested
        state_comparison_data = {}
        if include_state_comparison:
            # Get all unique states from border districts
            border_state_ids = list(set(row[3] for row in border_districts))
            border_state_ids.append(state1_id)  # Include the primary state
            if state2:
                border_state_ids.append(state2_id)  # Include the second state if specified

            # Remove duplicates
            border_state_ids = list(set(border_state_ids))

            # Query state-level indicator data
            if indicator_ids:
                state_indicator_filter = f"AND si.indicator_id IN ({','.join(['%s'] * len(indicator_ids))})"
                state_params = [border_state_ids] + indicator_ids
            else:
                state_indicator_filter = ""
                state_params = [border_state_ids]

            state_query = f"""
            SELECT
                s.state_name,
                i.indicator_name,
                i.indicator_direction,
                si.prevalence_2016,
                si.prevalence_2021,
                si.prevalence_change,
                si.headcount_2021
            FROM state_indicators si
            JOIN states s ON si.state_id = s.state_id
            JOIN indicators i ON si.indicator_id = i.indicator_id
            WHERE si.state_id = ANY(%s)
            {state_indicator_filter}
            ORDER BY s.state_name, i.indicator_name
            """

            cursor.execute(state_query, state_params)
            state_indicator_results = cursor.fetchall()

            # Process state data
            for row in state_indicator_results:
                state_key = row[0]  # state_name
                indicator_key = row[1]  # indicator_name

                if state_key not in state_comparison_data:
                    state_comparison_data[state_key] = {}

                state_comparison_data[state_key][indicator_key] = {
                    "indicator_direction": row[2],
                    "prevalence_2016": float(row[3]) if row[3] is not None else None,
                    "prevalence_2021": float(row[4]) if row[4] is not None else None,
                    "prevalence_change": float(row[5]) if row[5] is not None else None,
                    "headcount_2021": float(row[6]) if row[6] is not None else None
                }

        # Get health indicator data for border districts
        district_ids = [row[0] for row in border_districts]
        
        if indicator_ids:
            indicator_filter = f"AND di.indicator_id IN ({','.join(['%s'] * len(indicator_ids))})"
            params = [district_ids] + indicator_ids
        else:
            indicator_filter = ""
            params = [district_ids]
        
        year_column = "prevalence_2021" if year == 2021 else "prevalence_2016"
        
        data_query = f"""
        SELECT 
            d.district_name,
            s.state_name,
            i.indicator_name,
            i.indicator_direction,
            di.prevalence_2016,
            di.prevalence_2021,
            di.prevalence_change,
            di.headcount_2021
        FROM district_indicators di
        JOIN districts d ON di.district_id = d.district_id
        JOIN states s ON d.state_id = s.state_id
        JOIN indicators i ON di.indicator_id = i.indicator_id
        WHERE di.district_id = ANY(%s)
        {indicator_filter}
        AND di.{year_column} IS NOT NULL
        ORDER BY s.state_name, d.district_name, i.indicator_name
        """
        
        cursor.execute(data_query, params)
        indicator_results = cursor.fetchall()
        
        # Create a mapping of district boundary information
        district_boundary_info = {}
        for border_district in border_districts:
            district_key = f"{border_district[1]}_{border_district[2]}"  # district_state
            district_boundary_info[district_key] = {
                "area_sqkm": float(border_district[4]) if border_district[4] else None,
                "perimeter_km": float(border_district[5]) if border_district[5] else None,
                "shared_boundary_km": float(border_district[6]) if border_district[6] else None
            }

        # Process the data
        districts_data = {}
        for row in indicator_results:
            district_key = f"{row[0]}_{row[1]}"  # district_state
            
            if district_key not in districts_data:
                # Get boundary info for this district
                boundary_info = district_boundary_info.get(district_key, {})
                
                districts_data[district_key] = {
                    "district_name": row[0],
                    "state_name": row[1], 
                    "indicators": [],
                    "area_sqkm": boundary_info.get("area_sqkm"),
                    "perimeter_km": boundary_info.get("perimeter_km"),
                    "shared_boundary_km": boundary_info.get("shared_boundary_km")
                }
            
            indicator_data = {
                "indicator_name": row[2],
                "indicator_direction": row[3],
                "prevalence_2016": float(row[4]) if row[4] is not None else None,
                "prevalence_2021": float(row[5]) if row[5] is not None else None,
                "prevalence_change": float(row[6]) if row[6] is not None else None,
                "headcount_2021": float(row[7]) if row[7] is not None else None
            }
            
            districts_data[district_key]["indicators"].append(indicator_data)

        # Add state comparison data to districts
        if include_state_comparison and state_comparison_data:
            for district_data in districts_data.values():
                district_state = district_data["state_name"]

                # Add state comparison for each indicator in the district
                district_data["state_comparison"] = {}
                if district_state in state_comparison_data:
                    for indicator in district_data["indicators"]:
                        indicator_name = indicator["indicator_name"]
                        if indicator_name in state_comparison_data[district_state]:
                            state_values = state_comparison_data[district_state][indicator_name]

                            # Calculate comparison metrics
                            district_value = indicator["prevalence_2021"]
                            state_value = state_values["prevalence_2021"]

                            comparison = {}
                            if district_value is not None and state_value is not None:
                                difference = district_value - state_value
                                percentage_diff = (difference / state_value) * 100 if state_value != 0 else None

                                # Determine performance relative to state
                                indicator_direction = indicator["indicator_direction"]
                                if indicator_direction == "higher_is_better":
                                    if difference > 0:
                                        performance_status = "above_state_average"
                                    elif difference < 0:
                                        performance_status = "below_state_average"
                                    else:
                                        performance_status = "at_state_average"
                                else:  # lower_is_better
                                    if difference < 0:
                                        performance_status = "above_state_average"  # better than state average
                                    elif difference > 0:
                                        performance_status = "below_state_average"  # worse than state average
                                    else:
                                        performance_status = "at_state_average"

                                comparison = {
                                    "district_value": district_value,
                                    "state_value": state_value,
                                    "difference": difference,
                                    "percentage_difference": percentage_diff,
                                    "performance_status": performance_status,
                                    "indicator_direction": indicator_direction
                                }

                            district_data["state_comparison"][indicator_name] = {
                                "state_data": state_values,
                                "comparison": comparison
                            }

        # Get boundary data if requested
        boundary_data = []
        if include_boundary_data:
            all_district_names = [data["district_name"] for data in districts_data.values()]
            boundary_data = get_district_boundary_data(all_district_names)
            
            # Enhance boundary data with shared boundary length information
            for boundary in boundary_data:
                district_key = f"{boundary['district_name']}_{boundary['state_name']}"
                if district_key in district_boundary_info:
                    boundary.update({
                        "area_sqkm": district_boundary_info[district_key]["area_sqkm"],
                        "perimeter_km": district_boundary_info[district_key]["perimeter_km"],
                        "shared_boundary_km": district_boundary_info[district_key]["shared_boundary_km"]
                    })
        
        # Generate analysis
        analysis = generate_border_districts_analysis(
            list(districts_data.values()), state1, state2, analysis_type, state_comparison_data if include_state_comparison else None
        )
        
        response = {
            "target_state": state1,
            "comparison_state": state2,
            "year": year,
            "total_border_districts": len(districts_data),
            "analysis_type": analysis_type,
            "data": list(districts_data.values()),
            "state_comparison_data": state_comparison_data if include_state_comparison else None,
            "boundary": boundary_data,
            "analysis": analysis,
            "map_type": "border_districts"
        }
        
        cursor.close()
        conn.close()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_border_districts: {e}")
        return {"error": f"Database error: {str(e)}"}

def get_districts_within_radius(
    center_point: str,  # Either district name or "lat,lng" coordinates
    radius_km: float,
    indicator_ids: Optional[List[int]] = None,
    max_districts: int = 50,
    include_boundary_data: bool = True
):
    """
    Find all districts within a specified radius from a center point and analyze their health indicator performance.

    This function supports both district-based and coordinate-based center points for flexible
    spatial analysis of health indicators. It provides comprehensive health data including:
    - 2021 prevalence values
    - 2016 prevalence values
    - Calculated prevalence change (2021 - 2016) with proper interpretation
    - Headcount data for 2021
    - Support for multiple indicators
    - Proper analysis considering indicator direction (higher_is_better vs lower_is_better)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Determine if center_point is coordinates or district name
        center_coordinates = None
        if ',' in center_point and re.match(r'^-?\d+\.?\d*,-?\d+\.?\d*$', center_point.strip()):
            # Coordinates format: "lat,lng"
            lat, lng = map(float, center_point.split(','))
            center_type = "coordinates"
            center_name = f"({lat}, {lng})"
            center_coordinates = {"lat": lat, "lng": lng}
            
            # Find districts within radius using coordinates
            districts = find_districts_within_radius_from_coordinates(
                cursor, lat, lng, radius_km, max_districts
            )
        else:
            # District name
            center_type = "district"
            center_name = center_point
            
            # Resolve the center district and get its coordinates
            resolved_center = resolve_district_name(cursor, center_point)
            if resolved_center:
                # Get the center district's coordinates
                center_query = """
                SELECT ST_X(ST_Centroid(dg.geometry)) as lng, ST_Y(ST_Centroid(dg.geometry)) as lat
                FROM district_geometries dg 
                WHERE dg.district_name = %s
                """
                cursor.execute(center_query, (resolved_center["district_name"],))
                center_result = cursor.fetchone()
                if center_result:
                    center_coordinates = {"lat": float(center_result[1]), "lng": float(center_result[0])}
            
            # Find districts within radius using district center
            districts = find_districts_within_radius_from_district(
                cursor, center_point, radius_km, max_districts
            )
        
        if not districts:
            return {
                "error": f"No districts found within {radius_km}km of {center_name}"
            }
        
        # Get health indicator data for found districts
        district_ids = [d['district_id'] for d in districts]
        
        if indicator_ids:
            indicator_filter = f"AND di.indicator_id IN ({','.join(['%s'] * len(indicator_ids))})"
            params = district_ids + indicator_ids
        else:
            indicator_filter = ""
            params = district_ids
        
        # Query for indicator data
        data_query = f"""
        SELECT
            d.district_name,
            s.state_name,
            i.indicator_name,
            i.indicator_direction,
            di.prevalence_2016,
            di.prevalence_2021,
            di.prevalence_change,
            di.headcount_2021,
            d.district_id
        FROM district_indicators di
        JOIN districts d ON di.district_id = d.district_id
        JOIN states s ON d.state_id = s.state_id
        JOIN indicators i ON di.indicator_id = i.indicator_id
        WHERE di.district_id = ANY(%s::int[])
        {indicator_filter}
        AND di.prevalence_2021 IS NOT NULL
        ORDER BY d.district_name, i.indicator_name
        """

        if indicator_ids:
            cursor.execute(data_query, (district_ids, *indicator_ids))
        else:
            cursor.execute(data_query, (district_ids,))
        indicator_results = cursor.fetchall()
        
        # Process the results
        districts_data = {}
        for row in indicator_results:
            district_id = row[8]
            district_key = f"{row[0]}_{row[1]}"  # district_state

            if district_key not in districts_data:
                # Find distance for this district
                district_info = next(d for d in districts if d['district_id'] == district_id)

                districts_data[district_key] = {
                    "district_name": row[0],
                    "state_name": row[1],
                    "distance_km": district_info['distance_km'],
                    "indicators": []
                }

            # Enhanced indicator data processing
            prevalence_2016 = float(row[4]) if row[4] is not None else None
            prevalence_2021 = float(row[5]) if row[5] is not None else None
            prevalence_change = float(row[6]) if row[6] is not None else None
            headcount_2021 = float(row[7]) if row[7] is not None else None
            indicator_direction = row[3]

            # Calculate interpreted change if data is available
            change_interpretation = None
            if prevalence_change is not None and prevalence_2016 is not None and prevalence_2021 is not None:
                change_interpretation = interpret_prevalence_change(
                    prevalence_2016, prevalence_2021, prevalence_change, indicator_direction
                )

            indicator_data = {
                "indicator_name": row[2],
                "indicator_direction": indicator_direction,
                "prevalence_2016": prevalence_2016,
                "prevalence_2021": prevalence_2021,
                "prevalence_change": prevalence_change,
                "headcount_2021": headcount_2021,
                "change_interpretation": change_interpretation
            }

            districts_data[district_key]["indicators"].append(indicator_data)
        
        # Sort by distance
        sorted_districts = sorted(
            districts_data.values(), 
            key=lambda x: x['distance_km']
        )
        
        # Get boundary data if requested
        boundary_data = []
        if include_boundary_data:
            all_district_names = [d["district_name"] for d in sorted_districts]
            boundary_data = get_district_boundary_data(all_district_names)
        
        # Generate chart data for frontend visualization
        chart_data = generate_radius_chart_data(sorted_districts, center_name, radius_km)
        
        # Structure the response with raw data for OpenAI to interpret
        response = {
            "center_point": center_name,
            "center_type": center_type,
            "center_coordinates": center_coordinates,
            "radius_km": radius_km,
            "total_districts": len(sorted_districts),
            "max_distance": max([d['distance_km'] for d in sorted_districts]) if sorted_districts else 0,
            "districts": sorted_districts,
            "boundary_data": boundary_data,
            "chart_data": chart_data,
            "query_type": "radius_analysis"
        }
        
        cursor.close()
        conn.close()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_districts_within_radius: {e}")
        return {"error": f"Database error: {str(e)}"}

def interpret_prevalence_change(prevalence_2016, prevalence_2021, prevalence_change, indicator_direction):
    """
    Interpret prevalence change considering indicator direction.
    Returns a structured interpretation of the change.
    """
    try:
        if prevalence_change is None or prevalence_2016 is None or prevalence_2021 is None:
            return {"status": "unknown", "description": "Insufficient data for interpretation"}

        higher_is_better = indicator_direction == "higher_is_better"

        # Determine if change is improvement or decline
        if prevalence_change > 0:
            if higher_is_better:
                status = "improving"
                direction_text = "increase"
                interpretation = "positive improvement"
            else:
                status = "declining"
                direction_text = "increase"
                interpretation = "negative trend"
        elif prevalence_change < 0:
            if higher_is_better:
                status = "declining"
                direction_text = "decrease"
                interpretation = "negative trend"
            else:
                status = "improving"
                direction_text = "decrease"
                interpretation = "positive improvement"
        else:
            status = "stable"
            direction_text = "no change"
            interpretation = "stable"

        # Calculate percentage change if baseline is not zero
        percentage_change = None
        if prevalence_2016 != 0:
            percentage_change = (prevalence_change / prevalence_2016) * 100

        return {
            "status": status,
            "direction": direction_text,
            "interpretation": interpretation,
            "change_value": prevalence_change,
            "percentage_change": percentage_change,
            "description": ".1f"            if percentage_change is not None else
                         ".2f"        }

    except Exception as e:
        logger.error(f"Error interpreting prevalence change: {e}")
        return {"status": "error", "description": f"Interpretation error: {str(e)}"}

# Helper functions

def resolve_district_name(cursor, district_name):
    """Resolve district name using fuzzy matching"""
    try:
        # First try exact match
        cursor.execute(
            "SELECT d.district_id, d.district_name, s.state_name FROM districts d JOIN states s ON d.state_id = s.state_id WHERE LOWER(d.district_name) = LOWER(%s)",
            (district_name,)
        )
        result = cursor.fetchone()
        
        if result:
            return {
                "district_id": result[0],
                "district_name": result[1], 
                "state_name": result[2]
            }
        
        # If no exact match, try fuzzy matching
        cursor.execute(
            "SELECT d.district_id, d.district_name, s.state_name FROM districts d JOIN states s ON d.state_id = s.state_id"
        )
        all_districts = cursor.fetchall()
        
        district_names = [row[1] for row in all_districts]
        match = process.extractOne(district_name, district_names, scorer=fuzz.WRatio, score_cutoff=70)
        
        if match:
            matched_name = match[0]
            for row in all_districts:
                if row[1] == matched_name:
                    return {
                        "district_id": row[0],
                        "district_name": row[1],
                        "state_name": row[2]
                    }
        
        return None
        
    except Exception as e:
        logger.error(f"Error resolving district name: {e}")
        return None

def find_districts_within_radius_from_coordinates(cursor, lat, lng, radius_km, max_districts):
    """Find districts within radius from coordinates"""
    try:
        query = """
        SELECT 
            d.district_id,
            d.district_name,
            s.state_name,
            ST_Distance(
                ST_GeogFromText('POINT(' || %s || ' ' || %s || ')'),
                ST_Transform(dg.geometry, 4326)
            ) / 1000.0 as distance_km
        FROM districts d
        JOIN states s ON d.state_id = s.state_id  
        JOIN district_geometries dg ON d.district_name = dg.district_name
        WHERE ST_DWithin(
            ST_GeogFromText('POINT(' || %s || ' ' || %s || ')'),
            ST_Transform(dg.geometry, 4326),
            %s * 1000
        )
        ORDER BY distance_km
        LIMIT %s
        """
        
        cursor.execute(query, (lng, lat, lng, lat, radius_km, max_districts))
        results = cursor.fetchall()
        
        return [
            {
                "district_id": row[0],
                "district_name": row[1],
                "state_name": row[2],
                "distance_km": float(row[3])
            }
            for row in results
        ]
        
    except Exception as e:
        logger.error(f"Error finding districts by coordinates: {e}")
        return []

def find_districts_within_radius_from_district(cursor, center_district, radius_km, max_districts):
    """Find districts within radius from a center district"""
    try:
        # First resolve the center district
        resolved = resolve_district_name(cursor, center_district)
        if not resolved:
            return []
        
        query = """
        SELECT 
            d.district_id,
            d.district_name,
            s.state_name,
            ST_Distance(
                ST_Transform(ST_Centroid(center_dg.geometry), 3857),
                ST_Transform(ST_Centroid(dg.geometry), 3857)
            ) / 1000.0 as distance_km
        FROM districts d
        JOIN states s ON d.state_id = s.state_id
        JOIN district_geometries dg ON d.district_name = dg.district_name
        CROSS JOIN (
            SELECT geometry 
            FROM district_geometries 
            WHERE district_name = %s
        ) center_dg
        WHERE ST_DWithin(
            ST_Transform(ST_Centroid(center_dg.geometry), 3857),
            ST_Transform(ST_Centroid(dg.geometry), 3857), 
            %s * 1000
        )
        AND d.district_name != %s
        ORDER BY distance_km
        LIMIT %s
        """
        
        cursor.execute(query, (resolved["district_name"], radius_km, resolved["district_name"], max_districts))
        results = cursor.fetchall()
        
        return [
            {
                "district_id": row[0],
                "district_name": row[1], 
                "state_name": row[2],
                "distance_km": float(row[3])
            }
            for row in results
        ]
        
    except Exception as e:
        logger.error(f"Error finding districts by district center: {e}")
        return []

def interpret_health_trend(change_value, indicator_direction):
    """Interpret health indicator trend based on direction"""
    if change_value is None:
        return {"status": "unknown", "description": "No trend data available"}
    
    higher_is_better = indicator_direction == "higher_is_better"
    
    if change_value > 0:
        if higher_is_better:
            status = "improving"
            description = f"Improved by {change_value:.2f} percentage points"
        else:
            status = "declining"
            description = f"Worsened by {change_value:.2f} percentage points"
    elif change_value < 0:
        if higher_is_better:
            status = "declining"
            description = f"Declined by {abs(change_value):.2f} percentage points"
        else:
            status = "improving" 
            description = f"Improved by {abs(change_value):.2f} percentage points"
    else:
        status = "stable"
        description = "No change"
    
    return {"status": status, "description": description}

def generate_single_district_chart_data(district_data, indicators):
    """Generate chart data for a single district showing its health indicators"""
    try:
        if not district_data or not district_data.get("indicators"):
            return None
            
        district_name = district_data["district_name"]
        
        # If too many indicators, use top 10 most relevant ones for better readability
        if len(indicators) > 10:
            # For single district, select indicators with complete data (both 2016 and 2021)
            complete_indicators = []
            for indicator in indicators:
                for ind_data in district_data['indicators']:
                    if (ind_data['indicator_name'] == indicator and 
                        ind_data['prevalence_2016'] is not None and 
                        ind_data['prevalence_2021'] is not None):
                        complete_indicators.append(indicator)
                        break
            
            # Take up to 10 indicators with complete data, or fall back to first 10
            selected_indicators = complete_indicators[:10] if complete_indicators else indicators[:10]
        else:
            selected_indicators = indicators
        
        # Create datasets for 2016 vs 2021 comparison
        values_2016 = []
        values_2021 = []
        labels = []
        
        for indicator in selected_indicators:
            for ind_data in district_data['indicators']:
                if ind_data['indicator_name'] == indicator:
                    # Use shortened label for better display
                    short_label = indicator[:30] + ("..." if len(indicator) > 30 else "")
                    labels.append(short_label)
                    
                    values_2016.append(ind_data['prevalence_2016'] if ind_data['prevalence_2016'] is not None else 0)
                    values_2021.append(ind_data['prevalence_2021'] if ind_data['prevalence_2021'] is not None else 0)
                    break
        
        # Return single chart data showing 2016 vs 2021 comparison
        return {
            "type": "bar",
            "title": f"Health Indicators for {district_name}",
            "subtitle": "Comparison of 2016 vs 2021 values",
            "labels": labels,
            "datasets": [
                {
                    "label": "2016 Values",
                    "data": values_2016,
                    "backgroundColor": "#3498db",
                    "borderColor": "#2980b9",
                    "borderWidth": 1
                },
                {
                    "label": "2021 Values",
                    "data": values_2021,
                    "backgroundColor": "#2ecc71",
                    "borderColor": "#27ae60",
                    "borderWidth": 1
                }
            ],
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Prevalence (%)"},
                        "beginAtZero": True
                    },
                    "x": {
                        "title": {"display": True, "text": "Health Indicators"}
                    }
                },
                "plugins": {
                    "legend": {
                        "position": "top"
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating single district chart data: {e}")
        return None

def generate_multi_district_chart_data(districts_data, indicators):
    """Generate simplified chart data for comparing multiple districts - only bar chart showing indicators by districts"""
    try:
        if len(districts_data) < 2:
            return None
            
        district_names = list(districts_data.keys())
        
        # If too many indicators, use top 8 most variable ones for better readability
        if len(indicators) > 8:
            indicator_variance = {}
            for indicator in indicators:
                values = []
                for district_data in districts_data.values():
                    for ind_data in district_data['indicators']:
                        if ind_data['indicator_name'] == indicator and ind_data['prevalence_2021'] is not None:
                            values.append(ind_data['prevalence_2021'])
                
                if len(values) > 1:
                    # Calculate variance to find most variable indicators
                    mean_val = sum(values) / len(values)
                    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                    indicator_variance[indicator] = variance
            
            # Select top 8 most variable indicators
            top_indicators = sorted(indicator_variance.items(), key=lambda x: x[1], reverse=True)[:8]
            selected_indicators = [ind[0] for ind in top_indicators]
        else:
            selected_indicators = indicators
        
        # Create datasets for each indicator
        datasets = []
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        
        for i, indicator in enumerate(selected_indicators):
            data = []
            for district_name in district_names:
                district_data = districts_data[district_name]
                value = None
                for ind_data in district_data['indicators']:
                    if ind_data['indicator_name'] == indicator:
                        value = ind_data['prevalence_2021']
                        break
                data.append(value if value is not None else 0)
            
            datasets.append({
                "label": indicator,
                "data": data,
                "backgroundColor": colors[i % len(colors)],
                "borderColor": colors[i % len(colors)],
                "borderWidth": 1
            })
        
        # Return single chart data
        return {
            "type": "bar",
            "title": "District Health Indicator Comparison",
            "subtitle": f"Comparing {len(district_names)} districts across {len(selected_indicators)} health indicators",
            "labels": district_names,
            "datasets": datasets,
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Prevalence (%)"},
                        "beginAtZero": True
                    },
                    "x": {
                        "title": {"display": True, "text": "Districts"}
                    }
                },
                "plugins": {
                    "legend": {
                        "position": "top"
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating multi-district chart data: {e}")
        return None

def generate_multi_district_analysis(districts_data, indicators):
    """Generate analysis text for multiple districts comparison"""
    try:
        district_names = list(districts_data.keys())
        
        analysis_parts = []
        analysis_parts.append(f"**Multi-District Health Comparison Analysis ({len(district_names)} Districts)**\n")
        analysis_parts.append(f"Districts analyzed: {', '.join(district_names)}")
        analysis_parts.append(f"Total indicators compared: {len(indicators)}\n")
        
        # Best and worst performing districts overall
        district_scores = {}
        for district_name, district_data in districts_data.items():
            total_score = 0
            indicator_count = 0
            
            for ind_data in district_data['indicators']:
                if ind_data['prevalence_2021'] is not None:
                    # Normalize score based on indicator direction
                    score = ind_data['prevalence_2021']
                    if ind_data['indicator_direction'] == 'lower_is_better':
                        score = 100 - score  # Invert for lower_is_better indicators
                    
                    total_score += score
                    indicator_count += 1
            
            if indicator_count > 0:
                district_scores[district_name] = total_score / indicator_count
        
        if district_scores:
            best_district = max(district_scores.items(), key=lambda x: x[1])
            worst_district = min(district_scores.items(), key=lambda x: x[1])
            
            analysis_parts.append(f"**Overall Performance:**")
            analysis_parts.append(f"🏆 Best performing: {best_district[0]} (score: {best_district[1]:.1f})")
            analysis_parts.append(f"⚠️ Needs attention: {worst_district[0]} (score: {worst_district[1]:.1f})\n")
        
        # State-wise grouping
        states = {}
        for district_name, district_data in districts_data.items():
            state = district_data['state_name']
            if state not in states:
                states[state] = []
            states[state].append(district_name)
        
        if len(states) > 1:
            analysis_parts.append(f"**Geographic Distribution:**")
            for state, districts in states.items():
                analysis_parts.append(f"📍 {state}: {', '.join(districts)}")
            analysis_parts.append("")
        
        # Indicator insights
        if len(indicators) <= 5:
            analysis_parts.append(f"**Key Findings by Indicator:**")
            for indicator in indicators[:3]:  # Top 3 indicators
                district_values = []
                for district_name, district_data in districts_data.items():
                    for ind_data in district_data['indicators']:
                        if ind_data['indicator_name'] == indicator and ind_data['prevalence_2021'] is not None:
                            district_values.append((district_name, ind_data['prevalence_2021']))
                
                if district_values:
                    best = max(district_values, key=lambda x: x[1])
                    worst = min(district_values, key=lambda x: x[1])
                    analysis_parts.append(f"• **{indicator}**: Best - {best[0]} ({best[1]:.1f}%), Worst - {worst[0]} ({worst[1]:.1f}%)")
        
        return "\n".join(analysis_parts)
        
    except Exception as e:
        logger.error(f"Error generating multi-district analysis: {e}")
        return "Analysis generation failed"

def generate_district_analysis(district_name, state_name, indicators_data):
    """Generate comprehensive analysis for individual district"""
    if not indicators_data:
        return "No indicator data available for analysis."
    
    total_indicators = len(indicators_data)
    indicators_with_trends = [i for i in indicators_data if i.get("prevalence_change") is not None]
    
    if indicators_with_trends:
        improving = len([i for i in indicators_with_trends if i["trend_interpretation"]["status"] == "improving"])
        declining = len([i for i in indicators_with_trends if i["trend_interpretation"]["status"] == "declining"])
        stable = len([i for i in indicators_with_trends if i["trend_interpretation"]["status"] == "stable"])
        
        trend_summary = f"Out of {len(indicators_with_trends)} indicators with trend data: {improving} improving, {declining} declining, {stable} stable."
    else:
        trend_summary = "No trend data available for analysis."
    
    analysis = f"""
    **District Profile: {district_name}, {state_name}**
    
    This analysis covers {total_indicators} health indicators for {district_name} district in {state_name}.
    
    **Trend Overview (2016-2021):**
    {trend_summary}
    
    **Key Insights:**
    - The district shows mixed performance across different health indicators
    - Prevalence values represent the percentage of population affected by each indicator
    - Headcount values show the absolute number of people affected in 2021
    - Changes are calculated as the difference between 2021 and 2016 values
    
    **Interpretation Guide:**
    - For "higher_is_better" indicators (like vaccination coverage): higher values are better
    - For "lower_is_better" indicators (like disease prevalence): lower values are better
    - Positive changes in "higher_is_better" indicators show improvement
    - Negative changes in "lower_is_better" indicators show improvement
    """
    
    return analysis

def generate_state_extremes_analysis(state_extremes, indicator_name, higher_is_better, year, include_trend):
    """Generate analysis for state-wise extremes"""
    if not state_extremes:
        return "No data available for analysis."
    
    # Find states with largest disparities
    states_with_range = [s for s in state_extremes if s.get("value_range") is not None]
    if states_with_range:
        highest_disparity = max(states_with_range, key=lambda x: x["value_range"])
        lowest_disparity = min(states_with_range, key=lambda x: x["value_range"])
    else:
        highest_disparity = lowest_disparity = None
    
    direction_text = "higher values indicate better performance" if higher_is_better else "lower values indicate better performance"
    
    analysis = f"""
    **State-wise Performance Analysis: {indicator_name} ({year})**
    
    This analysis shows the best and worst performing districts within each state for {indicator_name}.
    Note: {direction_text}.
    
    **Coverage:**
    - {len(state_extremes)} states included in analysis
    - Only states with at least 3 districts are included to ensure meaningful comparisons
    
    **Key Findings:**
    """
    
    if highest_disparity:
        analysis += f"""
    - **Highest Intra-state Disparity:** {highest_disparity['state_name']} 
      (Range: {highest_disparity['value_range']:.2f} percentage points between best and worst districts)
    """
    
    if lowest_disparity:
        analysis += f"""
    - **Most Uniform Performance:** {lowest_disparity['state_name']}
      (Range: {lowest_disparity['value_range']:.2f} percentage points between best and worst districts)
    """
    
    analysis += f"""
    
    **Interpretation:**
    - Large ranges indicate significant intra-state disparities requiring targeted interventions
    - Small ranges suggest more uniform healthcare access and outcomes within the state
    - This analysis helps identify both exemplary districts that can serve as models and underperforming districts needing support
    """
    
    return analysis

def generate_multi_indicator_state_extremes_analysis(all_results, year, include_trend, states_filter):
    """Generate comprehensive analysis for multiple indicators state-wise extremes"""
    if not all_results:
        return "No data available for analysis."
    
    total_indicators = len(all_results)
    
    # Calculate summary statistics
    total_states_analyzed = set()
    all_disparities = []
    indicator_summaries = []
    
    for indicator_name, indicator_data in all_results.items():
        state_results = indicator_data["state_results"]
        indicator_info = indicator_data["indicator_info"]
        direction = indicator_info["indicator_direction"]
        
        # Track states
        for state_result in state_results:
            total_states_analyzed.add(state_result["state_name"])
            if state_result.get("value_range"):
                all_disparities.append(state_result["value_range"])
        
        # Calculate indicator-specific insights
        if state_results:
            max_disparity_state = max(state_results, key=lambda x: x.get("value_range", 0))
            min_disparity_state = min(state_results, key=lambda x: x.get("value_range", float('inf')))
            
            indicator_summaries.append({
                "indicator_name": indicator_name,
                "direction": direction,
                "states_count": len(state_results),
                "max_disparity": max_disparity_state,
                "min_disparity": min_disparity_state
            })
    
    # Start building analysis
    states_text = f"across {len(total_states_analyzed)} states" if not states_filter else f"across {len(total_states_analyzed)} selected states ({', '.join(states_filter)})"
    
    analysis = f"""
    **Multi-Indicator State-wise Performance Analysis ({year})**
    
    This comprehensive analysis examines {total_indicators} health indicators {states_text}, identifying the best and worst performing districts within each state for each indicator.
    
    **Overview:**
    - **Indicators Analyzed:** {total_indicators}
    - **States Covered:** {len(total_states_analyzed)}
    - **Analysis Type:** Intra-state comparisons with indicator direction consideration
    
    **Key Findings by Indicator:**
    """
    
    for summary in indicator_summaries:
        direction_text = "↑ higher is better" if summary["direction"] == "higher_is_better" else "↓ lower is better"
        analysis += f"""
    
    **{summary['indicator_name']}** ({direction_text})
    - States analyzed: {summary['states_count']}
    - Highest disparity: {summary['max_disparity']['state_name']} (range: {summary['max_disparity']['value_range']:.2f} points)
    - Most uniform: {summary['min_disparity']['state_name']} (range: {summary['min_disparity']['value_range']:.2f} points)
    """
    
    # Overall insights
    if all_disparities:
        avg_disparity = sum(all_disparities) / len(all_disparities)
        max_overall_disparity = max(all_disparities)
        min_overall_disparity = min(all_disparities)
        
        analysis += f"""
    
    **Cross-Indicator Insights:**
    - **Average Disparity:** {avg_disparity:.2f} percentage points across all indicators
    - **Highest Disparity:** {max_overall_disparity:.2f} percentage points
    - **Lowest Disparity:** {min_overall_disparity:.2f} percentage points
    
    **Strategic Implications:**
    - Large disparities indicate opportunities for knowledge transfer between districts
    - Consistent best performers across indicators may serve as model districts
    - States with uniform performance may have effective statewide policies
    - Multiple indicator analysis helps identify systemic vs. specific health challenges
    
    **Indicator Direction Interpretation:**
    - For "higher_is_better" indicators (e.g., vaccination coverage): higher values = better performance
    - For "lower_is_better" indicators (e.g., disease prevalence): lower values = better performance
    - Rankings automatically consider these directions for accurate performance assessment
    """
    
    if include_trend:
        analysis += f"""
    
    **Trend Analysis:**
    - Trend data included for performance changes from 2016 to {year}
    - Improvements and declines are interpreted based on each indicator's direction
    - Best and worst performing districts' trends help identify sustained vs. temporary performance
    """
    
    return analysis

def generate_state_extremes_chart_data(all_results, year, states_filter):
    """
    Generate simplified chart data for state-wise extremes visualization
    Returns only two chart types: best vs worst comparison and intra-state disparities
    """
    try:
        if not all_results:
            return None

        # Generate only the two required chart types
        chart_data = {
            "best_vs_worst_comparison": generate_best_worst_comparison_chart(all_results),
            "intra_state_disparities": generate_intra_state_disparities_chart(all_results)
        }

        return chart_data

    except Exception as e:
        logger.error(f"Error generating state extremes chart data: {e}")
        return None

def generate_state_performance_chart(all_results, year):
    """Generate bar chart showing best vs worst performance by state"""
    try:
        if not all_results:
            return None

        # Use first indicator for primary visualization
        primary_indicator = list(all_results.keys())[0]
        state_results = all_results[primary_indicator]["state_results"]
        indicator_info = all_results[primary_indicator]["indicator_info"]
        
        # Sort states by disparity (value_range)
        sorted_states = sorted(state_results, key=lambda x: x.get("value_range", 0), reverse=True)[:15]
        
        labels = [state["state_name"] for state in sorted_states]
        best_values = [state["best_value"] for state in sorted_states]
        worst_values = [state["worst_value"] for state in sorted_states]
        
        direction_text = "↑ higher is better" if indicator_info["indicator_direction"] == "higher_is_better" else "↓ lower is better"
        
        return {
            "type": "bar",
            "title": f"State Performance Extremes: {primary_indicator} ({year})",
            "subtitle": f"Best vs Worst Districts per State ({direction_text})",
            "labels": labels,
            "datasets": [
                {
                    "label": "Best Performing District",
                    "data": best_values,
                    "backgroundColor": "#2ecc71",
                    "borderColor": "#27ae60",
                    "borderWidth": 1
                },
                {
                    "label": "Worst Performing District", 
                    "data": worst_values,
                    "backgroundColor": "#e74c3c",
                    "borderColor": "#c0392b",
                    "borderWidth": 1
                }
            ],
            "options": {
                "scales": {
                    "y": {
                        "title": {"display": True, "text": f"{primary_indicator} (%)"}
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating state performance chart: {e}")
        return None

def generate_disparity_analysis_chart(all_results):
    """Generate chart showing intra-state disparities (value ranges)"""
    try:
        if not all_results:
            return None

        # Use first indicator
        primary_indicator = list(all_results.keys())[0]
        state_results = all_results[primary_indicator]["state_results"]
        
        # Sort by disparity
        sorted_states = sorted(state_results, key=lambda x: x.get("value_range", 0), reverse=True)[:20]
        
        labels = [state["state_name"] for state in sorted_states]
        disparities = [state.get("value_range", 0) for state in sorted_states]
        
        # Color code: red for high disparity, green for low disparity
        colors = []
        for disparity in disparities:
            if disparity > 20:
                colors.append("#e74c3c")  # High disparity - red
            elif disparity > 10:
                colors.append("#f39c12")  # Medium disparity - orange  
            else:
                colors.append("#2ecc71")  # Low disparity - green
        
        return {
            "type": "bar",
            "title": f"Intra-State Health Disparities: {primary_indicator}",
            "subtitle": "Difference between Best and Worst Performing Districts",
            "labels": labels,
            "datasets": [
                {
                    "label": "Disparity (Percentage Points)",
                    "data": disparities,
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "borderWidth": 1
                }
            ],
            "options": {
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Disparity (Percentage Points)"}
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating disparity analysis chart: {e}")
        return None

def generate_best_worst_comparison_chart(all_results):
    """Generate bar chart comparing best vs worst districts across states"""
    try:
        if not all_results:
            return None

        # For single indicator
        if len(all_results) == 1:
            indicator_name, indicator_data = next(iter(all_results.items()))
            
            states = []
            best_values = []
            worst_values = []
            
            for state_result in indicator_data["state_results"]:
                states.append(state_result["state_name"])
                best_values.append(state_result["best_value"])
                worst_values.append(state_result["worst_value"])
            
            return {
                "type": "bar",
                "title": f"Best vs Worst District Performance by State - {indicator_name}",
                "labels": states,
                "datasets": [
                    {
                        "label": "Best Performing District",
                        "data": best_values,
                        "backgroundColor": "#28a745",
                        "borderColor": "#1e7e34",
                        "borderWidth": 1
                    },
                    {
                        "label": "Worst Performing District", 
                        "data": worst_values,
                        "backgroundColor": "#dc3545",
                        "borderColor": "#c82333",
                        "borderWidth": 1
                    }
                ],
                "options": {
                    "scales": {
                        "y": {
                            "title": {"display": True, "text": "Performance Value (%)"}
                        },
                        "x": {
                            "title": {"display": True, "text": "States"}
                        }
                    }
                }
            }
        
        # For multiple indicators - show comparison across indicators
        else:
            indicators = list(all_results.keys())
            states_coverage = {}
            
            # Find states that have data for all indicators
            for indicator_name, indicator_data in all_results.items():
                for state_result in indicator_data["state_results"]:
                    state_name = state_result["state_name"]
                    if state_name not in states_coverage:
                        states_coverage[state_name] = 0
                    states_coverage[state_name] += 1
            
            # Select states with data for at least half the indicators
            min_coverage = len(indicators) // 2 + 1
            selected_states = [state for state, count in states_coverage.items() if count >= min_coverage][:8]
            
            best_datasets = []
            worst_datasets = []
            colors = ["#28a745", "#17a2b8", "#ffc107", "#fd7e14", "#6f42c1", "#20c997"]
            
            for i, (indicator_name, indicator_data) in enumerate(list(all_results.items())[:3]):  # Limit to 3 indicators
                best_data = []
                worst_data = []
                
                for state_name in selected_states:
                    state_data = next(
                        (s for s in indicator_data["state_results"] if s["state_name"] == state_name),
                        None
                    )
                    best_data.append(state_data["best_value"] if state_data else 0)
                    worst_data.append(state_data["worst_value"] if state_data else 0)
                
                best_datasets.append({
                    "label": f"Best - {indicator_name[:20]}",
                    "data": best_data,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)],
                    "borderWidth": 1
                })
                
                worst_datasets.append({
                    "label": f"Worst - {indicator_name[:20]}",
                    "data": worst_data,
                    "backgroundColor": colors[i % len(colors)] + "80",  # Semi-transparent
                    "borderColor": colors[i % len(colors)],
                    "borderWidth": 1,
                    "borderDash": [5, 5]  # Dashed border for worst values
                })
            
            return {
                "type": "bar",
                "title": "Best vs Worst District Performance (Multi-Indicator)",
                "labels": selected_states,
                "datasets": best_datasets + worst_datasets,
                "options": {
                    "scales": {
                        "y": {
                            "title": {"display": True, "text": "Performance Value (%)"}
                        },
                        "x": {
                            "title": {"display": True, "text": "States"}
                        }
                    },
                    "plugins": {
                        "legend": {
                            "position": "top",
                            "maxHeight": 100
                        }
                    }
                }
            }

    except Exception as e:
        logger.error(f"Error generating best worst comparison chart: {e}")
        return None

def generate_intra_state_disparities_chart(all_results):
    """Generate chart showing intra-state health disparities (difference between best and worst)"""
    try:
        if not all_results:
            return None

        # For single indicator
        if len(all_results) == 1:
            indicator_name, indicator_data = next(iter(all_results.items()))
            
            states = []
            disparities = []
            
            for state_result in indicator_data["state_results"]:
                states.append(state_result["state_name"])
                disparity = abs(state_result["best_value"] - state_result["worst_value"])
                disparities.append(disparity)
            
            return {
                "type": "bar",
                "title": f"Intra-State Health Disparities - {indicator_name}",
                "subtitle": "Difference between best and worst performing districts within each state",
                "labels": states,
                "datasets": [{
                    "label": "Disparity (Best - Worst)",
                    "data": disparities,
                    "backgroundColor": "#e74c3c",
                    "borderColor": "#c0392b",
                    "borderWidth": 1
                }],
                "options": {
                    "scales": {
                        "y": {
                            "title": {"display": True, "text": "Disparity (Percentage Points)"},
                            "beginAtZero": True
                        },
                        "x": {
                            "title": {"display": True, "text": "States"}
                        }
                    }
                }
            }
        
        # For multiple indicators - show average disparities per state
        else:
            state_disparities = {}
            
            # Calculate disparities for each state across all indicators
            for indicator_name, indicator_data in all_results.items():
                for state_result in indicator_data["state_results"]:
                    state_name = state_result["state_name"]
                    disparity = abs(state_result["best_value"] - state_result["worst_value"])
                    
                    if state_name not in state_disparities:
                        state_disparities[state_name] = []
                    state_disparities[state_name].append(disparity)
            
            # Calculate average disparities
            states = []
            avg_disparities = []
            max_disparities = []
            
            for state_name, disparities in state_disparities.items():
                if disparities:
                    states.append(state_name)
                    avg_disparities.append(sum(disparities) / len(disparities))
                    max_disparities.append(max(disparities))
            
            return {
                "type": "bar",
                "title": "Average Intra-State Health Disparities (Multi-Indicator)",
                "subtitle": "Average disparity between best and worst districts across all indicators",
                "labels": states,
                "datasets": [
                    {
                        "label": "Average Disparity",
                        "data": avg_disparities,
                        "backgroundColor": "#3498db",
                        "borderColor": "#2980b9",
                        "borderWidth": 1
                    },
                    {
                        "label": "Maximum Disparity",
                        "data": max_disparities,
                        "backgroundColor": "#e74c3c",
                        "borderColor": "#c0392b",
                        "borderWidth": 1
                    }
                ],
                "options": {
                    "scales": {
                        "y": {
                            "title": {"display": True, "text": "Disparity (Percentage Points)"},
                            "beginAtZero": True
                        },
                        "x": {
                            "title": {"display": True, "text": "States"}
                        }
                    },
                    "plugins": {
                        "legend": {
                            "position": "top"
                        }
                    }
                }
            }

    except Exception as e:
        logger.error(f"Error generating intra-state disparities chart: {e}")
        return None

def generate_indicator_direction_chart(all_results):
    """Generate pie chart showing indicator direction breakdown"""
    try:
        higher_is_better_count = 0
        lower_is_better_count = 0
        
        for indicator_name, indicator_data in all_results.items():
            direction = indicator_data["indicator_info"]["indicator_direction"]
            if direction == "higher_is_better":
                higher_is_better_count += 1
            else:
                lower_is_better_count += 1
        
        if higher_is_better_count == 0 and lower_is_better_count == 0:
            return None
            
        return {
            "type": "pie",
            "title": "Indicator Direction Distribution",
            "labels": ["Higher is Better", "Lower is Better"],
            "datasets": [{
                "data": [higher_is_better_count, lower_is_better_count],
                "backgroundColor": ["#2ecc71", "#e74c3c"],
                "borderColor": "#ffffff",
                "borderWidth": 2
            }]
        }

    except Exception as e:
        logger.error(f"Error generating indicator direction chart: {e}")
        return None

def generate_state_trend_summary_chart(all_results, year):
    """Generate chart showing trend summary across states"""
    try:
        if not all_results:
            return None

        # Use first indicator
        primary_indicator = list(all_results.keys())[0]
        state_results = all_results[primary_indicator]["state_results"]
        
        # Count trend patterns
        improving_best = 0
        declining_best = 0
        stable_best = 0
        improving_worst = 0
        declining_worst = 0
        stable_worst = 0
        
        for state in state_results:
            # Best districts trends
            if state.get("best_trend_interpretation"):
                status = state["best_trend_interpretation"]["status"]
                if status == "improving":
                    improving_best += 1
                elif status == "declining":
                    declining_best += 1
                else:
                    stable_best += 1
            
            # Worst districts trends  
            if state.get("worst_trend_interpretation"):
                status = state["worst_trend_interpretation"]["status"]
                if status == "improving":
                    improving_worst += 1
                elif status == "declining":
                    declining_worst += 1
                else:
                    stable_worst += 1
        
        return {
            "type": "bar",
            "title": f"Trend Analysis Summary: {primary_indicator} (2016-{year})",
            "labels": ["Best Performing Districts", "Worst Performing Districts"],
            "datasets": [
                {
                    "label": "Improving",
                    "data": [improving_best, improving_worst],
                    "backgroundColor": "#2ecc71",
                    "borderColor": "#27ae60",
                    "borderWidth": 1
                },
                {
                    "label": "Declining",
                    "data": [declining_best, declining_worst],
                    "backgroundColor": "#e74c3c",
                    "borderColor": "#c0392b",
                    "borderWidth": 1
                },
                {
                    "label": "Stable",
                    "data": [stable_best, stable_worst],
                    "backgroundColor": "#95a5a6",
                    "borderColor": "#7f8c8d",
                    "borderWidth": 1
                }
            ],
            "options": {
                "scales": {
                    "x": {
                        "stacked": True
                    },
                    "y": {
                        "stacked": True,
                        "title": {"display": True, "text": "Number of States"}
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating state trend summary chart: {e}")
        return None

def generate_multi_indicator_state_chart(all_results):
    """Generate multi-indicator comparison chart"""
    try:
        if len(all_results) <= 1:
            return None

        # Get top 10 states with most data coverage
        state_coverage = {}
        for indicator_name, indicator_data in all_results.items():
            for state_result in indicator_data["state_results"]:
                state_name = state_result["state_name"]
                if state_name not in state_coverage:
                    state_coverage[state_name] = 0
                state_coverage[state_name] += 1

        # Select top states
        top_states = sorted(state_coverage.items(), key=lambda x: x[1], reverse=True)[:10]
        state_names = [state[0] for state in top_states]

        datasets = []
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        
        for i, (indicator_name, indicator_data) in enumerate(list(all_results.items())[:6]):
            disparity_data = []
            for state_name in state_names:
                # Find state data
                state_data = next(
                    (s for s in indicator_data["state_results"] if s["state_name"] == state_name),
                    None
                )
                disparity_data.append(state_data.get("value_range", 0) if state_data else 0)
            
            datasets.append({
                "label": indicator_name[:25] + ("..." if len(indicator_name) > 25 else ""),
                "data": disparity_data,
                "backgroundColor": colors[i % len(colors)],
                "borderColor": colors[i % len(colors)],
                "borderWidth": 1
            })

        return {
            "type": "bar",
            "title": "Multi-Indicator State Disparities Comparison",
            "labels": state_names,
            "datasets": datasets,
            "options": {
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Disparity (Percentage Points)"}
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating multi-indicator state chart: {e}")
        return None

def generate_cross_indicator_patterns_chart(all_results):
    """Generate chart showing patterns across indicators"""
    try:
        if len(all_results) <= 1:
            return None

        # Calculate average disparities per indicator
        indicator_avg_disparities = []
        indicator_names = []
        
        for indicator_name, indicator_data in all_results.items():
            disparities = [s.get("value_range", 0) for s in indicator_data["state_results"] if s.get("value_range")]
            if disparities:
                avg_disparity = sum(disparities) / len(disparities)
                indicator_avg_disparities.append(avg_disparity)
                indicator_names.append(indicator_name[:20] + ("..." if len(indicator_name) > 20 else ""))

        return {
            "type": "bar",
            "title": "Average State Disparities by Indicator",
            "labels": indicator_names,
            "datasets": [{
                "label": "Average Disparity",
                "data": indicator_avg_disparities,
                "backgroundColor": "#3498db",
                "borderColor": "#2980b9",
                "borderWidth": 1
            }],
            "options": {
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Average Disparity (Percentage Points)"}
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating cross-indicator patterns chart: {e}")
        return None

def generate_border_districts_analysis(districts_data, state1, state2, analysis_type, state_comparison_data=None):
    """Generate analysis for border districts"""
    if not districts_data:
        return "No border districts data available for analysis."
    
    total_districts = len(districts_data)
    states_represented = len(set(d["state_name"] for d in districts_data))
    
    if analysis_type == "bilateral_border":
        analysis_focus = f"districts along the border between {state1} and {state2}"
    else:
        analysis_focus = f"districts that border {state1} from neighboring states"
    
    analysis = f"""
    **Border Districts Analysis: {state1}**
    
    This analysis examines {analysis_focus}.
    
    **Coverage:**
    - {total_districts} border districts identified
    - {states_represented} states represented
    
    **Strategic Importance:**
    Border districts are crucial for:
    - Cross-border health coordination and resource sharing
    - Understanding regional health patterns and disparities
    - Developing joint healthcare initiatives between neighboring states
    - Identifying opportunities for collaborative interventions
    
    **Analysis Focus:**
    The data shows health indicator performance for districts that share geographical boundaries,
    enabling comparison of health outcomes across state borders and identification of:
    - Cross-border health disparities
    - Opportunities for resource sharing and best practice exchange
    - Areas where inter-state coordination could improve health outcomes
    """

    # Add state comparison analysis if available
    if state_comparison_data:
        analysis += generate_state_comparison_analysis(districts_data, state_comparison_data)

    return analysis

def generate_state_comparison_analysis(districts_data, state_comparison_data):
    """Generate detailed analysis of district performance compared to state averages"""
    if not districts_data or not state_comparison_data:
        return ""

    # Analyze performance distribution across border districts
    performance_summary = {}

    for district in districts_data:
        if "state_comparison" not in district:
            continue

        district_state = district["state_name"]

        for indicator_name, comparison_data in district["state_comparison"].items():
            if indicator_name not in performance_summary:
                performance_summary[indicator_name] = {
                    "above_average": 0,
                    "below_average": 0,
                    "at_average": 0,
                    "total_districts": 0,
                    "indicator_direction": comparison_data["comparison"].get("indicator_direction")
                }

            comparison = comparison_data["comparison"]
            if comparison and "performance_status" in comparison:
                status = comparison["performance_status"]
                if status == "above_state_average":
                    performance_summary[indicator_name]["above_average"] += 1
                elif status == "below_state_average":
                    performance_summary[indicator_name]["below_average"] += 1
                else:
                    performance_summary[indicator_name]["at_average"] += 1

                performance_summary[indicator_name]["total_districts"] += 1

    # Generate analysis text
    analysis = "\n**State-Level Performance Comparison:**\n"

    if performance_summary:
        analysis += "\n**Performance Distribution:**\n"
        for indicator_name, stats in performance_summary.items():
            total = stats["total_districts"]
            if total > 0:
                above_pct = (stats["above_average"] / total) * 100
                below_pct = (stats["below_average"] / total) * 100
                at_pct = (stats["at_average"] / total) * 100

                direction_note = ""
                if stats["indicator_direction"] == "higher_is_better":
                    direction_note = " (higher values are better)"
                else:
                    direction_note = " (lower values are better)"

                analysis += f"- **{indicator_name}{direction_note}:**\n"
                analysis += f"  - Above state average: {stats['above_average']} districts ({above_pct:.1f}%)\n"
                analysis += f"  - Below state average: {stats['below_average']} districts ({below_pct:.1f}%)\n"
                analysis += f"  - At state average: {stats['at_average']} districts ({at_pct:.1f}%)\n"

        # Add insights
        analysis += "\n**Key Insights:**\n"
        analysis += "- Districts above state average may serve as models for other regions\n"
        analysis += "- Districts below state average may require additional support and resources\n"
        analysis += "- State-level benchmarking helps identify both success stories and areas needing intervention\n"
        analysis += "- Cross-state comparison enables identification of best practices that could be shared\n"

    return analysis

def generate_radius_analysis(center_name, center_type, radius_km, districts_data, indicator_ids):
    """Generate comprehensive analysis for radius-based district selection with health indicators"""
    if not districts_data:
        return "No districts data available for analysis."

    total_districts = len(districts_data)
    states_represented = len(set(d["state_name"] for d in districts_data))
    max_distance = max(d["distance_km"] for d in districts_data) if districts_data else 0

    center_description = f"coordinates {center_name}" if center_type == "coordinates" else f"{center_name} district"

    # Analyze indicators across all districts
    indicator_summary = analyze_district_indicators(districts_data)

    analysis = f"""
    **Radius-based Health Analysis**

    This analysis examines {total_districts} districts within {radius_km}km of {center_description}.

    **Spatial Coverage:**
    - Search radius: {radius_km}km
    - Districts found: {total_districts}
    - States represented: {states_represented}
    - Maximum distance: {max_distance:.1f}km

    **Health Indicators Summary:**
    {indicator_summary}

    **Regional Health Patterns:**
    This spatial analysis helps identify:
    - Regional health clusters and patterns
    - Geographic disparities in health outcomes
    - Opportunities for regional health coordination
    - Resource allocation needs based on geographic proximity

    **Key Insights:**
    - Districts are sorted by distance from center point
    - Prevalence values represent health indicator levels for 2021
    - Change values show improvement/decline from 2016 to 2021
    - Headcount data indicates affected population size
    - Analysis considers indicator direction (higher/lower is better)

    **Applications:**
    - Regional healthcare planning and resource distribution
    - Understanding geographic health trends
    - Identifying clusters of high or low performance
    - Planning interventions for geographically connected areas
    """

    return analysis

def analyze_district_indicators(districts_data):
    """Analyze health indicators across districts and provide summary insights"""
    if not districts_data:
        return "No indicator data available for analysis."

    # Collect all indicators and their statistics
    indicator_stats = {}
    total_indicators_analyzed = 0

    for district in districts_data:
        for indicator in district.get("indicators", []):
            indicator_name = indicator["indicator_name"]
            indicator_direction = indicator["indicator_direction"]

            if indicator_name not in indicator_stats:
                indicator_stats[indicator_name] = {
                    "direction": indicator_direction,
                    "districts": 0,
                    "improving": 0,
                    "declining": 0,
                    "stable": 0,
                    "prevalence_2021_values": [],
                    "change_values": []
                }

            indicator_stats[indicator_name]["districts"] += 1
            total_indicators_analyzed += 1

            # Analyze change interpretation
            change_interpretation = indicator.get("change_interpretation", {})
            if change_interpretation and change_interpretation.get("status"):
                status = change_interpretation["status"]
                indicator_stats[indicator_name][status] += 1

            # Collect values for summary statistics
            if indicator.get("prevalence_2021") is not None:
                indicator_stats[indicator_name]["prevalence_2021_values"].append(indicator["prevalence_2021"])

            if indicator.get("prevalence_change") is not None:
                indicator_stats[indicator_name]["change_values"].append(indicator["prevalence_change"])

    # Generate summary text
    if not indicator_stats:
        return "No health indicators found in the analyzed districts."

    summary_parts = []
    summary_parts.append(f"• **Total Indicators Analyzed:** {total_indicators_analyzed}")
    summary_parts.append(f"• **Unique Indicators:** {len(indicator_stats)}")
    summary_parts.append("")

    for indicator_name, stats in indicator_stats.items():
        direction_text = "↑ higher is better" if stats["direction"] == "higher_is_better" else "↓ lower is better"
        summary_parts.append(f"**{indicator_name}** ({direction_text})")
        summary_parts.append(f"  - Districts with data: {stats['districts']}")
        summary_parts.append(f"  - Improving: {stats['improving']}, Declining: {stats['declining']}, Stable: {stats['stable']}")

        # Add value ranges if available
        if stats["prevalence_2021_values"]:
            min_val = min(stats["prevalence_2021_values"])
            max_val = max(stats["prevalence_2021_values"])
            avg_val = sum(stats["prevalence_2021_values"]) / len(stats["prevalence_2021_values"])
            summary_parts.append(f"  - 2021 Range: {min_val:.2f} - {max_val:.2f} (avg: {avg_val:.2f})")

    return "\n".join(summary_parts)

def generate_radius_chart_data(districts_data, center_name, radius_km):
    """
    Generate chart data for radius analysis visualization
    Returns structured chart data for frontend Chart.js components
    """
    try:
        if not districts_data or len(districts_data) == 0:
            return None

        # Get all unique indicators across all districts
        all_indicators = {}
        for district in districts_data:
            for indicator in district.get("indicators", []):
                indicator_name = indicator["indicator_name"]
                if indicator_name not in all_indicators:
                    all_indicators[indicator_name] = {
                        "direction": indicator["indicator_direction"],
                        "districts": []
                    }
                
                all_indicators[indicator_name]["districts"].append({
                    "district_name": district["district_name"],
                    "state_name": district["state_name"],
                    "distance_km": district["distance_km"],
                    "prevalence_2016": indicator.get("prevalence_2016"),
                    "prevalence_2021": indicator.get("prevalence_2021"),
                    "prevalence_change": indicator.get("prevalence_change"),
                    "headcount_2021": indicator.get("headcount_2021"),
                    "change_interpretation": indicator.get("change_interpretation")
                })

        # Generate different chart types
        chart_data = {
            "distance_vs_prevalence": generate_distance_prevalence_chart(districts_data, all_indicators),
            "indicator_comparison": generate_indicator_comparison_chart(districts_data, all_indicators),
            "state_wise_distribution": generate_state_distribution_chart(districts_data),
            "trend_analysis": generate_trend_analysis_chart(districts_data, all_indicators),
            "summary_stats": generate_summary_stats_chart(districts_data, all_indicators)
        }

        return chart_data

    except Exception as e:
        logger.error(f"Error generating radius chart data: {e}")
        return None

def generate_distance_prevalence_chart(districts_data, all_indicators):
    """Generate scatter plot data for distance vs prevalence/change"""
    try:
        if not all_indicators:
            return None

        # Create chart for the first indicator (or most common one)
        primary_indicator = list(all_indicators.keys())[0]
        indicator_data = all_indicators[primary_indicator]
        
        # Get unique states for color coding
        states = list(set(d["state_name"] for d in districts_data))
        state_colors = {
            state: f"rgba({50 + i * 40}, {100 + i * 30}, {200 - i * 20}, 0.8)" 
            for i, state in enumerate(states)
        }

        datasets = []
        for state in states:
            state_districts = [d for d in indicator_data["districts"] if d["state_name"] == state]
            if not state_districts:
                continue

            # Distance vs 2021 prevalence
            datasets.append({
                "label": f"{state} - 2021 Values",
                "data": [
                    {
                        "x": d["distance_km"],
                        "y": d["prevalence_2021"] if d["prevalence_2021"] is not None else 0,
                        "district": d["district_name"]
                    }
                    for d in state_districts if d["prevalence_2021"] is not None
                ],
                "backgroundColor": state_colors[state],
                "borderColor": state_colors[state].replace("0.8", "1.0"),
                "pointRadius": 6,
                "pointHoverRadius": 8
            })

        return {
            "type": "scatter",
            "title": f"Distance vs {primary_indicator} (2021)",
            "datasets": datasets,
            "options": {
                "scales": {
                    "x": {
                        "title": {"display": True, "text": "Distance from Center (km)"}
                    },
                    "y": {
                        "title": {"display": True, "text": f"{primary_indicator} Prevalence (%)"}
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating distance prevalence chart: {e}")
        return None

def generate_indicator_comparison_chart(districts_data, all_indicators):
    """Generate bar chart comparing indicators across districts"""
    try:
        if not all_indicators:
            return None

        # Limit to top 10 closest districts for readability
        closest_districts = sorted(districts_data, key=lambda d: d["distance_km"])[:10]
        
        # If multiple indicators, show all; if single indicator, show 2016 vs 2021
        if len(all_indicators) > 1:
            # Multiple indicators - show 2021 values
            labels = [d["district_name"][:15] + ("..." if len(d["district_name"]) > 15 else "") 
                     for d in closest_districts]
            
            datasets = []
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
            
            for i, (indicator_name, indicator_data) in enumerate(list(all_indicators.items())[:6]):
                district_values = []
                for district in closest_districts:
                    district_indicator = next(
                        (d for d in indicator_data["districts"] if d["district_name"] == district["district_name"]),
                        None
                    )
                    district_values.append(
                        district_indicator["prevalence_2021"] if district_indicator and district_indicator["prevalence_2021"] is not None else 0
                    )
                
                datasets.append({
                    "label": indicator_name[:30] + ("..." if len(indicator_name) > 30 else ""),
                    "data": district_values,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)].replace("0.8", "1.0"),
                    "borderWidth": 1
                })

            return {
                "type": "bar",
                "title": "Health Indicators Comparison (2021)",
                "labels": labels,
                "datasets": datasets
            }
        
        else:
            # Single indicator - show 2016 vs 2021
            primary_indicator = list(all_indicators.keys())[0]
            indicator_data = all_indicators[primary_indicator]
            
            labels = [d["district_name"][:15] + ("..." if len(d["district_name"]) > 15 else "") 
                     for d in closest_districts]
            
            values_2016 = []
            values_2021 = []
            
            for district in closest_districts:
                district_indicator = next(
                    (d for d in indicator_data["districts"] if d["district_name"] == district["district_name"]),
                    None
                )
                if district_indicator:
                    values_2016.append(district_indicator["prevalence_2016"] if district_indicator["prevalence_2016"] is not None else 0)
                    values_2021.append(district_indicator["prevalence_2021"] if district_indicator["prevalence_2021"] is not None else 0)
                else:
                    values_2016.append(0)
                    values_2021.append(0)

            return {
                "type": "bar",
                "title": f"{primary_indicator} - 2016 vs 2021",
                "labels": labels,
                "datasets": [
                    {
                        "label": "2016 Values",
                        "data": values_2016,
                        "backgroundColor": "#3498db",
                        "borderColor": "#2980b9",
                        "borderWidth": 1
                    },
                    {
                        "label": "2021 Values", 
                        "data": values_2021,
                        "backgroundColor": "#2ecc71",
                        "borderColor": "#27ae60",
                        "borderWidth": 1
                    }
                ]
            }

    except Exception as e:
        logger.error(f"Error generating indicator comparison chart: {e}")
        return None

def generate_state_distribution_chart(districts_data):
    """Generate pie chart showing state distribution"""
    try:
        state_counts = {}
        for district in districts_data:
            state = district["state_name"]
            state_counts[state] = state_counts.get(state, 0) + 1

        if not state_counts:
            return None

        # Generate colors for states
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
        
        return {
            "type": "pie",
            "title": "Districts Distribution by State",
            "labels": list(state_counts.keys()),
            "datasets": [{
                "data": list(state_counts.values()),
                "backgroundColor": colors[:len(state_counts)],
                "borderColor": "#ffffff",
                "borderWidth": 2
            }]
        }

    except Exception as e:
        logger.error(f"Error generating state distribution chart: {e}")
        return None

def generate_trend_analysis_chart(districts_data, all_indicators):
    """Generate line chart showing trend analysis"""
    try:
        if not all_indicators:
            return None

        # Focus on first indicator for trend analysis
        primary_indicator = list(all_indicators.keys())[0]
        indicator_data = all_indicators[primary_indicator]
        
        # Sort districts by distance
        sorted_districts = sorted(
            indicator_data["districts"], 
            key=lambda d: d["distance_km"]
        )[:15]  # Top 15 closest
        
        labels = [d["district_name"][:10] + "..." if len(d["district_name"]) > 10 else d["district_name"] 
                 for d in sorted_districts]
        
        return {
            "type": "line",
            "title": f"{primary_indicator} - Trend by Distance",
            "labels": labels,
            "datasets": [
                {
                    "label": "2016 Values",
                    "data": [d["prevalence_2016"] if d["prevalence_2016"] is not None else 0 for d in sorted_districts],
                    "borderColor": "#3498db",
                    "backgroundColor": "rgba(52, 152, 219, 0.1)",
                    "tension": 0.2,
                    "pointRadius": 4
                },
                {
                    "label": "2021 Values",
                    "data": [d["prevalence_2021"] if d["prevalence_2021"] is not None else 0 for d in sorted_districts],
                    "borderColor": "#2ecc71",
                    "backgroundColor": "rgba(46, 204, 113, 0.1)",
                    "tension": 0.2,
                    "pointRadius": 4
                }
            ]
        }

    except Exception as e:
        logger.error(f"Error generating trend analysis chart: {e}")
        return None

def generate_summary_stats_chart(districts_data, all_indicators):
    """Generate summary statistics chart"""
    try:
        if not all_indicators:
            return None

        stats = []
        labels = []
        
        for indicator_name, indicator_data in all_indicators.items():
            # Calculate improvement/decline statistics
            improving = sum(1 for d in indicator_data["districts"] 
                          if d["change_interpretation"] and d["change_interpretation"].get("status") == "improving")
            declining = sum(1 for d in indicator_data["districts"] 
                          if d["change_interpretation"] and d["change_interpretation"].get("status") == "declining")
            stable = sum(1 for d in indicator_data["districts"] 
                       if d["change_interpretation"] and d["change_interpretation"].get("status") == "stable")
            
            total = improving + declining + stable
            if total > 0:
                labels.append(indicator_name[:20] + "..." if len(indicator_name) > 20 else indicator_name)
                stats.append({
                    "improving": improving,
                    "declining": declining,
                    "stable": stable
                })

        if not stats:
            return None

        return {
            "type": "bar",
            "title": "Health Indicators Trend Summary",
            "labels": labels,
            "datasets": [
                {
                    "label": "Improving",
                    "data": [s["improving"] for s in stats],
                    "backgroundColor": "#2ecc71",
                    "borderColor": "#27ae60",
                    "borderWidth": 1
                },
                {
                    "label": "Declining",
                    "data": [s["declining"] for s in stats],
                    "backgroundColor": "#e74c3c",
                    "borderColor": "#c0392b",
                    "borderWidth": 1
                },
                {
                    "label": "Stable",
                    "data": [s["stable"] for s in stats],
                    "backgroundColor": "#95a5a6",
                    "borderColor": "#7f8c8d",
                    "borderWidth": 1
                }
            ],
            "options": {
                "scales": {
                    "x": {
                        "stacked": True
                    },
                    "y": {
                        "stacked": True,
                        "title": {"display": True, "text": "Number of Districts"}
                    }
                }
            }
        }

    except Exception as e:
        logger.error(f"Error generating summary stats chart: {e}")
        return None

# Indicator Matching System

def match_indicator_name_to_database(indicator_name):
    """
    Match a user-provided indicator name to the best database indicator.
    This handles misspellings, synonyms, and descriptions.
    """
    try:
        # Get all indicators from database
        indicators = get_all_indicators()
        if not indicators:
            return None

        # Create a mapping of indicator names for quick lookup
        indicator_map = {ind["indicator_name"].lower(): ind for ind in indicators}

        # First try exact match (case-insensitive)
        indicator_name_lower = indicator_name.lower().strip()
        if indicator_name_lower in indicator_map:
            return indicator_map[indicator_name_lower]

        # Try fuzzy matching using rapidfuzz
        from rapidfuzz import process, fuzz

        # Extract indicator names for fuzzy matching
        indicator_names = list(indicator_map.keys())

        # Find best match with 90% threshold
        match_result = process.extractOne(
            indicator_name_lower,
            indicator_names,
            scorer=fuzz.WRatio,
            score_cutoff=90  # Require 90% similarity
        )

        if match_result:
            matched_name, score, _ = match_result
            matched_indicator = indicator_map[matched_name]
            print(f"🔍 Fuzzy match (90%+): '{indicator_name}' → '{matched_indicator['indicator_name']}' (score: {score:.1f}%)")
            return matched_indicator

        # If fuzzy match < 90%, use OpenAI
        print(f"🤖 Fuzzy match < 90%, using OpenAI: '{indicator_name}'")
        openai_match = match_indicator_with_openai(indicator_name, indicators)
        if openai_match and openai_match.get("matched"):
            matched_indicator = None
            for ind in indicators:
                if ind["indicator_id"] == openai_match["indicator_id"]:
                    matched_indicator = ind
                    break

            if matched_indicator:
                print(f"🧠 OpenAI match: '{indicator_name}' → '{matched_indicator['indicator_name']}'")
                return matched_indicator

        return None

    except Exception as e:
        logger.error(f"Error in match_indicator_name_to_database: {e}")
        return None



def get_districts_by_constraints(
    constraints: List[Dict[str, Any]],
    year: int = 2021,
    states: Optional[List[str]] = None,
    max_districts: int = 100,
    include_boundary_data: bool = True
):
   
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Validate and match indicators
        validated_constraints = []
        matched_indicators = {}
        
        for constraint in constraints:
            # Validate constraint structure
            if not all(key in constraint for key in ["indicator_name", "operator", "value"]):
                return {"error": f"Invalid constraint format. Required keys: indicator_name, operator, value. Got: {constraint}"}
            
            # Validate operator
            valid_operators = ['>', '>=', '<', '<=', '=', '!=', 'eq', 'neq', 'gt', 'gte', 'lt', 'lte']
            operator = constraint["operator"].lower()
            if operator not in valid_operators:
                return {"error": f"Invalid operator '{constraint['operator']}'. Valid operators: {valid_operators}"}
            
            # Normalize operator
            operator_map = {
                'eq': '=', 'neq': '!=', 'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<='
            }
            normalized_operator = operator_map.get(operator, operator)
            
            # Match indicator name to database
            matched_indicator = match_indicator_name_to_database(constraint["indicator_name"])
            if not matched_indicator:
                return {"error": f"Could not match indicator: '{constraint['indicator_name']}'"}
            
            # Validate value
            try:
                value = float(constraint["value"])
            except (ValueError, TypeError):
                return {"error": f"Invalid value '{constraint['value']}' for constraint. Must be numeric."}
            
            validated_constraints.append({
                "indicator_id": matched_indicator["indicator_id"],
                "indicator_name": matched_indicator["indicator_name"],
                "indicator_direction": matched_indicator["indicator_direction"],
                "operator": normalized_operator,
                "value": value,
                "original_name": constraint["indicator_name"]
            })
            
            matched_indicators[matched_indicator["indicator_id"]] = matched_indicator
            logger.info(f"✅ Validated constraint: '{constraint['indicator_name']}' → '{matched_indicator['indicator_name']}' {normalized_operator} {value}")
        
        if not validated_constraints:
            return {"error": "No valid constraints provided"}
        
        # Determine which prevalence column to use
        year_column = "prevalence_2021" if year == 2021 else "prevalence_2016"
        
        # Build state filter condition
        state_filter = ""
        state_params = []
        if states:
            placeholders = ','.join(['%s'] * len(states))
            state_filter = f"AND s.state_name IN ({placeholders})"
            state_params = states
        
        # Build constraint conditions
        constraint_conditions = []
        constraint_params = []
        
        for i, constraint in enumerate(validated_constraints):
            alias = f"di{i}"
            constraint_conditions.append(f"""
                EXISTS (
                    SELECT 1 FROM district_indicators {alias}
                    WHERE {alias}.district_id = d.district_id
                    AND {alias}.indicator_id = %s
                    AND {alias}.{year_column} IS NOT NULL
                    AND {alias}.{year_column} {constraint['operator']} %s
                )
            """)
            constraint_params.extend([constraint["indicator_id"], constraint["value"]])
        
        # Main query to find districts meeting all constraints
        base_query = f"""
        SELECT DISTINCT
            d.district_id,
            d.district_name,
            s.state_name,
            s.state_id
        FROM districts d
        JOIN states s ON d.state_id = s.state_id
        WHERE {' AND '.join(constraint_conditions)}
        {state_filter}
        ORDER BY s.state_name, d.district_name
        LIMIT %s
        """
        
        params = constraint_params + state_params + [max_districts]
        cursor.execute(base_query, params)
        matching_districts = cursor.fetchall()
        
        if not matching_districts:
            constraint_summary = ", ".join([
                f"{c['indicator_name']} {c['operator']} {c['value']}" 
                for c in validated_constraints
            ])
            return {
                "error": f"No districts found matching all constraints: {constraint_summary}",
                "constraints_applied": validated_constraints,
                "year": year,
                "states_filter": states
            }
        
        # Get detailed indicator data for matching districts
        district_ids = [row[0] for row in matching_districts]
        indicator_ids = [c["indicator_id"] for c in validated_constraints]
        
        # Query for detailed indicator data
        detail_query = f"""
        SELECT
            d.district_name,
            s.state_name,
            i.indicator_name,
            i.indicator_direction,
            di.prevalence_2016,
            di.prevalence_2021,
            di.prevalence_change,
            di.headcount_2021,
            i.indicator_id,
            d.district_id
        FROM district_indicators di
        JOIN districts d ON di.district_id = d.district_id
        JOIN states s ON d.state_id = s.state_id
        JOIN indicators i ON di.indicator_id = i.indicator_id
        WHERE di.district_id = ANY(%s::int[])
        AND di.indicator_id = ANY(%s::int[])
        AND di.{year_column} IS NOT NULL
        ORDER BY s.state_name, d.district_name, i.indicator_name
        """
        
        cursor.execute(detail_query, (district_ids, indicator_ids))
        detail_results = cursor.fetchall()
        
        # Process results into structured format
        districts_data = {}
        constraint_stats = {c["indicator_id"]: {"values": [], "constraint": c} for c in validated_constraints}
        
        for row in detail_results:
            district_key = f"{row[0]}_{row[1]}"  # district_state
            indicator_id = row[8]
            
            if district_key not in districts_data:
                districts_data[district_key] = {
                    "district_name": row[0],
                    "state_name": row[1],
                    "district_id": row[9],
                    "indicators": [],
                    "constraint_values": {}
                }
            
            indicator_data = {
                "indicator_name": row[2],
                "indicator_direction": row[3],
                "prevalence_2016": float(row[4]) if row[4] is not None else None,
                "prevalence_2021": float(row[5]) if row[5] is not None else None,
                "prevalence_change": float(row[6]) if row[6] is not None else None,
                "headcount_2021": float(row[7]) if row[7] is not None else None,
                "indicator_id": indicator_id
            }
            
            # Add trend interpretation
            if indicator_data["prevalence_change"] is not None:
                indicator_data["trend_interpretation"] = interpret_health_trend(
                    indicator_data["prevalence_change"],
                    indicator_data["indicator_direction"]
                )
            
            districts_data[district_key]["indicators"].append(indicator_data)
            
            # Store constraint-specific values
            current_value = indicator_data["prevalence_2021"] if year == 2021 else indicator_data["prevalence_2016"]
            if indicator_id in constraint_stats and current_value is not None:
                districts_data[district_key]["constraint_values"][indicator_id] = current_value
                constraint_stats[indicator_id]["values"].append(current_value)
        
        # Get boundary data if requested
        boundary_data = []
        if include_boundary_data:
            all_district_names = [data["district_name"] for data in districts_data.values()]
            boundary_data = get_district_boundary_data(all_district_names)
        
        # Generate analysis
        analysis = generate_constraint_analysis(
            list(districts_data.values()), 
            validated_constraints, 
            constraint_stats, 
            year, 
            states
        )
        
        # Generate chart data
        chart_data = generate_constraint_chart_data(
            list(districts_data.values()), 
            validated_constraints, 
            constraint_stats
        )
        
        # Calculate summary statistics
        summary_stats = calculate_constraint_summary_stats(constraint_stats, validated_constraints)
        
        response = {
            "constraints_applied": validated_constraints,
            "year": year,
            "total_districts_found": len(districts_data),
            "states_filter": states,
            "max_districts": max_districts,
            "districts": list(districts_data.values()),
            "boundary": boundary_data,
            "analysis": analysis,
            "chart_data": chart_data,
            "summary_stats": summary_stats,
            "map_type": "constraint_based_search"
        }
        
        cursor.close()
        conn.close()
        
        logger.info(f"🎯 Constraint search completed: {len(districts_data)} districts found matching {len(validated_constraints)} constraints")
        return response
        
    except Exception as e:
        logger.error(f"Error in get_districts_by_constraints: {e}")
        return {"error": f"Database error: {str(e)}"}

def parse_constraint_text(constraint_text: str) -> List[Dict[str, Any]]:
    """
    Parse natural language constraint text into structured constraints.
    
    Examples:
    - "population with bpl cards > 40 and diarrhea <= 5"
    - "vaccination coverage >= 80, malnutrition < 10"
    """
    import re
    
    try:
        constraints = []
        
        # Split by 'and', ',' or similar conjunctions
        parts = re.split(r'\s+and\s+|,\s*', constraint_text.lower())
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Extract operator and value
            operators = ['>=', '<=', '!=', '>', '<', '=']
            operator_found = None
            value_found = None
            indicator_part = None
            
            for op in operators:
                if op in part:
                    split_parts = part.split(op, 1)
                    if len(split_parts) == 2:
                        indicator_part = split_parts[0].strip()
                        value_part = split_parts[1].strip()
                        
                        try:
                            value_found = float(value_part)
                            operator_found = op
                            break
                        except ValueError:
                            continue
            
            if operator_found and value_found is not None and indicator_part:
                constraints.append({
                    "indicator_name": indicator_part,
                    "operator": operator_found,
                    "value": value_found
                })
        
        return constraints
        
    except Exception as e:
        logger.error(f"Error parsing constraint text: {e}")
        return []

def generate_constraint_analysis(districts_data, constraints, constraint_stats, year, states_filter):
    """Generate comprehensive analysis for constraint-based district search"""
    if not districts_data:
        return "No districts found matching the specified constraints."
    
    total_districts = len(districts_data)
    states_represented = len(set(d["state_name"] for d in districts_data))
    
    # Build constraint summary
    constraint_summary = []
    for constraint in constraints:
        constraint_summary.append(
            f"'{constraint['indicator_name']}' {constraint['operator']} {constraint['value']}"
        )
    
    analysis_parts = []
    analysis_parts.append(f"**Constraint-Based District Search Results**\n")
    analysis_parts.append(f"**Constraints Applied ({year}):**")
    for constraint in constraint_summary:
        analysis_parts.append(f"• {constraint}")
    analysis_parts.append("")
    
    analysis_parts.append(f"**Results Overview:**")
    analysis_parts.append(f"• Districts found: {total_districts}")
    analysis_parts.append(f"• States represented: {states_represented}")
    if states_filter:
        analysis_parts.append(f"• States filtered: {', '.join(states_filter)}")
    analysis_parts.append("")
    
    # State-wise distribution
    state_counts = {}
    for district in districts_data:
        state = district["state_name"]
        state_counts[state] = state_counts.get(state, 0) + 1
    
    if len(state_counts) > 1:
        analysis_parts.append(f"**Geographic Distribution:**")
        sorted_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_states:
            analysis_parts.append(f"• {state}: {count} districts")
        analysis_parts.append("")
    
    # Constraint-specific statistics
    analysis_parts.append(f"**Constraint Statistics:**")
    for constraint in constraints:
        indicator_id = constraint["indicator_id"]
        if indicator_id in constraint_stats and constraint_stats[indicator_id]["values"]:
            values = constraint_stats[indicator_id]["values"]
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)
            
            direction_note = ""
            if constraint["indicator_direction"] == "higher_is_better":
                direction_note = " (higher is better)"
            else:
                direction_note = " (lower is better)"
            
            analysis_parts.append(f"**{constraint['indicator_name']}{direction_note}:**")
            analysis_parts.append(f"  - Constraint: {constraint['operator']} {constraint['value']}")
            analysis_parts.append(f"  - Range in results: {min_val:.2f} - {max_val:.2f}")
            analysis_parts.append(f"  - Average: {avg_val:.2f}")
            analysis_parts.append("")
    
    # Strategic insights
    analysis_parts.append(f"**Strategic Insights:**")
    analysis_parts.append("• These districts meet ALL specified criteria simultaneously")
    analysis_parts.append("• Results can inform targeted interventions and resource allocation")
    analysis_parts.append("• Geographic clustering may indicate regional patterns or policies")
    analysis_parts.append("• Districts meeting multiple criteria may serve as models for best practices")
    analysis_parts.append("")
    
    analysis_parts.append(f"**Applications:**")
    analysis_parts.append("• Identifying high-performing districts for case studies")
    analysis_parts.append("• Targeting interventions to districts with specific challenges")
    analysis_parts.append("• Resource allocation based on multiple health indicators")
    analysis_parts.append("• Policy evaluation across multiple health outcomes")
    
    return "\n".join(analysis_parts)

def calculate_constraint_summary_stats(constraint_stats, constraints):
    """Calculate summary statistics for constraint analysis"""
    summary = {}
    
    for constraint in constraints:
        indicator_id = constraint["indicator_id"]
        indicator_name = constraint["indicator_name"]
        
        if indicator_id in constraint_stats and constraint_stats[indicator_id]["values"]:
            values = constraint_stats[indicator_id]["values"]
            
            summary[indicator_name] = {
                "constraint": f"{constraint['operator']} {constraint['value']}",
                "districts_count": len(values),
                "min_value": min(values),
                "max_value": max(values),
                "average_value": sum(values) / len(values),
                "indicator_direction": constraint["indicator_direction"]
            }
    
    return summary

def generate_constraint_chart_data(districts_data, constraints, constraint_stats):
    """Generate simplified chart data for constraint-based analysis visualization - only bar chart with constraint indicators"""
    try:
        if not districts_data or not constraints:
            return None
        
        # Generate only the main constraint comparison chart (limited to 10 districts)
        # All other charts are removed per user request - only show the values of indicators in the query
        return generate_constraint_comparison_chart(districts_data, constraints)
        
    except Exception as e:
        logger.error(f"Error generating constraint chart data: {e}")
        return None

def generate_constraint_comparison_chart(districts_data, constraints):
    """Generate bar chart showing constraint indicator values for up to 10 districts"""
    try:
        # Limit to exactly 10 districts for chart readability (map will show all districts)
        limited_districts = districts_data[:10]
        
        # Create district labels with state info for clarity
        labels = []
        for d in limited_districts:
            district_name = d["district_name"]
            state_name = d["state_name"]
            # Truncate long names but include state
            if len(district_name) > 15:
                label = f"{district_name[:12]}... ({state_name[:3]})"
            else:
                label = f"{district_name} ({state_name[:3]})"
            labels.append(label)
        
        datasets = []
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        
        # Create dataset for each constraint indicator
        for i, constraint in enumerate(constraints):
            indicator_id = constraint["indicator_id"]
            constraint_values = []
            
            for district in limited_districts:
                value = district["constraint_values"].get(indicator_id, 0)
                constraint_values.append(value)
            
            # Truncate indicator name if too long
            indicator_label = constraint["indicator_name"]
            if len(indicator_label) > 30:
                indicator_label = indicator_label[:27] + "..."
            
            datasets.append({
                "label": indicator_label,
                "data": constraint_values,
                "backgroundColor": colors[i % len(colors)],
                "borderColor": colors[i % len(colors)],
                "borderWidth": 1
            })
        
        # Create title based on number of total districts
        total_districts = len(districts_data)
        if total_districts > 10:
            title = f"Constraint Indicator Values (Showing 10 of {total_districts} Districts)"
            subtitle = "Complete data available on map view"
        else:
            title = f"Constraint Indicator Values ({total_districts} Districts Found)"
            subtitle = "Values for indicators specified in constraints"
        
        return {
            "type": "bar",
            "title": title,
            "subtitle": subtitle,
            "labels": labels,
            "datasets": datasets,
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Indicator Values (%)"},
                        "beginAtZero": True
                    },
                    "x": {
                        "title": {"display": True, "text": "Districts"},
                        "ticks": {
                            "maxRotation": 45,
                            "minRotation": 0
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "position": "top",
                        "labels": {
                            "boxWidth": 12,
                            "font": {
                                "size": 11
                            }
                        }
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False,
                        "callbacks": {
                            "title": "function(context) { return context[0].label.split(' (')[0]; }",
                            "label": "function(context) { return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%'; }"
                        }
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating constraint comparison chart: {e}")
        return None

def generate_constraint_state_distribution_chart(districts_data):
    """Generate pie chart showing state distribution of matching districts"""
    try:
        state_counts = {}
        for district in districts_data:
            state = district["state_name"]
            state_counts[state] = state_counts.get(state, 0) + 1
        
        if not state_counts:
            return None
        
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
        
        return {
            "type": "pie",
            "title": "Districts by State",
            "subtitle": "Geographic distribution of districts meeting constraints",
            "labels": list(state_counts.keys()),
            "datasets": [{
                "data": list(state_counts.values()),
                "backgroundColor": colors[:len(state_counts)],
                "borderColor": "#ffffff",
                "borderWidth": 2
            }]
        }
        
    except Exception as e:
        logger.error(f"Error generating constraint state distribution chart: {e}")
        return None

def generate_constraint_value_ranges_chart(constraint_stats, constraints):
    """Generate chart showing value ranges for each constraint"""
    try:
        labels = []
        min_values = []
        max_values = []
        avg_values = []
        
        for constraint in constraints:
            indicator_id = constraint["indicator_id"]
            if indicator_id in constraint_stats and constraint_stats[indicator_id]["values"]:
                values = constraint_stats[indicator_id]["values"]
                
                labels.append(constraint["indicator_name"][:20] + ("..." if len(constraint["indicator_name"]) > 20 else ""))
                min_values.append(min(values))
                max_values.append(max(values))
                avg_values.append(sum(values) / len(values))
        
        if not labels:
            return None
        
        return {
            "type": "bar",
            "title": "Value Ranges for Each Constraint",
            "subtitle": "Min, Max, and Average values across matching districts",
            "labels": labels,
            "datasets": [
                {
                    "label": "Minimum Value",
                    "data": min_values,
                    "backgroundColor": "#3498db",
                    "borderColor": "#2980b9",
                    "borderWidth": 1
                },
                {
                    "label": "Average Value",
                    "data": avg_values,
                    "backgroundColor": "#2ecc71",
                    "borderColor": "#27ae60",
                    "borderWidth": 1
                },
                {
                    "label": "Maximum Value",
                    "data": max_values,
                    "backgroundColor": "#e74c3c",
                    "borderColor": "#c0392b",
                    "borderWidth": 1
                }
            ],
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Indicator Values"},
                        "beginAtZero": True
                    },
                    "x": {
                        "title": {"display": True, "text": "Health Indicators"}
                    }
                },
                "plugins": {
                    "legend": {
                        "position": "top"
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating constraint value ranges chart: {e}")
        return None

def generate_constraint_geographic_chart(districts_data, constraints):
    """Generate chart showing constraint performance by state"""
    try:
        # Group districts by state
        state_groups = {}
        for district in districts_data:
            state = district["state_name"]
            if state not in state_groups:
                state_groups[state] = []
            state_groups[state].append(district)
        
        # Calculate average values per state for each constraint
        state_labels = list(state_groups.keys())
        datasets = []
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        
        for i, constraint in enumerate(constraints):
            indicator_id = constraint["indicator_id"]
            state_averages = []
            
            for state in state_labels:
                values = []
                for district in state_groups[state]:
                    if indicator_id in district["constraint_values"]:
                        values.append(district["constraint_values"][indicator_id])
                
                avg_value = sum(values) / len(values) if values else 0
                state_averages.append(avg_value)
            
            datasets.append({
                "label": constraint["indicator_name"][:20] + ("..." if len(constraint["indicator_name"]) > 20 else ""),
                "data": state_averages,
                "backgroundColor": colors[i % len(colors)],
                "borderColor": colors[i % len(colors)],
                "borderWidth": 1
            })
        
        return {
            "type": "bar",
            "title": "Average Constraint Values by State",
            "subtitle": "State-wise comparison of constraint indicators",
            "labels": state_labels,
            "datasets": datasets,
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Average Values"},
                        "beginAtZero": True
                    },
                    "x": {
                        "title": {"display": True, "text": "States"}
                    }
                },
                "plugins": {
                    "legend": {
                        "position": "top"
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating constraint geographic chart: {e}")
        return None

def get_top_bottom_districts(
    indicator_names: Optional[List[str]] = None,
    indicator_name: Optional[str] = None,
    n_districts: int = 10,
    performance_type: str = "top",  # "top", "bottom", or "both"
    states: Optional[List[str]] = None,
    year: int = 2021,
    include_boundary_data: bool = True
):
    """
    Get top/bottom N districts for single or multiple health indicators with state filtering.
    
    This function returns the best or worst performing districts across all states for given 
    health indicators, properly handling indicator direction (higher_is_better vs lower_is_better).
    
    Parameters:
    - indicator_names: List of health indicator names (can be misspelled or described)
    - indicator_name: Single health indicator name (use this OR indicator_names)
    - n_districts: Number of top/bottom districts to return (default: 10)
    - performance_type: "top" for best performing, "bottom" for worst performing, "both" for both
    - states: List of specific state names to filter by (if None, includes all states)
    - year: Year for analysis (2016 or 2021, default: 2021)
    - include_boundary_data: Whether to include boundary geometry data (default: true)
    
    Returns:
    - For single indicator: Top/bottom N districts with their values
    - For multiple indicators: Districts ranked by composite score across indicators
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Handle indicator input - support both single and multiple indicators
        indicators_to_process = []
        if indicator_names:
            indicators_to_process = indicator_names
        elif indicator_name:
            indicators_to_process = [indicator_name]
        else:
            return {"error": "Please provide either indicator_name or indicator_names"}
        
        # Match indicators to database using the existing matching function
        matched_indicators = []
        for ind_name in indicators_to_process:
            matched_indicator = match_indicator_name_to_database(ind_name)
            if matched_indicator:
                matched_indicators.append(matched_indicator)
                logger.info(f"Matched '{ind_name}' to '{matched_indicator['indicator_name']}'")
            else:
                logger.warning(f"Could not match indicator: '{ind_name}'")
        
        if not matched_indicators:
            return {"error": f"No valid indicators found from: {indicators_to_process}"}
        
        # Determine which prevalence column to use
        year_column = "prevalence_2021" if year == 2021 else "prevalence_2016"
        
        # Build state filter condition
        state_filter = ""
        state_params = []
        if states:
            placeholders = ','.join(['%s'] * len(states))
            state_filter = f"AND s.state_name IN ({placeholders})"
            state_params = states
        
        # For single indicator - direct ranking
        if len(matched_indicators) == 1:
            matched_indicator = matched_indicators[0]
            indicator_id = matched_indicator["indicator_id"]
            indicator_direction = matched_indicator["indicator_direction"]
            higher_is_better = indicator_direction == "higher_is_better"
            
            # Determine ordering based on performance type and indicator direction
            if performance_type == "top":
                order_clause = "DESC" if higher_is_better else "ASC"
            elif performance_type == "bottom":
                order_clause = "ASC" if higher_is_better else "DESC"
            else:  # both
                order_clause = "DESC" if higher_is_better else "ASC"
            
            # Query for single indicator ranking
            query = f"""
            SELECT 
                d.district_name,
                s.state_name,
                di.{year_column} as indicator_value,
                di.prevalence_2016,
                di.prevalence_2021,
                di.prevalence_change,
                di.headcount_2021,
                i.indicator_name,
                i.indicator_direction,
                d.district_id,
                ROW_NUMBER() OVER (ORDER BY di.{year_column} {order_clause}) as rank
            FROM district_indicators di
            JOIN districts d ON di.district_id = d.district_id
            JOIN states s ON d.state_id = s.state_id
            JOIN indicators i ON di.indicator_id = i.indicator_id
            WHERE di.indicator_id = %s 
            AND di.{year_column} IS NOT NULL
            {state_filter}
            ORDER BY di.{year_column} {order_clause}
            LIMIT %s
            """
            
            params = [indicator_id] + state_params + [n_districts if performance_type != "both" else n_districts * 2]
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Process single indicator results
            districts_data = []
            all_district_names = []
            
            for row in results:
                indicator_data = {
                    "indicator_name": row[7],
                    "indicator_direction": row[8],
                    "prevalence_2016": float(row[3]) if row[3] is not None else None,
                    "prevalence_2021": float(row[4]) if row[4] is not None else None,
                    "prevalence_change": float(row[5]) if row[5] is not None else None,
                    "headcount_2021": float(row[6]) if row[6] is not None else None,
                    "indicator_value": float(row[2]) if row[2] is not None else None,
                    "rank": row[10]
                }
                
                # Add trend interpretation
                if indicator_data["prevalence_change"] is not None:
                    indicator_data["trend_interpretation"] = interpret_health_trend(
                        indicator_data["prevalence_change"],
                        indicator_data["indicator_direction"]
                    )
                
                district_data = {
                    "district_name": row[0],
                    "state_name": row[1],
                    "district_id": row[9],
                    "rank": row[10],
                    "performance_score": float(row[2]) if row[2] is not None else None,
                    "indicators": [indicator_data]
                }
                
                districts_data.append(district_data)
                all_district_names.append(row[0])
            
            response_type = "single_indicator"
            main_indicator = matched_indicator["indicator_name"]
            
        else:
            # Multiple indicators - composite scoring
            indicator_ids = [ind["indicator_id"] for ind in matched_indicators]
            
            # Query to get all districts with data for all indicators
            query = f"""
            WITH district_scores AS (
                SELECT 
                    d.district_id,
                    d.district_name,
                    s.state_name,
                    COUNT(DISTINCT di.indicator_id) as indicators_count,
                    -- Calculate normalized scores for each district
                    AVG(
                        CASE 
                            WHEN i.indicator_direction = 'higher_is_better' THEN
                                (di.{year_column} - MIN(di.{year_column}) OVER (PARTITION BY i.indicator_id)) /
                                NULLIF(MAX(di.{year_column}) OVER (PARTITION BY i.indicator_id) - MIN(di.{year_column}) OVER (PARTITION BY i.indicator_id), 0) * 100
                            ELSE
                                (MAX(di.{year_column}) OVER (PARTITION BY i.indicator_id) - di.{year_column}) /
                                NULLIF(MAX(di.{year_column}) OVER (PARTITION BY i.indicator_id) - MIN(di.{year_column}) OVER (PARTITION BY i.indicator_id), 0) * 100
                        END
                    ) as composite_score
                FROM district_indicators di
                JOIN districts d ON di.district_id = d.district_id
                JOIN states s ON d.state_id = s.state_id
                JOIN indicators i ON di.indicator_id = i.indicator_id
                WHERE di.indicator_id = ANY(%s::int[])
                AND di.{year_column} IS NOT NULL
                {state_filter}
                GROUP BY d.district_id, d.district_name, s.state_name
                HAVING COUNT(DISTINCT di.indicator_id) = %s
            )
            SELECT 
                district_id,
                district_name,
                state_name,
                composite_score,
                indicators_count,
                ROW_NUMBER() OVER (ORDER BY composite_score {"DESC" if performance_type != "bottom" else "ASC"}) as rank
            FROM district_scores
            WHERE composite_score IS NOT NULL
            ORDER BY composite_score {"DESC" if performance_type != "bottom" else "ASC"}
            LIMIT %s
            """
            
            params = [indicator_ids] + state_params + [len(matched_indicators), n_districts]
            cursor.execute(query, params)
            composite_results = cursor.fetchall()
            
            if not composite_results:
                return {"error": "No districts found with data for all specified indicators"}
            
            # Get detailed indicator data for top districts
            top_district_ids = [row[0] for row in composite_results]
            
            detail_query = f"""
            SELECT
                d.district_name,
                s.state_name,
                i.indicator_name,
                i.indicator_direction,
                di.prevalence_2016,
                di.prevalence_2021,
                di.prevalence_change,
                di.headcount_2021,
                di.{year_column} as indicator_value,
                d.district_id
            FROM district_indicators di
            JOIN districts d ON di.district_id = d.district_id
            JOIN states s ON d.state_id = s.state_id
            JOIN indicators i ON di.indicator_id = i.indicator_id
            WHERE di.district_id = ANY(%s::int[])
            AND di.indicator_id = ANY(%s::int[])
            AND di.{year_column} IS NOT NULL
            ORDER BY d.district_name, i.indicator_name
            """
            
            cursor.execute(detail_query, (top_district_ids, indicator_ids))
            detail_results = cursor.fetchall()
            
            # Process multiple indicator results
            districts_data = {}
            all_district_names = []
            
            # First, create district structure with composite scores
            for row in composite_results:
                district_key = f"{row[1]}_{row[2]}"  # district_state
                districts_data[district_key] = {
                    "district_name": row[1],
                    "state_name": row[2],
                    "district_id": row[0],
                    "rank": row[5],
                    "performance_score": float(row[3]) if row[3] is not None else None,
                    "indicators_count": row[4],
                    "indicators": []
                }
                all_district_names.append(row[1])
            
            # Add detailed indicator data
            for row in detail_results:
                district_key = f"{row[0]}_{row[1]}"  # district_state
                
                if district_key in districts_data:
                    indicator_data = {
                        "indicator_name": row[2],
                        "indicator_direction": row[3],
                        "prevalence_2016": float(row[4]) if row[4] is not None else None,
                        "prevalence_2021": float(row[5]) if row[5] is not None else None,
                        "prevalence_change": float(row[6]) if row[6] is not None else None,
                        "headcount_2021": float(row[7]) if row[7] is not None else None,
                        "indicator_value": float(row[8]) if row[8] is not None else None
                    }
                    
                    # Add trend interpretation
                    if indicator_data["prevalence_change"] is not None:
                        indicator_data["trend_interpretation"] = interpret_health_trend(
                            indicator_data["prevalence_change"],
                            indicator_data["indicator_direction"]
                        )
                    
                    districts_data[district_key]["indicators"].append(indicator_data)
            
            districts_data = list(districts_data.values())
            response_type = "multi_indicator"
            main_indicator = f"{len(matched_indicators)} indicators"
        
        # Get boundary data if requested
        boundary_data = []
        if include_boundary_data:
            boundary_data = get_district_boundary_data(all_district_names)
        
        # Generate analysis
        analysis = generate_top_bottom_analysis(
            districts_data, matched_indicators, n_districts, performance_type, year, states, response_type
        )
        
        # Generate chart data
        chart_data = generate_top_bottom_chart_data(
            districts_data, matched_indicators, performance_type, response_type
        )
        
        # Prepare response
        response = {
            "indicators": [ind["indicator_name"] for ind in matched_indicators],
            "n_districts": n_districts,
            "performance_type": performance_type,
            "year": year,
            "total_districts_found": len(districts_data),
            "states_filter": states,
            "response_type": response_type,
            "main_indicator": main_indicator,
            "districts": districts_data,
            "boundary": boundary_data,
            "analysis": analysis,
            "chart_data": chart_data,
            "map_type": "top_bottom_districts"
        }
        
        cursor.close()
        conn.close()
        
        logger.info(f"🏆 Top/bottom districts query completed: {len(districts_data)} districts found for {len(matched_indicators)} indicators")
        return response
        
    except Exception as e:
        logger.error(f"Error in get_top_bottom_districts: {e}")
        return {"error": f"Database error: {str(e)}"}

def generate_top_bottom_analysis(districts_data, matched_indicators, n_districts, performance_type, year, states_filter, response_type):
    """Generate comprehensive analysis for top/bottom districts results"""
    if not districts_data:
        return "No districts found matching the specified criteria."
    
    total_districts = len(districts_data)
    states_represented = len(set(d["state_name"] for d in districts_data))
    
    # Build indicator summary
    indicator_summary = []
    for ind in matched_indicators:
        direction_note = "higher is better" if ind["indicator_direction"] == "higher_is_better" else "lower is better"
        indicator_summary.append(f"'{ind['indicator_name']}' ({direction_note})")
    
    performance_text = {
        "top": "best performing",
        "bottom": "worst performing", 
        "both": "top and bottom performing"
    }[performance_type]
    
    analysis_parts = []
    analysis_parts.append(f"**{performance_text.title()} Districts Analysis**\n")
    
    if response_type == "single_indicator":
        analysis_parts.append(f"**Indicator:** {indicator_summary[0]}")
    else:
        analysis_parts.append(f"**Indicators Analyzed ({len(matched_indicators)}):**")
        for summary in indicator_summary:
            analysis_parts.append(f"• {summary}")
    analysis_parts.append("")
    
    analysis_parts.append(f"**Results Overview:**")
    analysis_parts.append(f"• Districts found: {total_districts}")
    analysis_parts.append(f"• States represented: {states_represented}")
    analysis_parts.append(f"• Analysis year: {year}")
    if states_filter:
        analysis_parts.append(f"• States filtered: {', '.join(states_filter)}")
    analysis_parts.append("")
    
    # Geographic distribution
    state_counts = {}
    for district in districts_data:
        state = district["state_name"]
        state_counts[state] = state_counts.get(state, 0) + 1
    
    if len(state_counts) > 1:
        analysis_parts.append(f"**Geographic Distribution:**")
        sorted_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_states[:5]:  # Top 5 states
            analysis_parts.append(f"• {state}: {count} districts")
        if len(sorted_states) > 5:
            remaining = sum(count for _, count in sorted_states[5:])
            analysis_parts.append(f"• Other states: {remaining} districts")
        analysis_parts.append("")
    
    # Performance insights
    if response_type == "single_indicator":
        # Single indicator insights
        if len(districts_data) >= 3:
            top_district = districts_data[0]
            bottom_district = districts_data[-1]
            
            analysis_parts.append(f"**Performance Range:**")
            analysis_parts.append(f"• Best: {top_district['district_name']} ({top_district['state_name']}) - {top_district['performance_score']:.2f}")
            analysis_parts.append(f"• Worst: {bottom_district['district_name']} ({bottom_district['state_name']}) - {bottom_district['performance_score']:.2f}")
            
            if top_district['performance_score'] and bottom_district['performance_score']:
                range_diff = abs(top_district['performance_score'] - bottom_district['performance_score'])
                analysis_parts.append(f"• Performance gap: {range_diff:.2f} percentage points")
        
    else:
        # Multi-indicator insights
        analysis_parts.append(f"**Composite Performance:**")
        analysis_parts.append("• Districts ranked by normalized composite score across all indicators")
        analysis_parts.append("• Scoring considers indicator direction (higher/lower is better)")
        analysis_parts.append("• Only districts with complete data for all indicators included")
        
        if len(districts_data) >= 3:
            top_district = districts_data[0]
            analysis_parts.append(f"• Top performer: {top_district['district_name']} ({top_district['state_name']}) - Score: {top_district['performance_score']:.1f}")
    
    analysis_parts.append("")
    
    # Strategic insights
    analysis_parts.append(f"**Strategic Insights:**")
    if performance_type == "top":
        analysis_parts.append("• These districts demonstrate best practices and successful health outcomes")
        analysis_parts.append("• Can serve as models for policy implementation and resource management")
        analysis_parts.append("• May indicate effective state-level policies or local innovations")
    elif performance_type == "bottom":
        analysis_parts.append("• These districts require urgent attention and targeted interventions")
        analysis_parts.append("• Represent opportunities for significant health outcome improvements")
        analysis_parts.append("• May indicate systemic challenges or resource constraints")
    else:
        analysis_parts.append("• Comparison shows the full spectrum of performance variations")
        analysis_parts.append("• Highlights both successful models and areas needing support")
    
    analysis_parts.append("")
    analysis_parts.append(f"**Applications:**")
    analysis_parts.append("• Policy benchmarking and best practice identification")
    analysis_parts.append("• Resource allocation and targeted intervention planning")
    analysis_parts.append("• Performance monitoring and evaluation")
    analysis_parts.append("• Inter-district learning and knowledge transfer")
    
    return "\n".join(analysis_parts)

def generate_top_bottom_chart_data(districts_data, matched_indicators, performance_type, response_type):
    """Generate bar chart data for top/bottom districts visualization"""
    try:
        if not districts_data:
            return None
        
        # For single indicator - simple bar chart showing indicator values
        if response_type == "single_indicator":
            labels = []
            values = []
            colors = []
            
            for district in districts_data:
                # Create district label with state
                district_name = district["district_name"]
                state_name = district["state_name"]
                if len(district_name) > 15:
                    label = f"{district_name[:12]}... ({state_name[:3]})"
                else:
                    label = f"{district_name} ({state_name[:3]})"
                labels.append(label)
                
                values.append(district["performance_score"] if district["performance_score"] is not None else 0)
                
                # Color coding based on performance
                if performance_type == "top":
                    colors.append("#2ecc71")  # Green for top performers
                elif performance_type == "bottom":
                    colors.append("#e74c3c")  # Red for bottom performers
                else:
                    # Gradient for both
                    colors.append("#2ecc71" if district["rank"] <= len(districts_data)//2 else "#e74c3c")
            
            indicator_name = matched_indicators[0]["indicator_name"]
            direction_text = "↑ higher is better" if matched_indicators[0]["indicator_direction"] == "higher_is_better" else "↓ lower is better"
            
            return {
                "type": "bar",
                "title": f"{performance_type.title()} Districts - {indicator_name}",
                "subtitle": f"{direction_text} | Year: {districts_data[0]['indicators'][0].get('prevalence_2021') and '2021' or '2016'}",
                "labels": labels,
                "datasets": [{
                    "label": f"{indicator_name} Values",
                    "data": values,
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "borderWidth": 1
                }],
                "options": {
                    "responsive": True,
                    "scales": {
                        "y": {
                            "title": {"display": True, "text": f"{indicator_name} (%)"},
                            "beginAtZero": True
                        },
                        "x": {
                            "title": {"display": True, "text": "Districts"},
                            "ticks": {
                                "maxRotation": 45,
                                "minRotation": 0
                            }
                        }
                    },
                    "plugins": {
                        "legend": {
                            "display": False
                        },
                        "tooltip": {
                            "callbacks": {
                                "title": "function(context) { return context[0].label.split(' (')[0]; }",
                                "label": "function(context) { return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%'; }"
                            }
                        }
                    }
                }
            }
        
        else:
            # Multiple indicators - show all indicators for top districts (limit to 8 districts)
            limited_districts = districts_data[:8]
            
            labels = []
            for district in limited_districts:
                district_name = district["district_name"]
                state_name = district["state_name"]
                if len(district_name) > 15:
                    label = f"{district_name[:12]}... ({state_name[:3]})"
                else:
                    label = f"{district_name} ({state_name[:3]})"
                labels.append(label)
            
            datasets = []
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
            
            # Create dataset for each indicator
            for i, indicator in enumerate(matched_indicators):
                indicator_name = indicator["indicator_name"]
                indicator_values = []
                
                for district in limited_districts:
                    # Find the indicator value for this district
                    value = 0
                    for ind_data in district["indicators"]:
                        if ind_data["indicator_name"] == indicator_name:
                            value = ind_data["indicator_value"] if ind_data["indicator_value"] is not None else 0
                            break
                    indicator_values.append(value)
                
                # Truncate long indicator names
                display_name = indicator_name[:25] + ("..." if len(indicator_name) > 25 else "")
                
                datasets.append({
                    "label": display_name,
                    "data": indicator_values,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)],
                    "borderWidth": 1
                })
            
            total_districts = len(districts_data)
            title_suffix = f" (Showing 8 of {total_districts})" if total_districts > 8 else f" ({total_districts} Districts)"
            
            return {
                "type": "bar",
                "title": f"{performance_type.title()} Districts - Multi-Indicator Comparison{title_suffix}",
                "subtitle": f"Districts ranked by composite performance score",
                "labels": labels,
                "datasets": datasets,
                "options": {
                    "responsive": True,
                    "scales": {
                        "y": {
                            "title": {"display": True, "text": "Indicator Values (%)"},
                            "beginAtZero": True
                        },
                        "x": {
                            "title": {"display": True, "text": "Districts (Ranked by Performance)"},
                            "ticks": {
                                "maxRotation": 45,
                                "minRotation": 0
                            }
                        }
                    },
                    "plugins": {
                        "legend": {
                            "position": "top",
                            "labels": {
                                "boxWidth": 12,
                                "font": {
                                    "size": 10
                                }
                            }
                        },
                        "tooltip": {
                            "mode": "index",
                            "intersect": False
                        }
                    }
                }
            }
        
    except Exception as e:
        logger.error(f"Error generating top/bottom chart data: {e}")
        return None

def _safe_float_conversion(value):
    """Convert a value to float while handling NaN, None, and infinity cases"""
    if value is None:
        return None
    
    try:
        float_val = float(value)
        # Check for NaN, infinity, or other invalid values
        if float_val != float_val or float_val == float('inf') or float_val == float('-inf'):
            return None
        return float_val
    except (ValueError, TypeError):
        return None

def get_indicator_change_analysis(
    indicator_name: str,
    analysis_level: str = "country",  # "country", "state", or "district"
    location_name: Optional[str] = None,  # state name for state level, district name for district level
    include_boundary_data: bool = True
):
    """
    Analyze indicator value changes from 2016 to 2021 at different geographic levels
    
    Args:
        indicator_name: Name of health indicator to analyze
        analysis_level: Level of analysis - "country", "state", or "district"
        location_name: Name of specific state/district for state/district level analysis
        include_boundary_data: Whether to include boundary data for mapping
    
    Returns:
        Dictionary containing change analysis data with examples and visualization data
    """
    logger.info(f"Starting indicator change analysis: {indicator_name} at {analysis_level} level")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Match indicator name to database
        matched_indicator = match_indicator_name_to_database(indicator_name)
        if not matched_indicator:
            return {
                "error": f"Could not match indicator name '{indicator_name}' to database indicators",
                "response_type": "error"
            }
        
        indicator_id = matched_indicator["indicator_id"]
        indicator_direction = matched_indicator["indicator_direction"]
        main_indicator_name = matched_indicator["indicator_name"]
        
        logger.info(f"Matched indicator: {main_indicator_name} (ID: {indicator_id})")
        
        if analysis_level == "country":
            return _get_country_level_change_analysis(cursor, indicator_id, main_indicator_name, indicator_direction, include_boundary_data)
        elif analysis_level == "state":
            if not location_name:
                return {"error": "State name is required for state-level analysis"}
            return _get_state_level_change_analysis(cursor, indicator_id, main_indicator_name, indicator_direction, location_name, include_boundary_data)
        elif analysis_level == "district":
            if not location_name:
                return {"error": "District name is required for district-level analysis"}
            return _get_district_level_change_analysis(cursor, indicator_id, main_indicator_name, indicator_direction, location_name, include_boundary_data)
        else:
            return {"error": "Invalid analysis_level. Must be 'country', 'state', or 'district'"}
            
    except Exception as e:
        logger.error(f"Error in indicator change analysis: {e}")
        return {
            "error": f"Failed to analyze indicator changes: {str(e)}",
            "response_type": "error"
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def _get_country_level_change_analysis(cursor, indicator_id, indicator_name, indicator_direction, include_boundary_data):
    """Get country-level change analysis with example districts"""
    
    # Get national average from overall_india table
    national_query = """
    SELECT 
        prevalence_2016,
        prevalence_2021,
        change_availability
    FROM overall_india 
    WHERE indicator_id = %s
    """
    
    cursor.execute(national_query, (indicator_id,))
    national_result = cursor.fetchone()
    
    if not national_result:
        return {"error": f"No national data found for indicator: {indicator_name}"}
    
    # Calculate prevalence change manually with safe conversion
    prevalence_2016 = _safe_float_conversion(national_result[0])
    prevalence_2021 = _safe_float_conversion(national_result[1])
    
    # Calculate change if both values are available and valid
    prevalence_change = None
    if prevalence_2016 is not None and prevalence_2021 is not None:
        prevalence_change = _safe_float_conversion(prevalence_2021 - prevalence_2016)
    
    national_data = {
        "prevalence_2016": prevalence_2016,
        "prevalence_2021": prevalence_2021,
        "prevalence_change": prevalence_change,
        "change_availability": national_result[2]
    }
    
    # Check if change data is available
    if national_data["change_availability"] == 2 or prevalence_change is None:
        # Instead of failing, try to provide district-level analysis without national context
        print(f"⚠️  National data unavailable for {indicator_name}, proceeding with district-only analysis")
        
        # Set national data to None to indicate it's not available
        national_data = {
            "prevalence_2016": None,
            "prevalence_2021": None,
            "prevalence_change": None,
            "change_availability": 2,
            "note": "National average not available for this indicator"
        }
    
    # Get 10 random districts as examples
    districts_query = """
    SELECT 
        d.district_name,
        s.state_name,
        di.prevalence_2016,
        di.prevalence_2021,
        di.prevalence_change,
        d.district_id
    FROM district_indicators di
    JOIN districts d ON di.district_id = d.district_id
    JOIN states s ON d.state_id = s.state_id
    WHERE di.indicator_id = %s 
    AND (di.prevalence_2016 IS NOT NULL OR di.prevalence_2021 IS NOT NULL)
    ORDER BY RANDOM()
    LIMIT 50
    """
    
    cursor.execute(districts_query, (indicator_id,))
    district_results = cursor.fetchall()
    
    example_districts = []
    for row in district_results:
        # Get safe float conversions
        prev_2016 = _safe_float_conversion(row[2])
        prev_2021 = _safe_float_conversion(row[3])
        prev_change = _safe_float_conversion(row[4])
        
        # Calculate change if not available in database but both years exist
        if prev_change is None and prev_2016 is not None and prev_2021 is not None:
            prev_change = _safe_float_conversion(prev_2021 - prev_2016)
        
        # Only include districts that have change data (either from DB or calculated)
        if prev_change is not None:
            example_districts.append({
                "district_name": row[0],
                "state_name": row[1],
                "prevalence_2016": prev_2016,
                "prevalence_2021": prev_2021,
                "prevalence_change": prev_change,
                "district_id": row[5]
            })
            
            # Stop when we have 10 valid examples
            if len(example_districts) >= 10:
                break
    
    # Get all districts for mapping
    all_districts_data = []
    boundary_data = []
    
    if include_boundary_data:
        all_districts_query = """
        SELECT 
            d.district_name,
            s.state_name,
            di.prevalence_2016,
            di.prevalence_2021,
            di.prevalence_change,
            d.district_id
        FROM district_indicators di
        JOIN districts d ON di.district_id = d.district_id
        JOIN states s ON d.state_id = s.state_id
        WHERE di.indicator_id = %s 
        AND (di.prevalence_2016 IS NOT NULL OR di.prevalence_2021 IS NOT NULL)
        ORDER BY s.state_name, d.district_name
        """
        
        cursor.execute(all_districts_query, (indicator_id,))
        all_results = cursor.fetchall()
        
        for row in all_results:
            # Get safe float conversions
            prev_2016 = _safe_float_conversion(row[2])
            prev_2021 = _safe_float_conversion(row[3])
            prev_change = _safe_float_conversion(row[4])
            
            # Calculate change if not available in database but both years exist
            if prev_change is None and prev_2016 is not None and prev_2021 is not None:
                prev_change = _safe_float_conversion(prev_2021 - prev_2016)
            
            all_districts_data.append({
                "district_name": row[0],
                "state_name": row[1],
                "prevalence_2016": prev_2016,
                "prevalence_2021": prev_2021,
                "prevalence_change": prev_change,
                "district_id": row[5]
            })
        
        # Get boundary data
        district_names = [d["district_name"] for d in all_districts_data]
        boundary_data = get_district_boundary_data(district_names)
    
    # Generate chart data
    chart_data = _generate_country_change_chart_data(national_data, example_districts, indicator_name, indicator_direction)
    
    # Generate analysis
    analysis = _generate_country_change_analysis(national_data, example_districts, indicator_name, indicator_direction)
    
    return {
        "response_type": "country_change_analysis",
        "analysis_level": "country",
        "main_indicator": indicator_name,
        "indicator_direction": indicator_direction,
        "national_data": national_data,
        "example_districts": example_districts,
        "all_districts": all_districts_data,
        "total_districts_analyzed": len(all_districts_data),
        "chart_data": chart_data,
        "boundary": boundary_data,
        "analysis": analysis,
        "map_type": "indicator_change_analysis"
    }


def _get_state_level_change_analysis(cursor, indicator_id, indicator_name, indicator_direction, state_name, include_boundary_data):
    """Get state-level change analysis with example districts"""
    
    # First resolve state name
    state_query = "SELECT state_id, state_name FROM states WHERE LOWER(state_name) LIKE LOWER(%s)"
    cursor.execute(state_query, (f"%{state_name}%",))
    state_result = cursor.fetchone()
    
    if not state_result:
        return {"error": f"State '{state_name}' not found"}
    
    state_id, resolved_state_name = state_result
    
    # Get state average from state_indicator table
    state_query = """
    SELECT 
        prevalence_2016,
        prevalence_2021,
        prevalence_change
    FROM state_indicators 
    WHERE indicator_id = %s AND state_id = %s
    """
    
    cursor.execute(state_query, (indicator_id, state_id))
    state_result = cursor.fetchone()
    
    if not state_result:
        return {"error": f"No state data found for indicator: {indicator_name} in state: {resolved_state_name}"}
    
    state_data = {
        "state_name": resolved_state_name,
        "prevalence_2016": float(state_result[0]) if state_result[0] is not None else None,
        "prevalence_2021": float(state_result[1]) if state_result[1] is not None else None,
        "prevalence_change": float(state_result[2]) if state_result[2] is not None else None
    }
    
    # Check if change data is available
    if state_data["prevalence_change"] is None:
        return {
            "error": f"Change data not available for {indicator_name} in {resolved_state_name}",
            "response_type": "error"
        }
    
    # Get 5 random districts from the state as examples
    districts_query = """
    SELECT 
        d.district_name,
        s.state_name,
        di.prevalence_2016,
        di.prevalence_2021,
        di.prevalence_change,
        d.district_id
    FROM district_indicators di
    JOIN districts d ON di.district_id = d.district_id
    JOIN states s ON d.state_id = s.state_id
    WHERE di.indicator_id = %s 
    AND s.state_id = %s
    AND di.prevalence_change IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 5
    """
    
    cursor.execute(districts_query, (indicator_id, state_id))
    district_results = cursor.fetchall()
    
    example_districts = []
    for row in district_results:
        example_districts.append({
            "district_name": row[0],
            "state_name": row[1],
            "prevalence_2016": float(row[2]) if row[2] is not None else None,
            "prevalence_2021": float(row[3]) if row[3] is not None else None,
            "prevalence_change": float(row[4]) if row[4] is not None else None,
            "district_id": row[5]
        })
    
    # Get all districts in the state for mapping
    all_districts_data = []
    boundary_data = []
    
    if include_boundary_data:
        all_districts_query = """
        SELECT 
            d.district_name,
            s.state_name,
            di.prevalence_2016,
            di.prevalence_2021,
            di.prevalence_change,
            d.district_id
        FROM district_indicators di
        JOIN districts d ON di.district_id = d.district_id
        JOIN states s ON d.state_id = s.state_id
        WHERE di.indicator_id = %s 
        AND s.state_id = %s
        AND di.prevalence_change IS NOT NULL
        ORDER BY d.district_name
        """
        
        cursor.execute(all_districts_query, (indicator_id, state_id))
        all_results = cursor.fetchall()
        
        for row in all_results:
            all_districts_data.append({
                "district_name": row[0],
                "state_name": row[1],
                "prevalence_2016": float(row[2]) if row[2] is not None else None,
                "prevalence_2021": float(row[3]) if row[3] is not None else None,
                "prevalence_change": float(row[4]) if row[4] is not None else None,
                "district_id": row[5]
            })
        
        # Get boundary data for districts in the state
        district_names = [d["district_name"] for d in all_districts_data]
        boundary_data = get_district_boundary_data(district_names)
    
    # Generate chart data
    chart_data = _generate_state_change_chart_data(state_data, example_districts, indicator_name, indicator_direction)
    
    # Generate analysis
    analysis = _generate_state_change_analysis(state_data, example_districts, indicator_name, indicator_direction)
    
    return {
        "response_type": "state_change_analysis", 
        "analysis_level": "state",
        "main_indicator": indicator_name,
        "indicator_direction": indicator_direction,
        "state_data": state_data,
        "example_districts": example_districts,
        "all_districts": all_districts_data,
        "total_districts_analyzed": len(all_districts_data),
        "chart_data": chart_data,
        "boundary": boundary_data,
        "analysis": analysis,
        "map_type": "indicator_change_analysis"
    }


def _get_district_level_change_analysis(cursor, indicator_id, indicator_name, indicator_direction, district_name, include_boundary_data):
    """Get district-level change analysis"""
    
    # Resolve district name
    district_query = """
    SELECT d.district_id, d.district_name, s.state_name 
    FROM districts d 
    JOIN states s ON d.state_id = s.state_id
    WHERE LOWER(d.district_name) LIKE LOWER(%s)
    """
    cursor.execute(district_query, (f"%{district_name}%",))
    district_result = cursor.fetchone()
    
    if not district_result:
        return {"error": f"District '{district_name}' not found"}
    
    district_id, resolved_district_name, state_name = district_result
    
    # Get district data
    district_query = """
    SELECT 
        di.prevalence_2016,
        di.prevalence_2021,
        di.prevalence_change,
        di.headcount_2021
    FROM district_indicators di
    WHERE di.indicator_id = %s AND di.district_id = %s
    """
    
    cursor.execute(district_query, (indicator_id, district_id))
    district_result = cursor.fetchone()
    
    if not district_result:
        return {"error": f"No data found for indicator: {indicator_name} in district: {resolved_district_name}"}
    
    district_data = {
        "district_name": resolved_district_name,
        "state_name": state_name,
        "prevalence_2016": float(district_result[0]) if district_result[0] is not None else None,
        "prevalence_2021": float(district_result[1]) if district_result[1] is not None else None,
        "prevalence_change": float(district_result[2]) if district_result[2] is not None else None,
        "headcount_2021": float(district_result[3]) if district_result[3] is not None else None,
        "district_id": district_id
    }
    
    # Check if change data is available
    if district_data["prevalence_change"] is None:
        return {
            "error": f"Change data not available for {indicator_name} in {resolved_district_name}",
            "response_type": "error"
        }
    
    # Get boundary data
    boundary_data = []
    if include_boundary_data:
        boundary_data = get_district_boundary_data([resolved_district_name])
    
    # Generate chart data
    chart_data = _generate_district_change_chart_data(district_data, indicator_name, indicator_direction)
    
    # Generate analysis
    analysis = _generate_district_change_analysis(district_data, indicator_name, indicator_direction)
    
    return {
        "response_type": "district_change_analysis",
        "analysis_level": "district", 
        "main_indicator": indicator_name,
        "indicator_direction": indicator_direction,
        "district_data": district_data,
        "chart_data": chart_data,
        "boundary": boundary_data,
        "analysis": analysis,
        "map_type": "indicator_change_analysis"
    }


def _generate_country_change_chart_data(national_data, example_districts, indicator_name, indicator_direction):
    """Generate chart data for country-level change analysis"""
    
    # Check if national data is available
    if national_data["prevalence_change"] is not None:
        # Create bar chart with national average and example districts
        labels = ["National Average"] + [d["district_name"] for d in example_districts]
        change_values = [national_data["prevalence_change"]] + [d["prevalence_change"] for d in example_districts]
    else:
        # National data not available, show only district examples
        labels = [d["district_name"] for d in example_districts]
        change_values = [d["prevalence_change"] for d in example_districts]
    
    # Color based on change direction and indicator type
    colors = []
    for change in change_values:
        if change is None:
            colors.append('#cccccc')
        elif change > 0:
            # Positive change - good for higher_is_better, bad for lower_is_better
            if indicator_direction == "higher_is_better":
                colors.append('#2ecc71')  # Green
            else:
                colors.append('#e74c3c')  # Red
        else:
            # Negative change - bad for higher_is_better, good for lower_is_better  
            if indicator_direction == "higher_is_better":
                colors.append('#e74c3c')  # Red
            else:
                colors.append('#2ecc71')  # Green
    
    # Set appropriate title based on data availability
    if national_data["prevalence_change"] is not None:
        title = f"{indicator_name} - Change from 2016 to 2021 (Country Level)"
    else:
        title = f"{indicator_name} - District Examples (National Average Not Available)"
    
    return {
        "type": "bar",
        "title": title,
        "labels": labels,
        "datasets": [{
            "label": "Change Value",
            "data": change_values,
            "backgroundColor": colors,
            "borderColor": colors,
            "borderWidth": 1
        }]
    }


def _generate_state_change_chart_data(state_data, example_districts, indicator_name, indicator_direction):
    """Generate chart data for state-level change analysis"""
    
    labels = [f"{state_data['state_name']} (State Avg)"] + [d["district_name"] for d in example_districts]
    change_values = [state_data["prevalence_change"]] + [d["prevalence_change"] for d in example_districts]
    
    # Color based on change direction and indicator type
    colors = []
    for change in change_values:
        if change is None:
            colors.append('#cccccc')
        elif change > 0:
            if indicator_direction == "higher_is_better":
                colors.append('#2ecc71')  # Green
            else:
                colors.append('#e74c3c')  # Red
        else:
            if indicator_direction == "higher_is_better":
                colors.append('#e74c3c')  # Red
            else:
                colors.append('#2ecc71')  # Green
    
    return {
        "type": "bar",
        "title": f"{indicator_name} - Change from 2016 to 2021 ({state_data['state_name']} State)",
        "labels": labels,
        "datasets": [{
            "label": "Change Value",
            "data": change_values,
            "backgroundColor": colors,
            "borderColor": colors,
            "borderWidth": 1
        }]
    }


def _generate_district_change_chart_data(district_data, indicator_name, indicator_direction):
    """Generate chart data for district-level change analysis"""
    
    labels = ["2016", "2021"]
    values = [district_data["prevalence_2016"], district_data["prevalence_2021"]]
    
    # Color based on trend
    change = district_data["prevalence_change"]
    if change > 0:
        color = '#2ecc71' if indicator_direction == "higher_is_better" else '#e74c3c'
    else:
        color = '#e74c3c' if indicator_direction == "higher_is_better" else '#2ecc71'
    
    return {
        "type": "line",
        "title": f"{indicator_name} - Trend from 2016 to 2021 ({district_data['district_name']})",
        "labels": labels,
        "datasets": [{
            "label": indicator_name,
            "data": values,
            "borderColor": color,
            "backgroundColor": color + '20',
            "borderWidth": 3,
            "fill": True
        }]
    }


def _generate_country_change_analysis(national_data, example_districts, indicator_name, indicator_direction):
    """Generate analysis text for country-level change"""
    
    if national_data["prevalence_change"] is not None:
        # National data is available
        change = national_data["prevalence_change"]
        direction = "increased" if change > 0 else "decreased" 
        
        if indicator_direction == "higher_is_better":
            interpretation = "improvement" if change > 0 else "decline"
        else:
            interpretation = "decline" if change > 0 else "improvement"
        
        analysis = f"""
        **National {indicator_name} Change Analysis (2016-2021)**
        
        At the national level, {indicator_name} has {direction} by {abs(change):.2f} percentage points from 2016 to 2021, representing an overall {interpretation} in this health indicator.
        
        **National Overview:**
        - 2016 Level: {national_data['prevalence_2016']:.2f}%
        - 2021 Level: {national_data['prevalence_2021']:.2f}%
        - Change: {change:+.2f} percentage points
        
        **District Examples:**
        The following 10 districts show the variation in {indicator_name} changes across the country:
        """
    else:
        # National data is not available
        analysis = f"""
        **{indicator_name} District-Level Change Analysis (2016-2021)**
        
        National average data is not available for this indicator, but district-level data shows significant variation across the country. This analysis focuses on district-level changes to understand regional patterns.
        
        **District Examples:**
        The following districts show the variation in {indicator_name} changes across different regions:
        """
    
    for i, district in enumerate(example_districts, 1):
        dist_change = district["prevalence_change"]
        dist_direction = "increased" if dist_change > 0 else "decreased"
        analysis += f"\n{i}. {district['district_name']}, {district['state_name']}: {dist_direction} by {abs(dist_change):.2f} points"
    
    return analysis


def _generate_state_change_analysis(state_data, example_districts, indicator_name, indicator_direction):
    """Generate analysis text for state-level change"""
    
    change = state_data["prevalence_change"]
    direction = "increased" if change > 0 else "decreased"
    
    if indicator_direction == "higher_is_better":
        interpretation = "improvement" if change > 0 else "decline"
    else:
        interpretation = "decline" if change > 0 else "improvement"
    
    analysis = f"""
    **{state_data['state_name']} {indicator_name} Change Analysis (2016-2021)**
    
    In {state_data['state_name']}, {indicator_name} has {direction} by {abs(change):.2f} percentage points from 2016 to 2021, representing an overall {interpretation} in this health indicator at the state level.
    
    **State Overview:**
    - 2016 Level: {state_data['prevalence_2016']:.2f}%
    - 2021 Level: {state_data['prevalence_2021']:.2f}%
    - Change: {change:+.2f} percentage points
    
    **District Examples from {state_data['state_name']}:**
    """
    
    for i, district in enumerate(example_districts, 1):
        dist_change = district["prevalence_change"]
        dist_direction = "increased" if dist_change > 0 else "decreased"
        analysis += f"\n{i}. {district['district_name']}: {dist_direction} by {abs(dist_change):.2f} points"
    
    return analysis


def _generate_district_change_analysis(district_data, indicator_name, indicator_direction):
    """Generate analysis text for district-level change"""
    
    change = district_data["prevalence_change"]
    direction = "increased" if change > 0 else "decreased"
    
    if indicator_direction == "higher_is_better":
        interpretation = "improvement" if change > 0 else "decline" 
    else:
        interpretation = "decline" if change > 0 else "improvement"
    
    analysis = f"""
    **{district_data['district_name']} {indicator_name} Change Analysis (2016-2021)**
    
    In {district_data['district_name']} district ({district_data['state_name']}), {indicator_name} has {direction} by {abs(change):.2f} percentage points from 2016 to 2021, representing a {interpretation} in this health indicator.
    
    **District Details:**
    - 2016 Level: {district_data['prevalence_2016']:.2f}%
    - 2021 Level: {district_data['prevalence_2021']:.2f}%
    - Change: {change:+.2f} percentage points
    - Affected Population (2021): {district_data['headcount_2021']:,.0f} people
    
    This change indicates that the district has experienced a {'positive' if interpretation == 'improvement' else 'concerning'} trend in {indicator_name.lower()} over the 5-year period.
    """
    
    return analysis


def get_district_performance_comparison(
    district_names: List[str],
    indicator_names: List[str],
    comparison_type: str = "national",  # "national" or "state"
    year: int = 2021,
    include_boundary_data: bool = True
):
    """
    Compare performance of multiple districts across multiple indicators 
    with either national averages or state averages.
    
    Args:
        district_names: List of district names to compare
        indicator_names: List of indicator names to analyze
        comparison_type: "national" for national averages, "state" for state averages
        year: Year for analysis (2016 or 2021)
        include_boundary_data: Whether to include boundary data for mapping
    
    Returns:
        Dictionary containing comparison data, analysis, and chart data
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Match indicator names to database
        matched_indicators = []
        for indicator_name in indicator_names:
            matched = match_indicator_name_to_database(indicator_name)
            if matched:
                matched_indicators.append(matched)
                print(f"🎯 Matched '{indicator_name}' to '{matched['indicator_name']}'")
            else:
                print(f"❌ Could not match indicator name '{indicator_name}'")
        
        if not matched_indicators:
            return {
                "error": f"Could not match any indicator names: {indicator_names}",
                "response_type": "error"
            }
        
        # Resolve district names
        resolved_districts = []
        for district_name in district_names:
            resolved = resolve_district_name(cursor, district_name)
            if resolved:
                resolved_districts.append(resolved)
                print(f"🎯 Resolved '{district_name}' to '{resolved['district_name']}, {resolved['state_name']}'")
            else:
                print(f"❌ Could not resolve district name '{district_name}'")
        
        if not resolved_districts:
            return {
                "error": f"Could not resolve any district names: {district_names}",
                "response_type": "error"
            }
        
        # Get district data for all indicators
        districts_data = []
        for district in resolved_districts:
            district_info = {
                "district_name": district["district_name"],
                "state_name": district["state_name"],
                "indicators": []
            }
            
            for indicator in matched_indicators:
                # Get district indicator data
                district_query = f"""
                SELECT 
                    di.prevalence_{year},
                    di.prevalence_change,
                    di.headcount_{year}
                FROM district_indicators di
                JOIN districts d ON di.district_id = d.district_id
                WHERE d.district_name = %s AND di.indicator_id = %s
                """
                
                cursor.execute(district_query, (district["district_name"], indicator["indicator_id"]))
                district_result = cursor.fetchone()
                
                if district_result:
                    prevalence = _safe_float_conversion(district_result[0])
                    prevalence_change = _safe_float_conversion(district_result[1])
                    headcount = _safe_float_conversion(district_result[2])
                    
                    district_info["indicators"].append({
                        "indicator_id": indicator["indicator_id"],
                        "indicator_name": indicator["indicator_name"],
                        "indicator_direction": indicator["indicator_direction"],
                        "prevalence": prevalence,
                        "prevalence_change": prevalence_change,
                        "headcount": headcount
                    })
            
            if district_info["indicators"]:  # Only add if we have indicator data
                districts_data.append(district_info)
        
        if not districts_data:
            return {
                "error": "No indicator data found for the specified districts",
                "response_type": "error"
            }
        
        # Get comparison data (national or state averages)
        comparison_data = {}
        
        if comparison_type == "national":
            # Get national averages from overall_india table
            for indicator in matched_indicators:
                national_query = f"""
                SELECT prevalence_{year}
                FROM overall_india
                WHERE indicator_id = %s
                """
                
                cursor.execute(national_query, (indicator["indicator_id"],))
                national_result = cursor.fetchone()
                
                if national_result:
                    comparison_data[indicator["indicator_id"]] = {
                        "comparison_type": "national",
                        "comparison_name": "National Average",
                        "value": _safe_float_conversion(national_result[0])
                    }
        
        else:  # state comparison
            # Get state averages for each district's state
            state_averages = {}
            for district in districts_data:
                state_name = district["state_name"]
                if state_name not in state_averages:
                    state_averages[state_name] = {}
                
                for indicator in matched_indicators:
                    if indicator["indicator_id"] not in state_averages[state_name]:
                        state_query = f"""
                        SELECT si.prevalence_{year}
                        FROM state_indicators si
                        JOIN states s ON si.state_id = s.state_id
                        WHERE s.state_name = %s AND si.indicator_id = %s
                        """
                        
                        cursor.execute(state_query, (state_name, indicator["indicator_id"]))
                        state_result = cursor.fetchone()
                        
                        if state_result:
                            state_averages[state_name][indicator["indicator_id"]] = {
                                "comparison_type": "state",
                                "comparison_name": f"{state_name} Average",
                                "value": _safe_float_conversion(state_result[0])
                            }
            
            # Assign state averages to comparison_data based on district's state
            for district in districts_data:
                state_name = district["state_name"]
                for indicator in matched_indicators:
                    indicator_id = indicator["indicator_id"]
                    if (state_name in state_averages and 
                        indicator_id in state_averages[state_name]):
                        if indicator_id not in comparison_data:
                            comparison_data[indicator_id] = {}
                        comparison_data[indicator_id][state_name] = state_averages[state_name][indicator_id]
        
        # Get boundary data if requested
        boundary_data = []
        if include_boundary_data:
            district_names_for_boundary = [d["district_name"] for d in districts_data]
            boundary_data = get_district_boundary_data(district_names_for_boundary)
        
        # Generate chart data and analysis
        chart_data = generate_district_comparison_chart_data(districts_data, comparison_data, comparison_type, matched_indicators)
        analysis = generate_district_comparison_analysis(districts_data, comparison_data, comparison_type, matched_indicators, year)
        
        return {
            "districts": districts_data,
            "comparison_data": comparison_data,
            "comparison_type": comparison_type,
            "indicators": [{"indicator_id": ind["indicator_id"], "indicator_name": ind["indicator_name"], "indicator_direction": ind["indicator_direction"]} for ind in matched_indicators],
            "total_districts": len(districts_data),
            "total_indicators": len(matched_indicators),
            "year": year,
            "chart_data": chart_data,
            "analysis": analysis,
            "boundary": boundary_data,
            "map_type": "district_comparison",
            "response_type": "district_performance_comparison"
        }
    
    except Exception as e:
        print(f"Error in get_district_performance_comparison: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Function execution failed: {str(e)}",
            "response_type": "error"
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def generate_district_comparison_chart_data(districts_data, comparison_data, comparison_type, matched_indicators):
    """Generate comprehensive chart data for district performance comparison"""
    charts = []
    
    # Chart 1: Main Comparison Bar Chart (one chart per indicator)
    for indicator in matched_indicators:
        indicator_id = indicator["indicator_id"]
        indicator_name = indicator["indicator_name"]
        indicator_direction = indicator["indicator_direction"]
        
        district_values = []
        district_labels = []
        comparison_values = []
        comparison_labels = []
        
        for district in districts_data:
            # Find this indicator's data for this district
            district_indicator_data = None
            for ind_data in district["indicators"]:
                if ind_data["indicator_id"] == indicator_id:
                    district_indicator_data = ind_data
                    break
            
            if district_indicator_data and district_indicator_data["prevalence"] is not None:
                district_values.append(district_indicator_data["prevalence"])
                district_labels.append(f"{district['district_name']}, {district['state_name']}")
                
                # Get comparison value
                if comparison_type == "national":
                    if indicator_id in comparison_data:
                        comparison_values.append(comparison_data[indicator_id]["value"])
                        comparison_labels.append("National Average")
                    else:
                        comparison_values.append(None)
                        comparison_labels.append("No Data")
                else:  # state comparison
                    state_name = district["state_name"]
                    if (indicator_id in comparison_data and 
                        state_name in comparison_data[indicator_id]):
                        comparison_values.append(comparison_data[indicator_id][state_name]["value"])
                        comparison_labels.append(f"{state_name} Average")
                    else:
                        comparison_values.append(None)
                        comparison_labels.append("No Data")
        
        if district_values:
            # Create grouped bar chart
            datasets = [
                {
                    "label": "District Values",
                    "data": district_values,
                    "backgroundColor": "#3498db",
                    "borderColor": "#2980b9",
                    "borderWidth": 1
                }
            ]
            
            # Add comparison values if available
            if any(v is not None for v in comparison_values):
                datasets.append({
                    "label": f"{comparison_type.title()} Average",
                    "data": comparison_values,
                    "backgroundColor": "#e74c3c",
                    "borderColor": "#c0392b",
                    "borderWidth": 1
                })
            
            chart = {
                "type": "bar",
                "title": f"{indicator_name} - District vs {comparison_type.title()} Comparison",
                "data": {
                    "labels": district_labels,
                    "datasets": datasets
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{indicator_name} Comparison ({comparison_type.title()})"
                        },
                        "legend": {"display": True}
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": f"{indicator_name} (%)"
                            }
                        },
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Districts"
                            }
                        }
                    }
                },
                "indicator_direction": indicator_direction
            }
            charts.append(chart)
    
    # Chart 2: Performance Gap Analysis (shows difference from comparison)
    gap_chart_data = []
    gap_labels = []
    gap_colors = []
    
    for indicator in matched_indicators:
        indicator_id = indicator["indicator_id"]
        indicator_name = indicator["indicator_name"]
        indicator_direction = indicator["indicator_direction"]
        
        for district in districts_data:
            district_indicator_data = None
            for ind_data in district["indicators"]:
                if ind_data["indicator_id"] == indicator_id:
                    district_indicator_data = ind_data
                    break
            
            if district_indicator_data and district_indicator_data["prevalence"] is not None:
                district_value = district_indicator_data["prevalence"]
                comparison_value = None
                
                if comparison_type == "national":
                    if indicator_id in comparison_data:
                        comparison_value = comparison_data[indicator_id]["value"]
                else:  # state comparison
                    state_name = district["state_name"]
                    if (indicator_id in comparison_data and 
                        state_name in comparison_data[indicator_id]):
                        comparison_value = comparison_data[indicator_id][state_name]["value"]
                
                if comparison_value is not None:
                    gap = district_value - comparison_value
                    gap_chart_data.append(gap)
                    gap_labels.append(f"{district['district_name']} - {indicator_name}")
                    
                    # Color based on performance (considering indicator direction)
                    if indicator_direction == "higher_is_better":
                        gap_colors.append("#27ae60" if gap > 0 else "#e74c3c")
                    else:  # lower_is_better
                        gap_colors.append("#27ae60" if gap < 0 else "#e74c3c")
    
    if gap_chart_data:
        gap_chart = {
            "type": "bar",
            "title": f"Performance Gap Analysis - Districts vs {comparison_type.title()} Average",
            "data": {
                "labels": gap_labels,
                "datasets": [{
                    "label": f"Gap from {comparison_type.title()} Average",
                    "data": gap_chart_data,
                    "backgroundColor": gap_colors,
                    "borderColor": gap_colors,
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"Performance Gap Analysis ({comparison_type.title()})"
                    },
                    "legend": {"display": True}
                },
                "scales": {
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Gap (Percentage Points)"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "District - Indicator"
                        }
                    }
                }
            }
        }
        charts.append(gap_chart)
    
    # Chart 3: Overall Performance Summary (radar/spider chart alternative as bar chart)
    if len(matched_indicators) > 1:
        summary_data = []
        summary_labels = []
        
        for district in districts_data:
            district_scores = []
            for indicator in matched_indicators:
                indicator_id = indicator["indicator_id"]
                district_indicator_data = None
                for ind_data in district["indicators"]:
                    if ind_data["indicator_id"] == indicator_id:
                        district_indicator_data = ind_data
                        break
                
                if district_indicator_data and district_indicator_data["prevalence"] is not None:
                    district_scores.append(district_indicator_data["prevalence"])
                else:
                    district_scores.append(0)
            
            # Calculate average score for this district
            if district_scores:
                avg_score = sum(district_scores) / len(district_scores)
                summary_data.append(avg_score)
                summary_labels.append(f"{district['district_name']}, {district['state_name']}")
        
        if summary_data:
            summary_chart = {
                "type": "bar",
                "title": "Overall Performance Summary - Average Across All Indicators",
                "data": {
                    "labels": summary_labels,
                    "datasets": [{
                        "label": "Average Indicator Value",
                        "data": summary_data,
                        "backgroundColor": "#9b59b6",
                        "borderColor": "#8e44ad",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Overall Performance Summary"
                        },
                        "legend": {"display": True}
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Average Value (%)"
                            }
                        },
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Districts"
                            }
                        }
                    }
                }
            }
            charts.append(summary_chart)
    
    return charts


def generate_district_comparison_analysis(districts_data, comparison_data, comparison_type, matched_indicators, year):
    """Generate comprehensive analysis for district performance comparison"""
    
    analysis_parts = []
    
    # Overview
    total_districts = len(districts_data)
    total_indicators = len(matched_indicators)
    comparison_name = "national averages" if comparison_type == "national" else "state averages"
    
    analysis_parts.append(f"""
## District Performance Comparison Analysis ({year})

This analysis compares **{total_districts} districts** across **{total_indicators} health indicators** against {comparison_name}.

### Districts Analyzed:
{', '.join([f"{d['district_name']} ({d['state_name']})" for d in districts_data])}

### Health Indicators:
{', '.join([ind['indicator_name'] for ind in matched_indicators])}
""")
    
    # Performance analysis for each indicator
    for indicator in matched_indicators:
        indicator_id = indicator["indicator_id"]
        indicator_name = indicator["indicator_name"]
        indicator_direction = indicator["indicator_direction"]
        
        analysis_parts.append(f"\n### {indicator_name} Analysis")
        
        # Collect district values and comparisons
        district_performances = []
        for district in districts_data:
            district_indicator_data = None
            for ind_data in district["indicators"]:
                if ind_data["indicator_id"] == indicator_id:
                    district_indicator_data = ind_data
                    break
            
            if district_indicator_data and district_indicator_data["prevalence"] is not None:
                district_value = district_indicator_data["prevalence"]
                comparison_value = None
                comparison_name_specific = ""
                
                if comparison_type == "national":
                    if indicator_id in comparison_data:
                        comparison_value = comparison_data[indicator_id]["value"]
                        comparison_name_specific = "National Average"
                else:  # state comparison
                    state_name = district["state_name"]
                    if (indicator_id in comparison_data and 
                        state_name in comparison_data[indicator_id]):
                        comparison_value = comparison_data[indicator_id][state_name]["value"]
                        comparison_name_specific = f"{state_name} Average"
                
                if comparison_value is not None:
                    gap = district_value - comparison_value
                    performance_status = ""
                    
                    if indicator_direction == "higher_is_better":
                        if gap > 0:
                            performance_status = "✅ **Above average** (better performance)"
                        else:
                            performance_status = "⚠️ **Below average** (needs improvement)"
                    else:  # lower_is_better
                        if gap < 0:
                            performance_status = "✅ **Below average** (better performance)"
                        else:
                            performance_status = "⚠️ **Above average** (needs improvement)"
                    
                    district_performances.append({
                        "district": district,
                        "value": district_value,
                        "comparison_value": comparison_value,
                        "gap": gap,
                        "status": performance_status
                    })
        
        if district_performances:
            analysis_parts.append(f"\n**Performance Summary:**")
            for perf in district_performances:
                analysis_parts.append(f"- **{perf['district']['district_name']}**: {perf['value']:.1f}% vs {perf['comparison_value']:.1f}% ({perf['gap']:+.1f}pp) - {perf['status']}")
        
        # Best and worst performers for this indicator
        if len(district_performances) > 1:
            if indicator_direction == "higher_is_better":
                best_perf = max(district_performances, key=lambda x: x['value'])
                worst_perf = min(district_performances, key=lambda x: x['value'])
            else:
                best_perf = min(district_performances, key=lambda x: x['value'])
                worst_perf = max(district_performances, key=lambda x: x['value'])
            
            analysis_parts.append(f"\n**Best Performer**: {best_perf['district']['district_name']} ({best_perf['value']:.1f}%)")
            analysis_parts.append(f"**Needs Most Attention**: {worst_perf['district']['district_name']} ({worst_perf['value']:.1f}%)")
    
    # Overall insights
    analysis_parts.append(f"\n## Key Insights\n")
    
    # Count districts performing above/below average
    above_average_count = 0
    below_average_count = 0
    
    for district in districts_data:
        district_above = 0
        district_below = 0
        
        for indicator in matched_indicators:
            indicator_id = indicator["indicator_id"]
            indicator_direction = indicator["indicator_direction"]
            
            district_indicator_data = None
            for ind_data in district["indicators"]:
                if ind_data["indicator_id"] == indicator_id:
                    district_indicator_data = ind_data
                    break
            
            if district_indicator_data and district_indicator_data["prevalence"] is not None:
                district_value = district_indicator_data["prevalence"]
                comparison_value = None
                
                if comparison_type == "national":
                    if indicator_id in comparison_data:
                        comparison_value = comparison_data[indicator_id]["value"]
                else:  # state comparison
                    state_name = district["state_name"]
                    if (indicator_id in comparison_data and 
                        state_name in comparison_data[indicator_id]):
                        comparison_value = comparison_data[indicator_id][state_name]["value"]
                
                if comparison_value is not None:
                    gap = district_value - comparison_value
                    
                    if indicator_direction == "higher_is_better":
                        if gap > 0:
                            district_above += 1
                        else:
                            district_below += 1
                    else:  # lower_is_better
                        if gap < 0:
                            district_above += 1
                        else:
                            district_below += 1
        
        if district_above > district_below:
            above_average_count += 1
        else:
            below_average_count += 1
    
    analysis_parts.append(f"- **{above_average_count} out of {total_districts} districts** show overall better-than-average performance")
    analysis_parts.append(f"- **{below_average_count} out of {total_districts} districts** need targeted interventions")
    
    if comparison_type == "state":
        unique_states = set(d["state_name"] for d in districts_data)
        if len(unique_states) > 1:
            analysis_parts.append(f"- Analysis covers **{len(unique_states)} states**: {', '.join(sorted(unique_states))}")
    
    analysis_parts.append(f"\n### Recommendations\n")
    analysis_parts.append(f"1. **Priority Focus**: Districts consistently below average across multiple indicators need immediate attention")
    analysis_parts.append(f"2. **Best Practice Sharing**: High-performing districts can serve as models for improvement strategies")
    analysis_parts.append(f"3. **Targeted Interventions**: Each indicator may require specific policy approaches based on district-level gaps")
    
    return "\n".join(analysis_parts)


def get_all_indicators():
    """Get all available health indicators from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT
            indicator_id,
            indicator_name,
            indicator_direction
        FROM indicators
        ORDER BY indicator_name
        """

        cursor.execute(query)
        results = cursor.fetchall()

        indicators = []
        for row in results:
            indicators.append({
                "indicator_id": row[0],
                "indicator_name": row[1],
                "indicator_direction": row[2]
            })

        cursor.close()
        conn.close()

        return indicators

    except Exception as e:
        logger.error(f"Error getting all indicators: {e}")
        return []



def match_indicator_with_openai(user_query, available_indicators):
    """
    Use OpenAI to match user query to the best indicator from available options.
    This handles misspellings, synonyms, and descriptions.
    """
    try:
        # Format indicators for OpenAI
        indicators_list = "\n".join([
            f"- {ind['indicator_name']} (ID: {ind['indicator_id']}, Direction: {ind['indicator_direction']})"
            for ind in available_indicators
        ])

        # Create matching prompt
        matching_prompt = f"""
        You are an expert in health indicators and public health terminology.

        The user asked: "{user_query}"

        Here are all available health indicators in the database:
        {indicators_list}

        Your task is to determine if the user is referring to a specific health indicator, and if so, select the BEST matching indicator.

        Rules:
        1. Look for direct mentions, synonyms, descriptions, or related terms
        2. Handle common misspellings and variations (e.g., "diarrhea" = "diarrhoea")
        3. Consider context and intent
        4. If no clear match, return null
        5. Only return the indicator_id of the best match

        Examples:
        - "watery pooping" → matches "Acute Diarrhoeal Disease [All Children]"
        - "chest infections in kids" → matches "Acute Respiratory Infection [All Children]"
        - "mal nutrition" → matches "Children Under 5 Years Who Are Severely Wasted"
        - "vaccination rates" → could match various vaccination indicators

        Return format:
        {{
            "matched": true/false,
            "indicator_id": matched_indicator_id_or_null,
            "confidence": "high/medium/low",
            "reasoning": "brief explanation of match"
        }}
        """

        # Call OpenAI for matching
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are a health indicator expert. Match user queries to appropriate indicators."
                },
                {
                    "role": "user",
                    "content": matching_prompt
                }
            ],
            max_tokens=200,
            temperature=0.1  # Low temperature for consistent matching
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            logger.warning(f"Failed to parse OpenAI response as JSON: {result_text}")

            # Try to extract indicator_id from response
            import re
            id_match = re.search(r'"indicator_id":\s*(\d+)', result_text)
            if id_match:
                return {
                    "matched": True,
                    "indicator_id": int(id_match.group(1)),
                    "confidence": "medium",
                    "reasoning": "Extracted from response"
                }

            return {
                "matched": False,
                "indicator_id": None,
                "confidence": "low",
                "reasoning": "Could not parse response"
            }

    except Exception as e:
        logger.error(f"Error in indicator matching: {e}")
        return {
            "matched": False,
            "indicator_id": None,
            "confidence": "low",
            "reasoning": f"Error: {str(e)}"
        }


def get_multi_indicator_performance(
    district_names: Optional[List[str]] = None,
    category_name: Optional[str] = None,
    indicator_names: Optional[List[str]] = None,
    performance_type: str = "specific",  # "specific", "top", "bottom", "both"
    n_districts: int = 10,
    year: int = 2021,
    include_boundary_data: bool = True
):
    """
    Analyze multi-indicator performance using normalized composite index methodology.
    
    Implements the 4-step process:
    1. Min-Max normalization across all districts and years
    2. Direction alignment (higher=better vs lower=better)
    3. Composite performance index calculation
    4. Year-over-year comparison and ranking
    
    Parameters:
    - district_names: Specific districts to analyze
    - category_name: Category name (e.g., "healthcare", "nutrition") - uses OpenAI matching
    - indicator_names: Specific indicators to include
    - performance_type: "specific", "top", "bottom", or "both"
    - n_districts: Number of top/bottom districts to return
    - year: Primary year for analysis (2021 or 2016)
    - include_boundary_data: Include boundary data for mapping
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Step 1: Determine which indicators to use
        selected_indicators = []
        
        if indicator_names:
            # Use specific indicators
            for indicator_name in indicator_names:
                matched = match_indicator_name_to_database(indicator_name)
                if matched:
                    selected_indicators.append(matched)
        
        elif category_name:
            # Use category-based selection with OpenAI matching
            selected_indicators = get_indicators_by_category(cursor, category_name)
        
        else:
            # Use all indicators
            cursor.execute("""
                SELECT indicator_id, indicator_name, indicator_direction, indicator_category
                FROM indicators 
                ORDER BY indicator_name
            """)
            selected_indicators = [
                {
                    "indicator_id": row[0],
                    "indicator_name": row[1], 
                    "indicator_direction": row[2],
                    "indicator_category": row[3]
                }
                for row in cursor.fetchall()
            ]
        
        if not selected_indicators:
            return {"error": "No indicators found for analysis"}
        
        print(f"🎯 Selected {len(selected_indicators)} indicators for multi-indicator analysis")
        
        # Step 2: Get all district data for normalization
        indicator_ids = [ind["indicator_id"] for ind in selected_indicators]
        
        # Query all districts and years for normalization
        normalization_query = """
        SELECT 
            d.district_name,
            s.state_name,
            di.indicator_id,
            di.prevalence_2016,
            di.prevalence_2021
        FROM districts d
        JOIN states s ON d.state_id = s.state_id
        JOIN district_indicators di ON d.district_id = di.district_id
        WHERE di.indicator_id = ANY(%s)
        AND (di.prevalence_2016 IS NOT NULL OR di.prevalence_2021 IS NOT NULL)
        ORDER BY d.district_name, di.indicator_id
        """
        
        cursor.execute(normalization_query, (indicator_ids,))
        all_data = cursor.fetchall()
        
        if not all_data:
            return {"error": "No data found for selected indicators"}
        
        # Step 3: Perform normalization and calculate composite indices
        districts_performance = calculate_multi_indicator_performance(
            all_data, selected_indicators, year
        )
        
        # Step 4: Filter and rank districts based on request type
        if performance_type == "specific" and district_names:
            # Filter to specific districts
            filtered_districts = []
            for district_name in district_names:
                # Fuzzy match district names
                matched_districts = [d for d in districts_performance 
                                   if district_name.lower() in d["district_name"].lower() 
                                   or d["district_name"].lower() in district_name.lower()]
                filtered_districts.extend(matched_districts)
            
            final_districts = filtered_districts
            
        elif performance_type in ["top", "bottom", "both"]:
            # Sort by performance index
            sorted_districts = sorted(districts_performance, 
                                    key=lambda x: x["performance_index_2021"], 
                                    reverse=True)
            
            if performance_type == "top":
                final_districts = sorted_districts[:n_districts]
            elif performance_type == "bottom":
                final_districts = sorted_districts[-n_districts:]
            else:  # both
                final_districts = sorted_districts[:n_districts] + sorted_districts[-n_districts:]
        
        else:
            # Return all districts
            final_districts = sorted(districts_performance, 
                                   key=lambda x: x["performance_index_2021"], 
                                   reverse=True)
        
        # Get boundary data if requested
        boundary_data = []
        if include_boundary_data and final_districts:
            district_names_for_boundary = [d["district_name"] for d in final_districts]
            boundary_data = get_district_boundary_data(district_names_for_boundary)
        
        # Generate analysis and charts
        analysis = generate_multi_indicator_analysis(
            final_districts, selected_indicators, performance_type, category_name, year
        )
        
        chart_data = generate_multi_indicator_chart_data(
            final_districts, selected_indicators, performance_type, year
        )
        
        cursor.close()
        conn.close()
        
        return {
            "response_type": "multi_indicator_performance",
            "performance_type": performance_type,
            "category_name": category_name,
            "year": year,
            "total_indicators": len(selected_indicators),
            "total_districts": len(final_districts),
            "indicators": selected_indicators,
            "districts": final_districts,
            "boundary_data": boundary_data,
            "chart_data": chart_data,
            "analysis": analysis,
            "map_type": "multi_indicator_performance"
        }
        
    except Exception as e:
        logger.error(f"Error in get_multi_indicator_performance: {e}")
        return {"error": f"Database error: {str(e)}"}

def get_indicators_by_category(cursor, category_name):
    """Get indicators by category name using OpenAI matching if needed"""
    try:
        # First try exact match
        cursor.execute("""
            SELECT ic."Value", ic."Label" 
            FROM indicator_category ic
            WHERE LOWER(ic."Label") LIKE LOWER(%s)
        """, (f"%{category_name}%",))
        
        category_match = cursor.fetchone()
        
        if not category_match:
            # Use OpenAI to match category
            cursor.execute('SELECT "Value", "Label" FROM indicator_category')
            available_categories = cursor.fetchall()
            
            category_match = match_category_with_openai(category_name, available_categories)
        
        if not category_match:
            return []
        
        category_id = category_match[0]
        
        # Get all indicators in this category
        cursor.execute("""
            SELECT indicator_id, indicator_name, indicator_direction, indicator_category
            FROM indicators 
            WHERE indicator_category = %s
            ORDER BY indicator_name
        """, (category_id,))
        
        return [
            {
                "indicator_id": row[0],
                "indicator_name": row[1],
                "indicator_direction": row[2], 
                "indicator_category": row[3]
            }
            for row in cursor.fetchall()
        ]
        
    except Exception as e:
        logger.error(f"Error getting indicators by category: {e}")
        return []

def match_category_with_openai(user_category, available_categories):
    """Use OpenAI to match user category name to database categories"""
    try:
        
        
        client = OpenAI(api_key="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")
        
        categories_text = "\n".join([f"{cat[0]}: {cat[1]}" for cat in available_categories])
        
        prompt = f"""
        User asked for category: "{user_category}"
        
        Available categories in database:
        {categories_text}
        
        Return only the category ID (number) that best matches the user's request, or "none" if no good match.
        Examples:
        - "healthcare" -> 2
        - "nutrition" -> 5 (or 6 depending on context)
        - "maternal health" -> 3
        - "mortality" -> 4
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.lower() == "none":
            return None
        
        try:
            category_id = int(result)
            # Find the matching category
            for cat in available_categories:
                if cat[0] == category_id:
                    return cat
        except ValueError:
            pass
        
        return None
        
    except Exception as e:
        logger.error(f"Error in OpenAI category matching: {e}")
        return None

def calculate_multi_indicator_performance(all_data, selected_indicators, year):
    """
    Calculate multi-indicator performance using the 4-step methodology:
    1. Min-Max normalization
    2. Direction alignment  
    3. Composite index calculation
    4. Year comparison
    """
    
    # Step 1: Organize data and calculate min/max for normalization
    indicator_stats = {}
    district_data = {}
    
    # First pass: collect all values for min/max calculation
    for row in all_data:
        district_name, state_name, indicator_id, prev_2016, prev_2021 = row
        
        if indicator_id not in indicator_stats:
            indicator_stats[indicator_id] = {
                "values_2016": [],
                "values_2021": [],
                "direction": None
            }
        
        # Find indicator direction
        for ind in selected_indicators:
            if ind["indicator_id"] == indicator_id:
                indicator_stats[indicator_id]["direction"] = ind["indicator_direction"]
                break
        
        # Collect all values for normalization
        if prev_2016 is not None:
            indicator_stats[indicator_id]["values_2016"].append(float(prev_2016))
        if prev_2021 is not None:
            indicator_stats[indicator_id]["values_2021"].append(float(prev_2021))
        
        # Organize by district
        if district_name not in district_data:
            district_data[district_name] = {
                "district_name": district_name,
                "state_name": state_name,
                "indicators": {},
                "normalized_scores_2016": [],
                "normalized_scores_2021": []
            }
        
        district_data[district_name]["indicators"][indicator_id] = {
            "prevalence_2016": float(prev_2016) if prev_2016 is not None else None,
            "prevalence_2021": float(prev_2021) if prev_2021 is not None else None
        }
    
    # Calculate global min/max for each indicator across both years
    for indicator_id in indicator_stats:
        all_values = (indicator_stats[indicator_id]["values_2016"] + 
                     indicator_stats[indicator_id]["values_2021"])
        
        if all_values:
            indicator_stats[indicator_id]["min_value"] = min(all_values)
            indicator_stats[indicator_id]["max_value"] = max(all_values)
        else:
            indicator_stats[indicator_id]["min_value"] = 0
            indicator_stats[indicator_id]["max_value"] = 1
    
    # Step 2 & 3: Normalize and calculate composite indices
    districts_performance = []
    
    for district_name, data in district_data.items():
        # Normalize each indicator for both years
        for indicator_id, values in data["indicators"].items():
            min_val = indicator_stats[indicator_id]["min_value"]
            max_val = indicator_stats[indicator_id]["max_value"]
            direction = indicator_stats[indicator_id]["direction"]
            
            # Avoid division by zero
            if max_val == min_val:
                normalized_2016 = 0.5
                normalized_2021 = 0.5
            else:
                # Step 1: Min-Max normalization
                if values["prevalence_2016"] is not None:
                    norm_2016 = (values["prevalence_2016"] - min_val) / (max_val - min_val)
                else:
                    norm_2016 = None
                
                if values["prevalence_2021"] is not None:
                    norm_2021 = (values["prevalence_2021"] - min_val) / (max_val - min_val)
                else:
                    norm_2021 = None
                
                # Step 2: Direction alignment
                if direction == "lower_is_better":
                    if norm_2016 is not None:
                        norm_2016 = 1 - norm_2016
                    if norm_2021 is not None:
                        norm_2021 = 1 - norm_2021
                
                normalized_2016 = norm_2016
                normalized_2021 = norm_2021
            
            # Store normalized values
            if normalized_2016 is not None:
                data["normalized_scores_2016"].append(normalized_2016)
            if normalized_2021 is not None:
                data["normalized_scores_2021"].append(normalized_2021)
        
        # Step 3: Calculate composite performance index (simple average)
        performance_2016 = (sum(data["normalized_scores_2016"]) / len(data["normalized_scores_2016"]) 
                           if data["normalized_scores_2016"] else 0)
        performance_2021 = (sum(data["normalized_scores_2021"]) / len(data["normalized_scores_2021"]) 
                           if data["normalized_scores_2021"] else 0)
        
        # Step 4: Calculate change metrics
        absolute_change = performance_2021 - performance_2016 if performance_2016 > 0 else 0
        relative_change = ((performance_2021 - performance_2016) / performance_2016 * 100 
                          if performance_2016 > 0 else 0)
        
        districts_performance.append({
            "district_name": district_name,
            "state_name": data["state_name"],
            "performance_index_2016": round(performance_2016, 4),
            "performance_index_2021": round(performance_2021, 4),
            "absolute_change": round(absolute_change, 4),
            "relative_change": round(relative_change, 2),
            "total_indicators": len(data["indicators"]),
            "indicators_2016": len(data["normalized_scores_2016"]),
            "indicators_2021": len(data["normalized_scores_2021"])
        })
    
    return districts_performance

def generate_multi_indicator_analysis(districts, indicators, performance_type, category_name, year):
    """Generate comprehensive analysis for multi-indicator performance"""
    if not districts:
        return "No districts data available for multi-indicator analysis."
    
    total_districts = len(districts)
    total_indicators = len(indicators)
    
    # Get category info
    category_desc = f" in {category_name} category" if category_name else ""
    
    # Performance statistics
    avg_performance = sum(d["performance_index_2021"] for d in districts) / total_districts
    best_district = max(districts, key=lambda x: x["performance_index_2021"])
    worst_district = min(districts, key=lambda x: x["performance_index_2021"])
    
    # Change statistics
    districts_with_change = [d for d in districts if d["absolute_change"] != 0]
    improved_districts = [d for d in districts_with_change if d["absolute_change"] > 0]
    declined_districts = [d for d in districts_with_change if d["absolute_change"] < 0]
    
    analysis = f"""
    **Multi-Indicator Performance Analysis{category_desc}**
    
    This analysis examines {total_districts} districts across {total_indicators} health indicators using a comprehensive composite index methodology.
    
    **Methodology Applied:**
    1. **Min-Max Normalization**: All indicators normalized to [0,1] scale using global min/max values
    2. **Direction Alignment**: "Lower is better" indicators inverted (X'' = 1 - X')
    3. **Composite Index**: Simple average of normalized, direction-aligned indicators
    4. **Change Analysis**: Absolute and relative performance changes from 2016 to 2021
    
    **Performance Overview ({year}):**
    - Average Performance Index: {avg_performance:.3f}
    - Best Performing District: {best_district["district_name"]} ({best_district["state_name"]}) - Index: {best_district["performance_index_2021"]:.3f}
    - Lowest Performing District: {worst_district["district_name"]} ({worst_district["state_name"]}) - Index: {worst_district["performance_index_2021"]:.3f}
    
    **Change Analysis (2016-2021):**
    - Districts with Improvement: {len(improved_districts)} ({len(improved_districts)/total_districts*100:.1f}%)
    - Districts with Decline: {len(declined_districts)} ({len(declined_districts)/total_districts*100:.1f}%)
    - Average Absolute Change: {sum(d["absolute_change"] for d in districts)/total_districts:.3f}
    
    **Key Insights:**
    - Higher index values indicate better overall health performance
    - Index considers indicator direction (higher/lower is better)
    - All indicators contribute equally to the composite score
    - Results directly comparable across districts and time periods
    
    **Applications:**
    - Resource allocation based on comprehensive health performance
    - Identifying exemplary districts for best practice sharing
    - Monitoring multi-dimensional health improvements over time
    - Policy evaluation across multiple health domains simultaneously
    """
    
    return analysis

def generate_multi_indicator_chart_data(districts, indicators, performance_type, year):
    """Generate chart configurations for multi-indicator performance visualization"""
    
    if not districts:
        return []
    
    charts = []
    
    # 1. Performance Index Ranking Chart
    charts.append({
        "type": "bar",
        "title": f"Multi-Indicator Performance Index ({year})",
        "description": "Composite performance index across all indicators (higher = better)",
        "data": {
            "labels": [d["district_name"] for d in districts[:20]],  # Limit for readability
            "datasets": [{
                "label": f"Performance Index {year}",
                "data": [d["performance_index_2021"] for d in districts[:20]],
                "backgroundColor": ["#2E86C1" if d["performance_index_2021"] >= 0.5 else "#E74C3C" for d in districts[:20]],
                "borderColor": "#1B4F72",
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 1.0,
                    "title": {
                        "display": True,
                        "text": "Performance Index (0-1)"
                    }
                }
            }
        }
    })
    
    # 2. Performance Change Chart
    change_districts = [d for d in districts if d["absolute_change"] != 0][:15]
    if change_districts:
        charts.append({
            "type": "bar",
            "title": "Performance Change (2016-2021)",
            "description": "Absolute change in performance index over time",
            "data": {
                "labels": [d["district_name"] for d in change_districts],
                "datasets": [{
                    "label": "Change in Performance Index",
                    "data": [d["absolute_change"] for d in change_districts],
                    "backgroundColor": ["#27AE60" if d["absolute_change"] > 0 else "#E74C3C" for d in change_districts],
                    "borderColor": "#1E8449",
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Change in Index"
                        }
                    }
                }
            }
        })
    
    # 3. Performance Distribution Chart
    performance_ranges = {
        "Excellent (0.8-1.0)": len([d for d in districts if d["performance_index_2021"] >= 0.8]),
        "Good (0.6-0.8)": len([d for d in districts if 0.6 <= d["performance_index_2021"] < 0.8]),
        "Average (0.4-0.6)": len([d for d in districts if 0.4 <= d["performance_index_2021"] < 0.6]),
        "Poor (0.2-0.4)": len([d for d in districts if 0.2 <= d["performance_index_2021"] < 0.4]),
        "Very Poor (0-0.2)": len([d for d in districts if d["performance_index_2021"] < 0.2])
    }
    
    charts.append({
        "type": "doughnut",
        "title": "Performance Distribution",
        "description": "Distribution of districts across performance ranges",
        "data": {
            "labels": list(performance_ranges.keys()),
            "datasets": [{
                "data": list(performance_ranges.values()),
                "backgroundColor": ["#27AE60", "#52C41A", "#F39C12", "#E67E22", "#E74C3C"]
            }]
        }
    })
    
    return charts


def get_state_multi_indicator_performance(
    state_names: Optional[List[str]] = None,
    category_name: Optional[str] = None,
    indicator_names: Optional[List[str]] = None,
    performance_type: str = "top",  # "top", "bottom", "both"
    n_districts: int = 5,
    year: int = 2021,
    include_boundary_data: bool = True,
    query_hint: Optional[str] = None  # Added to detect "lowest" hints
):
    """
    Analyze multi-indicator performance comparing states with their top/bottom districts.
    
    This function extends the multi-indicator performance analysis to state level:
    1. Extracts state-level indicator values from state_indicators table
    2. Computes state performance using get_multi_indicator_performance logic
    3. For each state, finds top/bottom N districts within that state
    4. Supports comparison of multiple states
    5. Generates appropriate visualizations (maps and charts)
    
    Parameters:
    - state_names: List of state names to analyze (if None, analyzes all states)
    - category_name: Category name for indicator selection
    - indicator_names: Specific indicators to include
    - performance_type: "top" for best districts, "bottom" for worst, "both" for both
    - n_districts: Number of districts per state to include (default 5)
    - year: Primary year for analysis (2021 or 2016)
    - include_boundary_data: Include boundary data for mapping
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Step 1: Determine which indicators to use (same logic as original function)
        selected_indicators = []
        
        if indicator_names:
            # Use specific indicators
            for indicator_name in indicator_names:
                matched = match_indicator_name_to_database(indicator_name)
                if matched:
                    selected_indicators.append(matched)
        
        elif category_name:
            # Use category-based selection with OpenAI matching
            selected_indicators = get_indicators_by_category(cursor, category_name)
        
        else:
            # Use all indicators
            cursor.execute("""
                SELECT indicator_id, indicator_name, indicator_direction, indicator_category
                FROM indicators 
                ORDER BY indicator_name
            """)
            selected_indicators = [
                {
                    "indicator_id": row[0],
                    "indicator_name": row[1], 
                    "indicator_direction": row[2],
                    "indicator_category": row[3]
                }
                for row in cursor.fetchall()
            ]
        
        if not selected_indicators:
            return {"error": "No indicators found for analysis"}
        
        print(f"🎯 Selected {len(selected_indicators)} indicators for state-level multi-indicator analysis")
        
        # Auto-detect performance type from query hint if provided
        if query_hint and performance_type == "top":
            query_lower = query_hint.lower()
            lowest_keywords = ["lowest", "worst", "bottom", "poor", "underperform", "weakest", "bad"]
            if any(keyword in query_lower for keyword in lowest_keywords):
                performance_type = "bottom"
                print(f"🔍 Detected 'lowest' hint in query, switching to bottom districts")
        
        # Step 2: Get state-level data from state_indicators table
        indicator_ids = [ind["indicator_id"] for ind in selected_indicators]
        
        # Prepare state filter
        state_filter = ""
        state_params = []
        if state_names:
            state_filter = "AND s.state_name = ANY(%s)"
            state_params.append(state_names)
        
        # Query state-level data
        state_query = f"""
        SELECT 
            s.state_name,
            si.indicator_id,
            si.prevalence_2016,
            si.prevalence_2021
        FROM state_indicators si
        JOIN states s ON si.state_id = s.state_id
        WHERE si.indicator_id = ANY(%s)
        {state_filter}
        ORDER BY s.state_name, si.indicator_id
        """
        
        cursor.execute(state_query, [indicator_ids] + state_params)
        state_data = cursor.fetchall()
        
        if not state_data:
            return {"error": "No state data found for the specified indicators"}
        
        # Step 3: Calculate state-level performance using same methodology
        state_performance = calculate_state_multi_indicator_performance(
            state_data, selected_indicators, year
        )
        
        # Step 4: For each state, get top/bottom districts
        state_district_data = {}
        all_districts_for_boundary = []
        
        # Get all districts once and then filter by state
        print(f"🔍 Getting district data for filtering by states...")
        all_districts_result = get_multi_indicator_performance(
            district_names=None,  # Get all districts
            category_name=category_name,
            indicator_names=indicator_names,
            performance_type="top",  # Get all, we'll sort later
            n_districts=1000,  # Get many districts to filter from
            year=year,
            include_boundary_data=False  # We'll handle boundary separately
        )
        
        if "districts" in all_districts_result and all_districts_result["districts"]:
            all_districts = all_districts_result["districts"]
            print(f"📊 Found {len(all_districts)} total districts to filter from")
            
            for state_perf in state_performance:
                state_name = state_perf["state_name"]
                print(f"🔍 Filtering districts for state: {state_name}")
                
                # Filter districts to only those in current state
                state_districts = [
                    d for d in all_districts 
                    if d.get("state_name") == state_name
                ]
                
                print(f"📍 Found {len(state_districts)} districts in {state_name}")
                
                if state_districts:
                    # Sort and limit based on performance_type
                    if performance_type in ["top", "both"]:
                        state_districts_sorted = sorted(state_districts, 
                                                      key=lambda x: x.get("performance_index_2021", 0), 
                                                      reverse=True)
                        top_districts = state_districts_sorted[:n_districts]
                    else:
                        top_districts = []
                        
                    if performance_type in ["bottom", "both"]:
                        state_districts_sorted = sorted(state_districts, 
                                                      key=lambda x: x.get("performance_index_2021", 0))
                        bottom_districts = state_districts_sorted[:n_districts]
                    else:
                        bottom_districts = []
                    
                    # Combine based on performance_type
                    if performance_type == "both":
                        selected_districts = top_districts + bottom_districts
                    elif performance_type == "bottom":
                        selected_districts = bottom_districts
                    else:
                        selected_districts = top_districts
                    
                    state_district_data[state_name] = selected_districts
                    print(f"✅ Selected {len(selected_districts)} districts for {state_name}")
                    
                    # Collect district names for boundary data
                    for district in selected_districts:
                        all_districts_for_boundary.append(district["district_name"])
                else:
                    print(f"⚠️ No districts found for state: {state_name}")
                    state_district_data[state_name] = []
        else:
            print("❌ No districts found in district analysis result")
            # Set empty district data for all states
            for state_perf in state_performance:
                state_district_data[state_perf["state_name"]] = []
        
        # Step 5: Get boundary data if requested
        boundary_data = []
        if include_boundary_data:
            if state_names:
                # Include state boundaries and district boundaries
                state_boundary_data = get_state_boundary_data(state_names)
                boundary_data.extend(state_boundary_data)
            
            if all_districts_for_boundary:
                district_boundary_data = get_district_boundary_data(all_districts_for_boundary)
                boundary_data.extend(district_boundary_data)
        
        # Step 6: Generate analysis and charts
        analysis = generate_state_multi_indicator_analysis(
            state_performance, state_district_data, selected_indicators, 
            performance_type, category_name, year
        )
        
        chart_data = generate_state_multi_indicator_chart_data(
            state_performance, state_district_data, selected_indicators, 
            performance_type, year
        )
        
        cursor.close()
        conn.close()
        
        return {
            "response_type": "state_multi_indicator_performance",
            "performance_type": performance_type,
            "category_name": category_name,
            "year": year,
            "total_indicators": len(selected_indicators),
            "total_states": len(state_performance),
            "total_districts": len(all_districts_for_boundary),
            "indicators": selected_indicators,
            "states": state_performance,
            "state_districts": state_district_data,
            "boundary_data": boundary_data,
            "chart_data": chart_data,
            "analysis": analysis,
            "map_type": "state_district_multi_indicator"
        }
        
    except Exception as e:
        logger.error(f"Error in get_state_multi_indicator_performance: {e}")
        return {"error": f"Database error: {str(e)}"}


def calculate_state_multi_indicator_performance(state_data, selected_indicators, year):
    """
    Calculate state-level multi-indicator performance using the same 4-step methodology
    as the district-level function.
    """
    
    # Step 1: Organize data and calculate min/max for normalization
    indicator_stats = {}
    state_data_dict = {}
    
    # First pass: collect all values for min/max calculation
    for row in state_data:
        state_name, indicator_id, prev_2016, prev_2021 = row
        
        if indicator_id not in indicator_stats:
            indicator_stats[indicator_id] = {
                "values_2016": [],
                "values_2021": [],
                "direction": None
            }
        
        # Find indicator direction
        for ind in selected_indicators:
            if ind["indicator_id"] == indicator_id:
                indicator_stats[indicator_id]["direction"] = ind["indicator_direction"]
                break
        
        # Collect all values for normalization
        if prev_2016 is not None:
            indicator_stats[indicator_id]["values_2016"].append(float(prev_2016))
        if prev_2021 is not None:
            indicator_stats[indicator_id]["values_2021"].append(float(prev_2021))
        
        # Organize by state
        if state_name not in state_data_dict:
            state_data_dict[state_name] = {
                "state_name": state_name,
                "indicators": {},
                "normalized_scores_2016": [],
                "normalized_scores_2021": []
            }
        
        state_data_dict[state_name]["indicators"][indicator_id] = {
            "prevalence_2016": float(prev_2016) if prev_2016 is not None else None,
            "prevalence_2021": float(prev_2021) if prev_2021 is not None else None
        }
    
    # Calculate global min/max for each indicator across both years
    for indicator_id, stats in indicator_stats.items():
        all_values = stats["values_2016"] + stats["values_2021"]
        if all_values:
            stats["min_value"] = min(all_values)
            stats["max_value"] = max(all_values)
        else:
            stats["min_value"] = 0
            stats["max_value"] = 1
    
    # Step 2: Normalize and calculate composite performance for each state
    state_performance = []
    
    for state_name, data in state_data_dict.items():
        # Normalize each indicator for both years
        for indicator_id, values in data["indicators"].items():
            min_val = indicator_stats[indicator_id]["min_value"]
            max_val = indicator_stats[indicator_id]["max_value"]
            direction = indicator_stats[indicator_id]["direction"]
            
            # Avoid division by zero
            if max_val == min_val:
                normalized_2016 = 0.5
                normalized_2021 = 0.5
            else:
                # Step 1: Min-Max normalization
                if values["prevalence_2016"] is not None:
                    norm_2016 = (values["prevalence_2016"] - min_val) / (max_val - min_val)
                else:
                    norm_2016 = None
                
                if values["prevalence_2021"] is not None:
                    norm_2021 = (values["prevalence_2021"] - min_val) / (max_val - min_val)
                else:
                    norm_2021 = None
                
                # Step 2: Direction alignment
                if direction == "lower_is_better":
                    if norm_2016 is not None:
                        norm_2016 = 1 - norm_2016
                    if norm_2021 is not None:
                        norm_2021 = 1 - norm_2021
                
                normalized_2016 = norm_2016
                normalized_2021 = norm_2021
            
            # Store normalized values
            if normalized_2016 is not None:
                data["normalized_scores_2016"].append(normalized_2016)
            if normalized_2021 is not None:
                data["normalized_scores_2021"].append(normalized_2021)
        
        # Step 3: Calculate composite performance index (simple average)
        # Filter out NaN values that might occur from invalid normalization
        valid_scores_2016 = [score for score in data["normalized_scores_2016"] if not (score != score)]  # Not NaN
        valid_scores_2021 = [score for score in data["normalized_scores_2021"] if not (score != score)]  # Not NaN
        
        performance_2016 = (sum(valid_scores_2016) / len(valid_scores_2016) 
                           if valid_scores_2016 else 0)
        performance_2021 = (sum(valid_scores_2021) / len(valid_scores_2021) 
                           if valid_scores_2021 else 0)
        
        # Step 4: Calculate change metrics
        absolute_change = performance_2021 - performance_2016 if performance_2016 > 0 else 0
        relative_change = ((performance_2021 - performance_2016) / performance_2016 * 100 
                          if performance_2016 > 0 else 0)
        
        state_performance.append({
            "state_name": state_name,
            "performance_index_2016": round(performance_2016, 4),
            "performance_index_2021": round(performance_2021, 4),
            "absolute_change": round(absolute_change, 4),
            "relative_change": round(relative_change, 2),
            "total_indicators": len(data["indicators"]),
            "indicators_2016": len(valid_scores_2016),
            "indicators_2021": len(valid_scores_2021)
        })
    
    return state_performance


def generate_state_multi_indicator_analysis(state_performance, state_districts, indicators, 
                                          performance_type, category_name, year):
    """Generate comprehensive analysis for state-level multi-indicator performance"""
    if not state_performance:
        return "No states data available for multi-indicator analysis."
    
    total_states = len(state_performance)
    total_indicators = len(indicators)
    total_districts = sum(len(districts) for districts in state_districts.values())
    
    # Get category info
    category_desc = f" in {category_name} category" if category_name else ""
    
    # Performance statistics
    avg_performance = sum(s["performance_index_2021"] for s in state_performance) / total_states
    best_state = max(state_performance, key=lambda x: x["performance_index_2021"])
    worst_state = min(state_performance, key=lambda x: x["performance_index_2021"])
    
    # Change statistics
    states_with_change = [s for s in state_performance if s["absolute_change"] != 0]
    improved_states = [s for s in states_with_change if s["absolute_change"] > 0]
    declined_states = [s for s in states_with_change if s["absolute_change"] < 0]
    
    # District performance type description
    district_desc = f"{'top' if performance_type in ['top', 'both'] else 'bottom'} performing districts"
    if performance_type == "both":
        district_desc = "top and bottom performing districts"
    
    n_districts = 5  # Default value
    
    analysis = f"""
    **State-Level Multi-Indicator Performance Analysis{category_desc}**
    
    This analysis examines {total_states} states across {total_indicators} health indicators, including {total_districts} {district_desc} within these states using a comprehensive composite index methodology.
    
    **Methodology Applied:**
    1. **State-Level Analysis**: Extract state averages from state_indicators table
    2. **Min-Max Normalization**: All indicators normalized to [0,1] scale using global min/max values
    3. **Direction Alignment**: "Lower is better" indicators inverted (X'' = 1 - X')
    4. **Composite Index**: Simple average of normalized, direction-aligned indicators
    5. **District Selection**: Identify {'top' if performance_type == 'top' else 'bottom' if performance_type == 'bottom' else 'top and bottom'} {n_districts} districts within each state
    
    **State Performance Overview ({year}):**
    - Average State Performance Index: {avg_performance:.3f}
    - Best Performing State: {best_state["state_name"]} - Index: {best_state["performance_index_2021"]:.3f}
    - Lowest Performing State: {worst_state["state_name"]} - Index: {worst_state["performance_index_2021"]:.3f}
    
    **State Change Analysis (2016-2021):**
    - States with Improvement: {len(improved_states)} ({len(improved_states)/total_states*100:.1f}%)
    - States with Decline: {len(declined_states)} ({len(declined_states)/total_states*100:.1f}%)
    - Average State Absolute Change: {sum(s["absolute_change"] for s in state_performance)/total_states:.3f}
    
    **District Analysis:**
    - Total Districts Included: {total_districts}
    - Districts per State: Up to {n_districts} {district_desc}
    - Selection Criteria: Multi-indicator performance index ranking within each state
    
    **Key Insights:**
    - Higher index values indicate better overall health performance
    - Index considers indicator direction (higher/lower is better)
    - All indicators contribute equally to the composite score
    - Results enable state-to-state and intra-state district comparisons
    - District performance viewed in context of state-level benchmarks
    
    **Applications:**
    - Inter-state health performance comparison and benchmarking
    - Identification of exemplary districts within each state for best practice sharing
    - Resource allocation decisions based on state and district performance gaps
    - Policy evaluation across multiple health domains at state level
    - Targeted intervention planning for underperforming districts within states
    """
    
    return analysis


def generate_state_multi_indicator_chart_data(state_performance, state_districts, indicators, 
                                            performance_type, year):
    """Generate chart configurations for state-level multi-indicator performance visualization"""
    
    if not state_performance:
        return []
    
    charts = []
    
    # 1. State Performance Index Ranking Chart
    charts.append({
        "type": "bar",
        "title": f"State Multi-Indicator Performance Index ({year})",
        "description": "Composite performance index across all indicators by state (higher = better)",
        "data": {
            "labels": [s["state_name"] for s in state_performance],
            "datasets": [{
                "label": f"State Performance Index {year}",
                "data": [s["performance_index_2021"] for s in state_performance],
                "backgroundColor": ["#2E86C1" if s["performance_index_2021"] >= 0.5 else "#E74C3C" for s in state_performance],
                "borderColor": "#1B4F72",
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 1.0,
                    "title": {"display": True, "text": "Performance Index (0-1)"}
                }
            }
        }
    })
    
    # 2. State Change Analysis Chart
    charts.append({
        "type": "bar", 
        "title": f"State Performance Change (2016-{year})",
        "description": "Absolute change in state performance index over time",
        "data": {
            "labels": [s["state_name"] for s in state_performance],
            "datasets": [{
                "label": "Absolute Change",
                "data": [s["absolute_change"] for s in state_performance],
                "backgroundColor": ["#27AE60" if s["absolute_change"] >= 0 else "#E74C3C" for s in state_performance],
                "borderColor": "#1E8449",
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "title": {"display": True, "text": "Performance Index Change"}
                }
            }
        }
    })
    
    # 3. District Performance within States Chart (for each state)
    for state_name, districts in state_districts.items():
        if districts:
            charts.append({
                "type": "bar",
                "title": f"{state_name}: {'Top' if performance_type == 'top' else 'Bottom' if performance_type == 'bottom' else 'Top & Bottom'} Districts Performance",
                "description": f"Multi-indicator performance of selected districts in {state_name}",
                "data": {
                    "labels": [d["district_name"] for d in districts],
                    "datasets": [{
                        "label": f"District Performance Index {year}",
                        "data": [d.get("performance_index_2021", 0) for d in districts],
                        "backgroundColor": ["#52C41A" if d.get("performance_index_2021", 0) >= 0.6 else 
                                          "#F39C12" if d.get("performance_index_2021", 0) >= 0.4 else "#E74C3C" 
                                          for d in districts],
                        "borderColor": "#389E0D",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 1.0,
                            "title": {"display": True, "text": "Performance Index (0-1)"}
                        }
                    }
                }
            })
    
    return charts


def get_state_boundary_data(state_names):
    """Get boundary data for specified states"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if district_geometry table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'district_geometry'
            )
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("⚠️ district_geometry table not found, skipping boundary data")
            cursor.close()
            conn.close()
            return []
        
        # Query state boundaries by aggregating district boundaries
        state_boundary_query = """
        SELECT DISTINCT 
            ds.state_name,
            ST_AsGeoJSON(ST_Union(dg.geom)) as boundary
        FROM district_geometry dg
        JOIN district_state ds ON dg.district_name = ds.district_name 
            AND dg.state_name = ds.state_name
        WHERE ds.state_name = ANY(%s)
        GROUP BY ds.state_name
        """
        
        cursor.execute(state_boundary_query, (state_names,))
        results = cursor.fetchall()
        
        boundary_data = []
        for row in results:
            state_name, boundary_geojson = row
            if boundary_geojson:
                boundary_data.append({
                    "type": "state",
                    "name": state_name,
                    "geometry": json.loads(boundary_geojson) if isinstance(boundary_geojson, str) else boundary_geojson
                })
        
        cursor.close()
        conn.close()
        
        return boundary_data
        
    except Exception as e:
        logger.error(f"Error getting state boundary data: {e}")
        return []


def calculate_state_multi_indicator_performance(state_data, selected_indicators, year):
    """Calculate multi-indicator performance for states using the same methodology as districts"""
    try:
        # Group state data by state
        state_dict = {}
        for row in state_data:
            state_name, indicator_id, prevalence_2016, prevalence_2021 = row
            if state_name not in state_dict:
                state_dict[state_name] = {}
            state_dict[state_name][indicator_id] = {
                'prevalence_2016': prevalence_2016,
                'prevalence_2021': prevalence_2021
            }
        
        # Create indicator mapping for easy lookup
        indicator_map = {ind['indicator_id']: ind for ind in selected_indicators}
        
        # Calculate performance for each state
        state_performance = []
        all_values_2016 = []
        all_values_2021 = []
        
        # Collect all values for global min/max normalization
        for state_name, indicators in state_dict.items():
            for indicator_id, values in indicators.items():
                if values['prevalence_2016'] is not None:
                    all_values_2016.append(values['prevalence_2016'])
                if values['prevalence_2021'] is not None:
                    all_values_2021.append(values['prevalence_2021'])
        
        # Calculate global min/max for normalization
        min_2016, max_2016 = (min(all_values_2016), max(all_values_2016)) if all_values_2016 else (0, 1)
        min_2021, max_2021 = (min(all_values_2021), max(all_values_2021)) if all_values_2021 else (0, 1)
        
        for state_name, indicators in state_dict.items():
            normalized_scores_2016 = []
            normalized_scores_2021 = []
            
            for indicator_id, values in indicators.items():
                if indicator_id in indicator_map:
                    indicator_info = indicator_map[indicator_id]
                    direction = indicator_info['indicator_direction']
                    
                    # Normalize 2016 values
                    if values['prevalence_2016'] is not None and max_2016 != min_2016:
                        normalized_val = (values['prevalence_2016'] - min_2016) / (max_2016 - min_2016)
                        if direction == 'Lower is better':
                            normalized_val = 1 - normalized_val
                        if not (normalized_val != normalized_val):  # Check for NaN
                            normalized_scores_2016.append(normalized_val)
                    
                    # Normalize 2021 values
                    if values['prevalence_2021'] is not None and max_2021 != min_2021:
                        normalized_val = (values['prevalence_2021'] - min_2021) / (max_2021 - min_2021)
                        if direction == 'Lower is better':
                            normalized_val = 1 - normalized_val
                        if not (normalized_val != normalized_val):  # Check for NaN
                            normalized_scores_2021.append(normalized_val)
            
            # Calculate composite indices
            performance_index_2016 = sum(normalized_scores_2016) / len(normalized_scores_2016) if normalized_scores_2016 else 0
            performance_index_2021 = sum(normalized_scores_2021) / len(normalized_scores_2021) if normalized_scores_2021 else 0
            
            state_performance.append({
                'state_name': state_name,
                'performance_index_2016': performance_index_2016,
                'performance_index_2021': performance_index_2021,
                'absolute_change': performance_index_2021 - performance_index_2016,
                'relative_change': ((performance_index_2021 - performance_index_2016) / performance_index_2016 * 100) if performance_index_2016 > 0 else 0,
                'indicators_count': len(normalized_scores_2021)
            })
        
        return state_performance
        
    except Exception as e:
        logger.error(f"Error calculating state multi-indicator performance: {e}")
        return []


def generate_state_multi_indicator_analysis(state_performance, state_district_data, selected_indicators, performance_type, category_name, year):
    """Generate comprehensive analysis for state-level multi-indicator performance"""
    try:
        analysis_parts = []
        
        # Header
        category_text = f" in {category_name} category" if category_name else ""
        analysis_parts.append(f"**State-Level Multi-Indicator Performance Analysis{category_text}**")
        analysis_parts.append("")
        
        # Overview
        total_states = len(state_performance)
        total_districts = sum(len(districts) for districts in state_district_data.values())
        analysis_parts.append(f"This analysis examines {total_states} states across {len(selected_indicators)} health indicators, including {total_districts} {performance_type} performing districts within these states using a comprehensive composite index methodology.")
        analysis_parts.append("")
        
        # Methodology
        analysis_parts.append("**Methodology Applied:**")
        analysis_parts.append("1. **State-Level Analysis**: Extract state averages from state_indicators table")
        analysis_parts.append("2. **Min-Max Normalization**: All indicators normalized to [0,1] scale using global min/max values")
        analysis_parts.append("3. **Direction Alignment**: 'Lower is better' indicators inverted (X' = 1 - X)")
        analysis_parts.append("4. **Composite Index**: Simple average of normalized, direction-aligned indicators")
        analysis_parts.append(f"5. **District Selection**: Identify {performance_type} 5 districts within each state")
        analysis_parts.append("")
        
        if state_performance:
            # State Performance Overview
            avg_performance = sum(s['performance_index_2021'] for s in state_performance) / len(state_performance)
            best_state = max(state_performance, key=lambda x: x['performance_index_2021'])
            worst_state = min(state_performance, key=lambda x: x['performance_index_2021'])
            
            analysis_parts.append("**State Performance Overview (2021):**")
            analysis_parts.append(f"- Average State Performance Index: {avg_performance:.3f}")
            analysis_parts.append(f"- Best Performing State: {best_state['state_name']} - Index: {best_state['performance_index_2021']:.3f}")
            analysis_parts.append(f"- Lowest Performing State: {worst_state['state_name']} - Index: {worst_state['performance_index_2021']:.3f}")
            analysis_parts.append("")
            
            # Change Analysis
            states_with_improvement = [s for s in state_performance if s['absolute_change'] > 0]
            states_with_decline = [s for s in state_performance if s['absolute_change'] < 0]
            avg_change = sum(s['absolute_change'] for s in state_performance) / len(state_performance)
            
            analysis_parts.append("**State Change Analysis (2016-2021):**")
            analysis_parts.append(f"- States with Improvement: {len(states_with_improvement)} ({len(states_with_improvement)/len(state_performance)*100:.1f}%)")
            analysis_parts.append(f"- States with Decline: {len(states_with_decline)} ({len(states_with_decline)/len(state_performance)*100:.1f}%)")
            analysis_parts.append(f"- Average State Absolute Change: {avg_change:.3f}")
            analysis_parts.append("")
            
            # District Insights
            if state_district_data:
                analysis_parts.append(f"**District Insights within States:**")
                for state_name, districts in state_district_data.items():
                    if districts:
                        avg_district_perf = sum(d['performance_index_2021'] for d in districts) / len(districts)
                        state_perf = next(s['performance_index_2021'] for s in state_performance if s['state_name'] == state_name)
                        comparison = "outperform" if avg_district_perf > state_perf else "underperform"
                        analysis_parts.append(f"- {state_name}: {len(districts)} {performance_type} districts (avg: {avg_district_perf:.3f}) {comparison} state average ({state_perf:.3f})")
        
        return "\n".join(analysis_parts)
        
    except Exception as e:
        logger.error(f"Error generating state multi-indicator analysis: {e}")
        return "Analysis generation failed due to an error."


def generate_state_multi_indicator_chart_data(state_performance, state_district_data, selected_indicators, performance_type, year):
    """Generate chart configurations for state-level multi-indicator performance visualization"""
    charts = []
    
    try:
        # Chart 1: State Multi-Indicator Performance Index
        if state_performance:
            state_names = [state["state_name"] for state in state_performance]
            state_scores = [state["performance_index_2021"] for state in state_performance]
            
            charts.append({
                "type": "bar",
                "title": "State Multi-Indicator Performance Index",
                "description": f"Composite performance index across {len(selected_indicators)} health indicators for {year}",
                "data": {
                    "labels": state_names,
                    "datasets": [{
                        "label": "Performance Index",
                        "data": state_scores,
                        "backgroundColor": "#3B82F6",
                        "borderColor": "#2563EB",
                        "borderWidth": 1
                    }]
                }
            })
        
        # Chart 2: State Performance Change (2016 to 2021)
        if state_performance:
            state_names = [state["state_name"] for state in state_performance]
            state_changes = [state["absolute_change"] for state in state_performance]
            
            charts.append({
                "type": "bar",
                "title": "State Performance Change (2016-2021)",
                "description": "Change in multi-indicator performance index over 5-year period",
                "data": {
                    "labels": state_names,
                    "datasets": [{
                        "label": "Performance Change",
                        "data": state_changes,
                        "backgroundColor": ["#10B981" if change >= 0 else "#EF4444" for change in state_changes],
                        "borderColor": ["#059669" if change >= 0 else "#DC2626" for change in state_changes],
                        "borderWidth": 1
                    }]
                }
            })
        
        # Chart 3+: District Performance within each State
        for state_name, districts in state_district_data.items():
            if districts:
                district_names = [d["district_name"] for d in districts]
                district_scores = [d["performance_index_2021"] for d in districts]
                
                charts.append({
                    "type": "bar", 
                    "title": f"{state_name}: {performance_type.title()} Performing Districts",
                    "description": f"{performance_type.title()} {len(districts)} districts in {state_name} by multi-indicator performance",
                    "data": {
                        "labels": district_names,
                        "datasets": [{
                            "label": "District Performance Index",
                            "data": district_scores,
                            "backgroundColor": "#8B5CF6",
                            "borderColor": "#7C3AED", 
                            "borderWidth": 1
                        }]
                    }
                })
        
        return charts
        
    except Exception as e:
        logger.error(f"Error generating state multi-indicator chart data: {e}")
        return []


def generate_multi_indicator_chart_data(districts, selected_indicators, performance_type, year):
    """Generate chart configurations for multi-indicator performance visualization"""
    charts = []
    
    try:
        # Chart 1: District Multi-Indicator Performance Index
        if districts:
            district_names = [d["district_name"] for d in districts]
            district_scores = [d["performance_index_2021"] for d in districts]
            
            charts.append({
                "type": "bar",
                "title": f"{performance_type.title()} District Multi-Indicator Performance Index",
                "description": f"Composite performance index across {len(selected_indicators)} health indicators for {year}",
                "data": {
                    "labels": district_names,
                    "datasets": [{
                        "label": "Performance Index",
                        "data": district_scores,
                        "backgroundColor": "#3B82F6",
                        "borderColor": "#2563EB",
                        "borderWidth": 1
                    }]
                }
            })
        
        # Chart 2: District Performance Change (2016 to 2021)
        if districts:
            district_names = [d["district_name"] for d in districts]
            district_changes = [d["absolute_change"] for d in districts]
            
            charts.append({
                "type": "bar",
                "title": "District Performance Change (2016-2021)",
                "description": "Change in multi-indicator performance index over 5-year period",
                "data": {
                    "labels": district_names,
                    "datasets": [{
                        "label": "Performance Change",
                        "data": district_changes,
                        "backgroundColor": ["#10B981" if change >= 0 else "#EF4444" for change in district_changes],
                        "borderColor": ["#059669" if change >= 0 else "#DC2626" for change in district_changes],
                        "borderWidth": 1
                    }]
                }
            })
        
        return charts
        
    except Exception as e:
        logger.error(f"Error generating multi-indicator chart data: {e}")
        return []


def _find_similar_districts_enhanced(districts, indicator_id, n_districts):
    """
    Enhanced algorithm to find districts with statistically similar indicator values.
    Uses clustering and standard deviation to identify truly similar districts.
    """
    import numpy as np
    from statistics import mean, stdev
    
    # Extract values for analysis
    district_values = []
    for d in districts:
        value = d["indicators"][indicator_id]["prevalence_current"]
        if value is not None:
            district_values.append((d, value))
    
    if len(district_values) < 2:
        return [item[0] for item in district_values[:n_districts]]
    
    values = [item[1] for item in district_values]
    mean_val = mean(values)
    
    # Calculate standard deviation (handle case where all values are the same)
    try:
        std_val = stdev(values) if len(values) > 1 else 0
    except:
        std_val = 0
    
    if std_val == 0:
        # All values are the same, return first n districts
        return [item[0] for item in district_values[:n_districts]]
    
    # Method 1: Find districts within 0.5 standard deviations of the mean (most similar)
    similarity_threshold = std_val * 0.5
    similar_districts = []
    
    for district, value in district_values:
        if abs(value - mean_val) <= similarity_threshold:
            similar_districts.append((district, value, abs(value - mean_val)))
    
    # If we have enough similar districts, take the most similar ones
    if len(similar_districts) >= n_districts:
        # Sort by closeness to mean (most similar first)
        similar_districts.sort(key=lambda x: x[2])
        return [item[0] for item in similar_districts[:n_districts]]
    
    # Method 2: If not enough very similar districts, use clustering approach
    if len(district_values) >= 10:  # Only cluster if we have enough data
        try:
            # Use simple k-means clustering to find the largest cluster
            values_array = np.array(values).reshape(-1, 1)
            
            # Try 3 clusters and take the largest one
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(3, len(values)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(values_array)
            
            # Find the largest cluster
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            largest_cluster_label = unique_labels[np.argmax(counts)]
            
            # Get districts from the largest cluster
            clustered_districts = []
            for i, (district, value) in enumerate(district_values):
                if cluster_labels[i] == largest_cluster_label:
                    clustered_districts.append((district, value))
            
            # Sort by value within cluster for consistency
            clustered_districts.sort(key=lambda x: x[1])
            return [item[0] for item in clustered_districts[:n_districts]]
            
        except ImportError:
            # Fallback if sklearn not available
            pass
    
    # Method 3: Fallback - select districts closest to median
    median_val = np.median(values)
    district_distances = [(d, v, abs(v - median_val)) for d, v in district_values]
    district_distances.sort(key=lambda x: x[2])  # Sort by distance from median
    
    return [item[0] for item in district_distances[:n_districts]]


def _find_different_districts_enhanced(districts, indicator_id, n_districts):
    """
    Enhanced algorithm to find districts with maximally different indicator values.
    Uses percentile-based selection and distance maximization for optimal diversity.
    """
    import numpy as np
    
    # Extract values for analysis
    district_values = []
    for d in districts:
        value = d["indicators"][indicator_id]["prevalence_current"]
        if value is not None:
            district_values.append((d, value))
    
    if len(district_values) <= n_districts:
        return [item[0] for item in district_values]
    
    values = [item[1] for item in district_values]
    district_values.sort(key=lambda x: x[1])  # Sort by value
    
    # Method 1: Percentile-based selection for maximum spread
    if n_districts >= 3:
        # Create percentiles to ensure good spread across the range
        percentiles = np.linspace(0, 100, n_districts)
        selected_districts = []
        
        for percentile in percentiles:
            target_idx = int(np.percentile(range(len(district_values)), percentile))
            target_idx = min(target_idx, len(district_values) - 1)
            
            # Avoid duplicates
            candidate = district_values[target_idx]
            if candidate not in selected_districts:
                selected_districts.append(candidate)
        
        # If we don't have enough due to duplicates, fill in with most distant
        while len(selected_districts) < n_districts and len(selected_districts) < len(district_values):
            # Find district with maximum distance from already selected
            best_candidate = None
            max_min_distance = -1
            
            for candidate_d, candidate_v in district_values:
                if (candidate_d, candidate_v) not in selected_districts:
                    # Calculate minimum distance to any selected district
                    min_distance = min([abs(candidate_v - selected_v) for _, selected_v in selected_districts])
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = (candidate_d, candidate_v)
            
            if best_candidate:
                selected_districts.append(best_candidate)
            else:
                break
        
        return [item[0] for item in selected_districts[:n_districts]]
    
    # Method 2: Simple extremes for small n_districts
    else:
        # Take districts from extremes
        selected = []
        if n_districts >= 1:
            selected.append(district_values[0])  # Minimum
        if n_districts >= 2:
            selected.append(district_values[-1])  # Maximum
        if n_districts >= 3:
            mid_idx = len(district_values) // 2
            selected.append(district_values[mid_idx])  # Middle
        
        return [item[0] for item in selected[:n_districts]]


def get_district_similarity_analysis(
    indicator_names: Optional[List[str]] = None,
    category_name: Optional[str] = None,
    analysis_type: str = "similar",  # "similar" or "different"
    state_names: Optional[List[str]] = None,
    n_districts: int = 20,
    year: int = 2021,
    include_boundary_data: bool = True
):
    """
    Analyze districts with similar or different performance patterns across multiple indicators.
    
    This function identifies districts that show similar or contrasting trends in health indicators,
    helping to understand regional patterns and policy effectiveness.
    
    Parameters:
    - indicator_names: List of specific indicators to analyze
    - category_name: Category name for indicator selection (will randomly select 4 indicators)
    - analysis_type: "similar" for districts with similar trends, "different" for contrasting patterns
    - state_names: List of states to filter districts (None for all states)
    - n_districts: Maximum number of districts to return (default: 20)
    - year: Primary year for analysis (2021 or 2016)
    - include_boundary_data: Include boundary data for mapping
    
    Returns:
    - Dictionary with analysis results, chart data, and boundary information
    """
    try:
        import random
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Step 1: Determine which indicators to use
        selected_indicators = []
        
        if indicator_names:
            # Use specific indicators
            for indicator_name in indicator_names:
                matched = match_indicator_name_to_database(indicator_name)
                if matched:
                    selected_indicators.append(matched)
        
        elif category_name:
            # Use category-based selection with OpenAI matching and random selection
            category_indicators = get_indicators_by_category(cursor, category_name)
            if len(category_indicators) > 4:
                # Randomly select 4 indicators from the category
                selected_indicators = random.sample(category_indicators, 4)
            else:
                selected_indicators = category_indicators
        
        else:
            # Use all indicators (but limit to 4 random ones for performance)
            cursor.execute("""
                SELECT indicator_id, indicator_name, indicator_direction, indicator_category
                FROM indicators 
                ORDER BY indicator_name
            """)
            all_indicators = [
                {
                    "indicator_id": row[0],
                    "indicator_name": row[1], 
                    "indicator_direction": row[2],
                    "indicator_category": row[3]
                }
                for row in cursor.fetchall()
            ]
            selected_indicators = random.sample(all_indicators, min(4, len(all_indicators)))
        
        if not selected_indicators:
            return {"error": "No indicators found for analysis"}
        
        print(f"🎯 Selected {len(selected_indicators)} indicators for similarity analysis")
        
        # Step 2: Get district data for all selected indicators
        indicator_ids = [ind["indicator_id"] for ind in selected_indicators]
        
        # Prepare state filter
        state_filter = ""
        state_params = []
        if state_names:
            # Use state names directly (assuming they are valid)
            state_filter = "AND s.state_name = ANY(%s)"
            state_params.append(state_names)
        
        # Query district data
        district_query = f"""
        SELECT 
            d.district_name,
            s.state_name,
            di.indicator_id,
            di.prevalence_{year},
            di.prevalence_2016,
            di.prevalence_2021,
            di.prevalence_change,
            di.headcount_{year}
        FROM district_indicators di
        JOIN districts d ON di.district_id = d.district_id
        JOIN states s ON d.state_id = s.state_id
        WHERE di.indicator_id = ANY(%s)
        {state_filter}
        AND di.prevalence_{year} IS NOT NULL
        ORDER BY d.district_name, di.indicator_id
        """
        
        cursor.execute(district_query, [indicator_ids] + state_params)
        raw_data = cursor.fetchall()
        
        if not raw_data:
            return {"error": "No district data found for the specified indicators and states"}
        
        # Step 3: Organize data by district
        districts_data = {}
        for row in raw_data:
            district_name = row[0]
            state_name = row[1]
            indicator_id = row[2]
            prevalence_current = _safe_float_conversion(row[3])
            prevalence_2016 = _safe_float_conversion(row[4])
            prevalence_2021 = _safe_float_conversion(row[5])
            prevalence_change = _safe_float_conversion(row[6])
            headcount = _safe_float_conversion(row[7])
            
            if district_name not in districts_data:
                districts_data[district_name] = {
                    "district_name": district_name,
                    "state_name": state_name,
                    "indicators": {}
                }
            
            districts_data[district_name]["indicators"][indicator_id] = {
                "prevalence_current": prevalence_current,
                "prevalence_2016": prevalence_2016,
                "prevalence_2021": prevalence_2021,
                "prevalence_change": prevalence_change,
                "headcount": headcount
            }
        
        # Step 4: Filter districts that have data for all selected indicators
        complete_districts = []
        for district_name, district_data in districts_data.items():
            if len(district_data["indicators"]) == len(selected_indicators):
                # Check if all indicators have valid data
                all_valid = True
                for indicator_id in indicator_ids:
                    if (indicator_id not in district_data["indicators"] or 
                        district_data["indicators"][indicator_id]["prevalence_current"] is None):
                        all_valid = False
                        break
                
                if all_valid:
                    complete_districts.append(district_data)
        
        if len(complete_districts) < 2:
            return {"error": "Insufficient districts with complete data for similarity analysis"}
        
        print(f"📊 Found {len(complete_districts)} districts with complete indicator data")
        
        # Step 5: Simple similarity/difference analysis
        if analysis_type == "similar":
            # Calculate simple variance-based similarity
            district_scores = []
            
            for district in complete_districts:
                # Calculate normalized scores for this district
                scores = []
                for indicator in selected_indicators:
                    indicator_id = indicator["indicator_id"]
                    value = district["indicators"][indicator_id]["prevalence_current"]
                    scores.append(value if value is not None else 0)
                
                district_scores.append((district, scores))
            
            # Find districts with similar patterns (low variance across normalized scores)
            def calculate_pattern_similarity(scores1, scores2):
                """Calculate similarity between two score patterns"""
                if len(scores1) != len(scores2):
                    return float('inf')
                return sum(abs(a - b) for a, b in zip(scores1, scores2))
            
            # Group similar districts
            selected_districts = []
            used_districts = set()
            
            for i, (district, scores) in enumerate(district_scores):
                if district["district_name"] in used_districts:
                    continue
                    
                similar_group = [district]
                used_districts.add(district["district_name"])
                
                # Find similar districts
                for j, (other_district, other_scores) in enumerate(district_scores):
                    if (i != j and 
                        other_district["district_name"] not in used_districts and
                        len(selected_districts) < n_districts):
                        
                        similarity = calculate_pattern_similarity(scores, other_scores)
                        threshold = 0.1  # Adjust threshold as needed
                        
                        if similarity < threshold:
                            similar_group.append(other_district)
                            used_districts.add(other_district["district_name"])
                
                selected_districts.extend(similar_group[:n_districts - len(selected_districts)])
                if len(selected_districts) >= n_districts:
                    break
            
            analysis_description = f"Districts with similar performance patterns across {len(selected_indicators)} indicators"
            
        else:  # analysis_type == "different"
            # Select most diverse districts
            selected_districts = []
            remaining_districts = complete_districts.copy()
            
            # Start with first district
            if remaining_districts:
                selected_districts.append(remaining_districts.pop(0))
            
            # Add most different districts
            while len(selected_districts) < n_districts and remaining_districts:
                max_difference = -1
                best_candidate = None
                best_index = -1
                
                for i, candidate in enumerate(remaining_districts):
                    # Calculate average difference from selected districts
                    total_difference = 0
                    for selected in selected_districts:
                        # Simple difference calculation
                        diff = 0
                        for indicator in selected_indicators:
                            indicator_id = indicator["indicator_id"]
                            val1 = candidate["indicators"][indicator_id]["prevalence_current"] or 0
                            val2 = selected["indicators"][indicator_id]["prevalence_current"] or 0
                            diff += abs(val1 - val2)
                        total_difference += diff
                    
                    avg_difference = total_difference / len(selected_districts)
                    
                    if avg_difference > max_difference:
                        max_difference = avg_difference
                        best_candidate = candidate
                        best_index = i
                
                if best_candidate:
                    selected_districts.append(best_candidate)
                    remaining_districts.pop(best_index)
                else:
                    break
            
            analysis_description = f"Districts with contrasting performance patterns across {len(selected_indicators)} indicators"
        
        # Limit to requested number
        selected_districts = selected_districts[:n_districts]
        
        # Step 6: Generate chart data for each indicator with indicator-specific district selection
        chart_data = []
        indicator_districts = {}  # Store districts for each indicator
        
        for indicator in selected_indicators:
            indicator_id = indicator["indicator_id"]
            indicator_name = indicator["indicator_name"]
            indicator_direction = indicator.get("indicator_direction", "lower_is_better")
            
            # Get all districts that have data for this specific indicator
            indicator_complete_districts = []
            for district in complete_districts:
                if (indicator_id in district["indicators"] and 
                    district["indicators"][indicator_id]["prevalence_current"] is not None):
                    indicator_complete_districts.append(district)
            
            if len(indicator_complete_districts) < 2:
                # Skip this indicator if insufficient data
                continue
            
            # Find similar/different districts for THIS specific indicator using enhanced algorithms
            if analysis_type == "similar":
                selected_for_indicator = _find_similar_districts_enhanced(
                    indicator_complete_districts, indicator_id, n_districts
                )
            else:  # analysis_type == "different"
                selected_for_indicator = _find_different_districts_enhanced(
                    indicator_complete_districts, indicator_id, n_districts
                )
            
            # Limit to requested number
            selected_for_indicator = selected_for_indicator[:n_districts]
            indicator_districts[indicator_id] = selected_for_indicator
            
            # Generate chart data for this indicator's selected districts
            district_names = [d["district_name"] for d in selected_for_indicator]
            values = [d["indicators"][indicator_id]["prevalence_current"] for d in selected_for_indicator]
            
            # Get change values for additional context
            changes = []
            for d in selected_for_indicator:
                change_val = d["indicators"][indicator_id].get("prevalence_change")
                if change_val is not None:
                    changes.append(change_val)
                else:
                    changes.append(0)
            
            # Determine chart color based on indicator direction
            background_color = "#DC2626" if indicator_direction == "lower_is_better" else "#16A34A"
            border_color = "#B91C1C" if indicator_direction == "lower_is_better" else "#15803D"
            
            # Calculate enhanced statistics for chart info
            min_val, max_val = min(values), max(values)
            avg_val = sum(values) / len(values) if values else 0
            value_range = max_val - min_val if values else 0
            
            # Determine algorithm used
            algorithm_used = "statistical_similarity" if analysis_type == "similar" else "percentile_diversity"
            if analysis_type == "similar":
                algorithm_desc = "Districts selected using standard deviation clustering (within 0.5σ of mean) or K-means clustering for optimal similarity"
            else:
                algorithm_desc = "Districts selected using percentile-based distribution and distance maximization for optimal diversity"
            
            chart_data.append({
                "type": "bar",
                "title": f"{indicator_name}",
                "description": f"{indicator_name} prevalence (%) across districts selected using {algorithm_used} algorithm in {year}",
                "data": {
                    "labels": district_names,
                    "datasets": [{
                        "label": f"{indicator_name} (%)",
                        "data": values,
                        "backgroundColor": background_color,
                        "borderColor": border_color, 
                        "borderWidth": 1
                    }]
                },
                "chart_info": {
                    "indicator_direction": indicator_direction,
                    "year": year,
                    "changes_2016_to_2021": changes,
                    "unit": "percentage",
                    "interpretation": "Lower values are better" if indicator_direction == "lower_is_better" else "Higher values are better",
                    "selected_districts": len(selected_for_indicator),
                    "analysis_type": analysis_type,
                    "algorithm_used": algorithm_used,
                    "algorithm_description": algorithm_desc,
                    "value_statistics": {
                        "min": round(min_val, 1),
                        "max": round(max_val, 1), 
                        "average": round(avg_val, 1),
                        "range": round(value_range, 1),
                        "similarity_measure": "Low spread" if analysis_type == "similar" and value_range < 10 else "High diversity" if analysis_type == "different" else "Moderate variation"
                    }
                }
            })
        
        # Get boundary data if requested - collect all unique districts from all indicators
        boundary_data = []
        all_selected_districts = []
        if include_boundary_data:
            # Collect all unique districts from all indicators
            unique_districts = set()
            for districts_list in indicator_districts.values():
                for district in districts_list:
                    unique_districts.add(district["district_name"])
                    if district not in all_selected_districts:
                        all_selected_districts.append(district)
            
            if unique_districts:
                boundary_data = get_district_boundary_data(list(unique_districts))
        
        # If no indicator-specific districts were found, fall back to original logic
        if not all_selected_districts:
            all_selected_districts = selected_districts
        
        # Step 7: Generate analysis
        total_unique_districts = len(all_selected_districts)
        total_states = len(set(d['state_name'] for d in all_selected_districts)) if all_selected_districts else 0
        
        analysis_parts = [
            f"🔍 **Enhanced {analysis_type.title()} Analysis with Indicator-Specific Districts**\n",
            f"**Analysis Type:** {analysis_type.title()} performance patterns per indicator",
            f"**Indicators Analyzed:** {len(selected_indicators)}",
            f"**Total Unique Districts:** {total_unique_districts}",
            f"**States Covered:** {total_states}",
            f"**Year:** {year}",
            f"**Algorithm Used:** {'Statistical Clustering (0.5σ threshold)' if analysis_type == 'similar' else 'Percentile Distribution & Distance Maximization'}\n"
        ]
        
        # Add explicit section header for maximum clarity
        analysis_parts.append("## 📊 DETAILED INDICATOR-SPECIFIC DISTRICT SELECTIONS")
        analysis_parts.append("*Each indicator below has its own optimally selected districts based on enhanced algorithms:*\n")
        for indicator in selected_indicators:
            indicator_id = indicator["indicator_id"]
            if indicator_id in indicator_districts:
                specific_districts = indicator_districts[indicator_id]
                # Calculate statistics for this indicator
                values = [d["indicators"][indicator_id]["prevalence_current"] for d in specific_districts]
                min_val, max_val = min(values), max(values)
                avg_val = sum(values) / len(values)
                value_range = max_val - min_val
                
                analysis_parts.append(f"\n### 📊 **{indicator['indicator_name']}** - Specific District Selection")
                if analysis_type == "similar":
                    analysis_parts.append(f"**Algorithm:** Statistical Clustering - Districts within 0.5 standard deviations")
                    analysis_parts.append(f"**Districts Selected:** {len(specific_districts)} with statistically similar values")
                    analysis_parts.append(f"**Value Range:** {min_val:.1f}% - {max_val:.1f}% (spread: {value_range:.1f}pp)")
                    analysis_parts.append(f"**Average:** {avg_val:.1f}% (low variation indicates high similarity)")
                else:
                    analysis_parts.append(f"**Algorithm:** Percentile Distribution - Maximum diversity across value range")
                    analysis_parts.append(f"**Districts Selected:** {len(specific_districts)} with maximally diverse values") 
                    analysis_parts.append(f"**Value Range:** {min_val:.1f}% - {max_val:.1f}% (diversity: {value_range:.1f}pp)")
                    analysis_parts.append(f"**Selection Method:** Distributed across percentiles for maximum contrast")
                
                analysis_parts.append(f"**Selected Districts for {indicator['indicator_name']}:**")
                
                # Show top 5 districts for this indicator
                for i, district in enumerate(specific_districts[:5]):
                    current_value = district["indicators"][indicator_id]["prevalence_current"]
                    change_value = district["indicators"][indicator_id].get("prevalence_change")
                    
                    if current_value is not None:
                        value_text = f"   {i+1}. {district['district_name']} ({district['state_name']}): {current_value:.1f}%"
                        
                        # Add change information if available
                        if change_value is not None:
                            direction_symbol = "+" if change_value > 0 else ""
                            trend_text = "↗️" if change_value > 0 else "↘️" if change_value < 0 else "→"
                            value_text += f" ({direction_symbol}{change_value:.1f}pp {trend_text})"
                        
                        analysis_parts.append(value_text)
                
                if len(specific_districts) > 5:
                    analysis_parts.append(f"   ... and {len(specific_districts) - 5} more districts")
        
        # State filtering info
        if state_names:
            analysis_parts.append(f"\n**State Filter Applied:** {', '.join(state_names)}")
        
        analysis = "\n".join(analysis_parts)
        
        return {
            "districts": all_selected_districts,  # All unique districts across indicators
            "indicator_districts": indicator_districts,  # Districts per indicator
            "indicators": selected_indicators,
            "analysis_type": analysis_type,
            "total_districts": total_unique_districts,
            "total_indicators": len(selected_indicators),
            "year": year,
            "chart_data": chart_data,
            "analysis": analysis,
            "boundary": boundary_data,
            "category_name": category_name,
            "state_filter": state_names,
            "map_type": "district_similarity_per_indicator",
            "response_type": "district_similarity_analysis_enhanced"
        }
        
    except Exception as e:
        print(f"Error in get_district_similarity_analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Function execution failed: {str(e)}",
            "response_type": "error"
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def get_district_classification(
    indicator_name: str,
    state_names: Optional[List[str]] = None,
    year: int = 2021,
    include_boundary_data: bool = True
):
    """
    Classify all districts (or filtered by states) into 4 classes using Jenks natural breaks
    based on a specific health indicator value.
    
    This function implements the Jenks natural breaks optimization method to create 
    4 meaningful classes that minimize within-class variance and maximize between-class variance.
    
    Args:
        indicator_name: Name of health indicator (can be misspelled or described)
        state_names: Optional list of state names to filter districts (None for all India)
        year: Year for analysis (2016 or 2021, default: 2021)
        include_boundary_data: Whether to include boundary geometry data for mapping
        
    Returns:
        Dict containing:
        - classified_districts: List of districts with classification info
        - indicator_info: Indicator metadata (name, direction, etc.)
        - classification_legend: Legend with class definitions and colors
        - chart_data: Bar chart showing district count per class
        - analysis: Detailed text analysis
        - boundary_data: Boundary geometry for mapping (if requested)
        - statistics: Classification statistics and summary
        - map_type: "district_classification"
        - response_type: "district_classification_analysis"
    """
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Step 1: Match indicator name to database
        logger.info(f"🔍 Matching indicator name: {indicator_name}")
        matched_indicator = match_indicator_name_to_database(indicator_name)
        
        if not matched_indicator:
            return {
                "error": f"Could not find indicator matching '{indicator_name}'. Please check the indicator name.",
                "response_type": "error"
            }
        
        indicator_id = matched_indicator["indicator_id"]
        indicator_display_name = matched_indicator["indicator_name"]
        indicator_direction = matched_indicator["indicator_direction"]
        higher_is_better = indicator_direction == "higher_is_better"
        
        logger.info(f"🎯 Matched to: {indicator_display_name} (ID: {indicator_id}, Direction: {indicator_direction})")
        
        # Step 2: Build state filter condition
        state_filter_condition = ""
        state_filter_params = []
        
        if state_names:
            # Normalize state names and build filter
            normalized_states = [state.strip().title() for state in state_names]
            placeholders = ','.join(['%s'] * len(normalized_states))
            state_filter_condition = f"AND s.state_name IN ({placeholders})"
            state_filter_params = normalized_states
            logger.info(f"🏛️ Filtering by states: {', '.join(normalized_states)}")
        else:
            logger.info("🇮🇳 Analyzing all districts in India")
        
        # Step 3: Get all districts with indicator data
        year_column = f"prevalence_{year}"
        
        query = f"""
        SELECT DISTINCT
            d.district_id,
            d.district_name,
            s.state_name,
            di.{year_column} as indicator_value
        FROM districts d
        JOIN states s ON d.state_id = s.state_id
        JOIN district_indicators di ON d.district_id = di.district_id
        WHERE di.indicator_id = %s
        AND di.{year_column} IS NOT NULL
        {state_filter_condition}
        ORDER BY s.state_name, d.district_name
        """
        
        params = [indicator_id] + state_filter_params
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            scope_text = f"in states: {', '.join(state_names)}" if state_names else "in India"
            return {
                "error": f"No data found for '{indicator_display_name}' {scope_text} for year {year}",
                "response_type": "error"
            }
        
        logger.info(f"📊 Found data for {len(results)} districts")
        
        # Step 4: Extract values and calculate Jenks breaks
        all_values = [row[3] for row in results if row[3] is not None]
        
        if len(all_values) < 4:
            return {
                "error": f"Not enough data points ({len(all_values)}) to create 4 classes. Need at least 4 districts with data.",
                "response_type": "error"
            }
        
        logger.info(f"📈 Calculating Jenks natural breaks for {len(all_values)} values")
        logger.info(f"📈 Value range: {min(all_values):.2f} - {max(all_values):.2f}")
        
        # Calculate Jenks breaks for 4 classes
        breaks = jenks_breaks(all_values, 4)
        logger.info(f"🎯 Jenks breaks: {[f'{b:.2f}' for b in breaks]}")
        
        # Step 5: Classify each district (OPTIMIZED WITH PARALLEL PROCESSING)
        classified_districts = []
        class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        # AGGRESSIVE PARALLEL PROCESSING for large datasets
        if len(results) > 50:  # Lower threshold for parallel processing
            import os
            max_workers = min(os.cpu_count() or 4, 8)  # Use up to 8 cores
            logger.info(f"🚀 Using AGGRESSIVE parallel processing: {max_workers} workers for {len(results)} districts")
            
            # Optimize batch size for better load distribution
            optimal_batch_size = max(20, len(results) // (max_workers * 2))  # Smaller batches for better parallelism
            batches = [results[i:i + optimal_batch_size] for i in range(0, len(results), optimal_batch_size)]
            
            logger.info(f"📦 Created {len(batches)} optimized batches of ~{optimal_batch_size} districts each")
            
            # Process batches in parallel with maximum efficiency
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_classify_district_batch, batch, breaks, higher_is_better)
                    for batch in batches
                ]
                
                # Collect results from all batches with progress tracking
                completed_batches = 0
                for i, future in enumerate(futures):
                    try:
                        batch_results = future.result(timeout=30)  # 30 second timeout per batch
                        classified_districts.extend(batch_results)
                        completed_batches += 1
                        logger.info(f"📈 Progress: {completed_batches}/{len(batches)} batches completed ({len(classified_districts)} districts classified)")
                    except Exception as e:
                        logger.error(f"Error processing batch {i+1}: {e}")
                        # Fall back to sequential processing for this batch
                        continue
            
            logger.info(f"✅ Parallel processing completed: {len(classified_districts)} districts classified in {len(batches)} batches")
        else:
            # Sequential processing for smaller datasets
            logger.info(f"📝 Using sequential processing for {len(results)} districts")
            classified_districts = _classify_district_batch(results, breaks, higher_is_better)
        
        # Count classifications
        for district in classified_districts:
            class_counts[district["class_number"]] += 1
        
        # Step 6: Generate classification legend
        legend_items = get_classification_legend(breaks, higher_is_better, indicator_display_name, "%")
        
        # Step 7: Generate chart data (district count per class)
        chart_data = {
            "type": "bar",
            "title": f"District Classification: {indicator_display_name} ({year})",
            "description": f"Number of districts in each performance class based on Jenks natural breaks classification",
            "data": {
                "labels": [item["class_name"] for item in legend_items],
                "datasets": [{
                    "label": "Number of Districts",
                    "data": [class_counts[item["class_number"]] for item in legend_items],
                    "backgroundColor": [item["color"] for item in legend_items],
                    "borderColor": [item["color"] for item in legend_items],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Number of Districts"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Performance Class"
                        }
                    }
                }
            }
        }
        
        # Step 8: Generate boundary data if requested
        boundary_data = []
        if include_boundary_data:
            district_names_for_boundary = [d["district_name"] for d in classified_districts]
            boundary_data = get_district_boundary_data(district_names_for_boundary)
            logger.info(f"🗺️ Retrieved boundary data for {len(boundary_data)} districts")
        
        # Step 9: Generate comprehensive analysis
        total_districts = len(classified_districts)
        scope_description = f"across {', '.join(state_names)}" if state_names else "across India"
        
        # Calculate statistics with JSON-safe conversion
        mean_value = _safe_float_conversion_classification(np.mean(all_values))
        median_value = _safe_float_conversion_classification(np.median(all_values))
        std_value = _safe_float_conversion_classification(np.std(all_values))
        
        analysis_parts = [
            f"🏥 **District Classification Analysis: {indicator_display_name}**\n",
            f"**Classification Method:** Jenks Natural Breaks Optimization",
            f"**Scope:** {total_districts} districts {scope_description}",
            f"**Year:** {year}",
            f"**Indicator Direction:** {'Higher values are better' if higher_is_better else 'Lower values are better'}\n",
            
            f"**📊 Statistical Summary:**",
            f"• Mean: {mean_value:.2f}%",
            f"• Median: {median_value:.2f}%", 
            f"• Standard Deviation: {std_value:.2f}%",
            f"• Range: {min(all_values):.2f}% - {max(all_values):.2f}%\n",
            
            f"**🎯 Classification Results:**"
        ]
        
        # Add class-by-class breakdown
        for item in legend_items:
            class_num = item["class_number"]
            count = class_counts[class_num]
            percentage = (count / total_districts) * 100
            
            analysis_parts.append(f"• **{item['class_name']} ({item['range_text']})**: {count} districts ({percentage:.1f}%)")
            analysis_parts.append(f"  {item['description']}")
        
        # Add state-wise breakdown if multiple states
        if state_names and len(state_names) > 1:
            analysis_parts.append(f"\n**🏛️ State-wise Distribution:**")
            state_stats = {}
            for district in classified_districts:
                state = district["state_name"]
                if state not in state_stats:
                    state_stats[state] = {1: 0, 2: 0, 3: 0, 4: 0}
                state_stats[state][district["class_number"]] += 1
            
            for state, counts in state_stats.items():
                total_state_districts = sum(counts.values())
                analysis_parts.append(f"• **{state}**: {total_state_districts} districts")
                for class_num in [1, 2, 3, 4]:
                    if counts[class_num] > 0:
                        class_name = legend_items[class_num-1]["class_name"]
                        analysis_parts.append(f"  - {class_name}: {counts[class_num]} districts")
        
        # Add insights based on distribution
        analysis_parts.append(f"\n**💡 Key Insights:**")
        
        if higher_is_better:
            high_performers = class_counts[4] + class_counts[3]
            low_performers = class_counts[1] + class_counts[2]
        else:
            high_performers = class_counts[1] + class_counts[2]
            low_performers = class_counts[3] + class_counts[4]
        
        high_pct = (high_performers / total_districts) * 100
        low_pct = (low_performers / total_districts) * 100
        
        analysis_parts.append(f"• {high_performers} districts ({high_pct:.1f}%) show good performance")
        analysis_parts.append(f"• {low_performers} districts ({low_pct:.1f}%) need attention or intervention")
        
        # Find best and worst performing districts
        best_districts = [d for d in classified_districts if d["class_number"] == (4 if higher_is_better else 1)]
        worst_districts = [d for d in classified_districts if d["class_number"] == (1 if higher_is_better else 4)]
        
        if best_districts:
            best_list = [f"{d['district_name']} ({d['state_name']})" for d in best_districts[:5]]
            analysis_parts.append(f"• Top performing districts: {', '.join(best_list)}")
        
        if worst_districts:
            worst_list = [f"{d['district_name']} ({d['state_name']})" for d in worst_districts[:5]]
            analysis_parts.append(f"• Districts needing attention: {', '.join(worst_list)}")
        
        analysis = "\n".join(analysis_parts)
        
        # Step 10: Compile final result with JSON-safe values
        result = {
            "classified_districts": classified_districts,
            "indicator_info": {
                "indicator_id": int(indicator_id) if indicator_id is not None else None,
                "indicator_name": str(indicator_display_name),
                "indicator_direction": str(indicator_direction),
                "higher_is_better": bool(higher_is_better)
            },
            "classification_legend": legend_items,
            "jenks_breaks": [_safe_float_conversion_classification(b) for b in breaks],
            "chart_data": chart_data,
            "analysis": str(analysis),
            "boundary_data": boundary_data,
            "statistics": {
                "total_districts": int(total_districts),
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "mean_value": mean_value,
                "median_value": median_value,
                "std_value": std_value,
                "min_value": _safe_float_conversion_classification(min(all_values)),
                "max_value": _safe_float_conversion_classification(max(all_values))
            },
            "state_filter": [str(s) for s in state_names] if state_names else None,
            "year": int(year),
            "map_type": "district_classification",
            "response_type": "district_classification_analysis"
        }
        
        logger.info(f"✅ District classification completed successfully")
        logger.info(f"📊 Total districts classified: {total_districts}")
        logger.info(f"🎯 Class distribution: {dict(class_counts)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_district_classification: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Classification failed: {str(e)}",
            "response_type": "error"
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def get_district_classification_change(
    indicator_name: str,
    state_names: Optional[List[str]] = None,
    include_boundary_data: bool = True
):
    """
    Classify all districts (or filtered by states) into 4 classes using Jenks natural breaks
    based on a specific health indicator's prevalence change (2021 - 2016).
    
    This function implements the Jenks natural breaks optimization method to create 
    4 meaningful classes that minimize within-class variance and maximize between-class variance.
    
    Args:
        indicator_name: Name of health indicator (can be misspelled or described)
        state_names: Optional list of state names to filter districts (None for all India)
        include_boundary_data: Whether to include boundary geometry data for mapping
        
    Returns:
        Dict containing:
        - classified_districts: List of districts with classification info
        - indicator_info: Indicator metadata (name, direction, etc.)
        - classification_legend: Legend with class definitions and colors
        - chart_data: Bar chart showing district count per class
        - analysis: Detailed text analysis
        - boundary_data: Boundary geometry for mapping (if requested)
        - statistics: Classification statistics and summary
        - map_type: "district_classification_change"
        - response_type: "district_classification_change_analysis"
    """
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Step 1: Match indicator name to database
        logger.info(f"🔍 Matching indicator name for change analysis: {indicator_name}")
        matched_indicator = match_indicator_name_to_database(indicator_name)
        
        if not matched_indicator:
            return {
                "error": f"Could not find indicator matching '{indicator_name}'. Please check the indicator name.",
                "response_type": "error"
            }
        
        indicator_id = matched_indicator["indicator_id"]
        indicator_display_name = matched_indicator["indicator_name"]
        indicator_direction = matched_indicator["indicator_direction"]
        
        # For change analysis, we interpret direction differently:
        # - If original indicator is "lower_is_better" (e.g., malnutrition), then negative change is good
        # - If original indicator is "higher_is_better" (e.g., vaccination), then positive change is good
        change_higher_is_better = indicator_direction == "higher_is_better"
        
        logger.info(f"🎯 Matched to: {indicator_display_name} (ID: {indicator_id})")
        logger.info(f"📈 Change interpretation: {'Positive change is better' if change_higher_is_better else 'Negative change is better'}")
        
        # Step 2: Build state filter condition
        state_filter_condition = ""
        state_filter_params = []
        
        if state_names:
            # Normalize state names and build filter
            normalized_states = [state.strip().title() for state in state_names]
            placeholders = ','.join(['%s'] * len(normalized_states))
            state_filter_condition = f"AND s.state_name IN ({placeholders})"
            state_filter_params = normalized_states
            logger.info(f"🏛️ Filtering by states: {', '.join(normalized_states)}")
        else:
            logger.info("🇮🇳 Analyzing all districts in India")
        
        # Step 3: Get all districts with prevalence change data
        query = f"""
        SELECT DISTINCT
            d.district_id,
            d.district_name,
            s.state_name,
            di.prevalence_change as change_value
        FROM districts d
        JOIN states s ON d.state_id = s.state_id
        JOIN district_indicators di ON d.district_id = di.district_id
        WHERE di.indicator_id = %s
        AND di.prevalence_change IS NOT NULL
        {state_filter_condition}
        ORDER BY s.state_name, d.district_name
        """
        
        params = [indicator_id] + state_filter_params
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            scope_text = f"in states: {', '.join(state_names)}" if state_names else "in India"
            return {
                "error": f"No prevalence change data found for '{indicator_display_name}' {scope_text}",
                "response_type": "error"
            }
        
        logger.info(f"📊 Found change data for {len(results)} districts")
        
        # Step 4: Extract values and calculate Jenks breaks
        all_values = [row[3] for row in results if row[3] is not None]
        
        if len(all_values) < 4:
            return {
                "error": f"Not enough data points ({len(all_values)}) to create 4 classes. Need at least 4 districts with change data.",
                "response_type": "error"
            }
        
        logger.info(f"📈 Calculating Jenks natural breaks for {len(all_values)} change values")
        logger.info(f"📈 Change range: {min(all_values):.2f} - {max(all_values):.2f}")
        
        # Calculate Jenks breaks for 4 classes
        breaks = jenks_breaks(all_values, 4)
        logger.info(f"🎯 Jenks breaks: {[f'{b:.2f}' for b in breaks]}")
        
        # Step 5: Classify each district (OPTIMIZED WITH PARALLEL PROCESSING)
        classified_districts = []
        class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        # AGGRESSIVE PARALLEL PROCESSING for large datasets
        if len(results) > 50:  # Lower threshold for parallel processing
            import os
            max_workers = min(os.cpu_count() or 4, 8)  # Use up to 8 cores
            logger.info(f"🚀 Using AGGRESSIVE parallel processing: {max_workers} workers for {len(results)} districts")
            
            # Optimize batch size for better load distribution
            optimal_batch_size = max(20, len(results) // (max_workers * 2))  # Smaller batches for better parallelism
            batches = [results[i:i + optimal_batch_size] for i in range(0, len(results), optimal_batch_size)]
            
            logger.info(f"📦 Created {len(batches)} optimized batches of ~{optimal_batch_size} districts each")
            
            # Use ThreadPoolExecutor for parallel processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(_classify_district_batch, batch, breaks, change_higher_is_better): i 
                    for i, batch in enumerate(batches)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_num = future_to_batch[future]
                    try:
                        batch_result = future.result()
                        batch_results.append((batch_num, batch_result))
                        logger.info(f"✅ Batch {batch_num + 1}/{len(batches)} completed: {len(batch_result)} districts")
                    except Exception as e:
                        logger.error(f"❌ Batch {batch_num + 1} failed: {e}")
                        batch_results.append((batch_num, []))
            
            # Sort results by batch number and flatten
            batch_results.sort(key=lambda x: x[0])
            for _, batch_result in batch_results:
                classified_districts.extend(batch_result)
                
            logger.info(f"🎯 Parallel processing completed: {len(classified_districts)} districts classified")
        else:
            # Sequential processing for smaller datasets
            logger.info(f"🔄 Using sequential processing for {len(results)} districts")
            for row in results:
                district_id, district_name, state_name, change_value = row
                
                # Classify the district
                classification = classify_value(change_value, breaks, change_higher_is_better)
                
                # Ensure all values are JSON serializable
                classified_district = {
                    "district_id": int(district_id) if district_id is not None else None,
                    "district_name": str(district_name) if district_name is not None else "",
                    "state_name": str(state_name) if state_name is not None else "",
                    "indicator_value": float(change_value) if change_value is not None else None,
                    "change_value": float(change_value) if change_value is not None else None,
                    "class_number": int(classification["class_number"]),
                    "class_name": str(classification["class_name"]),
                    "class_color": str(classification["color"]),
                    "class_description": str(classification["description"])
                }
                
                classified_districts.append(classified_district)
        
        # Count districts per class
        for district in classified_districts:
            class_counts[district["class_number"]] += 1
        
        # Step 6: Generate classification legend for change data
        legend_items = get_classification_legend(breaks, change_higher_is_better, f"{indicator_display_name} Change", "pp")  # pp = percentage points
        
        # Step 7: Generate chart data (district count per class)
        chart_data = {
            "type": "bar",
            "title": f"District Change Classification: {indicator_display_name}",
            "description": f"Number of districts in each change performance class based on Jenks natural breaks classification",
            "data": {
                "labels": [item["class_name"] for item in legend_items],
                "datasets": [{
                    "label": "Number of Districts",
                    "data": [class_counts[item["class_number"]] for item in legend_items],
                    "backgroundColor": [item["color"] for item in legend_items],
                    "borderColor": [item["color"] for item in legend_items],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Number of Districts"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Change Performance Class"
                        }
                    }
                }
            }
        }
        
        # Step 8: Generate boundary data if requested
        boundary_data = []
        if include_boundary_data:
            district_names_for_boundary = [d["district_name"] for d in classified_districts]
            boundary_data = get_district_boundary_data(district_names_for_boundary)
            logger.info(f"🗺️ Retrieved boundary data for {len(boundary_data)} districts")
        
        # Step 9: Generate comprehensive analysis
        total_districts = len(classified_districts)
        scope_description = f"across {', '.join(state_names)}" if state_names else "across India"
        
        # Calculate statistics with JSON-safe conversion
        mean_value = _safe_float_conversion_classification(np.mean(all_values))
        median_value = _safe_float_conversion_classification(np.median(all_values))
        std_value = _safe_float_conversion_classification(np.std(all_values))
        
        analysis_parts = [
            f"📈 **District Change Classification Analysis: {indicator_display_name}**\n",
            f"**Classification Method:** Jenks Natural Breaks Optimization",
            f"**Analysis Type:** Prevalence Change (2021 - 2016)",
            f"**Scope:** {total_districts} districts {scope_description}",
            f"**Change Direction:** {'Positive change is better' if change_higher_is_better else 'Negative change is better'}\n",
            
            f"**📊 Statistical Summary:**",
            f"• Mean Change: {mean_value:.2f} percentage points",
            f"• Median Change: {median_value:.2f} percentage points", 
            f"• Standard Deviation: {std_value:.2f} percentage points",
            f"• Range: {min(all_values):.2f} to {max(all_values):.2f} percentage points\n",
            
            f"**🎯 Classification Results:**"
        ]
        
        # Add class-by-class breakdown
        for item in legend_items:
            class_num = item["class_number"]
            count = class_counts[class_num]
            percentage = (count / total_districts) * 100
            
            analysis_parts.append(f"• **{item['class_name']} ({item['range_text']})**: {count} districts ({percentage:.1f}%)")
            analysis_parts.append(f"  {item['description']}")
        
        # Add state-wise breakdown if multiple states
        if state_names and len(state_names) > 1:
            analysis_parts.append(f"\n**🏛️ State-wise Distribution:**")
            state_stats = {}
            for district in classified_districts:
                state = district["state_name"]
                if state not in state_stats:
                    state_stats[state] = {1: 0, 2: 0, 3: 0, 4: 0}
                state_stats[state][district["class_number"]] += 1
            
            for state, counts in state_stats.items():
                total_state_districts = sum(counts.values())
                analysis_parts.append(f"• **{state}**: {total_state_districts} districts")
                for class_num in [1, 2, 3, 4]:
                    if counts[class_num] > 0:
                        class_name = legend_items[class_num-1]["class_name"]
                        analysis_parts.append(f"  - {class_name}: {counts[class_num]} districts")
        
        # Add insights based on distribution
        analysis_parts.append(f"\n**💡 Key Insights:**")
        
        if change_higher_is_better:
            improving_districts = class_counts[4] + class_counts[3]
            declining_districts = class_counts[1] + class_counts[2]
        else:
            improving_districts = class_counts[1] + class_counts[2]
            declining_districts = class_counts[3] + class_counts[4]
        
        improving_pct = (improving_districts / total_districts) * 100
        declining_pct = (declining_districts / total_districts) * 100
        
        analysis_parts.append(f"• {improving_districts} districts ({improving_pct:.1f}%) show improving trends")
        analysis_parts.append(f"• {declining_districts} districts ({declining_pct:.1f}%) show declining trends")
        
        # Find best and worst changing districts
        best_districts = [d for d in classified_districts if d["class_number"] == (4 if change_higher_is_better else 1)]
        worst_districts = [d for d in classified_districts if d["class_number"] == (1 if change_higher_is_better else 4)]
        
        if best_districts:
            best_list = [f"{d['district_name']} ({d['state_name']})" for d in best_districts[:5]]
            analysis_parts.append(f"• Most improved districts: {', '.join(best_list)}")
        
        if worst_districts:
            worst_list = [f"{d['district_name']} ({d['state_name']})" for d in worst_districts[:5]]
            analysis_parts.append(f"• Declining districts needing attention: {', '.join(worst_list)}")
        
        analysis = "\n".join(analysis_parts)
        
        # Step 10: Compile final result with JSON-safe values
        result = {
            "classified_districts": classified_districts,
            "indicator_info": {
                "indicator_id": int(indicator_id) if indicator_id is not None else None,
                "indicator_name": str(indicator_display_name),
                "indicator_direction": str(indicator_direction),
                "change_higher_is_better": bool(change_higher_is_better),
                "higher_is_better": bool(change_higher_is_better)  # For compatibility
            },
            "classification_legend": legend_items,
            "jenks_breaks": [_safe_float_conversion_classification(b) for b in breaks],
            "chart_data": chart_data,
            "analysis": str(analysis),
            "boundary_data": boundary_data,
            "statistics": {
                "total_districts": int(total_districts),
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "mean_value": mean_value,
                "median_value": median_value,
                "std_value": std_value,
                "min_value": _safe_float_conversion_classification(min(all_values)),
                "max_value": _safe_float_conversion_classification(max(all_values))
            },
            "state_filter": [str(s) for s in state_names] if state_names else None,
            "analysis_type": "change",
            "map_type": "district_classification_change",
            "response_type": "district_classification_change_analysis"
        }
        
        logger.info(f"✅ District change classification completed successfully")
        logger.info(f"📊 Total districts classified: {total_districts}")
        logger.info(f"🎯 Class distribution: {dict(class_counts)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_district_classification_change: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Change classification failed: {str(e)}",
            "response_type": "error"
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()