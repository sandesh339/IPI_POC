import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Bar, Radar, Scatter } from 'react-chartjs-2';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { registerMapInstance } from '../utils/saveUtils';


import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend
);
// Set Mapbox access token
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN;

// Map Component for State-wise Extremes
const StateWiseExtremesMap = ({ data }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (!data || map.current) return;
    
    // Handle complex nested data structures - trace through all possibilities
    console.log('🔍 Data structure analysis:', {
      dataKeys: Object.keys(data),
      hasData: !!data.data,
      hasStateResults: !!data.state_results,
      dataType: typeof data.data,
      stateResultsType: typeof data.state_results,
      fullDataStructure: data
    });
    
    // Try different paths to find the state results array
    let stateResults = null;
    let isMultiIndicator = false;
    
    // Path 1: Direct data.data (but check if it's the flattened single function result)
    if (data.data && Array.isArray(data.data) && data.data[0] && data.data[0].result) {
      // This is the function call structure, need to look deeper
      console.log('🔄 Function call structure detected, checking function result');
    } else if (data.data && Array.isArray(data.data)) {
      stateResults = data.data;
      console.log('✅ Found state results in data.data');
    }
    // Path 2: data.state_results
    else if (data.state_results && Array.isArray(data.state_results)) {
      stateResults = data.state_results;
      console.log('✅ Found state results in data.state_results');
    }
    // Path 3: Check for multi-indicator structure (data.indicator_results) - PRIORITY PATH
    if (!stateResults && data.indicator_results && typeof data.indicator_results === 'object') {
      console.log('🔄 Multi-indicator structure detected');
      isMultiIndicator = true;
      
      // Flatten multi-indicator results into a single state results array
      const flattenedResults = [];
      
      for (const [indicatorName, indicatorData] of Object.entries(data.indicator_results)) {
        if (indicatorData.state_results && Array.isArray(indicatorData.state_results)) {
          indicatorData.state_results.forEach(stateResult => {
            // Add indicator information to each state result
            flattenedResults.push({
              ...stateResult,
              indicator_name: indicatorData.indicator_info?.indicator_name || indicatorName,
              indicator_direction: indicatorData.indicator_info?.indicator_direction || 'higher_is_better'
            });
          });
        }
      }
      
      stateResults = flattenedResults;
      console.log(`✅ Flattened multi-indicator results: ${flattenedResults.length} state results`);
    }
    // Path 4: Check if data.data is an object with nested results
    else if (data.data && typeof data.data === 'object' && data.data.length === undefined) {
      // Check for nested arrays in data.data
      const dataKeys = Object.keys(data.data);
      for (const key of dataKeys) {
        if (Array.isArray(data.data[key])) {
          stateResults = data.data[key];
          console.log(`✅ Found state results in data.data.${key}`);
          break;
        }
      }
    }
    // Path 5: Check in the function results if this came through function call flattening
    else if (data.function_calls && data.data && data.data.length > 0) {
      // Check function result data
      const functionData = data.data[0];
      if (functionData && functionData.result) {
        if (functionData.result.data && Array.isArray(functionData.result.data)) {
          stateResults = functionData.result.data;
          console.log('✅ Found state results in function result data');
        } else if (functionData.result.state_results && Array.isArray(functionData.result.state_results)) {
          stateResults = functionData.result.state_results;
          console.log('✅ Found state results in function result state_results');
        } else if (functionData.result.indicator_results) {
          // Handle multi-indicator structure in function results
          console.log('🔄 Multi-indicator structure in function results');
          isMultiIndicator = true;
          
          const flattenedResults = [];
          for (const [indicatorName, indicatorData] of Object.entries(functionData.result.indicator_results)) {
            if (indicatorData.state_results && Array.isArray(indicatorData.state_results)) {
              indicatorData.state_results.forEach(stateResult => {
                flattenedResults.push({
                  ...stateResult,
                  indicator_name: indicatorData.indicator_info?.indicator_name || indicatorName,
                  indicator_direction: indicatorData.indicator_info?.indicator_direction || 'higher_is_better'
                });
              });
            }
          }
          
          stateResults = flattenedResults;
          console.log(`✅ Flattened function result multi-indicator: ${flattenedResults.length} state results`);
        }
      }
    }
    
    if (!stateResults || !Array.isArray(stateResults)) {
      console.log('❌ No valid state results found anywhere in data structure');
      return;
    }
    
    console.log('📊 Using state results:', {
      count: stateResults.length,
      firstState: stateResults[0],
      structure: stateResults[0] ? Object.keys(stateResults[0]) : 'empty'
    });

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: [78.9629, 20.5937], // Center of India
      zoom: 4,
      preserveDrawingBuffer: true // Essential for WebGL canvas capture
    });

    // Register the map instance for save functionality
    registerMapInstance('state-wise-extremes-map', map.current);
    
    // Also attach to window for global access
    window.mapboxMap = map.current;

    map.current.on('load', () => {
      // Find boundary data in complex nested structure
      let boundaryData = null;
      let hasBoundaryData = false;
      
      // Try different paths for boundary data
      if (data.boundary && Array.isArray(data.boundary)) {
        boundaryData = data.boundary;
        console.log('✅ Found boundary data in data.boundary');
      } else if (data.boundary_data && Array.isArray(data.boundary_data)) {
        boundaryData = data.boundary_data;
        console.log('✅ Found boundary data in data.boundary_data');
      } else if (data.data && data.data.length > 0 && data.data[0].result) {
        // Check in function result
        const functionResult = data.data[0].result;
        if (functionResult.boundary && Array.isArray(functionResult.boundary)) {
          boundaryData = functionResult.boundary;
          console.log('✅ Found boundary data in function result boundary');
        } else if (functionResult.boundary_data && Array.isArray(functionResult.boundary_data)) {
          boundaryData = functionResult.boundary_data;
          console.log('✅ Found boundary data in function result boundary_data');
        }
      }
      
      // Filter boundary data to only include districts that are in our state results
      // This ensures we only show boundaries for best/worst performers, not all districts
      if (boundaryData && stateResults) {
        const relevantDistrictNames = new Set();
        
        // Collect all best and worst district names from state results
        stateResults.forEach(stateResult => {
          if (stateResult.best_district) {
            relevantDistrictNames.add(stateResult.best_district.toLowerCase().trim());
          }
          if (stateResult.worst_district) {
            relevantDistrictNames.add(stateResult.worst_district.toLowerCase().trim());
          }
        });
        
        // Filter boundary data to only include relevant districts
        const filteredBoundaryData = boundaryData.filter(boundary => {
          const boundaryName = (boundary.district_name || boundary.district || boundary.name || '').toLowerCase().trim();
          return relevantDistrictNames.has(boundaryName) || 
                 Array.from(relevantDistrictNames).some(districtName => 
                   boundaryName.includes(districtName) || districtName.includes(boundaryName)
                 );
        });
        
        console.log('🎯 Filtered boundary data:', {
          originalCount: boundaryData.length,
          filteredCount: filteredBoundaryData.length,
          relevantDistricts: Array.from(relevantDistrictNames),
          filteredDistricts: filteredBoundaryData.map(b => b.district_name || b.district || b.name)
        });
        
        boundaryData = filteredBoundaryData;
      }
      
      console.log('🗺️ Boundary data analysis:', {
        hasBoundary: !!data.boundary,
        hasBoundaryData: !!data.boundary_data,
        boundaryLength: boundaryData ? boundaryData.length : 0,
        boundaryType: typeof boundaryData,
        isArray: Array.isArray(boundaryData),
        stateResultsLength: stateResults ? stateResults.length : 0,
        stateResultsType: typeof stateResults,
        firstBoundary: boundaryData && boundaryData[0] ? {
          keys: Object.keys(boundaryData[0]),
          district_name: boundaryData[0].district_name,
          hasGeometry: !!boundaryData[0].geometry
        } : null,
        firstStateResult: stateResults && stateResults[0] ? {
          keys: Object.keys(stateResults[0]),
          state_name: stateResults[0].state_name,
          best_district: stateResults[0].best_district,
          worst_district: stateResults[0].worst_district
        } : null
      });
      
      if (boundaryData && Array.isArray(boundaryData) && boundaryData.length > 0) {
        
        // Create GeoJSON for best performers
        const bestPerformersGeoJSON = {
          type: 'FeatureCollection',
          features: []
        };

        // Create GeoJSON for worst performers
        const worstPerformersGeoJSON = {
          type: 'FeatureCollection',
          features: []
        };

        console.log('Processing state results for map:', {
          stateResultsLength: stateResults.length,
          boundaryDataLength: boundaryData ? boundaryData.length : 0,
          boundaryDataType: typeof boundaryData,
          boundaryDataIsArray: Array.isArray(boundaryData),
          firstBoundary: boundaryData && boundaryData[0] ? boundaryData[0] : 'no boundary data',
          firstBoundaryKeys: boundaryData && boundaryData[0] ? Object.keys(boundaryData[0]) : 'no keys',
          firstStateResult: stateResults[0],
          boundaryDistrictNames: boundaryData ? boundaryData.slice(0, 5).map(b => {
            const name = b.district_name || b.district || b.name || 'unknown';
            console.log('Boundary item:', { name, keys: Object.keys(b) });
            return name;
          }) : ['no boundary data'],
          stateDistrictNames: stateResults.slice(0, 3).map(s => ({
            best: s.best_district || s.best_performer?.district_name,
            worst: s.worst_district || s.worst_performer?.district_name
          }))
        });

        // Process state results and match with boundary data
        stateResults.forEach((stateResult, index) => {
          // Use the actual data structure from the console output
          const bestDistrict = {
            district_name: stateResult.best_district,
            current_value: stateResult.best_value,
            annual_change: stateResult.best_trend,
            trend_interpretation: stateResult.best_trend_interpretation
          };
          
          const worstDistrict = {
            district_name: stateResult.worst_district,
            current_value: stateResult.worst_value,
            annual_change: stateResult.worst_trend,
            trend_interpretation: stateResult.worst_trend_interpretation
          };

          console.log(`Processing state: ${stateResult.state_name}`, {
            bestDistrict: bestDistrict.district_name,
            worstDistrict: worstDistrict.district_name,
            indicatorName: stateResult.indicator_name,
            indicatorDirection: stateResult.indicator_direction
          });

          // Find boundary data for best performer with flexible matching
          console.log(`🔍 Looking for boundary for best district: "${bestDistrict.district_name}" in state: "${stateResult.state_name}"`);
          
          // Debug available boundary names
          if (boundaryData && boundaryData.length > 0) {
            console.log('Available boundary names (first 10):', 
              boundaryData.slice(0, 10).map(b => ({
                district_name: b.district_name,
                district: b.district, 
                name: b.name,
                state_name: b.state_name
              }))
            );
          }
          
          const bestBoundary = boundaryData ? boundaryData.find(
            boundary => {
              const boundaryName = boundary.district_name || boundary.district || boundary.name;
              const targetName = bestDistrict.district_name;
              
              if (!boundaryName || !targetName) return false;
              
              // Try exact match first
              let match = boundaryName === targetName;
              
              // Try case-insensitive match
              if (!match) {
                match = boundaryName.toLowerCase() === targetName.toLowerCase();
              }
              
              // Try trimmed match
              if (!match) {
                match = boundaryName.trim().toLowerCase() === targetName.trim().toLowerCase();
              }
              
              // Try contains match (partial)
              if (!match) {
                match = boundaryName.toLowerCase().includes(targetName.toLowerCase()) ||
                        targetName.toLowerCase().includes(boundaryName.toLowerCase());
              }
              
              // Try removing common words and special characters
              if (!match) {
                const cleanBoundary = boundaryName.toLowerCase().replace(/[^a-z0-9]/g, '');
                const cleanTarget = targetName.toLowerCase().replace(/[^a-z0-9]/g, '');
                match = cleanBoundary === cleanTarget || 
                        cleanBoundary.includes(cleanTarget) || 
                        cleanTarget.includes(cleanBoundary);
              }
              
              if (match) {
                console.log(`✅ Found best boundary match: "${targetName}" -> "${boundaryName}"`);
              }
              return match;
            }
          ) : null;
          
          if (!bestBoundary) {
            console.log(`❌ No boundary found for best district: "${bestDistrict.district_name}"`);
            if (boundaryData && boundaryData.length > 0) {
              console.log('Available boundary names:', boundaryData.slice(0, 10).map(b => b.district_name || b.district || b.name || 'unnamed'));
            }
          }

          
          if (bestBoundary && (bestBoundary.geometry || bestBoundary.boundary)) {
            const geometry = bestBoundary.geometry || bestBoundary.boundary;
            if (geometry && (geometry.type || geometry.coordinates)) {
            bestPerformersGeoJSON.features.push({
              type: 'Feature',
              properties: {
                district_name: bestDistrict.district_name,
                state_name: stateResult.state_name,
                indicator_name: stateResult.indicator_name,
                indicator_direction: stateResult.indicator_direction,
                value: bestDistrict.current_value,
                trend: bestDistrict.annual_change,
                trend_interpretation: bestDistrict.trend_interpretation,
                performance_type: 'best',
                type: 'best'
              },
                geometry: geometry
            });
            }
          }

          // Find boundary data for worst performer with flexible matching
          console.log(`🔍 Looking for boundary for worst district: "${worstDistrict.district_name}" in state: "${stateResult.state_name}"`);
          const worstBoundary = boundaryData ? boundaryData.find(
            boundary => {
              const boundaryName = boundary.district_name || boundary.district || boundary.name;
              const targetName = worstDistrict.district_name;
              
              if (!boundaryName || !targetName) return false;
              
              // Try exact match first
              let match = boundaryName === targetName;
              
              // Try case-insensitive match
              if (!match) {
                match = boundaryName.toLowerCase() === targetName.toLowerCase();
              }
              
              // Try trimmed match
              if (!match) {
                match = boundaryName.trim().toLowerCase() === targetName.trim().toLowerCase();
              }
              
              // Try contains match (partial)
              if (!match) {
                match = boundaryName.toLowerCase().includes(targetName.toLowerCase()) ||
                        targetName.toLowerCase().includes(boundaryName.toLowerCase());
              }
              
              // Try removing common words and special characters
              if (!match) {
                const cleanBoundary = boundaryName.toLowerCase().replace(/[^a-z0-9]/g, '');
                const cleanTarget = targetName.toLowerCase().replace(/[^a-z0-9]/g, '');
                match = cleanBoundary === cleanTarget || 
                        cleanBoundary.includes(cleanTarget) || 
                        cleanTarget.includes(cleanBoundary);
              }
              
              if (match) {
                console.log(`✅ Found worst boundary match: "${targetName}" -> "${boundaryName}"`);
              }
              return match;
            }
          ) : null;
          
          if (!worstBoundary) {
            console.log(`❌ No boundary found for worst district: "${worstDistrict.district_name}"`);
          }
          if (worstBoundary && (worstBoundary.geometry || worstBoundary.boundary)) {
            const geometry = worstBoundary.geometry || worstBoundary.boundary;
            if (geometry && (geometry.type || geometry.coordinates)) {
            worstPerformersGeoJSON.features.push({
              type: 'Feature',
              properties: {
                district_name: worstDistrict.district_name,
                state_name: stateResult.state_name,
                indicator_name: stateResult.indicator_name,
                indicator_direction: stateResult.indicator_direction,
                value: worstDistrict.current_value,
                trend: worstDistrict.annual_change,
                trend_interpretation: worstDistrict.trend_interpretation,
                performance_type: 'worst',
                type: 'worst'
              },
                geometry: geometry
            });
            }
          }
        });

        console.log('📊 Finished processing districts:', {
          bestPerformersCount: bestPerformersGeoJSON.features.length,
          worstPerformersCount: worstPerformersGeoJSON.features.length,
          totalStatesProcessed: stateResults.length,
          totalBoundariesAvailable: boundaryData ? boundaryData.length : 0,
          bestPerformers: bestPerformersGeoJSON.features.map(f => ({
            district: f.properties.district_name,
            state: f.properties.state_name,
            value: f.properties.value
          })),
          worstPerformers: worstPerformersGeoJSON.features.map(f => ({
            district: f.properties.district_name,
            state: f.properties.state_name,
            value: f.properties.value
          }))
        });

        // Check if we have any boundary data to display
        hasBoundaryData = bestPerformersGeoJSON.features.length > 0 || worstPerformersGeoJSON.features.length > 0;
        
        // If no districts matched but boundary data exists, create enhanced visualization
        if (!hasBoundaryData && boundaryData && boundaryData.length > 0) {
          console.log('No district matches found, creating enhanced visualization with performance data overlay');
          
          // Create enhanced GeoJSON with performance classification
          const enhancedBoundariesGeoJSON = {
            type: 'FeatureCollection',
            features: []
          };
          
          // Process each boundary to find matching performance data
          boundaryData.forEach((boundary, index) => {
            const boundaryName = boundary.district_name || boundary.district || boundary.name || `District ${index}`;
            const geometry = boundary.geometry || boundary.boundary;
            
            if (!geometry) return; // Skip if no geometry
            
            // Find if this boundary matches any district in our performance data
            let performanceType = 'available';
            let performanceData = null;
            let stateInfo = null;
            
            // Check all states for this district with improved matching
            stateResults.forEach(stateResult => {
              const bestDistrict = stateResult.best_district;
              const worstDistrict = stateResult.worst_district;
              
              // Flexible matching logic
              const matchesDistrict = (targetDistrict, boundaryName) => {
                if (!targetDistrict || !boundaryName) return false;
                const target = targetDistrict.toLowerCase().trim();
                const boundary = boundaryName.toLowerCase().trim();
                
                return target === boundary || 
                       target.includes(boundary) || 
                       boundary.includes(target) ||
                       target.replace(/\s+/g, '') === boundary.replace(/\s+/g, '');
              };
              
              if (matchesDistrict(bestDistrict, boundaryName)) {
                performanceType = 'best';
                performanceData = {
                  value: stateResult.best_value,
                  trend: stateResult.best_trend,
                  trend_interpretation: stateResult.best_trend_interpretation
                };
                stateInfo = stateResult.state_name;
                console.log(`✅ Enhanced match: boundary "${boundaryName}" -> best district "${bestDistrict}" in ${stateInfo}`);
              } else if (matchesDistrict(worstDistrict, boundaryName)) {
                performanceType = 'worst';
                performanceData = {
                  value: stateResult.worst_value,
                  trend: stateResult.worst_trend,
                  trend_interpretation: stateResult.worst_trend_interpretation
                };
                stateInfo = stateResult.state_name;
                console.log(`✅ Enhanced match: boundary "${boundaryName}" -> worst district "${worstDistrict}" in ${stateInfo}`);
              }
            });
            
            // Get the first state result for indicator info
            const firstStateResult = stateResults[0] || {};
            
            enhancedBoundariesGeoJSON.features.push({
              type: 'Feature',
              properties: {
                district_name: boundaryName,
                state_name: stateInfo || 'Unknown State',
                indicator_name: firstStateResult.indicator_name || data.indicator_name || data.indicators?.[0] || 'Health Indicator',
                indicator_direction: firstStateResult.indicator_direction || data.indicator_direction || 'unknown',
                performance_type: performanceType,
                value: performanceData?.value || null,
                trend: performanceData?.trend || null,
                trend_interpretation: performanceData?.trend_interpretation || null,
                type: performanceType
              },
              geometry: geometry
            });
          });
          
          if (enhancedBoundariesGeoJSON.features.length > 0) {
            console.log(`Created enhanced visualization with ${enhancedBoundariesGeoJSON.features.length} boundaries`);
            
            // Separate features by performance type
            const bestFeatures = enhancedBoundariesGeoJSON.features.filter(f => f.properties.performance_type === 'best');
            const worstFeatures = enhancedBoundariesGeoJSON.features.filter(f => f.properties.performance_type === 'worst');
            const availableFeatures = enhancedBoundariesGeoJSON.features.filter(f => f.properties.performance_type === 'available');
            
            // Add sources for each type
            if (bestFeatures.length > 0) {
              map.current.addSource('enhanced-best', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: bestFeatures }
              });
              
              map.current.addLayer({
                id: 'enhanced-best-fill',
                type: 'fill',
                source: 'enhanced-best',
                paint: {
                  'fill-color': '#28a745',
                  'fill-opacity': 0.7
                }
              });
              
              map.current.addLayer({
                id: 'enhanced-best-stroke',
                type: 'line',
                source: 'enhanced-best',
                paint: {
                  'line-color': '#1e7e34',
                  'line-width': 2
                }
              });
            }
            
            if (worstFeatures.length > 0) {
              map.current.addSource('enhanced-worst', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: worstFeatures }
              });
              
              map.current.addLayer({
                id: 'enhanced-worst-fill',
                type: 'fill',
                source: 'enhanced-worst',
                paint: {
                  'fill-color': '#dc3545',
                  'fill-opacity': 0.7
                }
              });
              
              map.current.addLayer({
                id: 'enhanced-worst-stroke',
                type: 'line',
                source: 'enhanced-worst',
                paint: {
                  'line-color': '#c82333',
                  'line-width': 2
                }
              });
            }
            
            if (availableFeatures.length > 0) {
              map.current.addSource('enhanced-available', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: availableFeatures }
              });
              
              map.current.addLayer({
                id: 'enhanced-available-fill',
                type: 'fill',
                source: 'enhanced-available',
                paint: {
                  'fill-color': '#3498db',
                  'fill-opacity': 0.3
                }
              });
              
              map.current.addLayer({
                id: 'enhanced-available-stroke',
                type: 'line',
                source: 'enhanced-available',
                paint: {
                  'line-color': '#2980b9',
                  'line-width': 1
                }
              });
            }
            
            hasBoundaryData = true; // Set flag to true since we're showing boundaries
          }
        }

        // Add best/worst performer layers only if we have matching districts
        if (bestPerformersGeoJSON.features.length > 0 || worstPerformersGeoJSON.features.length > 0) {
        // Add sources
        map.current.addSource('best-performers', {
          type: 'geojson',
          data: bestPerformersGeoJSON
        });

        map.current.addSource('worst-performers', {
          type: 'geojson',
          data: worstPerformersGeoJSON
        });

        // Add layers for best performers
        map.current.addLayer({
          id: 'best-performers-fill',
          type: 'fill',
          source: 'best-performers',
          paint: {
            'fill-color': '#28a745',
            'fill-opacity': 0.7
          }
        });

        map.current.addLayer({
          id: 'best-performers-stroke',
          type: 'line',
          source: 'best-performers',
          paint: {
            'line-color': '#1e7e34',
            'line-width': 2
          }
        });

        // Add layers for worst performers
        map.current.addLayer({
          id: 'worst-performers-fill',
          type: 'fill',
          source: 'worst-performers',
          paint: {
            'fill-color': '#dc3545',
            'fill-opacity': 0.7
          }
        });

        map.current.addLayer({
          id: 'worst-performers-stroke',
          type: 'line',
          source: 'worst-performers',
          paint: {
            'line-color': '#c82333',
            'line-width': 2
          }
        });
        }

        // Add popups only if we have boundary data
        if (hasBoundaryData) {
                  const createEnhancedPopup = (layerId, defaultType = null) => {
            if (map.current.getLayer(layerId)) { // Only add popup if layer exists
          map.current.on('click', layerId, (e) => {
            const properties = e.features[0].properties;
                const performanceType = properties.performance_type || properties.type || defaultType;
                
                let popupContent;
                let headerColor = '#3498db';
                let headerIcon = '📍';
                let headerText = 'District Information';
                
                // Determine header styling based on performance type
                if (performanceType === 'best') {
                  headerColor = '#28a745';
                  headerIcon = '🏆';
                  headerText = 'Best Performer';
                } else if (performanceType === 'worst') {
                  headerColor = '#dc3545';
                  headerIcon = '⚠️';
                  headerText = 'Worst Performer';
                } else if (performanceType === 'available') {
                  headerColor = '#3498db';
                  headerIcon = '📍';
                  headerText = 'District Boundary';
                }
                
                // Enhanced popup for better information display
                const getValueDisplay = (value) => {
                  if (value === null || value === undefined) return 'N/A';
                  const numValue = parseFloat(value);
                  return isNaN(numValue) ? 'N/A' : `${numValue.toFixed(2)}%`;
                };
                
                const getTrendDisplay = (trend, direction) => {
                  if (trend === null || trend === undefined) return 'N/A';
                  const numTrend = parseFloat(trend);
                  if (isNaN(numTrend)) return 'N/A';
                  
                  const sign = numTrend >= 0 ? '+' : '';
                  const color = numTrend >= 0 ? '#28a745' : '#dc3545';
                  
                  // Determine if trend is good or bad based on direction
                  let trendQuality = '';
                  if (direction === 'higher_is_better') {
                    trendQuality = numTrend >= 0 ? '📈 Improving' : '📉 Declining';
                  } else {
                    trendQuality = numTrend <= 0 ? '📈 Improving' : '📉 Declining';
                  }
                  
                  return `<span style="color: ${color}; font-weight: 500;">${sign}${numTrend.toFixed(2)} points</span> ${trendQuality}`;
                };

                popupContent = `
                  <div style="
                    font-family: Arial, sans-serif !important; 
                    width: 260px !important; 
                    max-width: 260px !important; 
                    min-width: 260px !important;
                    box-sizing: border-box !important;
                    overflow: hidden !important;
                    word-wrap: break-word !important;
                    overflow-wrap: break-word !important;
                    white-space: normal !important;
                  ">
                    <h3 style="
                      margin: 0 0 12px 0; 
                      color: ${headerColor}; 
                      font-size: 15px; 
                      border-bottom: 2px solid ${headerColor}; 
                      padding-bottom: 6px;
                      overflow: hidden;
                      text-overflow: ellipsis;
                      white-space: nowrap;
                    ">
                      ${headerIcon} ${headerText}
                    </h3>
                    
                    <div style="margin-bottom: 12px !important; overflow: hidden !important; width: 100% !important;">
                      <p style="margin: 3px 0 !important; font-size: 13px !important; font-weight: 600 !important; word-wrap: break-word !important; overflow-wrap: break-word !important; white-space: normal !important; max-width: 100% !important;"><strong>📍 District:</strong> ${properties.district_name || 'Unknown'}</p>
                      <p style="margin: 3px 0 !important; font-size: 13px !important; word-wrap: break-word !important; overflow-wrap: break-word !important; white-space: normal !important; max-width: 100% !important;"><strong>🗺️ State:</strong> ${properties.state_name || 'Unknown'}</p>
                      <p style="margin: 3px 0 !important; font-size: 13px !important; word-wrap: break-word !important; overflow-wrap: break-word !important; white-space: normal !important; max-width: 100% !important;"><strong>🏥 Indicator:</strong> ${properties.indicator_name || 'Health Indicator'}</p>
                      ${isMultiIndicator ? `<p style="margin: 3px 0 !important; font-size: 11px !important; color: #666 !important; font-style: italic !important; line-height: 1.3 !important; word-wrap: break-word !important; overflow-wrap: break-word !important; white-space: normal !important; max-width: 100% !important;">Multi-indicator analysis - showing primary metric</p>` : ''}
                    </div>
                    
                    ${performanceType !== 'available' && properties.value !== null ? `
                      <div style="
                        margin-bottom: 12px; 
                        padding: 10px; 
                        background-color: ${headerColor}15; 
                        border-radius: 8px; 
                        border-left: 4px solid ${headerColor};
                        overflow: hidden;
                        box-sizing: border-box;
                      ">
                        <p style="margin: 3px 0 !important; font-size: 13px !important; font-weight: 600 !important; word-wrap: break-word !important; overflow-wrap: break-word !important; white-space: normal !important; max-width: 100% !important;"><strong>📊 Current Value (2021):</strong> ${getValueDisplay(properties.value)}</p>
                        <p style="margin: 3px 0 !important; font-size: 12px !important; color: #666 !important; word-wrap: break-word !important; overflow-wrap: break-word !important; white-space: normal !important; max-width: 100% !important;">
                          <strong>📈 Direction:</strong> ${properties.indicator_direction === 'higher_is_better' ? '↑ Higher is Better' : '↓ Lower is Better'}
                        </p>
                        ${properties.trend !== null && properties.trend !== undefined ? `
                          <p style="margin: 6px 0 3px 0 !important; font-size: 12px !important; word-wrap: break-word !important; overflow-wrap: break-word !important; white-space: normal !important; max-width: 100% !important;">
                            <strong>⏱️ 5-Year Trend:</strong> ${getTrendDisplay(properties.trend, properties.indicator_direction)}
                          </p>
                          ${properties.trend_interpretation ? `
                            <p style="
                              margin: 3px 0 !important; 
                              font-size: 11px !important; 
                              color: #666 !important; 
                              font-style: italic !important; 
                              line-height: 1.3 !important;
                              word-wrap: break-word !important;
                              overflow-wrap: break-word !important;
                              white-space: normal !important;
                              max-width: 100% !important;
                            ">
                              💡 ${typeof properties.trend_interpretation === 'object' ? 
                                properties.trend_interpretation.description || 'Trend data available' : 
                                properties.trend_interpretation}
                            </p>
                          ` : ''}
                        ` : ''}
                      </div>
                    ` : ''}
                    
                    ${performanceType === 'available' ? `
                      <div style="
                        margin-top: 12px; 
                        padding: 8px; 
                        background-color: #f8f9fa; 
                        border-radius: 6px; 
                        border: 1px solid #dee2e6;
                        overflow: hidden;
                        box-sizing: border-box;
                      ">
                        <p style="
                          margin: 0; 
                          font-size: 12px; 
                          color: #666; 
                          text-align: center; 
                          line-height: 1.4;
                          word-wrap: break-word;
                        ">
                          📊 Geographic boundary available<br/>
                          <span style="font-size: 11px;">Performance data not matched for this district</span>
                        </p>
                      </div>
                    ` : ''}
                    
                    <div style="
                      margin-top: 12px; 
                      font-size: 11px; 
                      color: #888; 
                      text-align: center; 
                      border-top: 1px solid #eee; 
                      padding-top: 6px; 
                      line-height: 1.3;
                      word-wrap: break-word;
                      overflow: hidden;
                    ">
                      ${performanceType === 'best' ? '🏆 Top performing district in this state' : 
                        performanceType === 'worst' ? '⚠️ Needs attention - lowest performing in state' : 
                        '📍 Geographic reference point'}
                      ${isMultiIndicator ? '<br/><span style="font-size: 10px;">Part of multi-indicator analysis</span>' : ''}
                    </div>
                  </div>
                `;

            const popup = new mapboxgl.Popup({
              maxWidth: '260px',
              className: 'custom-popup'
            })
              .setLngLat(e.lngLat)
              .setHTML(popupContent)
              .addTo(map.current);
              
            // Additional CSS override to ensure popup respects width
            setTimeout(() => {
              const popupElement = popup.getElement();
              if (popupElement) {
                const content = popupElement.querySelector('.mapboxgl-popup-content');
                if (content) {
                  content.style.maxWidth = '260px !important';
                  content.style.width = '260px !important';
                  content.style.padding = '8px !important';
                  content.style.overflow = 'hidden !important';
                  content.style.wordWrap = 'break-word !important';
                }
              }
            }, 0);
          });

          // Change cursor on hover
          map.current.on('mouseenter', layerId, () => {
            map.current.getCanvas().style.cursor = 'pointer';
          });

          map.current.on('mouseleave', layerId, () => {
            map.current.getCanvas().style.cursor = '';
          });
            }
          };

          // Add popups for all possible layers
          createEnhancedPopup('best-performers-fill', 'best');
          createEnhancedPopup('worst-performers-fill', 'worst');
          createEnhancedPopup('enhanced-best-fill', 'best');
          createEnhancedPopup('enhanced-worst-fill', 'worst');
          createEnhancedPopup('enhanced-available-fill', 'available');
          createEnhancedPopup('all-boundaries-fill', 'available');
        }
      } else {
        // Add a text overlay indicating no map data
        console.log('No boundary data available, showing no-data message');
      }
      
      // Add overlay message if no boundaries were matched
      if (!hasBoundaryData && boundaryData.length === 0) {
        const noDataDiv = document.createElement('div');
        noDataDiv.innerHTML = `
          <div style="
            position: absolute; 
            top: 50%; 
            left: 50%; 
            transform: translate(-50%, -50%);
            background: rgba(255,255,255,0.95);
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            font-family: Arial, sans-serif;
            color: #666;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 400px;
          ">
            <h4 style="margin: 0 0 15px 0; color: #333;">📍 Map Data Unavailable</h4>
            <p style="margin: 0 0 10px 0; font-size: 14px; line-height: 1.5;">Geographic boundary data is not available for this analysis.</p>
            <p style="margin: 0; font-size: 12px; color: #888;">Please refer to the charts for data visualization.</p>
          </div>
        `;
        mapContainer.current.appendChild(noDataDiv);
      } else if (!hasBoundaryData && boundaryData.length > 0) {
        const noMatchDiv = document.createElement('div');
        noMatchDiv.innerHTML = `
          <div style="
            position: absolute; 
            top: 50%; 
            left: 50%; 
            transform: translate(-50%, -50%);
            background: rgba(255,255,255,0.95);
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            font-family: Arial, sans-serif;
            color: #666;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 400px;
          ">
            <h4 style="margin: 0 0 15px 0; color: #333;">🔍 District Boundaries Not Found</h4>
            <p style="margin: 0 0 10px 0; font-size: 14px; line-height: 1.5;">District boundaries could not be matched for the current analysis.</p>
            <p style="margin: 0; font-size: 12px; color: #888;">Check the browser console for detailed matching information.</p>
          </div>
        `;
        mapContainer.current.appendChild(noMatchDiv);
      }
    });

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [data]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '500px' }}>
      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />
      
      {/* Dynamic Map Legend */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'white',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
        fontSize: '12px',
        minWidth: '180px'
      }}>
        <div style={{ 
          marginBottom: '8px', 
          fontWeight: 'bold', 
          fontSize: '13px',
          color: '#333'
        }}>
          State-wise Health Extremes
        </div>
        <div style={{ 
          marginBottom: '6px', 
          fontSize: '11px', 
          color: '#666',
          fontStyle: 'italic'
        }}>
          {(() => {
            // Get indicator name from multiple possible locations
            const stateResults = data.data || data.state_results;
            
            // Try direct access first
            if (stateResults && Array.isArray(stateResults) && stateResults[0]?.indicator_name) {
              return stateResults[0].indicator_name;
            }
            
            // Try function result access
            if (data.data && data.data[0]?.result?.data?.[0]?.indicator_name) {
              return data.data[0].result.data[0].indicator_name;
            }
            
            // Try top-level properties
            if (data.indicator_name) {
              return data.indicator_name;
            }
            
            // Fallback
            return 'Health Indicator';
          })()}
        </div>
        
        {/* Show legend based on what's actually rendered */}
        {(data.data || data.state_results) && (data.boundary_data || data.boundary) ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
          <div style={{ 
            width: '15px', 
            height: '15px', 
            backgroundColor: '#28a745', 
                marginRight: '8px',
                border: '1px solid #1e7e34',
                borderRadius: '2px'
          }}></div>
              <span style={{ fontSize: '11px' }}>Best Performers</span>
        </div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px' }}>
          <div style={{ 
            width: '15px', 
            height: '15px', 
            backgroundColor: '#dc3545', 
                marginRight: '8px',
                border: '1px solid #c82333',
                borderRadius: '2px'
          }}></div>
              <span style={{ fontSize: '11px' }}>Worst Performers</span>
        </div>
          </>
        ) : (
          <div style={{ 
            fontSize: '11px', 
            color: '#666',
            textAlign: 'center',
            padding: '4px'
          }}>
            No boundary data available
          </div>
        )}
        
        <div style={{ 
          borderTop: '1px solid #eee', 
          paddingTop: '6px', 
          marginTop: '6px',
          fontSize: '10px',
          color: '#888'
        }}>
          Click on districts for details
        </div>
      </div>
    </div>
  );
};

const StateWiseExtremes = ({ extremesData, mapOnly = false, chartOnly = false }) => {
  // Debug logging to understand data structure
  console.log('🚀 StateWiseExtremes received data:', {
    extremesData,
    hasStateResults: !!extremesData?.state_results,
    hasData: !!extremesData?.data,
    hasChartData: !!extremesData?.chart_data,
    dataType: typeof extremesData?.data,
    dataIsArray: Array.isArray(extremesData?.data),
    functionCalls: extremesData?.function_calls,
    keys: extremesData ? Object.keys(extremesData) : 'no data',
    chartDataKeys: extremesData?.chart_data ? Object.keys(extremesData.chart_data) : 'no chart data',
    boundaryLength: extremesData?.boundary ? extremesData.boundary.length : 0,
    boundaryDataLength: extremesData?.boundary_data ? extremesData.boundary_data.length : 0
  });
  
  // CRITICAL DEBUG: Log the exact chart data structure
  console.log('🔥 DEBUGGING CHART DATA PATHS:');
  console.log('🔥 extremesData.chart_data keys:', extremesData?.chart_data ? Object.keys(extremesData.chart_data) : 'NO CHART DATA');
  console.log('🔥 extremesData.data?.chart_data keys:', extremesData?.data?.chart_data ? Object.keys(extremesData.data.chart_data) : 'NO NESTED CHART DATA');
  console.log('🔥 extremesData.data[0]?.result?.chart_data keys:', 
    (extremesData?.data && Array.isArray(extremesData.data) && extremesData.data[0]?.result?.chart_data) ? 
    Object.keys(extremesData.data[0].result.chart_data) : 'NO FUNCTION RESULT CHART DATA');
  
  if (extremesData?.chart_data) {
    console.log('📊 CHART DATA STRUCTURE (TOP LEVEL):', extremesData.chart_data);
    Object.keys(extremesData.chart_data).forEach(chartKey => {
      const chart = extremesData.chart_data[chartKey];
      console.log(`📈 TOP LEVEL ${chartKey}:`, {
        available: !!chart,
        title: chart?.title,
        type: chart?.type,
        labelsCount: chart?.labels?.length,
        datasetsCount: chart?.datasets?.length
      });
    });
  }

  const [currentChartIndex, setCurrentChartIndex] = useState(0);

  // Get available backend charts - handle nested data structure
  const availableCharts = useMemo(() => {
    // Try to find chart data at different paths
    let chartData = null;
    
    // Path 1: Direct chart_data
    if (extremesData?.chart_data) {
      chartData = extremesData.chart_data;
    }
    // Path 2: Nested in data.chart_data (from function results)
    else if (extremesData?.data?.chart_data) {
      chartData = extremesData.data.chart_data;
    }
    // Path 3: In function result structure
    else if (extremesData?.data && Array.isArray(extremesData.data) && extremesData.data[0]?.result?.chart_data) {
      chartData = extremesData.data[0].result.chart_data;
    }
    
    console.log('🔍 Chart data search CLEAN:', {
      path1_direct: !!extremesData?.chart_data,
      path2_nested: !!extremesData?.data?.chart_data,
      path3_function: !!(extremesData?.data && Array.isArray(extremesData.data) && extremesData.data[0]?.result?.chart_data),
      selectedPath: chartData ? 'FOUND' : 'NOT FOUND',
      selectedChartDataKeys: chartData ? Object.keys(chartData) : 'no chart data found'
    });
    
    if (!chartData) return [];
    
    const charts = [];
    
    // Handle both old and new chart key names for backward compatibility
    
    // Chart 1: Best vs Worst Comparison
    const bestWorstChart = chartData.best_vs_worst_comparison || chartData.state_performance_comparison;
    if (bestWorstChart) {
      charts.push({
        key: 'best_vs_worst_comparison',
        title: 'Best vs Worst District Performance',
        data: bestWorstChart
      });
    }
    
    // Chart 2: Intra-State Disparities  
    const disparitiesChart = chartData.intra_state_disparities || chartData.disparity_analysis;
    if (disparitiesChart) {
      charts.push({
        key: 'intra_state_disparities',
        title: 'Intra-State Health Disparities',
        data: disparitiesChart
      });
    }
    
    // Debug: Log what chart data keys we're actually getting
    console.log('🔍 Raw chartData keys:', Object.keys(chartData));
    console.log('🔍 Charts found after processing:', charts.length);
    
    // If no charts found with new logic, log the raw data for debugging
    if (charts.length === 0) {
      console.log('❌ No charts found! Raw chartData:', chartData);
    }
    
    console.log('📊 Available charts found:', charts.map(c => c.key));
    console.log('📊 Charts details:', charts.map(c => ({
      key: c.key,
      title: c.title,
      hasData: !!c.data,
      dataType: c.data?.type,
      dataTitle: c.data?.title
    })));
    
    return charts;
  }, [extremesData]);

  // Render current chart
  const renderCurrentChart = () => {
    if (availableCharts.length === 0) {
      return (
        <div style={{ textAlign: 'center', padding: '40px', color: '#6c757d' }}>
          <h4>📊 No Chart Data Available</h4>
          <p>Backend chart data is not available for this analysis.</p>
        </div>
      );
    }

    const currentChart = availableCharts[currentChartIndex];
    if (!currentChart) return null;

    // Both chart types are now bar charts, so we'll use Bar component for all
    let ChartComponent = Bar;

    return (
      <div style={{ background: 'white', borderRadius: '12px', padding: '20px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h4 style={{ color: '#495057', margin: 0 }}>
            {currentChart.data.title || currentChart.title}
          </h4>
          
          {/* Chart Navigation */}
          {availableCharts.length > 1 && (
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <button
                onClick={() => setCurrentChartIndex(Math.max(0, currentChartIndex - 1))}
                disabled={currentChartIndex === 0}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #dee2e6',
                  borderRadius: '6px',
                  background: currentChartIndex === 0 ? '#f8f9fa' : 'white',
                  cursor: currentChartIndex === 0 ? 'not-allowed' : 'pointer',
                  color: currentChartIndex === 0 ? '#6c757d' : '#495057'
                }}
              >
                ← Previous
              </button>
              
              <span style={{ color: '#6c757d', fontSize: '14px' }}>
                {currentChartIndex + 1} of {availableCharts.length}
              </span>
              
              <button
                onClick={() => setCurrentChartIndex(Math.min(availableCharts.length - 1, currentChartIndex + 1))}
                disabled={currentChartIndex === availableCharts.length - 1}
                style={{
                  padding: '8px 12px',
                  border: '1px solid #dee2e6',
                  borderRadius: '6px',
                  background: currentChartIndex === availableCharts.length - 1 ? '#f8f9fa' : 'white',
                  cursor: currentChartIndex === availableCharts.length - 1 ? 'not-allowed' : 'pointer',
                  color: currentChartIndex === availableCharts.length - 1 ? '#6c757d' : '#495057'
                }}
              >
                Next →
              </button>
            </div>
          )}
        </div>
        
        {currentChart.data.subtitle && (
          <p style={{ color: '#6c757d', fontSize: '14px', marginBottom: '20px', margin: '0 0 20px 0' }}>
            {currentChart.data.subtitle}
          </p>
        )}
        
                <div style={{ height: '400px' }}>
        <ChartComponent 
            data={currentChart.data}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { 
                  display: true, 
                  position: currentChart.data.options?.plugins?.legend?.position || 'top',
                  maxHeight: currentChart.data.options?.plugins?.legend?.maxHeight || undefined
                },
                tooltip: {
                  callbacks: {
                    afterLabel: (context) => {
                      const value = context.raw;
                      return typeof value === 'number' ? `${value.toFixed(2)}%` : '';
                    }
                  }
                }
              },
              scales: {
                y: { 
                  beginAtZero: currentChart.data.options?.scales?.y?.beginAtZero !== false,
                  title: {
                    display: true,
                    text: currentChart.data.options?.scales?.y?.title?.text || 'Values'
                  },
                  ...(currentChart.data.options?.scales?.y || {})
                },
                x: {
                  title: { 
                    display: true, 
                    text: currentChart.data.options?.scales?.x?.title?.text || 'Categories'
                  },
                  ...(currentChart.data.options?.scales?.x || {})
                }
              },
              ...(currentChart.data.options || {})
            }}
            height={400}
          />
        </div>
      </div>
    );
  };



  const renderChart = () => {
    return renderCurrentChart();
  };







  const renderAnalysisText = () => {
    if (!extremesData?.analysis) return null;

    return (
      <div style={{
        background: 'white',
        border: '1px solid #dee2e6',
        borderRadius: '12px',
        padding: '20px',
        marginTop: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
      }}>
        <h4 style={{
          color: '#495057',
          fontSize: '16px',
          fontWeight: '600',
          marginBottom: '15px'
        }}>
          📝 Detailed Analysis
        </h4>
        <div style={{
          lineHeight: '1.6',
          color: '#495057',
          whiteSpace: 'pre-line'
        }}>
          {extremesData.analysis}
        </div>
      </div>
    );
  };

  if (!extremesData) {
    return (
      <div style={{ 
        padding: '20px', 
        textAlign: 'center',
        color: '#6c757d'
      }}>
        No state-wise extremes data available
      </div>
    );
  }

  if (mapOnly) {
    return (
      <div style={{ padding: '20px' }}>
        <div style={{
          background: 'white',
          border: '1px solid #dee2e6',
          borderRadius: '12px',
          padding: '20px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
        }}>
          <h4 style={{
            color: '#495057',
            fontSize: '16px',
            fontWeight: '600',
            marginBottom: '15px'
          }}>
            🗺️ Geographic Distribution of State-wise Extremes
          </h4>
          <StateWiseExtremesMap data={extremesData} />
        </div>
      </div>
    );
  }

  if (chartOnly) {
    return (
      <div style={{ 
        padding: '10px',
        width: '100%',
        height: '80vh',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div style={{
          flex: 1,
          minHeight: '600px',
          background: 'white',
          borderRadius: '8px',
          padding: '20px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          {renderChart()}
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px' }}>
      {/* Header */}
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ 
          color: '#495057', 
          marginBottom: '10px',
          fontSize: '20px',
          fontWeight: '600'
        }}>
          🏆 State-wise Best & Worst Performers
        </h3>
        <div style={{ 
          color: '#6c757d',
          fontSize: '14px',
          marginBottom: '10px'
        }}>
          <strong>Indicator{(() => {
            // Check for indicators in different locations
            const indicators = extremesData.indicators || 
                             extremesData.data?.indicators ||
                             (extremesData.data && Array.isArray(extremesData.data) && extremesData.data[0]?.result?.indicators);
            return indicators && indicators.length > 1 ? 's' : '';
          })()}:</strong> {
            (() => {
              // Handle multi-indicator display from multiple data sources
              let indicators = extremesData.indicators;
              
              // Try nested data structure
              if (!indicators && extremesData.data?.indicators) {
                indicators = extremesData.data.indicators;
              }
              
              // Try function result structure
              if (!indicators && extremesData.data && Array.isArray(extremesData.data) && extremesData.data[0]?.result?.indicators) {
                indicators = extremesData.data[0].result.indicators;
              }
              
              if (indicators && indicators.length > 1) {
                return indicators.slice(0, 3).join(', ') + 
                       (indicators.length > 3 ? ` + ${indicators.length - 3} more` : '');
              }
              
              // Single indicator display - try multiple sources
              return extremesData.indicator_full_name || 
                     extremesData.indicator_name || 
                     extremesData.data?.indicator_name ||
                     (extremesData.data && Array.isArray(extremesData.data) && extremesData.data[0]?.result?.indicator_name) ||
                     (indicators && indicators[0]) || 
                     'Health Indicator';
            })()
          }
        </div>
        <div style={{ 
          color: '#6c757d',
          fontSize: '14px',
          display: 'flex',
          gap: '20px',
          flexWrap: 'wrap'
        }}>
          <span><strong>Year:</strong> {extremesData.year || extremesData.data?.year || 2021}</span>
          <span><strong>States:</strong> {extremesData.states_analyzed || extremesData.total_states || extremesData.data?.total_states || 'Multiple'}</span>
          <span><strong>Total Districts:</strong> {extremesData.total_districts || extremesData.data?.total_districts || 'Multiple'}</span>
          {(() => {
            const totalIndicators = extremesData.total_indicators || extremesData.data?.total_indicators;
            return totalIndicators ? (
              <span><strong>Total Indicators:</strong> {totalIndicators}</span>
            ) : null;
          })()}
          <span>
            <strong>Direction:</strong> {
              (() => {
                // Check for multi-indicator from different sources
                const indicators = extremesData.indicators || 
                                 extremesData.data?.indicators ||
                                 (extremesData.data && Array.isArray(extremesData.data) && extremesData.data[0]?.result?.indicators);
                
                if (indicators && indicators.length > 1) {
                  return 'Mixed (see individual charts)';
                }
                
                // Try different sources for indicator direction
                const higherIsBetter = extremesData.higher_is_better !== undefined 
                  ? extremesData.higher_is_better
                  : extremesData.data?.higher_is_better;
                  
                const indicatorDirection = extremesData.indicator_direction || extremesData.data?.indicator_direction;
                
                if (higherIsBetter !== undefined) {
                  return higherIsBetter ? 'Higher is Better' : 'Lower is Better';
                } else if (indicatorDirection) {
                  return indicatorDirection === 'higher_is_better' ? 'Higher is Better' : 'Lower is Better';
                }
                
                return 'Variable';
              })()
            }
          </span>
        </div>
      </div>

      {/* Charts */}
      <div style={{
        background: 'white',
        border: '1px solid #dee2e6',
        borderRadius: '12px',
        padding: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
      }}>
        {renderChart()}
      </div>

      {/* Analysis Text */}
      {renderAnalysisText()}
    </div>
  );
};

export default StateWiseExtremes; 