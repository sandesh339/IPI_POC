import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Bar, Line, Scatter } from 'react-chartjs-2';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { registerMapInstance, initializeMapForCapture } from '../utils/saveUtils';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Set Mapbox access token
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN;

// Color schemes for border districts
const BORDER_COLORS = {
  state1: '#3498db',          // Primary state color
  state2: '#e74c3c',          // Secondary state color (deprecated)
  neighboring: '#2ecc71',     // Neighboring state districts
  mixed: '#9b59b6',          // For mixed cases
  indicator_good: '#27ae60',
  indicator_poor: '#e67e22'
};

// Generate distinct colors for different states
const generateStateColors = (states, targetState) => {
  const colorPalette = [
    '#3498db', // Blue - for target state
    '#e74c3c', // Red
    '#2ecc71', // Green  
    '#f39c12', // Orange
    '#9b59b6', // Purple
    '#1abc9c', // Turquoise
    '#e67e22', // Dark Orange
    '#34495e', // Dark Blue
    '#f1c40f', // Yellow
    '#8e44ad', // Dark Purple
    '#16a085', // Dark Turquoise
    '#27ae60', // Dark Green
    '#d35400', // Dark Orange
    '#2c3e50', // Very Dark Blue
    '#c0392b', // Dark Red
  ];
  
  const stateColors = {};
  let colorIndex = 0;
  
  // Assign target state the first color
  if (targetState) {
    stateColors[targetState] = colorPalette[0];
    colorIndex = 1;
  }
  
  // Assign colors to other states
  states.forEach(state => {
    if (state !== targetState) {
      stateColors[state] = colorPalette[colorIndex % colorPalette.length];
      colorIndex++;
    }
  });
  
  return stateColors;
};

// Map Component for Border Districts
const BorderDistrictsMap = ({ data }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    
    
    // Handle nested data structure for map as well
    let actualBoundaryData = data?.boundary_data || data?.boundary;
    let actualDistrictData = data?.data;
    let actualTargetState = data?.target_state;
    
    
    
    // CRITICAL FIX FOR MAP: Check if data.data contains function call objects
    if (actualDistrictData && Array.isArray(actualDistrictData) && actualDistrictData[0]?.function === "get_border_districts") {
      const result = actualDistrictData[0].result;
      actualBoundaryData = result?.boundary_data || result?.boundary;
      actualDistrictData = result?.data;
      actualTargetState = result?.target_state;
      
    }
    
    // If data is wrapped in a districts array with result objects, extract it
    if (data?.districts && Array.isArray(data.districts) && data.districts[0]?.result) {
      const result = data.districts[0].result;
      actualBoundaryData = result.boundary_data || result.boundary;
      actualDistrictData = result.data;
      actualTargetState = result.target_state;
    }
    

    
    if (!actualBoundaryData) {
      console.log('No boundary data available:', data);
      return;
    }

    if (!actualDistrictData || !Array.isArray(actualDistrictData)) {
      console.log('No district data array available:', data);
      return;
    }


    // Get unique states and generate colors
    const allStates = [...new Set(actualDistrictData.map(d => d.state_name))];
    if (!allStates.length) {
      console.log('No states found in data');
      return;
    }
    const stateColors = generateStateColors(allStates, actualTargetState);

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: [78.9629, 20.5937], // Center of India
      zoom: 5,
      preserveDrawingBuffer: true // Essential for WebGL canvas capture
    });

    // Register the map instance for save functionality with multiple IDs for better capture
    registerMapInstance('border-districts-map', map.current);
    registerMapInstance('modal-map-content', map.current);
    registerMapInstance('modal-map-0', map.current);
    
    // Also attach to window for global access
    window.mapboxMap = map.current;
    
    // Initialize for capture with proper timing
    setTimeout(() => {
      initializeMapForCapture(map.current, 'border-districts-map');
    }, 1000);

    map.current.on('load', () => {
     
      
      // Initialize for capture once map is fully loaded
      setTimeout(() => {
        initializeMapForCapture(map.current, 'border-districts-map');
        // Trigger a repaint to ensure canvas is ready
        map.current.triggerRepaint();
      }, 500);
      
      // Create GeoJSON for border districts
      const borderGeoJSON = {
        type: 'FeatureCollection',
        features: actualBoundaryData.map((boundary, index) => {
          if (!boundary || !boundary.district_name || !boundary.state_name) {
            console.log('Invalid boundary data:', boundary);
            return null;
          }

          const districtData = actualDistrictData.find(d => 
            d.district_name === boundary.district_name && d.state_name === boundary.state_name
          );

          if (!districtData) {
            console.log('No matching district data found for:', boundary);
          }
          
          // Debug boundary data - only log if values are missing
          const boundaryDebug = {
            'boundary.area_sqkm': boundary.area_sqkm,
            'boundary.perimeter_km': boundary.perimeter_km, 
            'boundary.shared_boundary_km': boundary.shared_boundary_km,
            'districtData.area_sqkm': districtData?.area_sqkm,
            'districtData.perimeter_km': districtData?.perimeter_km,
            'districtData.shared_boundary_km': districtData?.shared_boundary_km
          };
          
          const hasGeometricData = (districtData?.area_sqkm || boundary.area_sqkm) > 0;
          if (!hasGeometricData) {
            console.warn(`Missing geometric data for ${boundary.district_name}:`, boundaryDebug);
          }

          // Calculate average performance for color coding and state comparison status
          let avgPerformance = 0;
          let indicatorCount = 0;
          let hasStateComparison = false;
          let aboveStateAverage = 0;
          let belowStateAverage = 0;
          
          if (districtData && districtData.indicators) {
            const validIndicators = districtData.indicators.filter(ind => 
              ind.prevalence_2021 !== null || ind.current_value !== null
            );
            if (validIndicators.length > 0) {
              avgPerformance = validIndicators.reduce((sum, ind) => 
                sum + (ind.prevalence_2021 || ind.current_value || 0), 0) / validIndicators.length;
              indicatorCount = validIndicators.length;
            }

            // Check state comparison data
            if (districtData.state_comparison) {
              hasStateComparison = true;
              Object.values(districtData.state_comparison).forEach(comparison => {
                if (comparison.comparison && comparison.comparison.performance_status) {
                  if (comparison.comparison.performance_status === 'above_state_average') {
                    aboveStateAverage++;
                  } else if (comparison.comparison.performance_status === 'below_state_average') {
                    belowStateAverage++;
                  }
                }
              });
            }
          }

          return {
            type: 'Feature',
            id: index,
            properties: {
              district_name: boundary.district_name || 'Unknown District',
              state_name: boundary.state_name || 'Unknown State',
              area_sqkm: districtData?.area_sqkm || boundary.area_sqkm || 0,
              perimeter_km: districtData?.perimeter_km || boundary.perimeter_km || 0,
              shared_boundary_km: districtData?.shared_boundary_km || boundary.shared_boundary_km || 0,
              avg_performance: avgPerformance,
              indicator_count: indicatorCount,
              has_data: !!(districtData && districtData.indicators && districtData.indicators.length > 0),
              has_state_comparison: hasStateComparison,
              above_state_average: aboveStateAverage,
              below_state_average: belowStateAverage,
              state_color: stateColors[boundary.state_name] || '#cccccc'
            },
            geometry: boundary.geometry
          };
        }).filter(feature => feature !== null) // Remove any invalid features
      };

      

      // Add source
      map.current.addSource('border-districts', {
        type: 'geojson',
        data: borderGeoJSON,
        generateId: true  // Let Mapbox generate feature IDs automatically
      });

      // Add fill layer with state-based coloring
      map.current.addLayer({
        id: 'border-districts-fill',
        type: 'fill',
        source: 'border-districts',
        paint: {
          'fill-color': ['get', 'state_color'],
          'fill-opacity': 0.7
        }
      });

      // Add stroke layer
      map.current.addLayer({
        id: 'border-districts-stroke',
        type: 'line',
        source: 'border-districts',
        paint: {
          'line-color': '#ffffff',
          'line-width': 2
        }
      });

      // Add hover effect
      map.current.addLayer({
        id: 'border-districts-hover',
        type: 'fill',
        source: 'border-districts',
        paint: {
          'fill-color': '#000000',
          'fill-opacity': [
            'case',
            ['boolean', ['feature-state', 'hover'], false],
            0.3,
            0
          ]
        }
      });

      let hoveredStateId = null;

      // Handle hover states
      map.current.on('mousemove', 'border-districts-fill', (e) => {
        if (e.features.length > 0) {
          if (hoveredStateId !== null) {
            map.current.setFeatureState(
              { source: 'border-districts', id: hoveredStateId },
              { hover: false }
            );
          }
          hoveredStateId = e.features[0].id;
          map.current.setFeatureState(
            { source: 'border-districts', id: hoveredStateId },
            { hover: true }
          );
        }
      });

      // Change cursor on hover
      map.current.on('mouseenter', 'border-districts-fill', () => {
        map.current.getCanvas().style.cursor = 'pointer';
      });

      map.current.on('mouseleave', 'border-districts-fill', () => {
        if (hoveredStateId !== null) {
          map.current.setFeatureState(
            { source: 'border-districts', id: hoveredStateId },
            { hover: false }
          );
        }
        hoveredStateId = null;
        map.current.getCanvas().style.cursor = '';
      });

      // Add popups
      map.current.on('click', 'border-districts-fill', (e) => {
        if (!e.features || !e.features[0]) {
          console.log('No feature data in click event:', e);
          return;
        }

        const properties = e.features[0].properties;
        if (!properties) {
          console.log('No properties in clicked feature:', e.features[0]);
          return;
        }

        const districtData = actualDistrictData.find(d => 
          d.district_name === properties.district_name && d.state_name === properties.state_name
        );

        let popupContent = `
          <div style="font-family: Arial, sans-serif; max-width: 350px; padding: 12px;">
            <h3 style="margin: 0 0 10px 0; color: #2c3e50; font-size: 16px; font-weight: bold;">
              ${properties.district_name || 'Unknown District'}
            </h3>
            <div style="background: linear-gradient(90deg, 
              ${stateColors[properties.state_name] || '#cccccc'}, 
              rgba(255,255,255,0.1)); 
              padding: 8px; 
              border-radius: 4px; 
              margin-bottom: 12px;">
              <strong style="color: white; font-size: 14px;">
                📍 ${properties.state_name || 'Unknown State'} State
              </strong>
              <div style="color: white; font-size: 12px; margin-top: 4px; opacity: 0.9;">
                ${properties.state_name === actualTargetState 
                    ? 'Target State' 
                    : `Neighboring District of ${actualTargetState || 'Unknown State'}`}
              </div>
            </div>

            ${properties.has_state_comparison ? `
              <div style="background: #e3f2fd; padding: 10px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #1976d2;">
                <h4 style="margin: 0 0 8px 0; color: #1976d2; font-size: 13px;">📊 Performance vs ${actualTargetState || 'Target State'}</h4>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                  <span style="color: #2e7d32;">
                    🔺 Better: <strong>${properties.above_state_average}</strong> indicators
                  </span>
                  <span style="color: #d32f2f;">
                    🔻 Worse: <strong>${properties.below_state_average}</strong> indicators
                  </span>
                </div>
                <div style="font-size: 11px; color: #555; margin-top: 4px; text-align: center;">
                  Compared to ${actualTargetState || 'target state'} averages
                </div>
              </div>
            ` : ''}
            
            <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 10px;">
              <tr>
                <td style="padding: 4px 0; color: #666; font-weight: 500;">Area:</td>
                <td style="padding: 4px 0; font-weight: 600;">${parseFloat(properties.area_sqkm || 0).toFixed(1)} km²</td>
              </tr>
              <tr>
                <td style="padding: 4px 0; color: #666; font-weight: 500;">Perimeter:</td>
                <td style="padding: 4px 0; font-weight: 600;">${parseFloat(properties.perimeter_km || 0).toFixed(1)} km</td>
              </tr>
              <tr>
                <td style="padding: 4px 0; color: #666; font-weight: 500;">Shared Boundary:</td>
                <td style="padding: 4px 0; font-weight: 600;">${parseFloat(properties.shared_boundary_km || 0).toFixed(1)} km</td>
              </tr>
            </table>
        `;

        if (districtData && districtData.indicators && districtData.indicators.length > 0) {
          popupContent += `
            <div style="border-top: 1px solid #eee; padding-top: 10px;">
              <h4 style="margin: 0 0 8px 0; color: #2c3e50; font-size: 14px;">Health Indicators (${districtData.indicators.length})</h4>
              <div style="max-height: 250px; overflow-y: auto;">
          `;

          // Show first 5 indicators with state comparison if available
          const indicatorsToShow = districtData.indicators.slice(0, 5);
          indicatorsToShow.forEach(indicator => {
            const value = parseFloat(indicator.prevalence_2021 || indicator.current_value || 0);
            const change = parseFloat(indicator.prevalence_change || indicator.actual_annual_change || 0);
            const changeColor = change > 0 ? BORDER_COLORS.indicator_good : BORDER_COLORS.indicator_poor;
            
            // Check for state comparison data
            const stateComparison = districtData.state_comparison && 
                                  districtData.state_comparison[indicator.indicator_name];
            
            popupContent += `
              <div style="margin-bottom: 8px; padding: 6px; background: #f8f9fa; border-radius: 3px;">
                <div style="font-weight: 600; font-size: 12px; color: #2c3e50;">${indicator.indicator_name}</div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                  <span>District: <strong>${value.toFixed(2)}</strong></span>
                  <span style="color: ${changeColor};">Change: ${change > 0 ? '+' : ''}${change.toFixed(2)}</span>
                </div>
                ${stateComparison && stateComparison.comparison ? `
                  <div style="border-top: 1px solid #ddd; padding-top: 4px; font-size: 10px;">
                    <div style="background: #f0f7ff; padding: 3px 6px; border-radius: 3px; margin-top: 2px;">
                      <div style="color: #1976d2; font-weight: 600; margin-bottom: 2px;">
                        📊 vs ${actualTargetState || 'Target State'} Average
                      </div>
                      <div style="display: flex; justify-content: space-between;">
                        <span>${actualTargetState || 'Target'}: <strong>${stateComparison.comparison.state_value?.toFixed(2) || 'N/A'}</strong></span>
                        <span style="color: ${stateComparison.comparison.performance_status === 'above_state_average' ? '#2e7d32' : 
                                              stateComparison.comparison.performance_status === 'below_state_average' ? '#d32f2f' : '#666'}; 
                                   font-weight: 600;">
                          ${stateComparison.comparison.performance_status === 'above_state_average' ? '🔺 Better' : 
                            stateComparison.comparison.performance_status === 'below_state_average' ? '🔻 Worse' : '➖ Same'} 
                          ${stateComparison.comparison.percentage_difference ? 
                            ' (' + (stateComparison.comparison.percentage_difference > 0 ? '+' : '') + 
                            stateComparison.comparison.percentage_difference.toFixed(1) + '%)' : ''}
                        </span>
                      </div>
                      <div style="font-size: 9px; color: #666; margin-top: 2px;">
                        Direction: ${stateComparison.comparison.indicator_direction === 'higher_is_better' ? 'Higher is better' : 'Lower is better'}
                      </div>
                    </div>
                  </div>
                ` : ''}
              </div>
            `;
          });

          if (districtData.indicators.length > 5) {
            popupContent += `<div style="color: #666; font-size: 11px; text-align: center;">... and ${districtData.indicators.length - 5} more indicators</div>`;
          }

          popupContent += `</div></div>`;
        } else {
          popupContent += `
            <div style="border-top: 1px solid #eee; padding-top: 10px; text-align: center; color: #666; font-style: italic;">
              No IPI data available for this district
            </div>
          `;
        }

        popupContent += `</div>`;

        new mapboxgl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(popupContent)
          .addTo(map.current);
      });

      // Fit map to border districts
      if (borderGeoJSON.features.length > 0) {
        const bounds = new mapboxgl.LngLatBounds();
        borderGeoJSON.features.forEach(feature => {
          if (feature.geometry.type === 'Polygon') {
            feature.geometry.coordinates[0].forEach(coord => {
              bounds.extend(coord);
            });
          } else if (feature.geometry.type === 'MultiPolygon') {
            feature.geometry.coordinates.forEach(polygon => {
              polygon[0].forEach(coord => {
                bounds.extend(coord);
              });
            });
          }
        });
        map.current.fitBounds(bounds, { padding: 50 });
      }

      // Add improved state-based legend
      const legend = document.createElement('div');
      legend.className = 'border-districts-legend';
      legend.style.cssText = `
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
        font-size: 12px;
        z-index: 1;
        border: 1px solid #ddd;
        min-width: 180px;
      `;
      
      const legendTitle = document.createElement('div');
      legendTitle.textContent = 'States of Border Districts';
      legendTitle.style.cssText = 'font-weight: bold; margin-bottom: 10px; font-size: 14px; color: #333;';
      legend.appendChild(legendTitle);
      
      // Add legend items for each state with their specific colors
      allStates.forEach((state, index) => {
        const stateItem = document.createElement('div');
        stateItem.style.cssText = 'display: flex; align-items: center; margin-bottom: 6px;';
        
        const stateColor = document.createElement('div');
        stateColor.style.cssText = `width: 16px; height: 16px; background-color: ${stateColors[state]}; margin-right: 8px; border-radius: 3px; flex-shrink: 0;`;
        
        const stateLabel = document.createElement('span');
        stateLabel.textContent = state === actualTargetState ? `${state} (Target)` : state;
        stateLabel.style.cssText = `color: #555; font-size: 12px; ${state === actualTargetState ? 'font-weight: 600;' : ''}`;
        
        stateItem.appendChild(stateColor);
        stateItem.appendChild(stateLabel);
        legend.appendChild(stateItem);
      });
      
      map.current.getContainer().appendChild(legend);
      
      // Final initialization for capture after all layers and legend are added
      setTimeout(() => {
        initializeMapForCapture(map.current, 'border-districts-map');
        map.current.triggerRepaint();
        
        // Set up canvas preservation
        const canvas = map.current.getCanvas();
        if (canvas) {
          const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
          if (gl) {
            console.log('WebGL context found, preserveDrawingBuffer:', gl.getContextAttributes().preserveDrawingBuffer);
          }
        }
      }, 1000);
    });

    return () => {
      if (map.current) {
        // Remove legend
        const legend = map.current.getContainer().querySelector('.border-districts-legend');
        if (legend) {
          legend.remove();
        }
        map.current.remove();
        map.current = null;
      }
    };
  }, [data]);

  return (
    <div style={{ height: '100vh', width: '100%', position: 'relative' }}>
      <div ref={mapContainer} style={{ height: '100%', width: '100%' }} />
    </div>
  );
};

const BorderDistrictsAnalysis = ({ borderData, mapOnly = false, chartOnly = false }) => {
  const [selectedIndicator, setSelectedIndicator] = useState(null);
  const [chartType, setChartType] = useState('district_performance');
  const [processedData, setProcessedData] = useState(null);
  const [districtData, setDistrictData] = useState([]);
  const [indicators, setIndicators] = useState([]);

  // Process the data when it changes
  useEffect(() => {

    // Handle nested data structure - check if data is in a result object
    let actualData = borderData?.data;
    
    
    // CRITICAL FIX: Check if borderData.data contains function call objects
    if (actualData && Array.isArray(actualData) && actualData[0]?.function === "get_border_districts") {
      console.log('Found function call wrapper, extracting from result...');
      console.log('Function call object:', actualData[0]);
      console.log('Function result structure:', actualData[0].result);
      console.log('Function result data:', actualData[0].result?.data);
      
      if (actualData[0].result?.data && actualData[0].result.data.length > 0) {
        const sampleDistrict = actualData[0].result.data[0];
        console.log('Sample district from function result:', sampleDistrict);
        console.log('Sample district keys:', Object.keys(sampleDistrict || {}));
        console.log('GEOMETRIC DATA IN SAMPLE:', {
          'area_sqkm': sampleDistrict?.area_sqkm,
          'perimeter_km': sampleDistrict?.perimeter_km,
          'shared_boundary_km': sampleDistrict?.shared_boundary_km
        });
      }
      
      actualData = actualData[0].result?.data;
      
    }
    
    // If data is wrapped in a districts array with result objects, extract it
    if (borderData?.districts && Array.isArray(borderData.districts) && borderData.districts[0]?.result) {
      actualData = borderData.districts[0].result.data;
    }
    
    // Also check if the data is directly at the top level from flattened response
    if (!actualData && borderData?.target_state) {
      actualData = borderData.data;
    }
    
    // Final check: if still no data, maybe it's structured differently
    if (!actualData) {
      console.log('No actualData found, checking all possible data locations:');

    }

    if (!actualData || !Array.isArray(actualData)) {
      
      return;
    }

    // Process the data
    const districts = actualData;
    
    
    // CRITICAL DEBUG: Check if districts have geometric data
    console.log('=== GEOMETRIC DATA CHECK ===');
    for (let i = 0; i < Math.min(3, districts.length); i++) {
      const district = districts[i];
      console.log(`District ${i} (${district?.district_name}):`, {
        'area_sqkm': district?.area_sqkm,
        'perimeter_km': district?.perimeter_km,
        'shared_boundary_km': district?.shared_boundary_km,
        'all_keys': Object.keys(district || {})
      });
    }
    
    const stateGroups = {};
    
    // Get all unique indicators
    const allIndicators = [];
    const indicatorSet = new Set();
    
    districts.forEach((district, index) => {
     
      
      const state = district.state_name;
      if (!stateGroups[state]) {
        stateGroups[state] = [];
      }
      stateGroups[state].push(district);

      // Collect unique indicators
      if (district.indicators && Array.isArray(district.indicators)) {
        
        district.indicators.forEach((indicator, indicatorIndex) => {
          
          if (indicator.indicator_name && !indicatorSet.has(indicator.indicator_name)) {
            indicatorSet.add(indicator.indicator_name);
            allIndicators.push(indicator);
            
          }
        });
      } else {
        console.log(`No valid indicators array for district ${district.district_name}`);
      }
    });

    const processedDataObj = {
      districts,
      stateGroups,
      indicators: allIndicators,
      totalDistricts: districts.length,
      totalIndicators: allIndicators.length
    };

   

    setProcessedData(processedDataObj);
    setIndicators(allIndicators);

    // Set initial indicator only if we don't have one
    if (!selectedIndicator && allIndicators.length > 0) {
      
      setSelectedIndicator(allIndicators[0]);
    }

  }, [borderData, selectedIndicator]); // Include selectedIndicator to prevent setting it repeatedly

  // Update district data when selected indicator changes
  useEffect(() => {
    if (!processedData?.districts || !selectedIndicator) return;

    const allStates = [...new Set(processedData.districts.map(d => d.state_name))];
    const stateColors = generateStateColors(allStates, borderData?.state);

    const newDistrictData = [];
    processedData.districts.forEach(district => {
      const indicator = district.indicators?.find(ind => 
        ind.indicator_name === selectedIndicator.indicator_name
      );
      if (indicator && (indicator.prevalence_2021 !== null || indicator.prevalence_2016 !== null)) {
        const boundaryKm = district.shared_boundary_km || 0;
        
        // Debug if boundary data is missing
        if (boundaryKm === 0) {
          console.warn(`District ${district.district_name} has 0 boundary length. District data:`, {
            'district.shared_boundary_km': district.shared_boundary_km,
            'district.area_sqkm': district.area_sqkm,
            'district.perimeter_km': district.perimeter_km
          });
        }
        
        newDistrictData.push({
          name: district.district_name,
          nfhs4Value: indicator.prevalence_2016 || 0,
          nfhs5Value: indicator.prevalence_2021 || 0,
          state: district.state_name,
          shared_boundary_km: boundaryKm,
          fullLabel: `${district.district_name} (${district.state_name}, ${boundaryKm.toFixed(1)} km)`
        });
      }
    });

    // Sort by shared boundary length
    newDistrictData.sort((a, b) => b.shared_boundary_km - a.shared_boundary_km);
    setDistrictData(newDistrictData);

  }, [processedData, selectedIndicator, borderData?.state]);

  // Generate state comparison chart with comparison to state averages
  const generateStateComparisonChart = () => {
    if (!processedData?.districts || !processedData?.stateGroups || !selectedIndicator) {
      console.log('Missing required data for state comparison chart');
      return null;
    }

    try {
      // Get unique states and generate colors
      const allStates = [...new Set(processedData.districts.map(d => d.state_name))];
      if (!allStates.length) {
        console.log('No states found in data');
        return null;
      }

      const stateColors = generateStateColors(allStates, borderData?.target_state);
      const stateAverages = {};
      const stateActualAverages = {}; // From state_comparison_data if available
      
      Object.entries(processedData.stateGroups).forEach(([state, districts]) => {
        const values = [];
        let totalBoundary = 0;
        
        districts.forEach(district => {
          const indicator = district.indicators?.find(ind => 
            ind.indicator_name === selectedIndicator.indicator_name
          );
          if (indicator && (indicator.prevalence_2021 !== null || indicator.current_value !== null)) {
            values.push(indicator.prevalence_2021 || indicator.current_value);
            totalBoundary += district.shared_boundary_km || 0;
          }
        });
        
        if (values.length > 0) {
          stateAverages[state] = {
            average: values.reduce((sum, val) => sum + val, 0) / values.length,
            districtCount: values.length,
            totalBoundary: totalBoundary
          };
        }

        // Get target state average from state comparison data for cross-border comparison
        let stateComparisonData = borderData?.state_comparison_data;
        if (!stateComparisonData && borderData?.data?.[0]?.result?.state_comparison_data) {
          stateComparisonData = borderData.data[0].result.state_comparison_data;
        }
        
        // Get the target state for comparison
        const targetState = borderData?.target_state || borderData?.data?.[0]?.result?.target_state;
        
        
        if (stateComparisonData && stateComparisonData[targetState]) {
          const targetStateData = stateComparisonData[targetState][selectedIndicator.indicator_name];
          if (targetStateData && targetStateData.prevalence_2021 !== null) {
            // Set the same target state value for all border state groups
            stateActualAverages[state] = targetStateData.prevalence_2021;
            
          }
        }
      });

      const states = Object.keys(stateAverages);
      if (!states.length) {
        console.log('No state averages calculated');
        return null;
      }

      const districtAverages = states.map(state => stateAverages[state].average);
      const datasets = [{
        label: `Border Districts Average - ${selectedIndicator.indicator_name}`,
        data: districtAverages,
        backgroundColor: states.map(state => stateColors[state] + '80'), // Add transparency
        borderColor: states.map(state => stateColors[state]),
        borderWidth: 2
      }];

      // Add target state average for comparison
      const targetState = borderData?.target_state || borderData?.data?.[0]?.result?.target_state;
      const actualAverages = states.map(state => stateActualAverages[state] || null);
      if (actualAverages.some(avg => avg !== null)) {
        datasets.push({
          label: `${targetState} State Average - ${selectedIndicator.indicator_name}`,
          data: actualAverages,
          backgroundColor: '#FF6B6B80', // Use a distinct color for target state comparison
          borderColor: '#FF6B6B',
          borderWidth: 3,
          borderDash: [5, 5] // Dashed line for target state average
        });
      }
      
      return {
        labels: states.map(state => `${state}\n(${stateAverages[state].districtCount} districts, ${stateAverages[state].totalBoundary.toFixed(1)} km)`),
        datasets: datasets
      };
    } catch (error) {
      console.error('Error generating state comparison chart:', error);
      return null;
    }
  };

  // Generate district performance chart
  const generateDistrictPerformanceChart = () => {
    if (!processedData?.districts || !selectedIndicator || !districtData.length) {
      console.log('Missing required data for district performance chart');
      return null;
    }

    try {
      // Get unique states and generate colors
      const allStates = [...new Set(processedData.districts.map(d => d.state_name))];
      if (!allStates.length) {
        console.log('No states found in data');
        return null;
      }

      const stateColors = generateStateColors(allStates, borderData?.target_state || borderData?.data?.[0]?.result?.target_state);
      
      // Create datasets for stacked bars
      const datasets = [];
      
      // Create base dataset (NFHS-4 values)
      datasets.push({
        label: 'NFHS-4 (2016)',
        data: districtData.map(d => d.nfhs4Value),
        backgroundColor: districtData.map(d => stateColors[d.state] + '80'), // Add transparency
        borderColor: districtData.map(d => stateColors[d.state]),
        borderWidth: 1,
        stack: 'Stack 0' // All bars will be in the same stack
      });

      // Create change dataset (difference between NFHS-5 and NFHS-4)
      datasets.push({
        label: 'Change to NFHS-5 (2021)',
        data: districtData.map(d => d.nfhs5Value - d.nfhs4Value),
        backgroundColor: districtData.map(d => stateColors[d.state]),
        borderColor: districtData.map(d => stateColors[d.state]),
        borderWidth: 1,
        stack: 'Stack 0' // Same stack as the base values
      });

      return {
        labels: districtData.map(d => d.fullLabel),
        datasets: datasets
      };
    } catch (error) {
      console.error('Error generating district performance chart:', error);
      return null;
    }
  };

  // Generate state performance summary chart showing above/below state average
  const generateStatePerformanceSummaryChart = () => {
    
    
    // Check if we have the data from function call result
    let stateComparisonData = borderData?.state_comparison_data;
    if (!stateComparisonData && borderData?.data?.[0]?.result?.state_comparison_data) {
      stateComparisonData = borderData.data[0].result.state_comparison_data;
    }
    
    if (!processedData?.districts || !stateComparisonData) {
      
      return null;
    }

    try {
      const states = Object.keys(stateComparisonData);
      
      
      const stateColors = generateStateColors(states, borderData?.target_state || borderData?.data?.[0]?.result?.target_state);
      
      const performanceData = states.map(state => {
        const districts = processedData.districts.filter(d => d.state_name === state);
        let totalAbove = 0;
        let totalBelow = 0;
        let totalIndicators = 0;

        districts.forEach(district => {
          if (district.state_comparison) {
            Object.values(district.state_comparison).forEach(comparison => {
              if (comparison.comparison && comparison.comparison.performance_status) {
                totalIndicators++;
                if (comparison.comparison.performance_status === 'above_state_average') {
                  totalAbove++;
                } else if (comparison.comparison.performance_status === 'below_state_average') {
                  totalBelow++;
                }
              }
            });
          }
        });

        return {
          state,
          abovePercentage: totalIndicators > 0 ? (totalAbove / totalIndicators) * 100 : 0,
          belowPercentage: totalIndicators > 0 ? (totalBelow / totalIndicators) * 100 : 0,
          totalIndicators,
          districtCount: districts.length
        };
      });

      return {
        labels: performanceData.map(d => `${d.state}\n(${d.districtCount} districts)`),
        datasets: [
          {
            label: 'Above State Average (%)',
            data: performanceData.map(d => d.abovePercentage),
            backgroundColor: performanceData.map(d => stateColors[d.state] + '80'),
            borderColor: performanceData.map(d => stateColors[d.state]),
            borderWidth: 2
          },
          {
            label: 'Below State Average (%)',
            data: performanceData.map(d => d.belowPercentage),
            backgroundColor: performanceData.map(d => stateColors[d.state] + '40'),
            borderColor: performanceData.map(d => stateColors[d.state]),
            borderWidth: 2,
            borderDash: [3, 3]
          }
        ]
      };
    } catch (error) {
      console.error('Error generating state performance summary chart:', error);
      return null;
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: selectedIndicator ? 
          `${selectedIndicator.indicator_name} - ${
            chartType === 'state_comparison' ? 'Border Districts vs Target State Average' :
            chartType === 'state_performance_summary' ? 'Border Districts Performance vs Target State' :
            'Border Districts Comparison (2016 vs 2021)'
          }` : 
          `Border Districts Analysis - ${
            chartType === 'state_comparison' ? 'Border Districts vs Target State Average' :
            chartType === 'state_performance_summary' ? 'Border Districts Performance vs Target State' :
            'Comparison (2016 vs 2021)'
          }`
      },
      tooltip: {
        callbacks: {
          title: function(context) {
            // Get the current chart type to customize tooltip
            if (chartType === 'district_performance') {
              const dataIndex = context[0].dataIndex;
              const district = districtData[dataIndex];
              if (district) {
                let tooltip = `${district.name} (${district.state})`;
                tooltip += `\nShared Boundary: ${district.shared_boundary_km.toFixed(1)} km`;
                const change = district.nfhs5Value - district.nfhs4Value;
                const changePercent = district.nfhs4Value > 0 ? ((change / district.nfhs4Value) * 100).toFixed(1) : 'N/A';
                tooltip += `\n\nNFHS-4 (2016): ${district.nfhs4Value.toFixed(2)}`;
                tooltip += `\nNFHS-5 (2021): ${district.nfhs5Value.toFixed(2)}`;
                tooltip += `\nChange: ${change > 0 ? '+' : ''}${change.toFixed(2)} (${changePercent}%)`;
                return tooltip;
              }
            } else if (chartType === 'state_comparison') {
              // For state comparison chart
              const label = context[0].label;
              const targetState = borderData?.target_state || borderData?.data?.[0]?.result?.target_state || 'Target State';
              return `${label}\nCompared to: ${targetState} State Average`;
            } else if (chartType === 'state_performance_summary') {
              // For state performance summary
              const label = context[0].label;
              const targetState = borderData?.target_state || borderData?.data?.[0]?.result?.target_state || 'Target State';
              return `${label}\nPerformance vs ${targetState} State Average`;
            }
            return context[0].label;
          },
          label: function(context) {
            const datasetLabel = context.dataset.label;
            const value = context.parsed.y;
            
            if (chartType === 'state_comparison') {
              const targetState = borderData?.target_state || borderData?.data?.[0]?.result?.target_state || 'Target State';
              if (datasetLabel.includes('Border Districts Average')) {
                return `Border Districts Average: ${value.toFixed(2)}`;
              } else if (datasetLabel.includes('State Average')) {
                return `${targetState} State Average: ${value.toFixed(2)}`;
              }
            } else if (chartType === 'state_performance_summary') {
              if (datasetLabel.includes('Above')) {
                return `Above ${borderData?.target_state || 'Target'} Average: ${value.toFixed(1)}%`;
              } else if (datasetLabel.includes('Below')) {
                return `Below ${borderData?.target_state || 'Target'} Average: ${value.toFixed(1)}%`;
              }
            } else if (chartType === 'district_performance') {
              if (datasetLabel === 'NFHS-4 (2016)') {
                return `Base value: ${value.toFixed(2)}`;
              } else {
                return `Change: ${value > 0 ? '+' : ''}${value.toFixed(2)}`;
              }
            }
            
            return `${datasetLabel}: ${value.toFixed(2)}`;
          },
          afterLabel: function(context) {
            if (chartType === 'state_comparison') {
              const dataIndex = context.dataIndex;
              const states = Object.keys(processedData?.stateGroups || {});
              const currentState = states[dataIndex];
              if (currentState && processedData?.stateGroups?.[currentState]) {
                const districts = processedData.stateGroups[currentState];
                return `Districts in analysis: ${districts.length}`;
              }
            }
            return '';
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: chartType === 'state_comparison' ? 'States' : 'Districts (sorted by shared boundary length)',
          font: {
            size: 12,
            weight: 'bold'
          }
        }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: selectedIndicator ? `${selectedIndicator.indicator_name} Value` : 'Indicator Value',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        stacked: true // Enable stacking for the Y axis
      },
    },
  };

  // Check if we have any usable data
  const hasUsableData = () => {
    if (!borderData) return false;
    
    // Check for direct data
    if (borderData.data && Array.isArray(borderData.data)) return true;
    
    // Check for nested data structure
    if (borderData.districts && Array.isArray(borderData.districts) && borderData.districts[0]?.result?.data) return true;
    
    return false;
  };

  // Early return if no data
  if (!hasUsableData()) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
        <h3>No Border Districts Data</h3>
        <p>No border districts data available for visualization.</p>
        <details style={{ marginTop: '20px', textAlign: 'left', background: '#f5f5f5', padding: '10px', borderRadius: '5px' }}>
          <summary>Debug Info</summary>
          <pre>{JSON.stringify(borderData, null, 2)}</pre>
        </details>
      </div>
    );
  }

  // Early return if no processed data
  if (!processedData) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
        <h3>Processing Data...</h3>
        <p>Processing border districts data...</p>
        <p>Districts count: {borderData?.data?.length || 0}</p>
      </div>
    );
  }

  // Early return if no indicators found
  if (!selectedIndicator) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
        <h3>No Indicators Found</h3>
        <p>No health indicators found in the border districts data.</p>
        <p>Total districts: {processedData?.totalDistricts || 0}</p>
        <p>Total indicators: {processedData?.totalIndicators || 0}</p>
        <details style={{ marginTop: '20px', textAlign: 'left', background: '#f5f5f5', padding: '10px', borderRadius: '5px' }}>
          <summary>Debug Info</summary>
          <pre>{JSON.stringify(processedData, null, 2)}</pre>
        </details>
      </div>
    );
  }

  if (mapOnly) {
    return (
      <div style={{ width: '100%', height: '100%' }}>
        <BorderDistrictsMap data={borderData} />
      </div>
    );
  }

  if (chartOnly) {
    const chartData = chartType === 'state_comparison' ? 
      generateStateComparisonChart() : 
      chartType === 'state_performance_summary' ?
      generateStatePerformanceSummaryChart() :
      generateDistrictPerformanceChart();

    return (
      <div style={{ width: '100%', height: '100%', padding: '20px' }}>
        {/* Indicator Selection */}
        <div style={{ 
          background: '#f8f9fa', 
          padding: '20px', 
          borderRadius: '12px', 
          marginBottom: '24px',
          border: '1px solid #dee2e6'
        }}>
          <h4 style={{ margin: '0 0 16px 0', color: '#495057' }}>Analysis Controls</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#495057' }}>
                Select Indicator:
              </label>
              <select
                value={selectedIndicator?.indicator_name || ''}
                onChange={(e) => {
                  const indicator = indicators.find(ind => 
                    ind.indicator_name === e.target.value
                  );
                  setSelectedIndicator(indicator);
                }}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  border: '1px solid #ced4da',
                  fontSize: '14px'
                }}
              >
                {indicators.map(indicator => (
                  <option key={indicator.indicator_name} value={indicator.indicator_name}>
                    {indicator.indicator_name} (IPI {indicator.sdg_goal})
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#495057' }}>
                Chart Type:
              </label>
              <select
                value={chartType}
                onChange={(e) => setChartType(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  border: '1px solid #ced4da',
                  fontSize: '14px'
                }}
              >
                <option value="state_comparison">Border Districts vs Target State</option>
                <option value="district_performance">District Performance</option>
                <option value="state_performance_summary">Performance vs Target State</option>
              </select>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div style={{ height: '400px', width: '100%' }}>
          {!chartData ? (
            <div style={{ 
              height: '100%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              color: '#666',
              backgroundColor: '#f8f9fa',
              borderRadius: '8px'
            }}>
              <p>Loading chart data...</p>
            </div>
          ) : (
            <Bar 
              data={chartData} 
              options={chartOptions}
            />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="border-analysis-container" style={{ padding: '20px' }}>
      {/* Summary Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
        gap: '16px', 
        marginBottom: '32px' 
      }}>
        <div style={{ 
          background: 'linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%)',
          padding: '20px', 
          borderRadius: '12px', 
          border: '1px solid #bee5eb',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#0c5460' }}>
            {processedData.totalDistricts}
          </div>
          <div style={{ fontSize: '14px', color: '#0c5460', marginTop: '4px' }}>
            Border Districts
          </div>
        </div>
        
        <div style={{ 
          background: 'linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%)',
          padding: '20px', 
          borderRadius: '12px', 
          border: '1px solid #ce93d8',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#4a148c' }}>
            {Object.keys(processedData.stateGroups).length}
          </div>
          <div style={{ fontSize: '14px', color: '#4a148c', marginTop: '4px' }}>
            States Involved
          </div>
        </div>
        
        <div style={{ 
          background: 'linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)',
          padding: '20px', 
          borderRadius: '12px', 
          border: '1px solid #a5d6a7',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1b5e20' }}>
            {processedData.totalIndicators}
          </div>
          <div style={{ fontSize: '14px', color: '#1b5e20', marginTop: '4px' }}>
            IPI Indicators
          </div>
        </div>
      </div>

      {/* Map */}
      <div style={{ marginBottom: '32px' }}>
        <h3 style={{ marginBottom: '16px', color: '#495057' }}>🗺️ Border Districts Map</h3>
        <div style={{ height: '500px', width: '100%', borderRadius: '8px', overflow: 'hidden' }}>
          <BorderDistrictsMap data={borderData} />
        </div>
      </div>
    </div>
  );
};


export default BorderDistrictsAnalysis; 
