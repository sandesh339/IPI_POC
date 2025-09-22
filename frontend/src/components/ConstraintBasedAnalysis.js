import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Bar } from 'react-chartjs-2';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { registerMapInstance, initializeMapForCapture } from '../utils/saveUtils';

// Import Chart.js components
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
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

// Color schemes for constraint analysis
const CONSTRAINT_COLORS = {
  district_fill: '#2ecc71',        // Green for districts meeting constraints
  district_stroke: '#ffffff',     // White stroke for boundaries
  district_hover: '#000000',      // Black for hover effect
  state_colors: [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", 
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    "#FF5722", "#8BC34A", "#607D8B", "#795548", "#FF9800"
  ]
};

// Map Component for Constraint-Based Analysis
const ConstraintBasedMap = ({ data }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (!data || !data.boundary || map.current) return;

    console.log('🗺️ Initializing constraint-based analysis map with data:', data);
    console.log('🔍 Map container element:', mapContainer.current);
    console.log('🔍 Mapbox token set:', !!mapboxgl.accessToken);
    console.log('🔍 Mapbox token value:', mapboxgl.accessToken?.substring(0, 10) + '...');
    console.log('🔍 Boundary data length:', data.boundary?.length);

    // Verify mapbox token
    if (!mapboxgl.accessToken) {
      console.error('❌ Mapbox access token not set!');
      return;
    }

    // Add a small delay to ensure DOM is ready
    const initializeMap = () => {
      // Check if container is available
      if (!mapContainer.current) {
        console.error('❌ Map container not found');
        return;
      }

      try {
        // Initialize map
        console.log('📍 Creating new mapbox map...');
        map.current = new mapboxgl.Map({
          container: mapContainer.current,
          style: 'mapbox://styles/mapbox/light-v11',
          center: [78.9629, 20.5937], // Center of India
          zoom: 5,
          preserveDrawingBuffer: true // Essential for WebGL canvas capture
        });
        
        console.log('✅ Mapbox map created successfully');

        // Add error handlers
        map.current.on('error', (e) => {
          console.error('❌ Mapbox map error:', e);
        });

        map.current.on('style.load', () => {
          console.log('🎨 Map style loaded successfully');
        });
      } catch (error) {
        console.error('❌ Error creating mapbox map:', error);
        return;
      }

      // Register the map instance for save functionality
      registerMapInstance('constraint-based-map', map.current);

      map.current.on('load', () => {
      console.log('Constraint map loaded, processing boundary data...');

      // Process boundary data
      let boundaryData = data.boundary || data.boundary_data || [];
      
      if (!Array.isArray(boundaryData) || boundaryData.length === 0) {
        console.log('No boundary data available for constraint analysis');
        return;
      }

      console.log(`Processing ${boundaryData.length} boundary features`);

      // Create state colors mapping
      const allStates = [...new Set(boundaryData.map(b => b.state_name || b.state).filter(Boolean))];
      const stateColors = {};
      allStates.forEach((state, index) => {
        stateColors[state] = CONSTRAINT_COLORS.state_colors[index % CONSTRAINT_COLORS.state_colors.length];
      });

      // Prepare GeoJSON for districts
      const districtsGeoJSON = {
        type: 'FeatureCollection',
        features: boundaryData
          .map((boundary, index) => {
            let parsedGeometry;
            try {
              parsedGeometry = typeof boundary.geometry === 'string'
                ? JSON.parse(boundary.geometry)
                : boundary.geometry;
            } catch (e) {
              console.warn('Failed to parse geometry for boundary:', boundary.district_name || boundary.district, e);
              return null;
            }

            if (!parsedGeometry || !parsedGeometry.type || !parsedGeometry.coordinates) {
              console.warn('Invalid geometry for boundary:', boundary.district_name || boundary.district);
              return null;
            }

            return {
              type: 'Feature',
              id: index,
              properties: {
                district_name: boundary.district_name || boundary.district || 'Unknown District',
                state_name: boundary.state_name || boundary.state || 'Unknown State',
                state_color: stateColors[boundary.state_name || boundary.state] || '#cccccc'
              },
              geometry: parsedGeometry
            };
          })
          .filter(Boolean)
      };

      console.log('District GeoJSON created with features:', districtsGeoJSON.features.length);

      // Add districts source
      map.current.addSource('constraint-districts', {
        type: 'geojson',
        data: districtsGeoJSON
      });

      // Add district fill layer with state colors
      map.current.addLayer({
        id: 'constraint-districts-fill',
        type: 'fill',
        source: 'constraint-districts',
        paint: {
          'fill-color': [
            'match',
            ['get', 'state_name'],
            ...allStates.reduce((acc, state) => [...acc, state, stateColors[state]], []),
            '#cccccc' // Default color for unknown states
          ],
          'fill-opacity': 0.7
        }
      });

      // Add district stroke layer
      map.current.addLayer({
        id: 'constraint-districts-stroke',
        type: 'line',
        source: 'constraint-districts',
        paint: {
          'line-color': CONSTRAINT_COLORS.district_stroke,
          'line-width': 2
        }
      });

      // Add hover effect
      map.current.addLayer({
        id: 'constraint-districts-hover',
        type: 'fill',
        source: 'constraint-districts',
        paint: {
          'fill-color': CONSTRAINT_COLORS.district_hover,
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
      map.current.on('mousemove', 'constraint-districts-fill', (e) => {
        if (e.features.length > 0) {
          if (hoveredStateId !== null) {
            map.current.setFeatureState(
              { source: 'constraint-districts', id: hoveredStateId },
              { hover: false }
            );
          }
          hoveredStateId = e.features[0].id;
          map.current.setFeatureState(
            { source: 'constraint-districts', id: hoveredStateId },
            { hover: true }
          );
        }
      });

      map.current.on('mouseleave', 'constraint-districts-fill', () => {
        if (hoveredStateId !== null) {
          map.current.setFeatureState(
            { source: 'constraint-districts', id: hoveredStateId },
            { hover: false }
          );
        }
        hoveredStateId = null;
      });

      // Add popup on click
      map.current.on('click', 'constraint-districts-fill', (e) => {
        const properties = e.features[0].properties;
        const districtName = properties.district_name;
        
        // Find the district data from the original data array
        const districtData = data.districts?.find(d => d.district_name === districtName);
        
        console.log('=== CONSTRAINT POPUP CLICK DEBUG ===');
        console.log('Clicked district:', districtName);
        console.log('Found district data:', districtData);

        // Create popup content
        let popupContent = `
          <div style="
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 400px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            color: white;
          ">
            <div style="
              padding: 20px;
              background: rgba(255,255,255,0.1);
              backdrop-filter: blur(10px);
            ">
              <h3 style="
                margin: 0 0 8px 0; 
                font-size: 18px; 
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
              ">
                📍 ${districtName}
              </h3>
              <p style="
                margin: 0; 
                font-size: 14px; 
                color: rgba(255,255,255,0.9);
                font-weight: 500;
              ">
                State: ${properties.state_name}
              </p>
            </div>
            
            <div style="padding: 20px;">
        `;

        if (districtData && districtData.indicators && districtData.indicators.length > 0) {
          // Show constraint indicator values
          const constraintIndicators = data.constraints_applied || [];
          
          if (constraintIndicators.length > 0) {
            popupContent += `
              <div style="margin-bottom: 16px;">
                <h4 style="
                  margin: 0 0 12px 0; 
                  font-size: 14px; 
                  color: rgba(255,255,255,0.9); 
                  font-weight: 600;
                  text-transform: uppercase;
                  letter-spacing: 0.5px;
                ">
                  🎯 Constraint Values
                </h4>
                <div style="display: flex; flex-direction: column; gap: 8px;">
            `;

            constraintIndicators.forEach(constraint => {
              // Find the matching indicator data
              const indicatorData = districtData.indicators.find(ind => 
                ind.indicator_id === constraint.indicator_id || 
                ind.indicator_name === constraint.indicator_name
              );

              if (indicatorData) {
                const value = indicatorData.prevalence_2021 || indicatorData.prevalence_2016 || 0;
                const direction = constraint.indicator_direction === 'higher_is_better' ? '↑' : '↓';
                const directionColor = constraint.indicator_direction === 'higher_is_better' ? '#4ade80' : '#f87171';

                popupContent += `
                  <div style="
                    background: rgba(255,255,255,0.1); 
                    padding: 12px; 
                    border-radius: 8px; 
                    border-left: 4px solid ${directionColor};
                  ">
                    <div style="
                      display: flex; 
                      justify-content: space-between; 
                      align-items: center; 
                      margin-bottom: 4px;
                    ">
                      <span style="font-weight: 600; font-size: 13px;">
                        ${constraint.indicator_name}
                      </span>
                      <span style="
                        background: ${directionColor}; 
                        color: white; 
                        padding: 2px 6px; 
                        border-radius: 12px; 
                        font-size: 10px; 
                        font-weight: bold;
                      ">
                        ${direction}
                      </span>
                    </div>
                    <div style="
                      display: flex; 
                      justify-content: space-between; 
                      align-items: center;
                    ">
                      <span style="
                        color: rgba(255,255,255,0.8); 
                        font-size: 11px;
                      ">
                        Constraint: ${constraint.operator} ${constraint.value}
                      </span>
                      <span style="
                        font-size: 16px; 
                        font-weight: bold; 
                        color: #fbbf24;
                      ">
                        ${value.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                `;
              }
            });

            popupContent += `
                </div>
              </div>
            `;
          }

          // Show additional indicators if available
          const otherIndicators = districtData.indicators.filter(ind => 
            !constraintIndicators.some(c => c.indicator_id === ind.indicator_id)
          ).slice(0, 3); // Show up to 3 additional indicators

          if (otherIndicators.length > 0) {
            popupContent += `
              <div>
                <h4 style="
                  margin: 0 0 12px 0; 
                  font-size: 14px; 
                  color: rgba(255,255,255,0.9); 
                  font-weight: 600;
                  text-transform: uppercase;
                  letter-spacing: 0.5px;
                ">
                  📊 Other Indicators
                </h4>
                <div style="display: flex; flex-direction: column; gap: 6px;">
            `;

            otherIndicators.forEach(indicator => {
              const value = indicator.prevalence_2021 || indicator.prevalence_2016 || 0;
              const direction = indicator.indicator_direction === 'higher_is_better' ? '↑' : '↓';
              const directionColor = indicator.indicator_direction === 'higher_is_better' ? '#4ade80' : '#f87171';

              popupContent += `
                <div style="
                  background: rgba(255,255,255,0.05); 
                  padding: 8px 12px; 
                  border-radius: 6px;
                  display: flex;
                  justify-content: space-between;
                  align-items: center;
                ">
                  <span style="
                    font-size: 12px; 
                    color: rgba(255,255,255,0.9);
                    max-width: 200px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                  ">
                    ${indicator.indicator_name}
                  </span>
                  <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="
                      color: ${directionColor}; 
                      font-size: 12px;
                      font-weight: bold;
                    ">
                      ${direction}
                    </span>
                    <span style="
                      font-size: 14px; 
                      font-weight: bold; 
                      color: #fbbf24;
                    ">
                      ${value.toFixed(1)}%
                    </span>
                  </div>
                </div>
              `;
            });

            popupContent += `
                </div>
              </div>
            `;
          }
        } else {
          popupContent += `
            <div style="
              color: rgba(255,255,255,0.8); 
              text-align: center; 
              padding: 20px;
              display: flex;
              flex-direction: column;
              align-items: center;
              gap: 8px;
            ">
              <div style="font-size: 32px; opacity: 0.6;">📊</div>
              <div style="font-size: 14px; font-weight: 600;">No Data Available</div>
              <div style="font-size: 12px; opacity: 0.8; max-width: 200px; line-height: 1.4;">
                This district doesn't meet the specified constraints or has no health data available.
              </div>
            </div>
          `;
        }

        popupContent += `
            </div>
          </div>
        `;

        // Create popup
        const popup = new mapboxgl.Popup({
          closeButton: true,
          closeOnClick: false,
          maxWidth: '420px',
          className: 'constraint-analysis-popup',
          anchor: 'top-left',
          offset: [15, 15]
        })
          .setLngLat(e.lngLat)
          .setHTML(popupContent)
          .addTo(map.current);

        // Style the close button
        const popupElement = popup.getElement();
        if (popupElement) {
          const closeButton = popupElement.querySelector('.mapboxgl-popup-close-button');
          if (closeButton) {
            closeButton.style.cssText = `
              background: rgba(0,0,0,0.8) !important;
              color: white !important;
              border-radius: 50% !important;
              width: 24px !important;
              height: 24px !important;
              font-size: 16px !important;
              font-weight: bold !important;
              border: 2px solid white !important;
              box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
              right: 8px !important;
              top: 8px !important;
              display: flex !important;
              align-items: center !important;
              justify-content: center !important;
              z-index: 1000 !important;
            `;
          }
        }
      });

      // Change cursor on hover
      map.current.on('mouseenter', 'constraint-districts-fill', () => {
        map.current.getCanvas().style.cursor = 'pointer';
      });

      map.current.on('mouseleave', 'constraint-districts-fill', () => {
        map.current.getCanvas().style.cursor = '';
      });

      // Initialize for capture after all layers are added
      setTimeout(() => {
        initializeMapForCapture(map.current, 'constraint-based-map');
      }, 1000);
    });
    };

    // Use a small delay to ensure DOM is ready
    const timer = setTimeout(initializeMap, 100);

    return () => {
      clearTimeout(timer);
      if (map.current) {
        map.current.remove();
      }
    };
  }, [data]);

  return (
    <div 
      ref={mapContainer} 
      style={{ 
        width: '100%', 
        height: '100%', 
        borderRadius: '8px', 
        overflow: 'hidden' 
      }} 
    />
  );
};

const ConstraintBasedAnalysis = ({ constraintData, mapOnly = false, chartOnly = false }) => {
  const [selectedDistrict, setSelectedDistrict] = useState(null);

  console.log('🎯 ConstraintBasedAnalysis - Input data:', constraintData);

  // Extract and process the data
  const processedData = useMemo(() => {
    if (!constraintData) {
      console.log('❌ No constraintData provided');
      return { districts: [], constraints: [], boundaryData: [], chartData: null };
    }

    // Extract districts data
    const districts = constraintData.districts || [];
    console.log('📍 Districts found:', districts.length);

    // Extract constraints applied
    const constraints = constraintData.constraints_applied || [];
    console.log('🔍 Constraints applied:', constraints.length);

    // Extract boundary data for mapping
    const boundaryData = constraintData.boundary || [];
    console.log('🗺️ Boundary data:', boundaryData.length);

    // Extract chart data
    const chartData = constraintData.chart_data || null;
    console.log('📊 Chart data available:', !!chartData);

    // Process districts for easier access
    const processedDistricts = districts.map(district => {
      const constraintValues = {};
      const indicatorDetails = {};

      // Extract constraint values for this district
      constraints.forEach(constraint => {
        const indicatorId = constraint.indicator_id;
        const value = district.constraint_values?.[indicatorId];
        if (value !== undefined) {
          constraintValues[constraint.indicator_name] = value;
          indicatorDetails[constraint.indicator_name] = {
            direction: constraint.indicator_direction,
            operator: constraint.operator,
            threshold: constraint.value
          };
        }
      });

      return {
        ...district,
        constraintValues,
        indicatorDetails
      };
    });

    return {
      districts: processedDistricts,
      constraints,
      boundaryData,
      chartData,
      totalFound: constraintData.total_districts_found || districts.length,
      year: constraintData.year || 2021,
      summaryStats: constraintData.summary_stats || {}
    };
  }, [constraintData]);



  // Prepare chart data from backend chart_data or generate from districts
  const chartOptions = useMemo(() => {
    if (!processedData.chartData) {
      return null;
    }

    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 20,
            font: {
              size: 12,
              weight: '600'
            }
          }
        },
        title: {
          display: true,
          text: processedData.chartData.title || 'Constraint-Based District Analysis',
          font: {
            size: 16,
            weight: 'bold'
          },
          color: '#2E7D32',
          padding: {
            bottom: 20
          }
        },
        tooltip: {
          backgroundColor: 'rgba(46, 125, 50, 0.9)',
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: '#4CAF50',
          borderWidth: 1,
          cornerRadius: 8,
          displayColors: true,
          callbacks: {
            label: function(context) {
              const value = context.parsed.y;
              const indicator = context.dataset.label;
              return `${indicator}: ${value.toFixed(2)}%`;
            },
            title: function(context) {
              return context[0].label;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: 'rgba(46, 125, 50, 0.1)'
          },
          ticks: {
            color: '#2E7D32',
            font: {
              size: 11
            },
            callback: function(value) {
              return value + '%';
            }
          },
          title: {
            display: true,
            text: 'Indicator Values (%)',
            color: '#2E7D32',
            font: {
              size: 12,
              weight: '600'
            }
          }
        },
        x: {
          grid: {
            display: false
          },
          ticks: {
            color: '#2E7D32',
            font: {
              size: 10
            },
            maxRotation: 45,
            minRotation: 0
          },
          title: {
            display: true,
            text: 'Districts',
            color: '#2E7D32',
            font: {
              size: 12,
              weight: '600'
            }
          }
        }
      },
      interaction: {
        intersect: false,
        mode: 'index'
      }
    };
  }, [processedData.chartData]);

  if (chartOnly && processedData.chartData) {
    return (
      <div style={{ width: '100%', height: '500px', padding: '20px' }}>
        <div style={{ 
          marginBottom: '20px', 
          padding: '16px', 
          backgroundColor: '#f8f9fa', 
          borderRadius: '8px',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ margin: '0 0 8px 0', color: '#2E7D32' }}>
            📊 Constraint-Based District Analysis
          </h3>
          <p style={{ margin: '0', color: '#666', fontSize: '14px' }}>
            
            {processedData.totalFound > processedData.districts.length && 
              ` (displaying ${processedData.districts.length} of ${processedData.totalFound} total)`
            }
          </p>
          <div style={{ marginTop: '8px' }}>
            {processedData.constraints.map((constraint, index) => (
              <span 
                key={index}
                style={{ 
                  display: 'inline-block',
                  margin: '2px 4px',
                  padding: '4px 8px',
                  backgroundColor: '#e8f5e8',
                  border: '1px solid #4CAF50',
                  borderRadius: '4px',
                  fontSize: '12px',
                  color: '#2E7D32'
                }}
              >
                {constraint.indicator_name} {constraint.operator} {constraint.value}
              </span>
            ))}
          </div>
        </div>
        <div style={{ height: 'calc(100% - 120px)' }}>
          <Bar data={processedData.chartData} options={chartOptions} />
        </div>
      </div>
    );
  }

  if (mapOnly) {
    return (
      <div style={{ width: '100%', height: '700px' }}>
        <div style={{ 
          marginBottom: '10px', 
          padding: '12px', 
          backgroundColor: '#f8f9fa', 
          borderRadius: '8px',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ margin: '0 0 4px 0', color: '#2E7D32', fontSize: '16px' }}>
            🗺️ Districts Meeting Constraints
          </h3>
          <p style={{ margin: '0', color: '#666', fontSize: '13px' }}>
            {processedData.districts.length} districts found • Click districts for details
          </p>
        </div>
        
        <div style={{ height: 'calc(100% - 70px)', borderRadius: '8px', overflow: 'hidden' }}>
          <ConstraintBasedMap data={{
            boundary: processedData.boundaryData,
            districts: processedData.districts,
            constraints_applied: processedData.constraints
          }} />
        </div>
      </div>
    );
  }

  // Full component with both map and chart
  return (
    <div style={{ width: '100%', padding: '20px' }}>
      {/* Header */}
      <div style={{ 
        marginBottom: '20px', 
        padding: '20px', 
        backgroundColor: '#f8f9fa', 
        borderRadius: '12px',
        border: '1px solid #e9ecef'
      }}>
        <h2 style={{ margin: '0 0 12px 0', color: '#2E7D32' }}>
          🎯 Constraint-Based District Analysis
        </h2>
        <p style={{ margin: '0 0 12px 0', color: '#666', fontSize: '16px' }}>
          Found <strong>{processedData.totalFound}</strong> districts meeting all specified constraints
        </p>
        
        {/* Constraints Summary */}
        <div style={{ marginTop: '12px' }}>
          <strong style={{ color: '#2E7D32', marginBottom: '8px', display: 'block' }}>
            Applied Constraints:
          </strong>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {processedData.constraints.map((constraint, index) => (
              <span 
                key={index}
                style={{ 
                  padding: '6px 12px',
                  backgroundColor: '#e8f5e8',
                  border: '1px solid #4CAF50',
                  borderRadius: '6px',
                  fontSize: '14px',
                  color: '#2E7D32',
                  fontWeight: '500'
                }}
              >
                {constraint.indicator_name} {constraint.operator} {constraint.value}
                <span style={{ fontSize: '12px', opacity: 0.8, marginLeft: '4px' }}>
                  ({constraint.indicator_direction === 'higher_is_better' ? '↑' : '↓'})
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Map Section */}
      <div style={{ marginBottom: '30px' }}>
        <h3 style={{ color: '#2E7D32', marginBottom: '15px' }}>
          🗺️ Geographic Distribution
        </h3>
        <div style={{ height: '600px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #e9ecef', position: 'relative' }}>
          <ConstraintBasedMap data={{
            boundary: processedData.boundaryData,
            districts: processedData.districts,
            constraints_applied: processedData.constraints
          }} />
        </div>
      </div>

      {/* Chart Section */}
      {processedData.chartData && (
        <div>
          <h3 style={{ color: '#2E7D32', marginBottom: '15px' }}>
            📊 Indicator Values Comparison
          </h3>
          <div style={{ 
            height: '500px', 
            padding: '20px', 
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #e9ecef'
          }}>
            <Bar data={processedData.chartData} options={chartOptions} />
          </div>
        </div>
      )}

      {/* Selected District Details */}
      {selectedDistrict && (
        <div style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          width: '300px',
          backgroundColor: '#ffffff',
          padding: '20px',
          borderRadius: '12px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
          border: '2px solid #4CAF50',
          zIndex: 2000
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
            <h4 style={{ margin: '0', color: '#2E7D32' }}>
              {selectedDistrict.district_name}
            </h4>
            <button
              onClick={() => setSelectedDistrict(null)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '18px',
                cursor: 'pointer',
                color: '#666'
              }}
            >
              ×
            </button>
          </div>
          <p style={{ margin: '0 0 12px 0', color: '#666' }}>
            📍 {selectedDistrict.state_name}
          </p>
          <div>
            <strong style={{ color: '#2E7D32', display: 'block', marginBottom: '8px' }}>
              Constraint Values:
            </strong>
            {Object.entries(selectedDistrict.constraintValues).map(([indicator, value]) => {
              const details = selectedDistrict.indicatorDetails[indicator];
              return (
                <div key={indicator} style={{ 
                  margin: '6px 0', 
                  padding: '8px', 
                  backgroundColor: '#f8f9fa', 
                  borderRadius: '6px',
                  border: '1px solid #e9ecef'
                }}>
                  <div style={{ fontWeight: '600', fontSize: '14px', color: '#2E7D32' }}>
                    {indicator} {details?.direction === 'higher_is_better' ? '↑' : '↓'}
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: '#2E7D32', margin: '2px 0' }}>
                    {value?.toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    Constraint: {details?.operator} {details?.threshold}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ConstraintBasedAnalysis;
