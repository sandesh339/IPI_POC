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

// Color schemes for top/bottom analysis
const PERFORMANCE_COLORS = {
  top_fill: '#2ecc71',           // Green for top performers
  bottom_fill: '#e74c3c',        // Red for bottom performers
  mixed_fill: '#f39c12',         // Orange for mixed/both
  district_stroke: '#ffffff',    // White stroke for boundaries
  district_hover: '#000000',     // Black for hover effect
  state_colors: [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", 
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    "#FF5722", "#8BC34A", "#607D8B", "#795548", "#FF9800"
  ]
};

// Map Component for Top/Bottom Districts Analysis
const TopBottomMap = ({ data }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (!data || !data.boundary || map.current) return;

    console.log('🗺️ Initializing top/bottom districts map with data:', data);
    console.log('🔍 Map container element:', mapContainer.current);
    console.log('🔍 Boundary data length:', data.boundary?.length);

    // Verify mapbox token
    if (!mapboxgl.accessToken) {
      console.error('❌ Mapbox access token not set!');
      return;
    }

    // Add a small delay to ensure DOM is ready
    const initializeMap = () => {
      if (!mapContainer.current) {
        console.error('❌ Map container not found');
        return;
      }

      try {
        console.log('📍 Creating new mapbox map...');
        map.current = new mapboxgl.Map({
          container: mapContainer.current,
          style: 'mapbox://styles/mapbox/light-v11',
          center: [78.9629, 20.5937], // Center of India
          zoom: 5,
          preserveDrawingBuffer: true
        });
        
        console.log('✅ Mapbox map created successfully');

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
      registerMapInstance('top-bottom-map', map.current);

      map.current.on('load', () => {
        console.log('Top/bottom map loaded, processing boundary data...');

        let boundaryData = data.boundary || data.boundary_data || [];
        
        if (!Array.isArray(boundaryData) || boundaryData.length === 0) {
          console.log('No boundary data available for top/bottom analysis');
          return;
        }

        console.log(`Processing ${boundaryData.length} boundary features`);

        // Determine colors based on performance type
        const performanceType = data.performance_type || 'top';
        let fillColor = PERFORMANCE_COLORS.top_fill;
        
        if (performanceType === 'bottom') {
          fillColor = PERFORMANCE_COLORS.bottom_fill;
        } else if (performanceType === 'both') {
          fillColor = PERFORMANCE_COLORS.mixed_fill;
        }

        // Create state colors mapping for additional visual distinction
        const allStates = [...new Set(boundaryData.map(b => b.state_name || b.state).filter(Boolean))];
        const stateColors = {};
        allStates.forEach((state, index) => {
          stateColors[state] = PERFORMANCE_COLORS.state_colors[index % PERFORMANCE_COLORS.state_colors.length];
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

              // Find corresponding district rank and performance
              const districtName = boundary.district_name || boundary.district;
              const districtData = data.districts?.find(d => d.district_name === districtName);
              
              return {
                type: 'Feature',
                id: index,
                properties: {
                  district_name: districtName || 'Unknown District',
                  state_name: boundary.state_name || boundary.state || 'Unknown State',
                  state_color: stateColors[boundary.state_name || boundary.state] || '#cccccc',
                  rank: districtData?.rank || index + 1,
                  performance_score: districtData?.performance_score || 0
                },
                geometry: parsedGeometry
              };
            })
            .filter(Boolean)
        };

        console.log('District GeoJSON created with features:', districtsGeoJSON.features.length);

        // Add districts source
        map.current.addSource('top-bottom-districts', {
          type: 'geojson',
          data: districtsGeoJSON
        });

        // Add district fill layer with performance-based colors
        map.current.addLayer({
          id: 'top-bottom-districts-fill',
          type: 'fill',
          source: 'top-bottom-districts',
          paint: {
            'fill-color': fillColor,
            'fill-opacity': 0.7
          }
        });

        // Add district stroke layer
        map.current.addLayer({
          id: 'top-bottom-districts-stroke',
          type: 'line',
          source: 'top-bottom-districts',
          paint: {
            'line-color': PERFORMANCE_COLORS.district_stroke,
            'line-width': 2
          }
        });

        // Add hover effect
        map.current.addLayer({
          id: 'top-bottom-districts-hover',
          type: 'fill',
          source: 'top-bottom-districts',
          paint: {
            'fill-color': PERFORMANCE_COLORS.district_hover,
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
        map.current.on('mousemove', 'top-bottom-districts-fill', (e) => {
          if (e.features.length > 0) {
            if (hoveredStateId !== null) {
              map.current.setFeatureState(
                { source: 'top-bottom-districts', id: hoveredStateId },
                { hover: false }
              );
            }
            hoveredStateId = e.features[0].id;
            map.current.setFeatureState(
              { source: 'top-bottom-districts', id: hoveredStateId },
              { hover: true }
            );
          }
        });

        map.current.on('mouseleave', 'top-bottom-districts-fill', () => {
          if (hoveredStateId !== null) {
            map.current.setFeatureState(
              { source: 'top-bottom-districts', id: hoveredStateId },
              { hover: false }
            );
          }
          hoveredStateId = null;
        });

        // Add popup on click
        map.current.on('click', 'top-bottom-districts-fill', (e) => {
          const properties = e.features[0].properties;
          const districtName = properties.district_name;
          
          // Find the district data from the original data array
          const districtData = data.districts?.find(d => d.district_name === districtName);
          
          console.log('=== TOP/BOTTOM POPUP CLICK DEBUG ===');
          console.log('Clicked district:', districtName);
          console.log('Found district data:', districtData);

          // Create popup content
          let popupContent = `
            <div style="
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
              max-width: 400px;
              background: linear-gradient(135deg, ${fillColor}dd 0%, ${fillColor}aa 100%);
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
                ${districtData ? `
                  <div style="
                    margin-top: 8px; 
                    padding: 8px 12px; 
                    background: rgba(255,255,255,0.2); 
                    border-radius: 6px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                  ">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.9);">Rank</span>
                    <span style="font-size: 16px; font-weight: bold; color: #fbbf24;">#${districtData.rank}</span>
                  </div>
                ` : ''}
              </div>
              
              <div style="padding: 20px;">
          `;

          if (districtData && districtData.indicators && districtData.indicators.length > 0) {
            // Show indicator values
            const indicators = data.indicators || [];
            
            if (indicators.length > 0) {
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
                    📊 Performance Indicators
                  </h4>
                  <div style="display: flex; flex-direction: column; gap: 8px;">
              `;

              districtData.indicators.forEach(indicator => {
                const value = indicator.indicator_value || indicator.prevalence_2021 || indicator.prevalence_2016 || 0;
                const direction = indicator.indicator_direction === 'higher_is_better' ? '↑' : '↓';
                const directionColor = indicator.indicator_direction === 'higher_is_better' ? '#4ade80' : '#f87171';

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
                        ${indicator.indicator_name}
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
                        ${data.performance_type === 'top' ? 'Top' : data.performance_type === 'bottom' ? 'Bottom' : 'Performance'} Rank: #${districtData.rank}
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
              });

              popupContent += `
                  </div>
                </div>
              `;
            }

            // Show performance score if available
            if (districtData.performance_score !== undefined) {
              popupContent += `
                <div>
                  <h4 style="
                    margin: 0 0 8px 0; 
                    font-size: 14px; 
                    color: rgba(255,255,255,0.9); 
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                  ">
                    🏆 Performance Score
                  </h4>
                  <div style="
                    background: rgba(255,255,255,0.1); 
                    padding: 12px; 
                    border-radius: 8px;
                    text-align: center;
                  ">
                    <div style="
                      font-size: 24px; 
                      font-weight: bold; 
                      color: #fbbf24;
                    ">
                      ${districtData.performance_score.toFixed(2)}
                    </div>
                    <div style="
                      font-size: 12px; 
                      color: rgba(255,255,255,0.8);
                      margin-top: 4px;
                    ">
                      ${data.response_type === 'multi_indicator' ? 'Composite Score' : 'Indicator Value'}
                    </div>
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
                  This district doesn't have performance data available for the analyzed indicators.
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
            className: 'top-bottom-analysis-popup',
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
        map.current.on('mouseenter', 'top-bottom-districts-fill', () => {
          map.current.getCanvas().style.cursor = 'pointer';
        });

        map.current.on('mouseleave', 'top-bottom-districts-fill', () => {
          map.current.getCanvas().style.cursor = '';
        });

        // Initialize for capture after all layers are added
        setTimeout(() => {
          initializeMapForCapture(map.current, 'top-bottom-map');
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

const TopBottomAnalysis = ({ topBottomData, mapOnly = false, chartOnly = false }) => {
  const [selectedDistrict, setSelectedDistrict] = useState(null);

  console.log('🏆 TopBottomAnalysis - Input data:', topBottomData);

  // Extract and process the data
  const processedData = useMemo(() => {
    if (!topBottomData) {
      console.log('❌ No topBottomData provided');
      return { districts: [], indicators: [], boundaryData: [], chartData: null };
    }

    // Extract districts data
    const districts = topBottomData.districts || [];
    console.log('📍 Districts found:', districts.length);

    // Extract indicators
    const indicators = topBottomData.indicators || [];
    console.log('📊 Indicators:', indicators.length);

    // Extract boundary data for mapping
    const boundaryData = topBottomData.boundary || [];
    console.log('🗺️ Boundary data:', boundaryData.length);

    // Extract chart data
    const chartData = topBottomData.chart_data || null;
    console.log('📊 Chart data available:', !!chartData);

    return {
      districts,
      indicators,
      boundaryData,
      chartData,
      performanceType: topBottomData.performance_type || 'top',
      nDistricts: topBottomData.n_districts || 10,
      totalFound: topBottomData.total_districts_found || districts.length,
      year: topBottomData.year || 2021,
      responseType: topBottomData.response_type || 'single_indicator',
      mainIndicator: topBottomData.main_indicator || 'Unknown',
      statesFilter: topBottomData.states_filter || null
    };
  }, [topBottomData]);

  // Prepare chart options
  const chartOptions = useMemo(() => {
    if (!processedData.chartData) {
      return null;
    }

    const performanceColor = processedData.performanceType === 'top' 
      ? '#2ecc71' 
      : processedData.performanceType === 'bottom' 
        ? '#e74c3c' 
        : '#f39c12';

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
          text: processedData.chartData.title || `${processedData.performanceType.charAt(0).toUpperCase() + processedData.performanceType.slice(1)} Districts Analysis`,
          font: {
            size: 16,
            weight: 'bold'
          },
          color: performanceColor,
          padding: {
            bottom: 20
          }
        },
        tooltip: {
          backgroundColor: `${performanceColor}dd`,
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: performanceColor,
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
            color: `${performanceColor}33`
          },
          ticks: {
            color: performanceColor,
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
            color: performanceColor,
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
            color: performanceColor,
            font: {
              size: 10
            },
            maxRotation: 45,
            minRotation: 0
          },
          title: {
            display: true,
            text: 'Districts (Ranked by Performance)',
            color: performanceColor,
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
  }, [processedData.chartData, processedData.performanceType]);

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
          <h3 style={{ margin: '0 0 8px 0', color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c' }}>
            🏆 {processedData.performanceType.charAt(0).toUpperCase() + processedData.performanceType.slice(1)} Districts Analysis
          </h3>
          <p style={{ margin: '0', color: '#666', fontSize: '14px' }}>
            {processedData.nDistricts} {processedData.performanceType} performing districts for {processedData.mainIndicator}
            {processedData.totalFound > processedData.districts.length && 
              ` (displaying ${processedData.districts.length} of ${processedData.totalFound} total)`
            }
          </p>
          <div style={{ marginTop: '8px' }}>
            {processedData.indicators.map((indicator, index) => (
              <span 
                key={index}
                style={{ 
                  display: 'inline-block',
                  margin: '2px 4px',
                  padding: '4px 8px',
                  backgroundColor: processedData.performanceType === 'top' ? '#e8f5e8' : '#fdeaea',
                  border: `1px solid ${processedData.performanceType === 'top' ? '#4CAF50' : '#f87171'}`,
                  borderRadius: '4px',
                  fontSize: '12px',
                  color: processedData.performanceType === 'top' ? '#2E7D32' : '#dc2626'
                }}
              >
                {indicator}
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
          <h3 style={{ margin: '0 0 4px 0', color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c', fontSize: '16px' }}>
            🗺️ {processedData.performanceType.charAt(0).toUpperCase() + processedData.performanceType.slice(1)} Performing Districts
          </h3>
          <p style={{ margin: '0', color: '#666', fontSize: '13px' }}>
            {processedData.districts.length} districts found • Click districts for details
          </p>
        </div>
        
        <div style={{ height: 'calc(100% - 70px)', borderRadius: '8px', overflow: 'hidden' }}>
          <TopBottomMap data={{
            boundary: processedData.boundaryData,
            districts: processedData.districts,
            indicators: processedData.indicators,
            performance_type: processedData.performanceType,
            response_type: processedData.responseType
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
        <h2 style={{ margin: '0 0 12px 0', color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c' }}>
          🏆 {processedData.performanceType.charAt(0).toUpperCase() + processedData.performanceType.slice(1)} Districts Performance Analysis
        </h2>
        <p style={{ margin: '0 0 12px 0', color: '#666', fontSize: '16px' }}>
          Found <strong>{processedData.totalFound}</strong> {processedData.performanceType} performing districts for <strong>{processedData.mainIndicator}</strong>
        </p>
        
        {/* Analysis Details */}
        <div style={{ marginTop: '12px', display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ 
            padding: '8px 12px',
            backgroundColor: processedData.performanceType === 'top' ? '#e8f5e8' : '#fdeaea',
            border: `1px solid ${processedData.performanceType === 'top' ? '#4CAF50' : '#f87171'}`,
            borderRadius: '6px',
            fontSize: '14px',
            color: processedData.performanceType === 'top' ? '#2E7D32' : '#dc2626',
            fontWeight: '500'
          }}>
            📊 Analysis Type: {processedData.responseType === 'single_indicator' ? 'Single Indicator' : 'Multi-Indicator Composite'}
          </div>
          <div style={{ 
            padding: '8px 12px',
            backgroundColor: '#e3f2fd',
            border: '1px solid #2196F3',
            borderRadius: '6px',
            fontSize: '14px',
            color: '#1565C0',
            fontWeight: '500'
          }}>
            📅 Year: {processedData.year}
          </div>
          {processedData.statesFilter && (
            <div style={{ 
              padding: '8px 12px',
              backgroundColor: '#fff3e0',
              border: '1px solid #FF9800',
              borderRadius: '6px',
              fontSize: '14px',
              color: '#F57C00',
              fontWeight: '500'
            }}>
              🌍 States: {processedData.statesFilter.join(', ')}
            </div>
          )}
        </div>
        
        {/* Indicators Summary */}
        {processedData.indicators.length > 0 && (
          <div style={{ marginTop: '12px' }}>
            <strong style={{ color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c', marginBottom: '8px', display: 'block' }}>
              Health Indicators Analyzed:
            </strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {processedData.indicators.map((indicator, index) => (
                <span 
                  key={index}
                  style={{ 
                    padding: '6px 12px',
                    backgroundColor: processedData.performanceType === 'top' ? '#e8f5e8' : '#fdeaea',
                    border: `1px solid ${processedData.performanceType === 'top' ? '#4CAF50' : '#f87171'}`,
                    borderRadius: '6px',
                    fontSize: '14px',
                    color: processedData.performanceType === 'top' ? '#2E7D32' : '#dc2626',
                    fontWeight: '500'
                  }}
                >
                  {indicator}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Map Section */}
      <div style={{ marginBottom: '30px' }}>
        <h3 style={{ color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c', marginBottom: '15px' }}>
          🗺️ Geographic Distribution
        </h3>
        <div style={{ height: '600px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #e9ecef', position: 'relative' }}>
          <TopBottomMap data={{
            boundary: processedData.boundaryData,
            districts: processedData.districts,
            indicators: processedData.indicators,
            performance_type: processedData.performanceType,
            response_type: processedData.responseType
          }} />
        </div>
      </div>

      {/* Chart Section */}
      {processedData.chartData && (
        <div>
          <h3 style={{ color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c', marginBottom: '15px' }}>
            📊 Performance Comparison
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
          border: `2px solid ${processedData.performanceType === 'top' ? '#4CAF50' : '#f87171'}`,
          zIndex: 2000
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
            <h4 style={{ margin: '0', color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c' }}>
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
            📍 {selectedDistrict.state_name} • Rank: #{selectedDistrict.rank}
          </p>
          {selectedDistrict.performance_score && (
            <div style={{ 
              margin: '6px 0', 
              padding: '8px', 
              backgroundColor: '#f8f9fa', 
              borderRadius: '6px',
              border: '1px solid #e9ecef'
            }}>
              <div style={{ fontWeight: '600', fontSize: '14px', color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c' }}>
                Performance Score
              </div>
              <div style={{ fontSize: '18px', fontWeight: '700', color: processedData.performanceType === 'top' ? '#2ecc71' : '#e74c3c', margin: '2px 0' }}>
                {selectedDistrict.performance_score.toFixed(2)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TopBottomAnalysis;
