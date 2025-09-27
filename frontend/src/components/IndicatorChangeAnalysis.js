import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Bar, Line } from 'react-chartjs-2';
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
  PointElement,
  LineElement,
  Filler,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Set Mapbox access token
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN;

// Color schemes for change analysis
const CHANGE_COLORS = {
  positive_improvement: '#2ecc71',    // Green for positive improvements
  negative_decline: '#e74c3c',       // Red for negative declines  
  positive_decline: '#e74c3c',       // Red for positive changes that are bad
  negative_improvement: '#2ecc71',   // Green for negative changes that are good
  neutral: '#95a5a6',                // Gray for neutral/no change
  background: '#ecf0f1'              // Light background
};

// Helper function to get color based on change and indicator direction
const getChangeColor = (change, indicatorDirection) => {
  if (change === 0 || change === null || change === undefined) {
    return CHANGE_COLORS.neutral;
  }
  
  if (change > 0) {
    // Positive change
    return indicatorDirection === "higher_is_better" 
      ? CHANGE_COLORS.positive_improvement 
      : CHANGE_COLORS.positive_decline;
  } else {
    // Negative change
    return indicatorDirection === "higher_is_better"
      ? CHANGE_COLORS.negative_decline
      : CHANGE_COLORS.negative_improvement;
  }
};

const IndicatorChangeMap = ({ data }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (!data || !data.boundary || map.current) return;

    const boundaryData = data.boundary;
    if (!boundaryData || boundaryData.length === 0) {
      
      return;
    }

    // Add a small delay to ensure DOM is ready
    const initializeMap = () => {
      if (!mapContainer.current) {
        
        return;
      }

      

      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/light-v11',
        center: [78.9629, 20.5937], // Center of India
        zoom: 4
      });

      // Store map instance for capture functionality
      registerMapInstance('indicator-change-map', map.current);

      map.current.on('load', () => {
        

        // Generate colors for states (for variety)
        const stateColors = {};
        const colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#34495e', '#f39c12', '#27ae60', '#e74c3c'];
        let colorIndex = 0;
        
        boundaryData.forEach(boundary => {
          const stateName = boundary.state_name || boundary.state;
          if (!stateColors[stateName]) {
            stateColors[stateName] = colors[colorIndex % colors.length];
            colorIndex++;
          }
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

              // Find corresponding district data for change information
              const districtName = boundary.district_name || boundary.district;
              let districtData = null;
              
              // Search in all districts data
              if (data.all_districts) {
                districtData = data.all_districts.find(d => d.district_name === districtName);
              } else if (data.example_districts) {
                districtData = data.example_districts.find(d => d.district_name === districtName);
              } else if (data.district_data && data.district_data.district_name === districtName) {
                districtData = data.district_data;
              }
              
              return {
                type: 'Feature',
                id: index,
                properties: {
                  district_name: districtName || 'Unknown District',
                  state_name: boundary.state_name || boundary.state || 'Unknown State',
                  state_color: stateColors[boundary.state_name || boundary.state] || '#cccccc',
                  prevalence_2016: districtData?.prevalence_2016 || null,
                  prevalence_2021: districtData?.prevalence_2021 || null,
                  prevalence_change: districtData?.prevalence_change || null,
                  change_available: districtData?.prevalence_change !== null && 
                                   districtData?.prevalence_change !== undefined && 
                                   !isNaN(districtData?.prevalence_change)
                },
                geometry: parsedGeometry
              };
            })
            .filter(Boolean)
        };

        

        // Add districts source (check if it already exists)
        if (!map.current.getSource('change-districts')) {
          map.current.addSource('change-districts', {
            type: 'geojson',
            data: districtsGeoJSON
          });
        } else {
          // Update existing source
          map.current.getSource('change-districts').setData(districtsGeoJSON);
        }

        // Add district fill layer with change-based colors (check if it already exists)
        if (!map.current.getLayer('districts-fill')) {
          map.current.addLayer({
            id: 'districts-fill',
            type: 'fill',
            source: 'change-districts',
            paint: {
            'fill-color': [
              'case',
              // First check if change data is not available
              ['==', ['get', 'change_available'], false],
              '#cccccc', // Gray for no data
              // Then check if the change value is null, undefined, or invalid
              ['any',
                ['==', ['get', 'prevalence_change'], null],
                ['!=', ['typeof', ['get', 'prevalence_change']], 'number']
              ],
              '#666666', // Dark gray for invalid data (prevents black)
              // Finally, interpolate for valid numeric values
              [
                'interpolate',
                ['linear'],
                ['get', 'prevalence_change'],
                -10, data.indicator_direction === 'higher_is_better' ? '#e74c3c' : '#2ecc71', // Large negative change
                -5, data.indicator_direction === 'higher_is_better' ? '#f39c12' : '#27ae60',   // Medium negative change
                0, '#95a5a6',    // No change
                5, data.indicator_direction === 'higher_is_better' ? '#27ae60' : '#f39c12',    // Medium positive change
                10, data.indicator_direction === 'higher_is_better' ? '#2ecc71' : '#e74c3c'   // Large positive change
              ]
            ],
              'fill-opacity': 0.7
            }
          });
        }

        // Add district border layer (check if it already exists)
        if (!map.current.getLayer('districts-border')) {
          map.current.addLayer({
            id: 'districts-border',
            type: 'line',
            source: 'change-districts',
            paint: {
              'line-color': '#2c3e50',
              'line-width': 1
            }
          });
        }

        // Add hover effect
        map.current.on('mouseenter', 'districts-fill', () => {
          map.current.getCanvas().style.cursor = 'pointer';
        });

        map.current.on('mouseleave', 'districts-fill', () => {
          map.current.getCanvas().style.cursor = '';
        });

        // Add click handler for district popups
        map.current.on('click', 'districts-fill', (e) => {
          const properties = e.features[0].properties;
          
          let popupContent = `
            <div style="font-family: Arial, sans-serif; max-width: 250px;">
              <h3 style="margin: 0 0 8px 0; color: #2c3e50; font-size: 14px;">
                ${properties.district_name}
              </h3>
              <p style="margin: 0 0 8px 0; color: #7f8c8d; font-size: 12px;">
                ${properties.state_name}
              </p>
          `;

          if (properties.change_available) {
            const change = properties.prevalence_change;
            
            // Validate that change is a valid number
            if (change !== null && change !== undefined && !isNaN(change)) {
              const direction = change > 0 ? 'increased' : 'decreased';
              const absChange = Math.abs(change);
              
              // Determine if this is good or bad based on indicator direction
              let interpretation = '';
              if (change > 0) {
                interpretation = data.indicator_direction === 'higher_is_better' ? 'improvement' : 'decline';
              } else {
                interpretation = data.indicator_direction === 'higher_is_better' ? 'decline' : 'improvement';
              }
              
              const color = getChangeColor(change, data.indicator_direction);
              
              popupContent += `
                <div style="border-top: 1px solid #ecf0f1; padding-top: 8px;">
                  <p style="margin: 0 0 4px 0; font-size: 12px;">
                    <strong>2016:</strong> ${properties.prevalence_2016 != null && !isNaN(properties.prevalence_2016) ? properties.prevalence_2016.toFixed(2) + '%' : 'N/A'}
                  </p>
                  <p style="margin: 0 0 4px 0; font-size: 12px;">
                    <strong>2021:</strong> ${properties.prevalence_2021 != null && !isNaN(properties.prevalence_2021) ? properties.prevalence_2021.toFixed(2) + '%' : 'N/A'}
                  </p>
                  <p style="margin: 0 0 4px 0; font-size: 12px;">
                    <strong>Change:</strong> 
                    <span style="color: ${color}; font-weight: bold;">
                      ${change > 0 ? '+' : ''}${change.toFixed(2)} points
                    </span>
                  </p>
                  <p style="margin: 0; font-size: 11px; color: ${color}; font-weight: bold;">
                    ${direction.charAt(0).toUpperCase() + direction.slice(1)} by ${absChange.toFixed(2)} points (${interpretation})
                  </p>
                </div>
              `;
            } else {
              // Change value is invalid
              popupContent += `
                <div style="border-top: 1px solid #ecf0f1; padding-top: 8px;">
                  <p style="margin: 0 0 4px 0; font-size: 12px;">
                    <strong>2016:</strong> ${properties.prevalence_2016 != null && !isNaN(properties.prevalence_2016) ? properties.prevalence_2016.toFixed(2) + '%' : 'N/A'}
                  </p>
                  <p style="margin: 0 0 4px 0; font-size: 12px;">
                    <strong>2021:</strong> ${properties.prevalence_2021 != null && !isNaN(properties.prevalence_2021) ? properties.prevalence_2021.toFixed(2) + '%' : 'N/A'}
                  </p>
                  <p style="margin: 0; font-size: 11px; color: #95a5a6;">
                    Change data is invalid or corrupted
                  </p>
                </div>
              `;
            }
          } else {
            popupContent += `
              <div style="border-top: 1px solid #ecf0f1; padding-top: 8px;">
                <p style="margin: 0; font-size: 11px; color: #95a5a6;">
                  Change data not available
                </p>
              </div>
            `;
          }

          popupContent += `</div>`;

          new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(popupContent)
            .addTo(map.current);
        });

        // Fit to bounds
        const bounds = new mapboxgl.LngLatBounds();
        districtsGeoJSON.features.forEach(feature => {
          if (feature.geometry.type === 'Polygon') {
            feature.geometry.coordinates[0].forEach(coord => bounds.extend(coord));
          } else if (feature.geometry.type === 'MultiPolygon') {
            feature.geometry.coordinates.forEach(polygon => {
              polygon[0].forEach(coord => bounds.extend(coord));
            });
          }
        });

        if (!bounds.isEmpty()) {
          map.current.fitBounds(bounds, { padding: 20 });
        }
      });

      map.current.on('error', (e) => {
        console.error('Mapbox error:', e);
      });
    };

    // Add a small delay to ensure DOM is ready
    setTimeout(initializeMap, 100);

    return () => {
      if (map.current) {
        // Clean up sources and layers before removing map
        try {
          if (map.current.getLayer('districts-border')) {
            map.current.removeLayer('districts-border');
          }
          if (map.current.getLayer('districts-fill')) {
            map.current.removeLayer('districts-fill');
          }
          if (map.current.getSource('change-districts')) {
            map.current.removeSource('change-districts');
          }
        } catch (e) {
          console.warn('Error cleaning up map layers/sources:', e);
        }
        
        map.current.remove();
        map.current = null;
      }
    };
  }, [data]);

  if (!data || !data.boundary || data.boundary.length === 0) {
    return (
      <div style={{ 
        width: '100%', 
        height: '500px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#f8f9fa',
        border: '1px solid #e9ecef',
        borderRadius: '8px'
      }}>
        <p style={{ color: '#6c757d', margin: 0 }}>
          No map data available for indicator change analysis
        </p>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div 
        ref={mapContainer} 
        id="indicator-change-map"
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '10px',
        right: '10px',
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        padding: '10px',
        borderRadius: '5px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
        fontSize: '12px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Change Legend</div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
          <div style={{ 
            width: '15px', 
            height: '15px', 
            backgroundColor: data.indicator_direction === 'higher_is_better' ? '#2ecc71' : '#e74c3c',
            marginRight: '5px' 
          }}></div>
          <span>Large Improvement</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
          <div style={{ 
            width: '15px', 
            height: '15px', 
            backgroundColor: '#95a5a6',
            marginRight: '5px' 
          }}></div>
          <span>No Change</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
          <div style={{ 
            width: '15px', 
            height: '15px', 
            backgroundColor: data.indicator_direction === 'higher_is_better' ? '#e74c3c' : '#2ecc71',
            marginRight: '5px' 
          }}></div>
          <span>Large Decline</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
          <div style={{ 
            width: '15px', 
            height: '15px', 
            backgroundColor: '#666666',
            marginRight: '5px' 
          }}></div>
          <span>Invalid Data</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: '15px', 
            height: '15px', 
            backgroundColor: '#cccccc',
            marginRight: '5px' 
          }}></div>
          <span>No Data</span>
        </div>
      </div>
    </div>
  );
};

const IndicatorChangeAnalysis = ({ changeData, mapOnly = false, chartOnly = false }) => {
  const [selectedLevel, setSelectedLevel] = useState(null);



  // Extract and process the data
  const processedData = useMemo(() => {
    if (!changeData) {
      console.log('No changeData provided');
      return { 
        analysisLevel: 'unknown', 
        mainIndicator: 'Unknown', 
        indicatorDirection: 'higher_is_better',
        nationalData: null,
        stateData: null,
        districtData: null,
        exampleDistricts: [],
        allDistricts: [],
        boundaryData: [],
        chartData: null 
      };
    }

    // Extract core information
    const analysisLevel = changeData.analysis_level || 'country';
    const mainIndicator = changeData.main_indicator || 'Unknown Indicator';
    const indicatorDirection = changeData.indicator_direction || 'higher_is_better';
    
    // Extract level-specific data
    const nationalData = changeData.national_data || null;
    const stateData = changeData.state_data || null;
    const districtData = changeData.district_data || null;
    
    // Extract districts and boundary data
    const exampleDistricts = changeData.example_districts || [];
    const allDistricts = changeData.all_districts || [];
    const boundaryData = changeData.boundary || [];
    
    // Extract chart data
    const chartData = changeData.chart_data || null;
    
   

    return {
      analysisLevel,
      mainIndicator,
      indicatorDirection,
      nationalData,
      stateData,
      districtData,
      exampleDistricts,
      allDistricts,
      boundaryData,
      chartData,
      totalAnalyzed: changeData.total_districts_analyzed || allDistricts.length,
      responseType: changeData.response_type || 'unknown'
    };
  }, [changeData]);

  // Prepare chart options
  const chartOptions = useMemo(() => {
    const isDistrictLevel = processedData.analysisLevel === 'district';
    
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: processedData.chartData?.title || `${processedData.mainIndicator} Change Analysis`,
          font: {
            size: 14,
            weight: 'bold'
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const value = context.parsed.y;
              if (isDistrictLevel) {
                return `${context.dataset.label}: ${value?.toFixed(2) || 'N/A'}%`;
              } else {
                const sign = value >= 0 ? '+' : '';
                return `Change: ${sign}${value?.toFixed(2) || 'N/A'} points`;
              }
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: !isDistrictLevel,
          title: {
            display: true,
            text: isDistrictLevel ? 'Prevalence (%)' : 'Change (percentage points)'
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)'
          }
        },
        x: {
          title: {
            display: true,
            text: isDistrictLevel ? 'Year' : 'Location'
          },
          grid: {
            display: false
          }
        }
      }
    };
  }, [processedData]);

  if (chartOnly && processedData.chartData) {
    const ChartComponent = processedData.chartData.type === 'line' ? Line : Bar;
    
    return (
      <div style={{ width: '100%', height: '400px', padding: '20px' }}>
        <div style={{ 
          marginBottom: '15px', 
          padding: '12px', 
          backgroundColor: '#f8f9fa', 
          borderRadius: '8px',
          border: '1px solid #e9ecef'
        }}>
          <h3 style={{ margin: '0 0 4px 0', color: '#2c3e50', fontSize: '16px' }}>
            📈 {processedData.mainIndicator} Change Analysis
          </h3>
          <p style={{ margin: '0', color: '#666', fontSize: '13px' }}>
            {processedData.analysisLevel.charAt(0).toUpperCase() + processedData.analysisLevel.slice(1)} level analysis • 
            {processedData.analysisLevel === 'district' ? ' Trend view' : ` ${processedData.exampleDistricts.length} examples`}
          </p>
        </div>
        
        <div style={{ height: 'calc(100% - 80px)' }}>
          <ChartComponent data={processedData.chartData} options={chartOptions} />
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
          <h3 style={{ margin: '0 0 4px 0', color: '#2c3e50', fontSize: '16px' }}>
            🗺️ {processedData.mainIndicator} Change Map
          </h3>
          <p style={{ margin: '0', color: '#666', fontSize: '13px' }}>
            {processedData.analysisLevel.charAt(0).toUpperCase() + processedData.analysisLevel.slice(1)} level • 
            {processedData.totalAnalyzed} districts analyzed • Click districts for details
          </p>
        </div>
        
        <div style={{ height: 'calc(100% - 70px)', borderRadius: '8px', overflow: 'hidden' }}>
          <IndicatorChangeMap data={{
            boundary: processedData.boundaryData,
            all_districts: processedData.allDistricts,
            example_districts: processedData.exampleDistricts,
            district_data: processedData.districtData,
            indicator_direction: processedData.indicatorDirection,
            analysis_level: processedData.analysisLevel
          }} />
        </div>
      </div>
    );
  }

  // Full component with both map and chart
  return (
    <div style={{ width: '100%' }}>
      {/* Header */}
      <div style={{ 
        marginBottom: '20px', 
        padding: '16px', 
        backgroundColor: '#f8f9fa', 
        borderRadius: '8px',
        border: '1px solid #e9ecef'
      }}>
        <h3 style={{ margin: '0 0 8px 0', color: '#2c3e50', fontSize: '18px' }}>
          📈 {processedData.mainIndicator} Change Analysis (2016-2021)
        </h3>
        <p style={{ margin: '0 0 8px 0', color: '#666', fontSize: '14px' }}>
          {processedData.analysisLevel.charAt(0).toUpperCase() + processedData.analysisLevel.slice(1)} level analysis
          {processedData.analysisLevel !== 'district' && ` • ${processedData.totalAnalyzed} districts analyzed`}
        </p>
        
        {/* Summary stats */}
        <div style={{ display: 'flex', gap: '20px', marginTop: '10px', flexWrap: 'wrap' }}>
          {processedData.nationalData && processedData.nationalData.prevalence_change !== null && (
            <div style={{ backgroundColor: 'white', padding: '8px 12px', borderRadius: '6px', border: '1px solid #dee2e6' }}>
              <div style={{ fontSize: '12px', color: '#666' }}>National Change</div>
              <div style={{ 
                fontSize: '14px', 
                fontWeight: 'bold', 
                color: getChangeColor(processedData.nationalData.prevalence_change, processedData.indicatorDirection)
              }}>
                {processedData.nationalData.prevalence_change > 0 ? '+' : ''}{processedData.nationalData.prevalence_change?.toFixed(2)} points
              </div>
            </div>
          )}
          
          {processedData.nationalData && processedData.nationalData.prevalence_change === null && (
            <div style={{ backgroundColor: '#fff3cd', padding: '8px 12px', borderRadius: '6px', border: '1px solid #ffeaa7' }}>
              <div style={{ fontSize: '12px', color: '#856404' }}>National Change</div>
              <div style={{ 
                fontSize: '14px', 
                fontWeight: 'bold', 
                color: '#856404'
              }}>
                Not Available
              </div>
            </div>
          )}
          
          {processedData.stateData && (
            <div style={{ backgroundColor: 'white', padding: '8px 12px', borderRadius: '6px', border: '1px solid #dee2e6' }}>
              <div style={{ fontSize: '12px', color: '#666' }}>{processedData.stateData.state_name} Change</div>
              <div style={{ 
                fontSize: '14px', 
                fontWeight: 'bold', 
                color: getChangeColor(processedData.stateData.prevalence_change, processedData.indicatorDirection)
              }}>
                {processedData.stateData.prevalence_change > 0 ? '+' : ''}{processedData.stateData.prevalence_change?.toFixed(2)} points
              </div>
            </div>
          )}
          
          {processedData.districtData && (
            <div style={{ backgroundColor: 'white', padding: '8px 12px', borderRadius: '6px', border: '1px solid #dee2e6' }}>
              <div style={{ fontSize: '12px', color: '#666' }}>{processedData.districtData.district_name} Change</div>
              <div style={{ 
                fontSize: '14px', 
                fontWeight: 'bold', 
                color: getChangeColor(processedData.districtData.prevalence_change, processedData.indicatorDirection)
              }}>
                {processedData.districtData.prevalence_change > 0 ? '+' : ''}{processedData.districtData.prevalence_change?.toFixed(2)} points
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chart Section */}
      {processedData.chartData && (
        <div style={{ marginBottom: '20px' }}>
          <h4 style={{ margin: '0 0 15px 0', color: '#2c3e50', fontSize: '16px' }}>
            📊 Change Visualization
          </h4>
          <div style={{ height: '400px', backgroundColor: 'white', padding: '15px', borderRadius: '8px', border: '1px solid #e9ecef' }}>
            {processedData.chartData.type === 'line' ? (
              <Line data={processedData.chartData} options={chartOptions} />
            ) : (
              <Bar data={processedData.chartData} options={chartOptions} />
            )}
          </div>
        </div>
      )}

      {/* Map Section */}
      {processedData.boundaryData.length > 0 && (
        <div>
          <h4 style={{ margin: '0 0 15px 0', color: '#2c3e50', fontSize: '16px' }}>
            🗺️ Geographic Change Distribution
          </h4>
          <div style={{ height: '500px', backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e9ecef', overflow: 'hidden' }}>
            <IndicatorChangeMap data={{
              boundary: processedData.boundaryData,
              all_districts: processedData.allDistricts,
              example_districts: processedData.exampleDistricts,
              district_data: processedData.districtData,
              indicator_direction: processedData.indicatorDirection,
              analysis_level: processedData.analysisLevel
            }} />
          </div>
        </div>
      )}
    </div>
  );
};

export default IndicatorChangeAnalysis;

