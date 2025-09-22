import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import Map, { Source, Layer, Popup } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { registerMapInstance, initializeReactMapGLForCapture } from '../utils/saveUtils';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN;

export default function DistrictComparisonAnalysis({ data = {}, chartOnly = false, mapOnly = false }) {
  console.log('DistrictComparisonAnalysis received data:', data);

  const [viewState, setViewState] = useState({
    longitude: 78.96,
    latitude: 20.59,
    zoom: 5
  });
  const mapRef = useRef(null);
  
  // Popup state
  const [popupInfo, setPopupInfo] = useState(null);
  const [hoveredDistrict, setHoveredDistrict] = useState(null);

  // Set up WebGL context preservation when map loads
  useEffect(() => {
    const initializeMap = async () => {
      if (mapRef.current) {
        try {
          await initializeReactMapGLForCapture(mapRef, 'district-comparison-map');
          console.log('District comparison map initialized for capture');
        } catch (error) {
          console.error('Error initializing district comparison map:', error);
        }
      }
    };
    
    initializeMap();
    const timer = setTimeout(initializeMap, 100);
    return () => clearTimeout(timer);
  }, []);

  // Extract the actual data from nested structure
  const actualData = useMemo(() => {
    // Handle flattened data structure from health_main.py
    if (data?.response_type === "district_performance_comparison" && data?.chart_data) {
      return data;
    }
    
    // Handle nested data structure
    if (data?.data && Array.isArray(data.data) && data.data.length > 0 && data.data[0]?.result) {
      const result = data.data[0].result;
      if (result.response_type === "district_performance_comparison") {
        return result;
      }
    }
    
    // If data is the direct result
    if (data && typeof data === 'object' && data.response_type === "district_performance_comparison") {
      return data;
    }
    
    return data;
  }, [data]);

  // Extract district comparison information
  const comparisonInfo = useMemo(() => {
    if (!actualData) return null;
    
    const info = {
      districts: actualData.districts || [],
      totalDistricts: actualData.total_districts || 0,
      totalIndicators: actualData.total_indicators || 0,
      indicators: actualData.indicators || [],
      comparisonType: actualData.comparison_type || 'national',
      comparisonData: actualData.comparison_data || {},
      year: actualData.year || 2021,
      chartData: actualData.chart_data || [],
      analysis: actualData.analysis || ""
    };
    
    console.log('Comparison info extracted:', info);
    
    return info;
  }, [actualData]);

  // Extract boundary data for mapping with performance information
  const boundaryData = useMemo(() => {
    if (!actualData?.boundary || !Array.isArray(actualData.boundary)) {
      console.log('No boundary data available:', actualData?.boundary);
      return null;
    }

    const features = actualData.boundary
      .filter(boundary => {
        // Filter out boundaries with invalid geometry
        const geometry = boundary.geometry || boundary;
        return geometry && (
          (typeof geometry === 'object' && geometry.type && geometry.coordinates) ||
          (typeof geometry === 'string' && geometry.length > 0)
        );
      })
      .map((boundary, index) => {
        try {
          // Find corresponding district data
          const districtData = comparisonInfo?.districts?.find(d => 
            d.district_name === boundary.district_name
          );

          // Calculate overall performance score for color coding
          let performanceScore = 0.5; // Default neutral
          if (districtData && districtData.indicators && districtData.indicators.length > 0) {
            const validIndicators = districtData.indicators.filter(ind => ind.prevalence !== null && ind.prevalence !== undefined);
            if (validIndicators.length > 0) {
              const avgValue = validIndicators.reduce((sum, ind) => sum + ind.prevalence, 0) / validIndicators.length;
              performanceScore = Math.min(Math.max(avgValue / 100, 0), 1); // Normalize to 0-1
            }
          }

          // Parse geometry safely
          let geometry = boundary.geometry || boundary;
          if (typeof geometry === 'string') {
            try {
              geometry = JSON.parse(geometry);
            } catch (e) {
              console.warn('Failed to parse geometry for district:', boundary.district_name, e);
              return null;
            }
          }

          return {
            type: "Feature",
            properties: {
              id: index,
              district_name: boundary.district_name || '',
              state_name: boundary.state_name || '',
              performance_score: performanceScore,
              comparison_type: comparisonInfo?.comparisonType || 'national'
            },
            geometry: geometry
          };
        } catch (error) {
          console.warn('Error processing boundary for district:', boundary.district_name, error);
          return null;
        }
      })
      .filter(feature => feature !== null); // Remove failed features

    if (features.length === 0) {
      console.warn('No valid boundary features after processing');
      return null;
    }

    return {
      type: "FeatureCollection",
      features: features
    };
  }, [actualData?.boundary, comparisonInfo?.districts, comparisonInfo?.comparisonType]);

  // Chart options for consistent styling
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          padding: 20,
          usePointStyle: true,
          font: {
            size: 12,
            family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: function(context) {
            const value = context.parsed.y;
            const label = context.dataset.label || '';
            return `${label}: ${value !== null ? value.toFixed(1) : 'N/A'}%`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          maxRotation: 45,
          minRotation: 0,
          font: {
            size: 11
          }
        }
      },
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          font: {
            size: 11
          },
          callback: function(value) {
            return value + '%';
          }
        }
      }
    }
  };

  // Event handlers for map interactions
  const onMapClick = (event) => {
    const feature = event.features && event.features[0];
    if (feature && feature.properties && event.lngLat) {
      const { district_name, state_name } = feature.properties;
      
      // Validate coordinates
      const lng = event.lngLat.lng;
      const lat = event.lngLat.lat;
      
      if (isNaN(lng) || isNaN(lat)) {
        console.warn('Invalid coordinates for popup:', lng, lat);
        return;
      }
      
      // Find the district data from our comparisonInfo instead of relying on serialized data
      const districtData = comparisonInfo?.districts?.find(d => 
        d.district_name === district_name
      );
      
      console.log('Popup click debug:', {
        district_name,
        state_name,
        districtData,
        coordinates: { lng, lat },
        comparisonInfo: comparisonInfo
      });
      
      setPopupInfo({
        longitude: lng,
        latitude: lat,
        district_name,
        state_name,
        district_data: districtData || null
      });
    }
  };

  const onMapHover = (event) => {
    const feature = event.features && event.features[0];
    if (feature && feature.properties) {
      setHoveredDistrict(feature.properties.district_name);
    } else {
      setHoveredDistrict(null);
    }
  };

  // If no data available
  if (!comparisonInfo || comparisonInfo.totalDistricts === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h3>No District Comparison Data Available</h3>
        <p>Unable to load district performance comparison data.</p>
      </div>
    );
  }

  return (
    <div className="district-comparison-analysis">
      {/* Analysis Summary */}
      {!chartOnly && !mapOnly && (
        <div style={{ 
          marginBottom: '24px',
          padding: '20px',
          background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
          borderRadius: '12px',
          border: '1px solid #dee2e6'
        }}>
          <h3 style={{ 
            margin: '0 0 16px 0',
            color: '#495057',
            fontSize: '18px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            🆚 District Performance Comparison Analysis
          </h3>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '16px',
            marginBottom: '16px'
          }}>
            <div style={{
              background: 'white',
              padding: '12px',
              borderRadius: '8px',
              border: '1px solid #e9ecef'
            }}>
              <div style={{ fontSize: '14px', color: '#6c757d', marginBottom: '4px' }}>Districts Compared</div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#495057' }}>{comparisonInfo.totalDistricts}</div>
            </div>
            
            <div style={{
              background: 'white',
              padding: '12px',
              borderRadius: '8px',
              border: '1px solid #e9ecef'
            }}>
              <div style={{ fontSize: '14px', color: '#6c757d', marginBottom: '4px' }}>Health Indicators</div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#495057' }}>{comparisonInfo.totalIndicators}</div>
            </div>
            
            <div style={{
              background: 'white',
              padding: '12px',
              borderRadius: '8px',
              border: '1px solid #e9ecef'
            }}>
              <div style={{ fontSize: '14px', color: '#6c757d', marginBottom: '4px' }}>Comparison Type</div>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#495057' }}>
                {comparisonInfo.comparisonType === 'national' ? '🇮🇳 National Average' : '🏛️ State Average'}
              </div>
            </div>
            
            <div style={{
              background: 'white',
              padding: '12px',
              borderRadius: '8px',
              border: '1px solid #e9ecef'
            }}>
              <div style={{ fontSize: '14px', color: '#6c757d', marginBottom: '4px' }}>Analysis Year</div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#495057' }}>{comparisonInfo.year}</div>
            </div>
          </div>

          <div style={{ fontSize: '14px', color: '#6c757d' }}>
            <strong>Districts:</strong> {comparisonInfo.districts.map(d => `${d.district_name} (${d.state_name})`).join(', ')}
          </div>
          <div style={{ fontSize: '14px', color: '#6c757d', marginTop: '4px' }}>
            <strong>Indicators:</strong> {comparisonInfo.indicators.map(i => i.indicator_name).join(', ')}
          </div>
        </div>
      )}

      {/* Charts Section */}
      {!mapOnly && comparisonInfo.chartData && comparisonInfo.chartData.length > 0 && (
        <div className="charts-section" style={{ marginBottom: '24px' }}>
          <h4 style={{ 
            margin: '0 0 20px 0',
            color: '#495057',
            fontSize: '16px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            📊 Performance Comparison Charts
          </h4>
          
          <div style={{ 
            display: 'grid', 
            gap: '24px',
            gridTemplateColumns: chartOnly ? '1fr' : 'repeat(auto-fit, minmax(500px, 1fr))'
          }}>
            {comparisonInfo.chartData.map((chart, index) => (
              <div key={index} style={{
                background: 'white',
                padding: '20px',
                borderRadius: '12px',
                border: '1px solid #e0e0e0',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}>
                <h5 style={{
                  margin: '0 0 16px 0',
                  fontSize: '14px',
                  fontWeight: '600',
                  color: '#495057',
                  textAlign: 'center'
                }}>
                  {chart.title}
                </h5>
                <div style={{ height: chartOnly ? '400px' : '300px' }}>
                  <Bar 
                    data={chart.data} 
                    options={{
                      ...chartOptions,
                      plugins: {
                        ...chartOptions.plugins,
                        title: {
                          display: false
                        }
                      }
                    }} 
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Map Section */}
      {!chartOnly && boundaryData && (
        <div className="map-section">
          <h4 style={{ 
            margin: '0 0 16px 0',
            color: '#495057',
            fontSize: '16px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            🗺️ Geographic Distribution
          </h4>
          <div style={{
            height: mapOnly ? '600px' : '500px',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '1px solid #e0e0e0',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <Map
              ref={mapRef}
              {...viewState}
              onMove={evt => setViewState(evt.viewState)}
              mapStyle="mapbox://styles/mapbox/light-v10"
              mapboxAccessToken={MAPBOX_TOKEN}
              attributionControl={false}
              style={{ width: '100%', height: '100%' }}
              onClick={onMapClick}
              onMouseMove={onMapHover}
              interactiveLayerIds={['district-fill']}
              cursor={hoveredDistrict ? 'pointer' : 'grab'}
            >
              {boundaryData && (
                <Source id="district-boundaries" type="geojson" data={boundaryData}>
                  <Layer
                    id="district-fill"
                    type="fill"
                    paint={{
                      'fill-color': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredDistrict || ''],
                        '#FF6B6B', // Hover color - bright red
                        [
                          'interpolate',
                          ['linear'],
                          ['get', 'performance_score'],
                          0, '#E74C3C',    // Poor performance - red
                          0.3, '#F39C12', // Below average - orange  
                          0.5, '#F1C40F', // Average - yellow
                          0.7, '#2ECC71', // Good - green
                          1, '#27AE60'     // Excellent - dark green
                        ]
                      ],
                      'fill-opacity': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredDistrict || ''],
                        0.9, // Higher opacity on hover
                        0.7  // Normal opacity
                      ]
                    }}
                  />
                  <Layer
                    id="district-stroke"
                    type="line"
                    paint={{
                      'line-color': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredDistrict || ''],
                        '#2C3E50', // Darker stroke on hover
                        '#34495E'  // Normal stroke
                      ],
                      'line-width': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredDistrict || ''],
                        3, // Thicker stroke on hover
                        1.5
                      ]
                    }}
                  />
                </Source>
              )}
              
              {/* Popup for district information */}
              {popupInfo && !isNaN(popupInfo.longitude) && !isNaN(popupInfo.latitude) && (
                <Popup
                  longitude={popupInfo.longitude}
                  latitude={popupInfo.latitude}
                  onClose={() => setPopupInfo(null)}
                  closeButton={true}
                  closeOnClick={false}
                  maxWidth="350px"
                  className="district-comparison-popup"
                >
                  <div style={{
                    padding: '12px',
                    minWidth: '280px',
                    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                  }}>
                    <h4 style={{
                      margin: '0 0 8px 0',
                      fontSize: '16px',
                      fontWeight: '600',
                      color: '#2C3E50',
                      borderBottom: '2px solid #3498DB',
                      paddingBottom: '6px'
                    }}>
                      {popupInfo.district_name}
                    </h4>
                    <p style={{
                      margin: '0 0 12px 0',
                      fontSize: '13px',
                      color: '#7F8C8D',
                      fontWeight: '500'
                    }}>
                      📍 {popupInfo.state_name}
                    </p>
                    
                    {popupInfo.district_data && popupInfo.district_data.indicators && popupInfo.district_data.indicators.length > 0 ? (
                      <div style={{ fontSize: '13px' }}>
                        <div style={{
                          marginBottom: '8px',
                          fontWeight: '600',
                          color: '#34495E'
                        }}>
                          🏥 Health Indicators vs {comparisonInfo.comparisonType === 'national' ? 'National' : 'State'} Average:
                        </div>
                        {popupInfo.district_data.indicators.slice(0, 4).map((indicator, idx) => {
                          // Debug indicator data
                          console.log('Popup indicator debug:', {
                            indicator,
                            comparisonType: comparisonInfo.comparisonType,
                            comparisonData: comparisonInfo.comparisonData,
                            stateName: popupInfo.state_name
                          });
                          
                          // Get comparison value
                          let comparisonValue = null;
                          if (comparisonInfo.comparisonType === 'national') {
                            comparisonValue = comparisonInfo.comparisonData?.[indicator.indicator_id]?.value;
                          } else {
                            // For state comparison, the structure might be different
                            const stateData = comparisonInfo.comparisonData?.[indicator.indicator_id];
                            if (stateData && typeof stateData === 'object') {
                              comparisonValue = stateData[popupInfo.state_name]?.value;
                            }
                          }
                          
                          console.log('Comparison value found:', comparisonValue);
                          
                          const gap = (comparisonValue !== null && indicator.prevalence !== null) 
                            ? (indicator.prevalence - comparisonValue) : null;
                          const isGoodGap = gap !== null && (
                            (indicator.indicator_direction === 'higher_is_better' && gap > 0) ||
                            (indicator.indicator_direction === 'lower_is_better' && gap < 0)
                          );
                          
                          return (
                            <div key={idx} style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              padding: '4px 0',
                              borderBottom: '1px solid #ECF0F1'
                            }}>
                              <span style={{ 
                                fontSize: '12px',
                                color: '#2C3E50',
                                maxWidth: '160px',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}>
                                {indicator.indicator_name}
                              </span>
                              <div style={{ textAlign: 'right' }}>
                                <span style={{
                                  fontSize: '12px',
                                  fontWeight: '600',
                                  color: '#2C3E50'
                                }}>
                                  {indicator.prevalence?.toFixed(1) || 'N/A'}%
                                </span>
                                {comparisonValue !== null && (
                                  <div style={{
                                    fontSize: '10px',
                                    color: '#7F8C8D',
                                    fontWeight: '500'
                                  }}>
                                    vs {comparisonValue.toFixed(1)}%
                                  </div>
                                )}
                                {gap !== null && (
                                  <div style={{
                                    fontSize: '10px',
                                    color: isGoodGap ? '#27AE60' : '#E74C3C',
                                    fontWeight: '600'
                                  }}>
                                    {gap > 0 ? '+' : ''}{gap.toFixed(1)}pp
                                  </div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                        {popupInfo.district_data.indicators.length > 4 && (
                          <div style={{
                            fontSize: '11px',
                            color: '#7F8C8D',
                            fontStyle: 'italic',
                            marginTop: '6px',
                            textAlign: 'center'
                          }}>
                            +{popupInfo.district_data.indicators.length - 4} more indicators
                          </div>
                        )}
                      </div>
                    ) : (
                      <div style={{
                        fontSize: '12px',
                        color: '#7F8C8D',
                        fontStyle: 'italic'
                      }}>
                        {popupInfo.district_data ? 
                          'No indicator data available for this district' :
                          'District not found in comparison data'
                        }
                        <br/>
                        <small style={{ fontSize: '10px' }}>
                          Debug: {popupInfo.district_data ? 'Has district data' : 'No district data'}
                        </small>
                      </div>
                    )}
                    
                    <div style={{
                      marginTop: '10px',
                      fontSize: '10px',
                      color: '#95A5A6',
                      textAlign: 'center'
                    }}>
                      💡 Click anywhere on map to close
                    </div>
                  </div>
                </Popup>
              )}
            </Map>
          </div>
        </div>
      )}

      {/* Analysis Text */}
      {!chartOnly && !mapOnly && comparisonInfo.analysis && (
        <div className="analysis-section" style={{ marginTop: '24px' }}>
          <h4 style={{ 
            margin: '0 0 16px 0',
            color: '#495057',
            fontSize: '16px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            📝 Detailed Analysis
          </h4>
          <div style={{
            background: 'white',
            padding: '20px',
            borderRadius: '12px',
            border: '1px solid #e0e0e0',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <div 
              style={{
                lineHeight: '1.6',
                color: '#495057',
                fontSize: '14px'
              }}
              dangerouslySetInnerHTML={{ 
                __html: comparisonInfo.analysis.replace(/\n/g, '<br/>') 
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
