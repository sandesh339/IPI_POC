import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Map, Source, Layer, Popup } from 'react-map-gl';
import { Bar } from 'react-chartjs-2';
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
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// Mapbox access token
const mapboxAccessToken = process.env.REACT_APP_MAPBOX_TOKEN;

const DistrictSimilarityAnalysis = ({ data, mapOnly = false, chartOnly = false }) => {
  const [popupInfo, setPopupInfo] = useState(null);
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const [currentChartPage, setCurrentChartPage] = useState(0);
  const mapRef = useRef();

  // Process the data for easier access
  const analysisInfo = useMemo(() => {
    if (!data) {
      return {
        districts: [],
        indicators: [],
        analysisType: 'similar',
        totalDistricts: 0,
        totalIndicators: 0,
        chartData: [],
        analysis: '',
        categoryName: null,
        stateFilter: null
      };
    }

    // Extract data from various possible structures
    let districts = [];
    let indicators = [];
    let chartData = [];
    
    // Handle different data structures
    if (data.districts && Array.isArray(data.districts)) {
      districts = data.districts;
      indicators = data.indicators || [];
      chartData = data.chart_data || [];
    }
    // Handle nested data structure
    else if (data.data && Array.isArray(data.data)) {
      for (const item of data.data) {
        if (item.result) {
          if (item.result.districts && Array.isArray(item.result.districts)) {
            districts = item.result.districts;
            indicators = item.result.indicators || [];
            chartData = item.result.chart_data || [];
            break;
          }
        }
      }
    }
    // Handle function result structure
    else if (data.data && !Array.isArray(data.data) && data.data.districts) {
      districts = data.data.districts;
      indicators = data.data.indicators || [];
      chartData = data.data.chart_data || [];
    }

    // Ensure chartData is an array
    if (chartData && !Array.isArray(chartData)) {
      chartData = Object.values(chartData).filter(chart => chart && typeof chart === 'object');
    }

    return {
      districts: districts || [],
      indicators: indicators || [],
      analysisType: data.analysis_type || 'similar',
      totalDistricts: data.total_districts || districts.length || 0,
      totalIndicators: data.total_indicators || indicators.length || 0,
      chartData: Array.isArray(chartData) ? chartData : [],
      analysis: data.analysis || '',
      categoryName: data.category_name,
      stateFilter: data.state_filter
    };
  }, [data]);

  // Process boundary data for map visualization
  const boundaryData = useMemo(() => {
    let boundaryData = data?.boundary || data?.boundary_data || [];
    
    // Handle case where boundary data might be nested
    if (!boundaryData.length && data?.data && Array.isArray(data.data)) {
      for (const item of data.data) {
        if (item.result && (item.result.boundary_data || item.result.boundary)) {
          boundaryData = item.result.boundary_data || item.result.boundary;
          break;
        }
      }
    }
    
    if (!boundaryData || !boundaryData.length) {
      return { 
        districtFeatures: { type: 'FeatureCollection', features: [] }
      };
    }

    try {
      const districtFeatures = [];

      boundaryData.forEach((boundary) => {
        try {
          const geometry = typeof boundary.geometry === 'string' 
            ? JSON.parse(boundary.geometry) 
            : boundary.geometry;

          if (!geometry || !geometry.coordinates || !Array.isArray(geometry.coordinates)) {
            return;
          }

          const districtName = boundary.district_name || boundary.district;
          const stateName = boundary.state_name || boundary.state;
          
          // Find district in analysis results
          const districtData = analysisInfo.districts.find(d => d.district_name === districtName);

          if (districtData) {
            // Calculate a simple performance score for visualization
            let performanceScore = 0.5; // Default middle value
            
            if (districtData.indicators && Object.keys(districtData.indicators).length > 0) {
              const values = Object.values(districtData.indicators)
                .map(ind => ind.prevalence_current)
                .filter(val => val !== null && val !== undefined);
              
              if (values.length > 0) {
                performanceScore = values.reduce((sum, val) => sum + val, 0) / values.length;
              }
            }

            districtFeatures.push({
              type: 'Feature',
              properties: {
                type: 'district',
                district_name: districtName,
                state_name: stateName || districtData.state_name,
                performance_score: Math.max(0, Math.min(1, performanceScore)),
                analysis_type: analysisInfo.analysisType,
                total_indicators: analysisInfo.totalIndicators
              },
              geometry: geometry
            });
          }
        } catch (error) {
          // Skip invalid boundaries
        }
      });

      return { 
        districtFeatures: { type: 'FeatureCollection', features: districtFeatures }
      };
    } catch (error) {
      return { 
        districtFeatures: { type: 'FeatureCollection', features: [] }
      };
    }
  }, [data?.boundary_data, data?.boundary, data?.data, analysisInfo.districts, analysisInfo.analysisType, analysisInfo.totalIndicators]);

  // Organize charts for pagination
  const chartPages = useMemo(() => {
    if (!analysisInfo.chartData || analysisInfo.chartData.length === 0) {
      return [];
    }

    // Group charts by indicator with enhanced styling and indicator-specific information
    const pages = analysisInfo.chartData.map((chart, index) => {
      const isFirstChart = index === 0;
      const isLastChart = index === analysisInfo.chartData.length - 1;
      
      // Get indicator-specific information
      const chartInfo = chart.chart_info || {};
      const selectedDistricts = chartInfo.selected_districts || 'N/A';
      const indicatorAnalysis = chartInfo.analysis_type || analysisInfo.analysisType;
      const algorithmUsed = chartInfo.algorithm_used || 'standard';
      const valueStats = chartInfo.value_statistics || {};
      const algorithmDesc = chartInfo.algorithm_description || '';
      
      return {
        title: `📊 ${chart.title}`,
        subtitle: chart.description || `Indicator ${index + 1} of ${analysisInfo.chartData.length}`,
        charts: [chart],
        description: `This chart shows ${chart.title.toLowerCase()} across ${selectedDistricts} districts using enhanced ${algorithmUsed} algorithm. ${algorithmDesc}`,
        gradient: isFirstChart ? "from-blue-500 to-blue-600" : 
                 isLastChart ? "from-purple-500 to-purple-600" : 
                 "from-green-500 to-green-600",
        icon: "📊",
        indicatorSpecific: true,
        selectedDistricts: selectedDistricts,
        analysisType: indicatorAnalysis,
        algorithmUsed: algorithmUsed,
        valueStats: valueStats,
        algorithmDescription: algorithmDesc
      };
    });

    return pages;
  }, [analysisInfo.chartData, analysisInfo.analysisType]);

  // Enhanced Chart component
  const EnhancedChart = ({ chartConfig, title = null }) => {
    if (!chartConfig) {
      return <div className="text-gray-500 p-4">No chart configuration provided</div>;
    }

    let chartData = chartConfig.data || chartConfig;

    if (!chartData || (!chartData.labels && !chartData.datasets)) {
      return <div className="text-gray-500 p-4">Invalid chart data structure</div>;
    }

    if (!chartData.datasets || !Array.isArray(chartData.datasets) || chartData.datasets.length === 0) {
      return <div className="text-gray-500 p-4">No chart datasets available</div>;
    }

    // Force bar charts based on user preference  
    const ChartComponent = Bar;
    
    // Calculate dynamic min/max values from actual data
    const allDataValues = chartData.datasets.flatMap(dataset => 
      dataset.data ? dataset.data.filter(v => v !== null && v !== undefined && !isNaN(v)) : []
    );
    
    const minValue = allDataValues.length > 0 ? Math.min(...allDataValues) : 0;
    const maxValue = allDataValues.length > 0 ? Math.max(...allDataValues) : 1;
    
    // Add 10% padding to min/max for better visualization
    const padding = (maxValue - minValue) * 0.1;
    const dynamicMin = Math.max(0, minValue - padding);
    const dynamicMax = maxValue + padding;

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      layout: {
        padding: {
          left: 10,
          right: 10,
          top: 30,
          bottom: 60
        }
      },
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: title || chartConfig.title || 'District Similarity Analysis Chart',
          font: {
            size: 22,
            weight: 'bold'
          },
          padding: {
            top: 10,
            bottom: 30
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          titleFont: {
            size: 16
          },
          bodyFont: {
            size: 14
          },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: function(context) {
              const label = context.dataset.label || '';
              const value = context.parsed.y;
              const formattedValue = `${value.toFixed(1)}%`;  // Values are already percentages
              return `${label}: ${formattedValue}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            font: {
              size: 11,
              weight: '500'
            },
            maxRotation: 45,
            minRotation: 0,
            padding: 5,
            maxTicksLimit: 15,
            callback: function(value, index, values) {
              const label = this.getLabelForValue(value);
              // Truncate long names
              return label && label.length > 15 ? label.substring(0, 12) + '...' : label;
            }
          },
          grid: {
            display: false
          },
          title: {
            display: true,
            text: 'Districts',
            font: {
              size: 12,
              weight: 'bold'
            }
          }
        },
        y: {
          ticks: {
            font: {
              size: 12,
              weight: '500'
            },
            padding: 8,
            stepSize: 5,  // Adjusted for percentage scale
            callback: function(value) {
              return `${value.toFixed(0)}%`;  // Values are already percentages
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)',
            lineWidth: 1
          },
          title: {
            display: true,
            text: 'Prevalence (%)',
            font: {
              size: 13,
              weight: 'bold'
            }
          },
          beginAtZero: true,
          min: dynamicMin,
          max: dynamicMax,
          grace: '5%'
        }
      },
      elements: {
        bar: {
          borderRadius: 6,
          borderSkipped: false,
        }
      },
      datasets: {
        bar: {
          maxBarThickness: 50,
          minBarLength: 5,
          categoryPercentage: 0.9,
          barPercentage: 0.8
        }
      },
      indexAxis: chartConfig.indexAxis || 'x',
      ...(chartConfig.options || {})
    };

    // Prepare final chart data with validation and data cleaning
    const finalChartData = {
      labels: chartData.labels || [],
      datasets: chartData.datasets.map(dataset => {
        const cleanData = (dataset.data || []).map(value => {
          const numValue = Number(value);
          return isNaN(numValue) ? 0 : numValue;
        });
        
        return {
          ...dataset,
          data: cleanData,
          backgroundColor: dataset.backgroundColor || '#3B82F6',
          borderColor: dataset.borderColor || '#2563EB',
          borderWidth: dataset.borderWidth || 1,
          minBarLength: 5
        };
      })
    };

    return (
      <div className="w-full h-full" style={{ position: 'relative', minHeight: '400px' }}>
        <ChartComponent 
          data={finalChartData} 
          options={chartOptions}
          style={{ maxHeight: '100%', maxWidth: '100%' }}
        />
      </div>
    );
  };

  // Map click handler
  const onMapClick = useCallback((event) => {
    if (!event.features || !event.features[0] || !event.lngLat) {
      return;
    }

    const lng = event.lngLat.lng;
    const lat = event.lngLat.lat;
    
    if (isNaN(lng) || isNaN(lat)) {
      return;
    }

    const feature = event.features[0];
    const featureType = feature.properties?.type;
    
    if (featureType === 'district') {
      const districtName = feature.properties?.district_name;
      const stateName = feature.properties?.state_name;
      
      // Find district data
      const districtData = analysisInfo.districts.find(d => d.district_name === districtName);

      if (districtData) {
        setPopupInfo({
          longitude: lng,
          latitude: lat,
          type: 'district',
          data: districtData,
          analysisType: analysisInfo.analysisType
        });
      }
    }
  }, [analysisInfo.districts, analysisInfo.analysisType]);

  // Map hover handlers
  const onMouseEnter = useCallback((event) => {
    if (event.features && event.features[0]) {
      const feature = event.features[0];
      const featureName = feature.properties?.district_name;
      setHoveredFeature(featureName);
      if (mapRef.current) {
        mapRef.current.getCanvas().style.cursor = 'pointer';
      }
    }
  }, []);

  const onMouseLeave = useCallback(() => {
    setHoveredFeature(null);
    if (mapRef.current) {
      mapRef.current.getCanvas().style.cursor = '';
    }
  }, []);

  if (chartOnly) {
    const currentPage = chartPages[currentChartPage] || null;

    return (
      <div className="space-y-6">
        {chartPages.length > 0 ? (
          <>
            {/* Enhanced Page Navigation Header */}
            <div className="bg-gradient-to-r from-gray-50 to-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
              <div className={`bg-gradient-to-r ${currentPage?.gradient || 'from-blue-500 to-purple-600'} p-6 text-white`}>
                <div className="flex justify-between items-start">
                  <div className="flex items-center gap-3">
                    <div className="text-3xl">{currentPage?.icon || '📊'}</div>
                    <div>
                      <h2 className="text-2xl font-bold">{currentPage?.title}</h2>
                      <p className="text-blue-100 mt-1">{currentPage?.subtitle}</p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="bg-white/20 backdrop-blur-sm rounded-lg px-3 py-1 text-sm">
                      Chart {currentChartPage + 1} of {chartPages.length}
                    </div>
                  </div>
                </div>
              </div>

              {/* Navigation Controls */}
              <div className="p-6 bg-white">
                <div className="flex justify-between items-center">
                  <button
                    onClick={() => setCurrentChartPage(Math.max(0, currentChartPage - 1))}
                    disabled={currentChartPage === 0}
                    className={`group flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                      currentChartPage === 0
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                    }`}
                  >
                    <span className="transition-transform group-hover:-translate-x-1">←</span>
                    Previous
                  </button>

                  <div className="flex gap-3">
                    {chartPages.map((page, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentChartPage(index)}
                        className={`group relative overflow-hidden rounded-xl transition-all duration-300 ${
                          index === currentChartPage
                            ? 'w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg'
                            : 'w-12 h-12 bg-gray-100 text-gray-600 hover:bg-gray-200 hover:shadow-md'
                        }`}
                        title={page.title}
                      >
                        <div className="absolute inset-0 flex items-center justify-center">
                          {index === currentChartPage ? (
                            <div className="text-center">
                              <div className="text-lg">{page.icon}</div>
                              <div className="text-xs font-bold">{index + 1}</div>
                            </div>
                          ) : (
                            <div className="font-semibold">{index + 1}</div>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>

                  <button
                    onClick={() => setCurrentChartPage(Math.min(chartPages.length - 1, currentChartPage + 1))}
                    disabled={currentChartPage === chartPages.length - 1}
                    className={`group flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                      currentChartPage === chartPages.length - 1
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-green-500 to-green-600 text-white hover:from-green-600 hover:to-green-700 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                    }`}
                  >
                    Next
                    <span className="transition-transform group-hover:translate-x-1">→</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Current Page Chart */}
            {currentPage && (
              <div className="bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
                <div className={`bg-gradient-to-r ${currentPage.gradient} p-6 text-white`}>
                  <div className="flex items-start gap-4">
                    <div className="text-4xl">{currentPage.icon}</div>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold mb-2">Indicator-Specific Analysis</h3>
                      <p className="text-blue-100 leading-relaxed mb-3">
                        {currentPage.description}
                      </p>
                      
                      {/* Enhanced indicator-specific information */}
                      {currentPage.indicatorSpecific && (
                        <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4 mt-3">
                          <div className="grid grid-cols-3 gap-4 text-sm mb-3">
                            <div>
                              <span className="font-semibold">Districts Selected:</span>
                              <div className="text-blue-100">{currentPage.selectedDistricts} districts</div>
                            </div>
                            <div>
                              <span className="font-semibold">Algorithm:</span>
                              <div className="text-blue-100 capitalize">{currentPage.algorithmUsed?.replace('_', ' ')}</div>
                            </div>
                            <div>
                              <span className="font-semibold">Selection Type:</span>
                              <div className="text-blue-100 capitalize">{currentPage.analysisType} values</div>
                            </div>
                          </div>
                          
                          {/* Value statistics */}
                          {currentPage.valueStats && Object.keys(currentPage.valueStats).length > 0 && (
                            <div className="grid grid-cols-4 gap-3 text-xs">
                              <div>
                                <span className="font-medium">Range:</span>
                                <div className="text-blue-100">{currentPage.valueStats.min}% - {currentPage.valueStats.max}%</div>
                              </div>
                              <div>
                                <span className="font-medium">Average:</span>
                                <div className="text-blue-100">{currentPage.valueStats.average}%</div>
                              </div>
                              <div>
                                <span className="font-medium">Spread:</span>
                                <div className="text-blue-100">{currentPage.valueStats.range}pp</div>
                              </div>
                              <div>
                                <span className="font-medium">Quality:</span>
                                <div className="text-blue-100">{currentPage.valueStats.similarity_measure}</div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="p-8">
                  {currentPage.charts.map((chart, index) => (
                    <div key={index} className="w-full">
                      <div className="relative">
                        <div className={`absolute inset-0 bg-gradient-to-r ${currentPage.gradient} rounded-xl opacity-10`}></div>
                        <div className="relative bg-white rounded-xl border-2 border-gray-100 p-6" style={{ minHeight: '600px' }}>
                          <div className="w-full h-full" style={{ height: '550px', position: 'relative' }}>
                            <EnhancedChart chartConfig={chart} />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No chart data available for district similarity visualization</p>
          </div>
        )}
      </div>
    );
  }

  if (mapOnly) {
    return (
      <div className="space-y-4">
        {/* Map visualization */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="h-96 relative">
            <Map
              ref={mapRef}
              mapboxAccessToken={mapboxAccessToken}
              initialViewState={{
                longitude: 78.9629,
                latitude: 20.5937,
                zoom: 4
              }}
              style={{ width: '100%', height: '100%' }}
              mapStyle="mapbox://styles/mapbox/light-v11"
              onClick={onMapClick}
              onMouseMove={onMouseEnter}
              onMouseLeave={onMouseLeave}
              interactiveLayerIds={['district-fill']}
              cursor="pointer"
            >
              {/* District boundaries */}
              {boundaryData.districtFeatures.features.length > 0 && (
                <Source id="district-boundaries" type="geojson" data={boundaryData.districtFeatures}>
                  <Layer
                    id="district-fill"
                    type="fill"
                    paint={{
                      'fill-color': analysisInfo.analysisType === 'similar' 
                        ? [
                            'interpolate',
                            ['linear'],
                            ['get', 'performance_score'],
                            0, '#E3F2FD',
                            0.2, '#BBDEFB',
                            0.4, '#90CAF9',
                            0.6, '#64B5F6',
                            0.8, '#42A5F5',
                            1.0, '#2196F3'
                          ]
                        : [
                            'interpolate',
                            ['linear'],
                            ['get', 'performance_score'],
                            0, '#FCE4EC',
                            0.2, '#F8BBD9',
                            0.4, '#F48FB1',
                            0.6, '#F06292',
                            0.8, '#EC407A',
                            1.0, '#E91E63'
                          ],
                      'fill-opacity': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredFeature || ''],
                        0.8,
                        0.6
                      ]
                    }}
                  />
                  <Layer
                    id="district-stroke"
                    type="line"
                    paint={{
                      'fill-color': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredFeature || ''],
                        '#1976D2',
                        '#757575'
                      ],
                      'line-width': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredFeature || ''],
                        2,
                        1
                      ]
                    }}
                  />
                </Source>
              )}

              {/* Popup for feature details */}
              {popupInfo && !isNaN(popupInfo.longitude) && !isNaN(popupInfo.latitude) && (
                <Popup
                  longitude={popupInfo.longitude}
                  latitude={popupInfo.latitude}
                  anchor="top"
                  onClose={() => setPopupInfo(null)}
                  maxWidth="400px"
                >
                  <div className="p-3">
                    <h3 className="font-bold text-base text-gray-800 mb-2">
                      🏙️ {popupInfo.data.district_name}
                    </h3>
                    <p className="text-sm text-gray-600 mb-3">
                      {popupInfo.data.state_name}
                    </p>
                    
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Analysis Type:</span>
                        <span className="font-semibold text-blue-600 capitalize">
                          {popupInfo.analysisType} Patterns
                        </span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-gray-600">Indicators Analyzed:</span>
                        <span className="font-semibold text-gray-700">
                          {analysisInfo.totalIndicators}
                        </span>
                      </div>
                    </div>

                    {/* Indicator values */}
                    {popupInfo.data.indicators && Object.keys(popupInfo.data.indicators).length > 0 && (
                      <div className="mt-3 p-2 bg-gray-50 rounded">
                        <div className="text-xs font-medium text-gray-700 mb-1">
                          Indicator Values:
                        </div>
                        <div className="text-xs text-gray-600 max-h-20 overflow-y-auto">
                          {Object.entries(popupInfo.data.indicators).map(([indicatorId, indicator], index) => {
                            const indicatorInfo = analysisInfo.indicators.find(ind => ind.indicator_id === parseInt(indicatorId));
                            const indicatorName = indicatorInfo ? indicatorInfo.indicator_name : `Indicator ${indicatorId}`;
                            const value = indicator.prevalence_current;
                            
                            return (
                              <div key={index} className="flex justify-between">
                                <span className="truncate mr-2">{indicatorName}:</span>
                                <span className="font-medium">
                                  {value !== null ? `${value.toFixed(1)}%` : 'N/A'}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </Popup>
              )}
            </Map>
          </div>
          
          {/* Legend */}
          <div className="p-4 bg-gray-50 border-t">
            <h4 className="text-sm font-semibold text-gray-800 mb-2">
              District {analysisInfo.analysisType === 'similar' ? 'Similarity' : 'Diversity'} Analysis
            </h4>
            <div className="flex flex-wrap gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div 
                  className="w-4 h-4 rounded" 
                  style={{ 
                    backgroundColor: analysisInfo.analysisType === 'similar' ? '#2196F3' : '#E91E63'
                  }}
                ></div>
                <span>
                  {analysisInfo.analysisType === 'similar' 
                    ? 'Districts with similar patterns' 
                    : 'Districts with diverse patterns'
                  }
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div 
                  className="w-4 h-4 rounded" 
                  style={{ 
                    backgroundColor: analysisInfo.analysisType === 'similar' ? '#E3F2FD' : '#FCE4EC'
                  }}
                ></div>
                <span>Lower performance range</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Full component with summary, map, and charts
  return (
    <div className="space-y-6">
      {/* Summary Information */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {analysisInfo.totalDistricts}
            </div>
            <div className="text-sm text-gray-600">Districts Found</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {analysisInfo.totalIndicators}
            </div>
            <div className="text-sm text-gray-600">Indicators Analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600 capitalize">
              {analysisInfo.analysisType}
            </div>
            <div className="text-sm text-gray-600">Analysis Type</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {analysisInfo.categoryName || 'Mixed Categories'}
            </div>
            <div className="text-sm text-gray-600">Category Focus</div>
          </div>
        </div>
      </div>

      {/* Map visualization */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="h-96 relative">
          <Map
            ref={mapRef}
            mapboxAccessToken={mapboxAccessToken}
            initialViewState={{
              longitude: 78.9629,
              latitude: 20.5937,
              zoom: 4
            }}
            style={{ width: '100%', height: '100%' }}
            mapStyle="mapbox://styles/mapbox/light-v11"
            onClick={onMapClick}
            onMouseMove={onMouseEnter}
            onMouseLeave={onMouseLeave}
            interactiveLayerIds={['district-fill']}
            cursor="pointer"
          >
            {/* District boundaries */}
            {boundaryData.districtFeatures.features.length > 0 && (
              <Source id="district-boundaries" type="geojson" data={boundaryData.districtFeatures}>
                <Layer
                  id="district-fill"
                  type="fill"
                  paint={{
                    'fill-color': analysisInfo.analysisType === 'similar' 
                      ? [
                          'interpolate',
                          ['linear'],
                          ['get', 'performance_score'],
                          0, '#E3F2FD',
                          0.2, '#BBDEFB',
                          0.4, '#90CAF9',
                          0.6, '#64B5F6',
                          0.8, '#42A5F5',
                          1.0, '#2196F3'
                        ]
                      : [
                          'interpolate',
                          ['linear'],
                          ['get', 'performance_score'],
                          0, '#FCE4EC',
                          0.2, '#F8BBD9',
                          0.4, '#F48FB1',
                          0.6, '#F06292',
                          0.8, '#EC407A',
                          1.0, '#E91E63'
                        ],
                    'fill-opacity': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredFeature || ''],
                      0.8,
                      0.6
                    ]
                  }}
                />
                <Layer
                  id="district-stroke"
                  type="line"
                  paint={{
                    'line-color': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredFeature || ''],
                      '#1976D2',
                      '#757575'
                    ],
                    'line-width': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredFeature || ''],
                      2,
                      1
                    ]
                  }}
                />
              </Source>
            )}

            {/* Popup for feature details */}
            {popupInfo && !isNaN(popupInfo.longitude) && !isNaN(popupInfo.latitude) && (
              <Popup
                longitude={popupInfo.longitude}
                latitude={popupInfo.latitude}
                anchor="top"
                onClose={() => setPopupInfo(null)}
                maxWidth="400px"
              >
                <div className="p-3">
                  <h3 className="font-bold text-base text-gray-800 mb-2">
                    🏙️ {popupInfo.data.district_name}
                  </h3>
                  <p className="text-sm text-gray-600 mb-3">
                    {popupInfo.data.state_name}
                  </p>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Analysis Type:</span>
                      <span className="font-semibold text-blue-600 capitalize">
                        {popupInfo.analysisType} Patterns
                      </span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-600">Indicators Analyzed:</span>
                      <span className="font-semibold text-gray-700">
                        {analysisInfo.totalIndicators}
                      </span>
                    </div>
                  </div>

                  {/* Indicator values */}
                  {popupInfo.data.indicators && Object.keys(popupInfo.data.indicators).length > 0 && (
                    <div className="mt-3 p-2 bg-gray-50 rounded">
                      <div className="text-xs font-medium text-gray-700 mb-1">
                        Indicator Values:
                      </div>
                      <div className="text-xs text-gray-600 max-h-20 overflow-y-auto">
                        {Object.entries(popupInfo.data.indicators).map(([indicatorId, indicator], index) => {
                          const indicatorInfo = analysisInfo.indicators.find(ind => ind.indicator_id === parseInt(indicatorId));
                          const indicatorName = indicatorInfo ? indicatorInfo.indicator_name : `Indicator ${indicatorId}`;
                          const value = indicator.prevalence_current;
                          
                          return (
                            <div key={index} className="flex justify-between">
                              <span className="truncate mr-2">{indicatorName}:</span>
                              <span className="font-medium">
                                {value !== null ? `${value.toFixed(1)}%` : 'N/A'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </Popup>
            )}
          </Map>
        </div>
        
        {/* Legend */}
        <div className="p-4 bg-gray-50 border-t">
          <h4 className="text-sm font-semibold text-gray-800 mb-2">
            District {analysisInfo.analysisType === 'similar' ? 'Similarity' : 'Diversity'} Analysis
          </h4>
          <div className="flex flex-wrap gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div 
                className="w-4 h-4 rounded" 
                style={{ 
                  backgroundColor: analysisInfo.analysisType === 'similar' ? '#2196F3' : '#E91E63'
                }}
              ></div>
              <span>
                {analysisInfo.analysisType === 'similar' 
                  ? 'Districts with similar patterns' 
                  : 'Districts with diverse patterns'
                }
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div 
                className="w-4 h-4 rounded" 
                style={{ 
                  backgroundColor: analysisInfo.analysisType === 'similar' ? '#E3F2FD' : '#FCE4EC'
                }}
              ></div>
              <span>Lower performance range</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chart visualizations */}
      {analysisInfo.chartData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {analysisInfo.chartData.map((chart, index) => (
            <div key={index} className="bg-white rounded-lg shadow p-4">
              <EnhancedChart chartConfig={chart} />
              {chart.description && (
                <p className="text-sm text-gray-600 mt-2 text-center">
                  {chart.description}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Analysis text */}
      {analysisInfo.analysis && (
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">📋 Analysis</h3>
          <div 
            className="text-gray-700 whitespace-pre-line leading-relaxed"
            dangerouslySetInnerHTML={{ 
              __html: analysisInfo.analysis.replace(/\n/g, '<br/>') 
            }}
          />
        </div>
      )}
    </div>
  );
};

export default DistrictSimilarityAnalysis;
