import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Map, Source, Layer, Popup } from 'react-map-gl';
import { Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

// Mapbox access token
const mapboxAccessToken = process.env.REACT_APP_MAPBOX_TOKEN;

const MultiIndicatorPerformance = ({ data, mapOnly = false, chartOnly = false }) => {
  const [popupInfo, setPopupInfo] = useState(null);
  const [hoveredDistrict, setHoveredDistrict] = useState(null);
  const [currentChartPage, setCurrentChartPage] = useState(0);
  const mapRef = useRef();

  // Process the data for easier access
  const performanceInfo = useMemo(() => {
    // Add comprehensive debugging
    console.log('🔍 MultiIndicatorPerformance RAW DATA:', data);
    
    if (!data) {
      console.log('❌ No data provided');
      return {
        districts: [],
        totalDistricts: 0,
        totalIndicators: 0,
        performanceType: 'unknown',
        categoryName: null,
        chartData: [],
        analysis: ''
      };
    }

    // Enhanced data extraction with multiple fallback paths
    let districts = [];
    let chartData = [];
    
    console.log('🔍 Data structure analysis:', {
      hasDistricts: !!data.districts,
      hasChartData: !!data.chart_data,
      hasData: !!data.data,
      dataType: typeof data.data,
      dataIsArray: Array.isArray(data.data),
      keys: Object.keys(data)
    });
    
    // Path 1: Direct access
    if (data.districts && Array.isArray(data.districts)) {
      console.log('✅ Path 1: Direct access');
      districts = data.districts;
      chartData = data.chart_data || [];
    }
    // Path 2: Nested in data array (from function_calls)
    else if (data.data && Array.isArray(data.data)) {
      console.log('🔍 Path 2: Checking nested data array');
      for (const item of data.data) {
        console.log('🔍 Checking item:', item);
        if (item.result) {
          if (item.result.districts && Array.isArray(item.result.districts)) {
            console.log('✅ Path 2: Found districts in nested structure');
            districts = item.result.districts;
            chartData = item.result.chart_data || [];
            break;
          }
        }
      }
    }
    // Path 3: Function result structure
    else if (data.data && !Array.isArray(data.data) && data.data.districts) {
      console.log('✅ Path 3: Function result structure');
      districts = data.data.districts;
      chartData = data.data.chart_data || [];
    }

    console.log('🔍 After initial extraction:', {
      districtsFound: districts.length,
      chartDataFound: Array.isArray(chartData) ? chartData.length : 'not array',
      chartDataType: typeof chartData
    });

    // Enhanced chart data extraction
    if (!chartData || !Array.isArray(chartData)) {
      console.log('🔍 Trying alternative chart data paths');
      // Try alternative paths for chart data
      chartData = data.chart_data || 
                 data.data?.chart_data || 
                 (data.data && Array.isArray(data.data) && data.data[0]?.result?.chart_data) || 
                 [];
      console.log('🔍 Alternative chart data result:', chartData);
    }

    // Convert chart data object to array if needed
    if (chartData && typeof chartData === 'object' && !Array.isArray(chartData)) {
      console.log('🔍 Converting chart data object to array');
      console.log('🔍 Original chart data object:', chartData);
      console.log('🔍 Object keys:', Object.keys(chartData));
      chartData = Object.values(chartData).filter(chart => chart && typeof chart === 'object');
      console.log('🔍 Converted chart data:', chartData);
      console.log('🔍 Each chart in converted data:');
      chartData.forEach((chart, index) => {
        console.log(`Chart ${index}:`, {
          title: chart.title,
          type: chart.type,
          hasData: !!chart.data,
          hasLabels: !!chart.data?.labels,
          hasDatasets: !!chart.data?.datasets,
          labelsLength: chart.data?.labels?.length,
          datasetsLength: chart.data?.datasets?.length
        });
      });
    }

    console.log('🔍 Final extraction result:', {
      districtsCount: districts.length,
      chartDataCount: Array.isArray(chartData) ? chartData.length : 0,
      chartDataIsArray: Array.isArray(chartData)
    });

    if (!districts || !districts.length) {
      console.log('❌ No districts found');
      return {
        districts: [],
        totalDistricts: 0,
        totalIndicators: 0,
        performanceType: 'unknown',
        categoryName: null,
        chartData: [],
        analysis: ''
      };
    }

    const result = {
      districts: districts || [],
      totalDistricts: data.total_districts || districts.length || 0,
      totalIndicators: data.total_indicators || 0,
      performanceType: data.performance_type || 'unknown',
      categoryName: data.category_name,
      chartData: Array.isArray(chartData) ? chartData : [],
      analysis: data.analysis || ''
    };

    console.log('✅ Final performanceInfo result:', result);
    return result;
  }, [data]);

  // Process boundary data for map visualization
  const boundaryData = useMemo(() => {
    let boundaryData = data?.boundary_data || data?.boundary || [];
    
    // Handle case where boundary data might be nested
    if (!boundaryData.length && data?.data && Array.isArray(data.data)) {
      for (const item of data.data) {
        if (item.result && (item.result.boundary_data || item.result.boundary)) {
          boundaryData = item.result.boundary_data || item.result.boundary;
          break;
        }
      }
    }
    
    
    if (!boundaryData || !boundaryData.length || !performanceInfo.districts) {
      return { type: 'FeatureCollection', features: [] };
    }

    try {
      const features = boundaryData
        .map((boundary) => {
          try {
            const geometry = typeof boundary.geometry === 'string' 
              ? JSON.parse(boundary.geometry) 
              : boundary.geometry;

            if (!geometry || !geometry.coordinates || !Array.isArray(geometry.coordinates)) {
              return null;
            }

            const districtName = boundary.district_name || boundary.district;
            const districtData = performanceInfo.districts.find(d => 
              d.district_name === districtName
            );

            if (!districtData) {
              return null;
            }

            // Calculate performance score for color coding (0-1 scale) with validation
            let performanceScore = 0;
            if (districtData.performance_index_2021 !== null && districtData.performance_index_2021 !== undefined) {
              performanceScore = Math.max(0, Math.min(1, districtData.performance_index_2021));
            } else if (districtData.performance_score !== null && districtData.performance_score !== undefined) {
              performanceScore = Math.max(0, Math.min(1, districtData.performance_score));
            }

            return {
              type: 'Feature',
              properties: {
                district_name: districtName,
                state_name: districtData.state_name,
                performance_index_2021: districtData.performance_index_2021,
                performance_index_2016: districtData.performance_index_2016,
                absolute_change: districtData.absolute_change,
                relative_change: districtData.relative_change,
                performance_score: performanceScore,
                total_indicators: districtData.total_indicators
              },
              geometry: geometry
            };
          } catch (error) {
            console.warn('Error processing boundary in multi-indicator analysis:', boundary.district_name || 'unknown', error);
            return null;
          }
        })
        .filter(feature => feature !== null);

      return { type: 'FeatureCollection', features };
    } catch (error) {
      console.error('Error processing boundary data for multi-indicator analysis:', error);
      return { type: 'FeatureCollection', features: [] };
    }
  }, [data?.boundary_data, data?.boundary, data?.data, performanceInfo.districts]);

  // Organize charts by category for pagination - always calculated to avoid conditional hooks
  const chartPages = useMemo(() => {
    console.log('🔍 Chart Pages Processing - Input:', {
      hasChartData: !!performanceInfo.chartData,
      chartDataLength: performanceInfo.chartData?.length,
      chartData: performanceInfo.chartData
    });
    
    if (!performanceInfo.chartData || performanceInfo.chartData.length === 0) {
      console.log('❌ No chart data found for pagination');
      return [];
    }

    // Categorize charts by type/purpose - focus on meaningful charts only
    const performanceChart = performanceInfo.chartData.find(chart => 
      chart.title?.toLowerCase().includes('performance index') && !chart.title?.toLowerCase().includes('change')
    );
    
    const changeChart = performanceInfo.chartData.find(chart => 
      chart.title?.toLowerCase().includes('change')
    );
    
    // Remove distribution chart as it's not meaningful for this analysis
    // const distributionChart = performanceInfo.chartData.find(chart => 
    //   chart.title?.toLowerCase().includes('distribution')
    // );

    console.log('🔍 Chart categorization:', {
      performanceChart: !!performanceChart,
      changeChart: !!changeChart,
      performanceChartTitle: performanceChart?.title,
      changeChartTitle: changeChart?.title
    });

    const pages = [];

    // Page 1: Performance Index (Best Performing)
    if (performanceChart) {
      pages.push({
        title: "District Performance Rankings",
        subtitle: "Top-performing districts in comprehensive health indicator analysis",
        charts: [performanceChart],
        description: "This chart shows the composite performance index for districts, calculated using min-max normalization and direction alignment across all health indicators. Higher values indicate better overall performance.",
        gradient: "from-blue-500 to-blue-600"
      });
    }

    // Page 2: Performance Change (Trend Analysis)
    if (changeChart) {
      pages.push({
        title: "📈 Performance Improvement Trends",
        subtitle: "District progress from 2016 to 2021",
        charts: [changeChart],
        description: "This chart displays how district performance has evolved over the 5-year period, highlighting significant improvements and identifying areas needing attention. Positive changes indicate progress in health outcomes.",
        icon: "📈",
        gradient: "from-green-500 to-green-600"
      });
    }

    console.log('🔍 Final chart pages result:', {
      pagesCount: pages.length,
      pages: pages.map(p => ({ title: p.title, chartsCount: p.charts?.length }))
    });

    return pages;
  }, [performanceInfo.chartData]);

  // Keyboard navigation for charts
  useEffect(() => {
    if (!chartOnly) return;

    const handleKeyPress = (event) => {
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
      
      if (event.key === 'ArrowLeft') {
        setCurrentChartPage(prev => Math.max(0, prev - 1));
      } else if (event.key === 'ArrowRight') {
        setCurrentChartPage(prev => Math.min((chartPages?.length || 1) - 1, prev + 1));
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [chartOnly, chartPages?.length]);

  // Map click handler
  const onMapClick = useCallback((event) => {
    if (!event.features || !event.features[0] || !event.lngLat) {
      return;
    }

    const lng = event.lngLat.lng;
    const lat = event.lngLat.lat;
    
    if (isNaN(lng) || isNaN(lat)) {
      console.warn('Invalid coordinates for multi-indicator popup:', lng, lat);
      return;
    }

    const feature = event.features[0];
    const districtName = feature.properties?.district_name;
    
    if (!districtName) {
      return;
    }

    // Find the complete district data
    const districtData = performanceInfo.districts.find(d => 
      d.district_name === districtName
    );

    if (districtData) {
      setPopupInfo({
        longitude: lng,
        latitude: lat,
        district: districtData
      });
    }
  }, [performanceInfo.districts]);

  // Map hover handlers
  const onMouseEnter = useCallback((event) => {
    if (event.features && event.features[0]) {
      setHoveredDistrict(event.features[0].properties?.district_name);
      if (mapRef.current) {
        mapRef.current.getCanvas().style.cursor = 'pointer';
      }
    }
  }, []);

  const onMouseLeave = useCallback(() => {
    setHoveredDistrict(null);
    if (mapRef.current) {
      mapRef.current.getCanvas().style.cursor = '';
    }
  }, []);

  // Enhanced Chart component
  const EnhancedChart = ({ chartConfig, title = null }) => {
    console.log('🎯 EnhancedChart called with:', {
      hasChartConfig: !!chartConfig,
      chartConfigKeys: chartConfig ? Object.keys(chartConfig) : 'none',
      title,
      chartConfig
    });

    // Comprehensive validation
    if (!chartConfig) {
      console.log('❌ No chart configuration provided');
      return <div className="text-gray-500 p-4">No chart configuration provided</div>;
    }

    // Handle different chart data structures
    let chartData = chartConfig.data || chartConfig;
    console.log('🎯 Chart data extracted:', {
      hasChartData: !!chartData,
      chartDataKeys: chartData ? Object.keys(chartData) : 'none',
      hasLabels: !!chartData?.labels,
      hasDatasets: !!chartData?.datasets,
      chartData
    });

    if (!chartData || (!chartData.labels && !chartData.datasets)) {
      console.log('❌ Invalid chart data structure');
      return <div className="text-gray-500 p-4">Invalid chart data structure</div>;
    }

    // Ensure we have valid datasets
    if (!chartData.datasets || !Array.isArray(chartData.datasets) || chartData.datasets.length === 0) {
      console.log('❌ No chart datasets available');
      return <div className="text-gray-500 p-4">No chart datasets available</div>;
    }

    // Debug: Log actual chart data values (only in development)
    if (process.env.NODE_ENV === 'development') {
      console.log('=== CHART DEBUG ===');
      console.log('Chart Title:', chartConfig.title || title);
      console.log('Chart Type:', chartConfig.type);
      console.log('Labels:', chartData.labels);
      console.log('Datasets:', chartData.datasets);
      if (chartData.datasets?.[0]?.data) {
        const validData = chartData.datasets[0].data.filter(v => v !== null && v !== undefined && !isNaN(v));
        if (validData.length > 0) {
          console.log('Data Values:', validData);
          console.log('Min Value:', Math.min(...validData));
          console.log('Max Value:', Math.max(...validData));
        }
      }
      console.log('==================');
    }

    // Force bar charts only based on user preference
    const ChartComponent = Bar;
    
    // Calculate dynamic min/max values from actual data
    const allDataValues = chartData.datasets.flatMap(dataset => 
      dataset.data ? dataset.data.filter(v => v !== null && v !== undefined && !isNaN(v)) : []
    );
    
    const minValue = allDataValues.length > 0 ? Math.min(...allDataValues) : 0;
    const maxValue = allDataValues.length > 0 ? Math.max(...allDataValues) : 1;
    const isChangeChart = (chartConfig.title || title || '').toLowerCase().includes('change');
    
    // Add 10% padding to min/max for better visualization
    const padding = (maxValue - minValue) * 0.1;
    const dynamicMin = isChangeChart ? Math.min(minValue - padding, -0.1) : Math.max(0, minValue - padding);
    const dynamicMax = isChangeChart ? Math.max(maxValue + padding, 0.1) : maxValue + padding;

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
          bottom: (chartConfig.title || title || '').includes('Performance Index') ? 100 : 60
        }
      },
      plugins: {
        legend: {
          display: false // Hide legend to avoid misleading dot indicators
        },
        title: {
          display: true,
          text: title || chartConfig.title || 'District Indicator Chart',
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
              const formattedValue = isChangeChart ? 
                `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%` :
                `${(value * 100).toFixed(1)}%`;
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
            maxRotation: 90,
            minRotation: 45,
            padding: 5,
            maxTicksLimit: 20,
            callback: function(value, index, values) {
              const label = this.getLabelForValue(value);
              // Truncate long district names
              return label && label.length > 12 ? label.substring(0, 10) + '...' : label;
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
            stepSize: isChangeChart ? 0.05 : 0.1,
            callback: function(value) {
              return isChangeChart ? 
                `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%` :
                `${(value * 100).toFixed(0)}%`;
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)',
            lineWidth: 1
          },
          title: {
            display: true,
            text: isChangeChart ? 'Change in Performance Index' : 'Performance Index',
            font: {
              size: 13,
              weight: 'bold'
            }
          },
          beginAtZero: !isChangeChart,
          min: dynamicMin,
          max: dynamicMax,
          grace: '5%',
          // Force display even with small values
          suggestedMin: isChangeChart ? -0.1 : 0,
          suggestedMax: isChangeChart ? 0.1 : 1
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
          maxBarThickness: 40,
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
        // Clean and validate data array
        const cleanData = (dataset.data || []).map(value => {
          // Convert to number and handle null/undefined
          const numValue = Number(value);
          return isNaN(numValue) ? 0 : numValue;
        });
        
        console.log('🎯 Dataset processing:', {
          originalData: dataset.data,
          cleanData,
          dataLength: cleanData.length,
          hasValidData: cleanData.some(v => v !== 0)
        });
        
        return {
          ...dataset,
          data: cleanData,
          backgroundColor: dataset.backgroundColor || '#3B82F6',
          borderColor: dataset.borderColor || '#2563EB',
          borderWidth: dataset.borderWidth || 1,
          // Force minimum bar height for visibility
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

  if (chartOnly) {
    console.log('🔍 Chart Only Mode - State:', {
      chartPagesLength: chartPages.length,
      currentChartPage,
      currentPageExists: !!chartPages[currentChartPage]
    });
    
    const currentPage = chartPages[currentChartPage] || null;

    return (
      <div className="space-y-6">
        {chartPages.length > 0 ? (
          <>
            {/* Enhanced Page Navigation Header */}
            <div className="bg-gradient-to-r from-gray-50 to-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
              {/* Header with gradient background */}
              <div className={`bg-gradient-to-r ${currentPage?.gradient || 'from-blue-500 to-purple-600'} p-6 text-white`}>
                <div className="flex justify-between items-start">
                  <div className="flex items-center gap-3">
                    <div className="text-3xl">{currentPage?.icon || '📊'}</div>
                    <div>
                      <h2 className="text-2xl font-bold">{currentPage?.title}</h2>
                      <p className="text-blue-100 mt-1">{currentPage?.subtitle}</p>
                    </div>
                  </div>
                  
                  {/* Enhanced Page Counter */}
                  <div className="text-right">
                    <div className="bg-white/20 backdrop-blur-sm rounded-lg px-3 py-1 text-sm">
                      Page {currentChartPage + 1} of {chartPages.length}
                    </div>
                    <div className="text-xs text-blue-100 mt-1 flex items-center gap-1">
                      <span>⌨️ Use</span>
                      <kbd className="bg-white/20 px-1 rounded text-xs">←</kbd>
                      <kbd className="bg-white/20 px-1 rounded text-xs">→</kbd>
                      <span>keys</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Navigation Controls */}
              <div className="p-6 bg-white">
                <div className="flex justify-between items-center">
                  {/* Previous Button */}
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

                  {/* Enhanced Page Indicators */}
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
                        {index === currentChartPage && (
                          <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                        )}
                      </button>
                    ))}
                  </div>

                  {/* Next Button */}
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

            {/* Enhanced Current Page Chart */}
            {currentPage && (
              <div className="bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
                {/* Enhanced Page Description */}
                <div className={`bg-gradient-to-r ${currentPage.gradient} p-6 text-white`}>
                  <div className="flex items-start gap-4">
                    <div className="text-4xl">{currentPage.icon}</div>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold mb-2">Analysis Insights</h3>
                      <p className="text-blue-100 leading-relaxed">
                        {currentPage.description}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Chart Display with Enhanced Styling */}
                <div className="p-8">
                  {currentPage.charts.map((chart, index) => (
                    <div key={index} className="w-full">
                      {/* Chart Container with Gradient Border */}
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
        ) : performanceInfo.chartData && performanceInfo.chartData.length > 0 ? (
          <>
            {/* Fallback: Direct chart rendering without categorization */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">📊 Multi-Indicator Performance Charts</h2>
              <p className="text-gray-600 text-sm mb-6">Direct chart display</p>
              
              <div className="grid grid-cols-1 gap-6">
                {performanceInfo.chartData.map((chart, index) => (
                  <div key={index} className="bg-white rounded-lg shadow-lg p-4">
                    <h3 className="text-lg font-semibold mb-2">Chart {index + 1}: {chart.title || 'Untitled'}</h3>
                    
                    {/* Debug info */}
                    <div className="mb-4 p-2 bg-gray-100 rounded text-xs">
                      <strong>Debug Info:</strong> 
                      Type: {chart.type}, 
                      Has Data: {!!chart.data ? 'Yes' : 'No'}, 
                      Labels: {chart.data?.labels?.length || 0}, 
                      Datasets: {chart.data?.datasets?.length || 0}
                    </div>
                    
                    <div className="w-full bg-gray-50 rounded-lg p-4" style={{ height: '500px', position: 'relative' }}>
                      <div className="w-full bg-white rounded-lg shadow-sm" style={{ height: '420px', position: 'relative', padding: '16px' }}>
                        {chart && chart.data && chart.data.datasets && chart.data.datasets.length > 0 ? (
                          <EnhancedChart chartConfig={chart} />
                        ) : (
                          // Test with a simple hardcoded chart to verify Chart.js is working
                          <div className="h-full w-full">
                            <h4 className="text-sm mb-2">Test Chart (Chart.js verification):</h4>
                            <div style={{ height: '350px', position: 'relative' }}>
                              <Bar
                                data={{
                                  labels: ['Test A', 'Test B', 'Test C'],
                                  datasets: [{
                                    label: 'Test Data',
                                    data: [0.5, 0.7, 0.3],
                                    backgroundColor: '#3B82F6',
                                    borderColor: '#2563EB',
                                    borderWidth: 1
                                  }]
                                }}
                                options={{
                                  responsive: true,
                                  maintainAspectRatio: false,
                                  scales: {
                                    y: {
                                      beginAtZero: true,
                                      max: 1
                                    }
                                  }
                                }}
                              />
                            </div>
                          </div>
                        )}
                        {!(chart && chart.data && chart.data.datasets && chart.data.datasets.length > 0) && (
                          <div className="flex items-center justify-center h-full">
                            <div className="text-center">
                              <p className="text-gray-500 mb-4">Chart data structure issue</p>
                              <p className="text-xs text-gray-400 mb-4">
                                Expected: chart.data.datasets, Found: {typeof chart?.data}
                              </p>
                              <div className="text-xs text-left bg-gray-100 p-2 rounded">
                                <strong>Available chart properties:</strong><br/>
                                {chart ? Object.keys(chart).join(', ') : 'None'}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Raw data display for debugging */}
                    <details className="mt-2">
                      <summary className="text-xs text-gray-500 cursor-pointer">Show Raw Chart Data</summary>
                      <pre className="text-xs bg-gray-100 p-2 mt-2 overflow-auto max-h-40">
                        {JSON.stringify(chart, null, 2)}
                      </pre>
                    </details>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No chart data available for visualization</p>
            <p className="text-xs mt-2">Debug: {JSON.stringify({
              hasData: !!data,
              hasPerformanceInfo: !!performanceInfo,
              chartDataLength: performanceInfo?.chartData?.length || 0
            })}</p>
          </div>
        )}

        {/* Analysis text */}
        {performanceInfo.analysis && (
          <div className="bg-gray-50 rounded-lg p-6 mt-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">📋 Detailed Analysis</h3>
            <div 
              className="text-gray-700 whitespace-pre-line leading-relaxed"
              dangerouslySetInnerHTML={{ 
                __html: performanceInfo.analysis.replace(/\n/g, '<br/>') 
              }}
            />
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
              interactiveLayerIds={['multi-indicator-fill']}
              cursor="pointer"
            >
              {/* District boundaries with performance-based coloring */}
              {boundaryData.features.length > 0 && (
                <Source id="multi-indicator-districts" type="geojson" data={boundaryData}>
                  <Layer
                    id="multi-indicator-fill"
                    type="fill"
                    paint={{
                      'fill-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'performance_score'],
                        0, '#E74C3C',    // Red for poor performance
                        0.2, '#F39C12',  // Orange for below average
                        0.4, '#F1C40F',  // Yellow for average
                        0.6, '#52C41A',  // Light green for good
                        0.8, '#27AE60',  // Green for very good
                        1.0, '#2ECC71'   // Bright green for excellent
                      ],
                      'fill-opacity': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredDistrict || ''],
                        0.8,
                        0.6
                      ]
                    }}
                  />
                  <Layer
                    id="multi-indicator-stroke"
                    type="line"
                    paint={{
                      'line-color': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredDistrict || ''],
                        '#2C3E50',
                        '#7F8C8D'
                      ],
                      'line-width': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredDistrict || ''],
                        3,
                        1
                      ]
                    }}
                  />
                </Source>
              )}

              {/* Popup for district details */}
              {popupInfo && !isNaN(popupInfo.longitude) && !isNaN(popupInfo.latitude) && (
                <Popup
                  longitude={popupInfo.longitude}
                  latitude={popupInfo.latitude}
                  anchor="top"
                  onClose={() => setPopupInfo(null)}
                  maxWidth="350px"
                >
                  <div className="p-3">
                    <h3 className="font-bold text-base text-gray-800 mb-2">
                      {popupInfo.district.district_name}
                    </h3>
                    <p className="text-sm text-gray-600 mb-3">
                      {popupInfo.district.state_name}
                    </p>
                    
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Performance Index (2021):</span>
                        <span className="font-semibold text-blue-600">
                          {(popupInfo.district.performance_index_2021 * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-gray-600">Performance Index (2016):</span>
                        <span className="font-semibold text-gray-700">
                          {(popupInfo.district.performance_index_2016 * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-gray-600">Change:</span>
                        <span className={`font-semibold ${
                          popupInfo.district.absolute_change > 0 ? 'text-green-600' : 
                          popupInfo.district.absolute_change < 0 ? 'text-red-600' : 'text-gray-600'
                        }`}>
                          {popupInfo.district.absolute_change > 0 ? '+' : ''}
                          {(popupInfo.district.absolute_change * 100).toFixed(1)}% 
                          ({popupInfo.district.relative_change.toFixed(1)}%)
                        </span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-gray-600">Total Indicators:</span>
                        <span className="font-semibold text-gray-700">
                          {popupInfo.district.total_indicators}
                        </span>
                      </div>
                    </div>

                    {/* Performance rating */}
                    <div className="mt-3 p-2 rounded" style={{
                      backgroundColor: popupInfo.district.performance_index_2021 >= 0.8 ? '#d4edda' :
                                     popupInfo.district.performance_index_2021 >= 0.6 ? '#fff3cd' :
                                     popupInfo.district.performance_index_2021 >= 0.4 ? '#ffeaa7' :
                                     '#f8d7da'
                    }}>
                      <div className="text-xs font-medium text-center">
                        {popupInfo.district.performance_index_2021 >= 0.8 ? 'Excellent Performance' :
                         popupInfo.district.performance_index_2021 >= 0.6 ? 'Good Performance' :
                         popupInfo.district.performance_index_2021 >= 0.4 ? 'Average Performance' :
                         'Needs Improvement'}
                      </div>
                    </div>
                  </div>
                </Popup>
              )}
            </Map>
          </div>
          
          {/* Legend */}
          <div className="p-4 bg-gray-50 border-t">
            <h4 className="text-sm font-semibold text-gray-800 mb-2">Performance Index Legend</h4>
            <div className="flex flex-wrap gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#2ECC71' }}></div>
                <span>Excellent (80-100%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#52C41A' }}></div>
                <span>Good (60-80%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F1C40F' }}></div>
                <span>Average (40-60%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F39C12' }}></div>
                <span>Below Average (20-40%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#E74C3C' }}></div>
                <span>Poor (0-20%)</span>
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
              {performanceInfo.totalDistricts}
            </div>
            <div className="text-sm text-gray-600">Districts Analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {performanceInfo.totalIndicators}
            </div>
            <div className="text-sm text-gray-600">Health Indicators</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {performanceInfo.performanceType.charAt(0).toUpperCase() + 
               performanceInfo.performanceType.slice(1)}
            </div>
            <div className="text-sm text-gray-600">Analysis Type</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {performanceInfo.categoryName || 'All Categories'}
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
            interactiveLayerIds={['multi-indicator-fill']}
            cursor="pointer"
          >
            {/* District boundaries with performance-based coloring */}
            {boundaryData.features.length > 0 && (
              <Source id="multi-indicator-districts" type="geojson" data={boundaryData}>
                <Layer
                  id="multi-indicator-fill"
                  type="fill"
                  paint={{
                    'fill-color': [
                      'interpolate',
                      ['linear'],
                      ['get', 'performance_score'],
                      0, '#E74C3C',    // Red for poor performance
                      0.2, '#F39C12',  // Orange for below average
                      0.4, '#F1C40F',  // Yellow for average
                      0.6, '#52C41A',  // Light green for good
                      0.8, '#27AE60',  // Green for very good
                      1.0, '#2ECC71'   // Bright green for excellent
                    ],
                    'fill-opacity': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredDistrict || ''],
                      0.8,
                      0.6
                    ]
                  }}
                />
                <Layer
                  id="multi-indicator-stroke"
                  type="line"
                  paint={{
                    'line-color': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredDistrict || ''],
                      '#2C3E50',
                      '#7F8C8D'
                    ],
                    'line-width': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredDistrict || ''],
                      3,
                      1
                    ]
                  }}
                />
              </Source>
            )}

            {/* Popup for district details */}
            {popupInfo && !isNaN(popupInfo.longitude) && !isNaN(popupInfo.latitude) && (
              <Popup
                longitude={popupInfo.longitude}
                latitude={popupInfo.latitude}
                anchor="top"
                onClose={() => setPopupInfo(null)}
                maxWidth="350px"
              >
                <div className="p-3">
                  <h3 className="font-bold text-base text-gray-800 mb-2">
                    {popupInfo.district.district_name}
                  </h3>
                  <p className="text-sm text-gray-600 mb-3">
                    {popupInfo.district.state_name}
                  </p>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Performance Index (2021):</span>
                      <span className="font-semibold text-blue-600">
                        {(popupInfo.district.performance_index_2021 * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-600">Performance Index (2016):</span>
                      <span className="font-semibold text-gray-700">
                        {(popupInfo.district.performance_index_2016 * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-600">Change:</span>
                      <span className={`font-semibold ${
                        popupInfo.district.absolute_change > 0 ? 'text-green-600' : 
                        popupInfo.district.absolute_change < 0 ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {popupInfo.district.absolute_change > 0 ? '+' : ''}
                        {(popupInfo.district.absolute_change * 100).toFixed(1)}% 
                        ({popupInfo.district.relative_change.toFixed(1)}%)
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Indicators:</span>
                      <span className="font-semibold text-gray-700">
                        {popupInfo.district.total_indicators}
                      </span>
                    </div>
                  </div>

                  {/* Performance rating */}
                  <div className="mt-3 p-2 rounded" style={{
                    backgroundColor: popupInfo.district.performance_index_2021 >= 0.8 ? '#d4edda' :
                                   popupInfo.district.performance_index_2021 >= 0.6 ? '#fff3cd' :
                                   popupInfo.district.performance_index_2021 >= 0.4 ? '#ffeaa7' :
                                   '#f8d7da'
                  }}>
                    <div className="text-xs font-medium text-center">
                      {popupInfo.district.performance_index_2021 >= 0.8 ? 'Excellent Performance' :
                       popupInfo.district.performance_index_2021 >= 0.6 ? 'Good Performance' :
                       popupInfo.district.performance_index_2021 >= 0.4 ? 'Average Performance' :
                       'Needs Improvement'}
                    </div>
                  </div>
                </div>
              </Popup>
            )}
          </Map>
        </div>
        
        {/* Legend */}
        <div className="p-4 bg-gray-50 border-t">
          <h4 className="text-sm font-semibold text-gray-800 mb-2">Performance Index Legend</h4>
          <div className="flex flex-wrap gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#2ECC71' }}></div>
              <span>Excellent (80-100%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#52C41A' }}></div>
              <span>Good (60-80%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F1C40F' }}></div>
              <span>Average (40-60%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F39C12' }}></div>
              <span>Below Average (20-40%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#E74C3C' }}></div>
              <span>Poor (0-20%)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chart visualizations */}
      {performanceInfo.chartData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {performanceInfo.chartData.map((chart, index) => (
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
      {performanceInfo.analysis && (
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Analysis</h3>
          <div 
            className="text-gray-700 whitespace-pre-line leading-relaxed"
            dangerouslySetInnerHTML={{ 
              __html: performanceInfo.analysis.replace(/\n/g, '<br/>') 
            }}
          />
        </div>
      )}
    </div>
  );
};

export default MultiIndicatorPerformance;
