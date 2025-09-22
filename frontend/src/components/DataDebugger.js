import React from 'react';

const DataDebugger = ({ data, title = "Data Structure Debug" }) => {
  const formatValue = (value, depth = 0) => {
    if (depth > 3) return "..."; // Prevent infinite recursion
    
    if (value === null) return "null";
    if (value === undefined) return "undefined";
    if (typeof value === "string") return `"${value}"`;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    
    if (Array.isArray(value)) {
      if (value.length === 0) return "[]";
      if (value.length > 5) {
        return `[${value.slice(0, 3).map(v => formatValue(v, depth + 1)).join(", ")}, ... +${value.length - 3} more]`;
      }
      return `[${value.map(v => formatValue(v, depth + 1)).join(", ")}]`;
    }
    
    if (typeof value === "object") {
      const keys = Object.keys(value);
      if (keys.length === 0) return "{}";
      if (keys.length > 10) {
        return `{${keys.slice(0, 5).map(k => `${k}: ${formatValue(value[k], depth + 1)}`).join(", ")}, ... +${keys.length - 5} more keys}`;
      }
      return `{${keys.map(k => `${k}: ${formatValue(value[k], depth + 1)}`).join(", ")}}`;
    }
    
    return String(value);
  };

  return (
    <div style={{
      background: '#f8f9fa',
      border: '1px solid #dee2e6',
      borderRadius: '8px',
      padding: '16px',
      margin: '16px 0',
      fontSize: '12px',
      fontFamily: 'Consolas, Monaco, monospace',
      maxHeight: '400px',
      overflow: 'auto'
    }}>
      <h4 style={{ margin: '0 0 12px 0', color: '#495057' }}>{title}</h4>
      <div style={{ color: '#212529' }}>
        <strong>Type:</strong> {typeof data}<br/>
        <strong>IsArray:</strong> {Array.isArray(data) ? 'Yes' : 'No'}<br/>
                 {data && typeof data === 'object' && (
           <>
             <strong>Keys:</strong> {Object.keys(data).join(', ')}<br/>
             {Array.isArray(data) && (
               <>
                 <strong>Length:</strong> {data.length}<br/>
               </>
             )}
           </>
         )}
        <hr style={{ margin: '12px 0', border: 'none', borderTop: '1px solid #ccc' }} />
        <strong>Structure:</strong><br/>
        <div style={{ 
          background: '#ffffff', 
          padding: '8px', 
          borderRadius: '4px',
          border: '1px solid #e9ecef',
          marginTop: '8px',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word'
        }}>
          {formatValue(data)}
        </div>
      </div>
    </div>
  );
};

export default DataDebugger; 