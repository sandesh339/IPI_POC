# New Frontend Implementation

A React-based chatbot frontend following the design patterns and architecture of the existing SDG chatbot implementation.

## Features

### Core Functionality
- **ChatGPT-style conversation interface** with message history
- **Typewriter effect** for bot responses
- **Voice input** with speech-to-text capability
- **Reaction system** (thumbs up/down) for user feedback
- **Real-time loading states** and typing indicators

### Visualization & Data
- **Chart integration** with Chart.js (Bar and Pie charts)
- **Modal expansion** for detailed visualizations
- **Dynamic data processing** with flexible data structure handling
- **Interactive chart controls** (chart type switching)



##  Project Structure

```
new-frontend/
├── src/
│   ├── components/
│   │   ├── ConversationView.js    # Main chat interface orchestrator
│   │   └── SampleChart.js         # Example visualization component
│   ├── App.js                     # Simple wrapper component
│   ├── index.css                  # Complete design system
│   └── index.js                   # App entry point
├── public/
│   └── index.html
├── package.json
└── README.md
```



## Installation & Setup

1. **Navigate to the project directory:**
   ```bash
   cd new-frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

##  Configuration

### API Integration
Update the API endpoint in `ConversationView.js`:
```javascript
const res = await fetch("http://127.0.0.1:8000/chatbot/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ 
    query,
    history: messages.map((msg) => ({
      role: msg.type === "user" ? "user" : "assistant",
      content: msg.text,
    })),
  }),
});
```

### Audio Transcription
Implement the `sendAudioToOpenAI` function in `ConversationView.js` to enable voice input functionality.

## Visualization Components

### Chart Component Pattern
```javascript
import SampleChart from './components/SampleChart';

// Usage in messages
{message.visualizations && (
  <SampleChart 
    data={message.visualizations} 
    chartOnly={false} 
  />
)}
```

### Data Structure Support
The components handle multiple data formats:
- Direct data objects
- Nested result structures
- Array of function call results



### Breakpoints
- **Desktop**: > 1024px (full features)
- **Tablet**: 768px - 1024px (optimized layout)
- **Mobile**: < 768px (compact interface)

### Mobile Optimizations
- Smaller avatars and padding
- Condensed input area
- Touch-friendly buttons
- Simplified chart controls

### Deployment

### Build for Production
```bash
npm run build
```

### Static File Serving
The build creates optimized static files in the `build/` directory ready for deployment to any static hosting service.

## Integration with Backend

Expected API response format:
```javascript
{
  "response": "Bot response text",
  "chart_data": {
    "labels": ["A", "B", "C"],
    "data": [1, 2, 3],
    "title": "Chart Title"
  },
  "boundary": [...], // Map boundary data
  "function_calls": [...] // Function call metadata
}
```

## Customization

### Theme Colors
Update color variables in `index.css`:
```css
/* Update primary colors */
.message-avatar.user {
  background: linear-gradient(135deg, #your-color-1, #your-color-2);
}
```

### Chart Styling
Modify chart colors in `SampleChart.js`:
```javascript
backgroundColor: [
  '#your-color-1',
  '#your-color-2',
  // ... more colors
]
```



## Development Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run test suite
- `npm run eject` - Eject from Create React App


