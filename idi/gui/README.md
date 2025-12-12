# IDI Synth Studio

A visually appealing, instrument-like GUI for the IDI Agent Parameterization Interface.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IDI Synth Studio                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐     ┌─────────────────────────────┐   │
│  │   Svelte Frontend   │────▶│   FastAPI Backend           │   │
│  │   (Tauri/Web)       │     │   (Python)                  │   │
│  │                     │     │                             │   │
│  │  • Control Surface  │     │  • Preset Service           │   │
│  │  • Preset Gallery   │     │  • Macro Engine             │   │
│  │  • Run Console      │     │  • Invariant Checker        │   │
│  │  • Invariants Panel │     │  • Synthesis Runner         │   │
│  └─────────────────────┘     └──────────────┬──────────────┘   │
│                                              │                  │
│                              ┌───────────────▼──────────────┐   │
│                              │   IDI Core                   │   │
│                              │   (Auto-QAgent, KRR, ZK)     │   │
│                              └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **Rust** (for Tauri desktop app, optional)

### 1. Start the Backend

```bash
# From the IDI root directory
cd idi/gui/backend

# Install Python dependencies
pip install fastapi uvicorn websockets

# Start the server
python main.py
```

The backend will run at `http://127.0.0.1:8765`

### 2. Start the Frontend (Development)

```bash
cd idi/gui/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The frontend will run at `http://localhost:5173`

### 3. (Optional) Build as Desktop App with Tauri

```bash
cd idi/gui/frontend

# Initialize Tauri
npm run tauri init

# Build the app
npm run tauri build
```

## Features

### 🎨 Preset Gallery
Browse and select from curated agent configurations:
- **Conservative Trader** - Low-risk, stable returns
- **Research Mode** - Experimental, multi-environment
- **Quick Test** - Fast iteration
- **Production Ready** - Balanced for deployment

### 🎛️ Control Surface
High-level macro controls that map to multiple underlying parameters:
- **Risk Appetite** - Learning rate, exploration, risk packs
- **Exploration Intensity** - Epsilon schedule, beam width
- **Training Time** - Episodes, generations, wallclock
- **Conservatism** - Discount factor, state size
- **Stability vs Reward** - Optimization objectives

### ▶️ Run Console
Real-time synthesis monitoring:
- Progress tracking
- Candidate discovery
- Log streaming
- Result exploration

### 🔒 Invariants Panel
Safety guarantee visualization:
- I1: State Size Bound
- I2: Discount Factor Bound
- I3: Learning Rate Bound
- I4: Exploration Decay Bound
- I5: Budget Sanity
- Tau Spec Preview

## Design System

### Synth Aesthetic

The UI follows a "modular synth" design language:
- **Dark theme** with warm amber accents
- **Knob controls** that respond to drag and scroll
- **Glass morphism** panels with subtle blur
- **Glow effects** on active elements
- **Smooth animations** for all transitions

### Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Background | `#1a1a2e` | Main background |
| Panel | `#252540` | Card backgrounds |
| Surface | `#2d2d4a` | Input backgrounds |
| Border | `#3d3d5c` | Borders, dividers |
| Accent | `#f9a826` | Primary actions, highlights |
| Success | `#4ade80` | Invariants OK, positive |
| Warning | `#fbbf24` | Caution states |
| Danger | `#ef4444` | Errors, violations |

## API Endpoints

### Presets
- `GET /api/presets` - List all presets
- `GET /api/presets/{id}` - Get preset with goal spec

### Macros
- `GET /api/macros` - List macro definitions
- `POST /api/macros/apply` - Apply macros to goal spec
- `POST /api/macros/preview` - Preview macro effects

### Invariants
- `POST /api/invariants/check` - Check all invariants
- `GET /api/invariants/descriptions` - Get invariant descriptions

### Runs
- `POST /api/runs/start` - Start synthesis run
- `GET /api/runs/{id}` - Get run status
- `POST /api/runs/{id}/stop` - Stop running synthesis
- `WS /ws/runs/{id}` - WebSocket for real-time updates

## Development

### Project Structure

```
idi/gui/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── api/                  # API routes (future)
│   └── services/
│       ├── presets.py        # Preset management
│       ├── invariants.py     # Invariant checking
│       └── macros.py         # Macro control engine
│
├── frontend/
│   ├── src/
│   │   ├── App.svelte        # Main app component
│   │   ├── app.css           # Global styles
│   │   └── lib/
│   │       ├── components/   # UI components
│   │       │   ├── Header.svelte
│   │       │   ├── Sidebar.svelte
│   │       │   ├── PresetGallery.svelte
│   │       │   ├── ControlSurface.svelte
│   │       │   ├── Knob.svelte
│   │       │   ├── RunConsole.svelte
│   │       │   └── InvariantsPanel.svelte
│   │       └── stores/
│   │           └── app.ts    # Svelte stores
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
└── README.md
```

### Adding New Components

1. Create component in `frontend/src/lib/components/`
2. Import in `App.svelte` or parent component
3. Use Tailwind classes with `synth-*` color palette
4. Follow existing component patterns for consistency

### Adding New API Endpoints

1. Add route in `backend/main.py`
2. Create Pydantic models for request/response
3. Implement business logic in `services/`
4. Update frontend stores to call new endpoint

## License

This GUI is part of the IDI project. See the main project LICENSE for details.
