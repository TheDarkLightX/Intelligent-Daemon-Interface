import { writable, derived } from 'svelte/store';

// Types
export interface Preset {
  id: string;
  name: string;
  description: string;
  icon: string;
  tags: string[];
  difficulty: string;
}

export interface MacroValue {
  id: string;
  value: number;
}

export interface InvariantStatus {
  id: string;
  label: string;
  ok: boolean;
  message: string;
  value?: number;
  threshold?: number;
}

export interface RunStatus {
  id: string;
  status: 'idle' | 'starting' | 'running' | 'completed' | 'failed' | 'stopped';
  progress: number;
  candidates: any[];
  logs: { time: number; message: string }[];
}

// Stores
export const currentView = writable<string>('home');
export const currentPreset = writable<Preset | null>(null);
export const goalSpec = writable<Record<string, any>>({});

export const macroValues = writable<Record<string, number>>({
  risk_appetite: 0.3,
  exploration_intensity: 0.4,
  training_time: 0.3,
  conservatism: 0.6,
  stability_reward: 0.5,
});

export const invariants = writable<InvariantStatus[]>([]);
export const runStatus = writable<RunStatus>({
  id: '',
  status: 'idle',
  progress: 0,
  candidates: [],
  logs: [],
});

// API base URL
export const API_BASE = 'http://127.0.0.1:8765';

// Derived stores
export const allInvariantsOk = derived(invariants, ($invariants) =>
  $invariants.every((inv) => inv.ok)
);

export const isRunning = derived(runStatus, ($run) =>
  ['starting', 'running'].includes($run.status)
);

// Actions
export async function loadPresets(): Promise<Preset[]> {
  try {
    const res = await fetch(`${API_BASE}/api/presets`);
    if (!res.ok) throw new Error('Failed to load presets');
    return await res.json();
  } catch (err) {
    console.error('Failed to load presets:', err);
    // Return mock presets for development
    return [
      {
        id: 'conservative_qagent',
        name: 'Conservative Trader',
        description: 'Low-risk agent with stable returns',
        icon: 'shield',
        tags: ['beginner', 'low-risk'],
        difficulty: 'beginner',
      },
      {
        id: 'research_qagent',
        name: 'Research Mode',
        description: 'Exploratory agent for testing',
        icon: 'flask',
        tags: ['intermediate', 'experimental'],
        difficulty: 'intermediate',
      },
      {
        id: 'quick_test',
        name: 'Quick Test',
        description: 'Fast iteration with minimal budget',
        icon: 'zap',
        tags: ['beginner', 'fast'],
        difficulty: 'beginner',
      },
      {
        id: 'production',
        name: 'Production Ready',
        description: 'Balanced configuration for deployment',
        icon: 'check-circle',
        tags: ['advanced', 'production'],
        difficulty: 'advanced',
      },
    ];
  }
}

export async function selectPreset(preset: Preset): Promise<void> {
  currentPreset.set(preset);
  currentView.set('control-surface');
  
  try {
    const res = await fetch(`${API_BASE}/api/presets/${preset.id}`);
    if (res.ok) {
      const data = await res.json();
      goalSpec.set(data.goal_spec);
    }
  } catch (err) {
    console.error('Failed to load preset details:', err);
  }
}

export async function checkInvariants(spec: Record<string, any>): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/api/invariants/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
    if (res.ok) {
      invariants.set(await res.json());
    }
  } catch (err) {
    console.error('Failed to check invariants:', err);
    // Set mock invariants for development
    invariants.set([
      { id: 'I1', label: 'State Size', ok: true, message: '100 ≤ 2048' },
      { id: 'I2', label: 'Discount', ok: true, message: '0.99 ≥ 0.5' },
      { id: 'I3', label: 'Learning Rate', ok: true, message: '0.1 ≤ 0.5' },
      { id: 'I4', label: 'Exploration', ok: true, message: '1000 > 0' },
      { id: 'I5', label: 'Budget', ok: true, message: '512 ≥ 64' },
    ]);
  }
}

export async function startRun(mode: 'preview' | 'full' = 'preview'): Promise<void> {
  let spec: Record<string, any> = {};
  goalSpec.subscribe((s) => (spec = s))();
  
  runStatus.set({
    id: '',
    status: 'starting',
    progress: 0,
    candidates: [],
    logs: [{ time: Date.now(), message: 'Starting synthesis...' }],
  });
  
  currentView.set('run-console');
  
  try {
    const res = await fetch(`${API_BASE}/api/runs/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal_spec: spec, mode }),
    });
    
    if (res.ok) {
      const { run_id } = await res.json();
      pollRunStatus(run_id);
    }
  } catch (err) {
    console.error('Failed to start run:', err);
    // Mock run for development
    mockRun();
  }
}

async function pollRunStatus(runId: string): Promise<void> {
  const poll = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/runs/${runId}`);
      if (res.ok) {
        const status = await res.json();
        runStatus.set(status);
        
        if (!['completed', 'failed', 'stopped'].includes(status.status)) {
          setTimeout(poll, 500);
        }
      }
    } catch (err) {
      console.error('Failed to poll run status:', err);
    }
  };
  poll();
}

function mockRun(): void {
  // Simulate a run for development
  let progress = 0;
  const interval = setInterval(() => {
    progress += 10;
    runStatus.update((r) => ({
      ...r,
      status: progress >= 100 ? 'completed' : 'running',
      progress,
      logs: [
        ...r.logs,
        { time: Date.now(), message: `Progress: ${progress}%` },
      ],
      candidates: progress >= 100
        ? [
            { id: 'agent_1', metrics: { reward: 0.85, risk: 0.12 } },
            { id: 'agent_2', metrics: { reward: 0.78, risk: 0.08 } },
            { id: 'agent_3', metrics: { reward: 0.72, risk: 0.05 } },
          ]
        : [],
    }));
    
    if (progress >= 100) {
      clearInterval(interval);
    }
  }, 500);
}
