<script lang="ts">
  import { macroValues, goalSpec, checkInvariants, startRun } from '../stores/app';
  import Knob from './Knob.svelte';

  const macros = [
    {
      id: 'risk_appetite',
      label: 'Risk Appetite',
      description: 'How much risk the agent should take',
      effects: ['Learning rate', 'Exploration', 'Risk packs'],
    },
    {
      id: 'exploration_intensity',
      label: 'Exploration',
      description: 'Explore new strategies vs exploit known ones',
      effects: ['Epsilon schedule', 'Beam width'],
    },
    {
      id: 'training_time',
      label: 'Training Time',
      description: 'Duration and thoroughness of training',
      effects: ['Episodes', 'Generations', 'Wallclock'],
    },
    {
      id: 'conservatism',
      label: 'Conservatism',
      description: 'How cautious the configuration should be',
      effects: ['Discount', 'State size'],
    },
    {
      id: 'stability_reward',
      label: 'Stability vs Reward',
      description: 'Balance returns and volatility',
      effects: ['Objectives'],
    },
  ];

  function handleMacroChange(id: string, value: number) {
    macroValues.update((m) => ({ ...m, [id]: value }));
    // Trigger invariant check
    checkInvariants($goalSpec);
  }

  function handleRunPreview() {
    startRun('preview');
  }

  function handleRunFull() {
    startRun('full');
  }
</script>

<div class="fade-in">
  <div class="mb-8">
    <h2 class="text-2xl font-bold mb-2">Control Surface</h2>
    <p class="text-synth-text-dim">Adjust these high-level controls to shape your agent's behavior.</p>
  </div>

  <!-- Macro Knobs Grid -->
  <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6 mb-8">
    {#each macros as macro}
      <div class="panel p-6 text-center">
        <Knob
          value={$macroValues[macro.id] || 0.5}
          on:change={(e) => handleMacroChange(macro.id, e.detail)}
        />
        <h4 class="font-semibold mt-4 mb-1">{macro.label}</h4>
        <p class="text-xs text-synth-text-dim mb-2">{macro.description}</p>
        <div class="flex flex-wrap justify-center gap-1">
          {#each macro.effects as effect}
            <span class="px-1.5 py-0.5 bg-synth-surface rounded text-[10px] text-synth-text-dim">
              {effect}
            </span>
          {/each}
        </div>
      </div>
    {/each}
  </div>

  <!-- Quick Stats -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
    <div class="panel p-4">
      <div class="text-sm text-synth-text-dim mb-1">Search Population</div>
      <div class="text-2xl font-mono text-synth-accent">8</div>
      <div class="text-xs text-synth-text-dim">agents</div>
    </div>
    <div class="panel p-4">
      <div class="text-sm text-synth-text-dim mb-1">Training Budget</div>
      <div class="text-2xl font-mono text-synth-accent">64</div>
      <div class="text-xs text-synth-text-dim">episodes/agent</div>
    </div>
    <div class="panel p-4">
      <div class="text-sm text-synth-text-dim mb-1">State Space</div>
      <div class="text-2xl font-mono text-synth-accent">100</div>
      <div class="text-xs text-synth-text-dim">10 × 10 bins</div>
    </div>
    <div class="panel p-4">
      <div class="text-sm text-synth-text-dim mb-1">Est. Runtime</div>
      <div class="text-2xl font-mono text-synth-accent">~2.5m</div>
      <div class="text-xs text-synth-text-dim">preview mode</div>
    </div>
  </div>

  <!-- Action Buttons -->
  <div class="flex gap-4 justify-center">
    <button
      class="btn-primary text-lg px-8 py-3 flex items-center gap-3"
      on:click={handleRunPreview}
    >
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M8 5v14l11-7z" />
      </svg>
      Run Preview
    </button>
    <button
      class="btn-secondary text-lg px-8 py-3 flex items-center gap-3"
      on:click={handleRunFull}
    >
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
      Full Run
    </button>
  </div>
</div>
