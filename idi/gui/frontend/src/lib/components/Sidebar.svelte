<script lang="ts">
  import { currentView } from '../stores/app';

  const navItems = [
    { id: 'home', label: 'Presets', icon: '🎨' },
    { id: 'control-surface', label: 'Controls', icon: '🎛️' },
    { id: 'run-console', label: 'Run', icon: '▶️' },
    { id: 'invariants', label: 'Invariants', icon: '🔒' },
  ];

  let showSettings = false;

  function handleSettings() {
    showSettings = !showSettings;
  }
</script>

<nav class="w-20 bg-synth-panel border-r border-synth-border flex flex-col items-center py-6 gap-2 relative">
  {#each navItems as item}
    <button
      class="w-14 h-14 rounded-xl flex flex-col items-center justify-center gap-1 transition-all duration-200 active:scale-95
             {$currentView === item.id 
               ? 'bg-synth-accent text-synth-bg shadow-glow-sm' 
               : 'text-synth-text-dim hover:bg-synth-surface hover:text-synth-text'}"
      on:click={() => currentView.set(item.id)}
    >
      <span class="text-xl">{item.icon}</span>
      <span class="text-[10px] font-medium">{item.label}</span>
    </button>
  {/each}

  <div class="flex-1" />

  <!-- Settings -->
  <button 
    class="w-14 h-14 rounded-xl flex flex-col items-center justify-center gap-1 transition-all duration-200 active:scale-95
           {showSettings ? 'bg-synth-accent text-synth-bg' : 'text-synth-text-dim hover:bg-synth-surface hover:text-synth-text'}"
    on:click={handleSettings}
  >
    <span class="text-xl">⚙️</span>
    <span class="text-[10px] font-medium">Settings</span>
  </button>

  <!-- Settings Popup -->
  {#if showSettings}
    <div class="absolute bottom-20 left-full ml-2 w-64 panel p-4 shadow-lg z-50">
      <h4 class="font-semibold mb-3">Settings</h4>
      <div class="space-y-3">
        <label class="flex items-center justify-between">
          <span class="text-sm">Dark Mode</span>
          <input type="checkbox" checked class="accent-synth-accent" />
        </label>
        <label class="flex items-center justify-between">
          <span class="text-sm">Sound Effects</span>
          <input type="checkbox" class="accent-synth-accent" />
        </label>
        <label class="flex items-center justify-between">
          <span class="text-sm">Auto-save</span>
          <input type="checkbox" checked class="accent-synth-accent" />
        </label>
      </div>
      <div class="mt-4 pt-3 border-t border-synth-border">
        <p class="text-xs text-synth-text-dim">IDI Synth Studio v0.1.0</p>
      </div>
    </div>
  {/if}
</nav>
