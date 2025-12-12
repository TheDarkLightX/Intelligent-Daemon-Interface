<script lang="ts">
  import { currentPreset, goalSpec } from '../stores/app';
  import { writable } from 'svelte/store';

  // Mode state
  let mode: 'simple' | 'advanced' = 'simple';
  
  // Toast notification
  let showToast = false;
  let toastMessage = '';

  function setMode(newMode: 'simple' | 'advanced') {
    mode = newMode;
    showNotification(`Switched to ${newMode} mode`);
  }

  function handleSave() {
    // Get current goal spec
    let spec: Record<string, any> = {};
    goalSpec.subscribe(s => spec = s)();
    
    // Create downloadable JSON
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `goal_spec_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    showNotification('Goal spec saved!');
  }

  function showNotification(message: string) {
    toastMessage = message;
    showToast = true;
    setTimeout(() => showToast = false, 2000);
  }
</script>

<header class="bg-synth-panel border-b border-synth-border px-6 py-4 flex items-center justify-between relative">
  <div class="flex items-center gap-4">
    <!-- Logo -->
    <div class="flex items-center gap-2">
      <div class="w-10 h-10 bg-gradient-to-br from-synth-accent to-synth-accent-dim rounded-lg flex items-center justify-center shadow-glow-sm">
        <span class="text-synth-bg font-bold text-lg">I</span>
      </div>
      <div>
        <h1 class="font-bold text-lg text-synth-text">IDI Synth Studio</h1>
        <p class="text-xs text-synth-text-dim">Agent Parameterization Interface</p>
      </div>
    </div>

    <!-- Current Patch -->
    {#if $currentPreset}
      <div class="ml-8 px-4 py-2 bg-synth-surface rounded-lg border border-synth-border">
        <span class="text-sm text-synth-text-dim">Current:</span>
        <span class="ml-2 font-medium text-synth-accent">{$currentPreset.name}</span>
      </div>
    {/if}
  </div>

  <div class="flex items-center gap-3">
    <!-- Save Button -->
    <button 
      class="btn-secondary flex items-center gap-2 active:scale-95 transition-transform"
      on:click={handleSave}
    >
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
      </svg>
      Save Patch
    </button>

    <!-- Mode Toggle -->
    <div class="flex items-center gap-2 px-3 py-2 bg-synth-surface rounded-lg border border-synth-border">
      <span class="text-sm text-synth-text-dim">Mode:</span>
      <button 
        class="px-2 py-1 text-sm rounded transition-all duration-200 {mode === 'simple' ? 'bg-synth-accent text-synth-bg' : 'text-synth-text-dim hover:text-synth-text hover:bg-synth-border/50'}"
        on:click={() => setMode('simple')}
      >
        Simple
      </button>
      <button 
        class="px-2 py-1 text-sm rounded transition-all duration-200 {mode === 'advanced' ? 'bg-synth-accent text-synth-bg' : 'text-synth-text-dim hover:text-synth-text hover:bg-synth-border/50'}"
        on:click={() => setMode('advanced')}
      >
        Advanced
      </button>
    </div>
  </div>

  <!-- Toast Notification -->
  {#if showToast}
    <div class="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-4 py-2 bg-synth-success text-synth-bg rounded-lg shadow-lg animate-fade-in z-50">
      {toastMessage}
    </div>
  {/if}
</header>

<style>
  @keyframes fade-in {
    from { opacity: 0; transform: translateX(-50%) translateY(-10px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
  }
  .animate-fade-in {
    animation: fade-in 0.2s ease-out;
  }
</style>
