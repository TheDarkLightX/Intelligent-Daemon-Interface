<script lang="ts">
  import Header from './lib/components/Header.svelte';
  import Sidebar from './lib/components/Sidebar.svelte';
  import ControlSurface from './lib/components/ControlSurface.svelte';
  import PresetGallery from './lib/components/PresetGallery.svelte';
  import RunConsole from './lib/components/RunConsole.svelte';
  import InvariantsPanel from './lib/components/InvariantsPanel.svelte';
  import { currentView, currentPreset } from './lib/stores/app';
</script>

<div class="min-h-screen bg-synth-bg flex flex-col">
  <!-- Header -->
  <Header />

  <div class="flex flex-1 overflow-hidden">
    <!-- Sidebar Navigation -->
    <Sidebar />

    <!-- Main Content -->
    <main class="flex-1 overflow-auto p-6">
      {#if $currentView === 'home'}
        <PresetGallery />
      {:else if $currentView === 'control-surface'}
        <ControlSurface />
      {:else if $currentView === 'run-console'}
        <RunConsole />
      {:else if $currentView === 'invariants'}
        <InvariantsPanel />
      {:else}
        <PresetGallery />
      {/if}
    </main>

    <!-- Right Panel: Live Preview -->
    <aside class="w-80 bg-synth-panel border-l border-synth-border p-4 overflow-auto hidden lg:block">
      <h3 class="text-sm font-semibold text-synth-text-dim uppercase tracking-wider mb-4">
        Live Preview
      </h3>
      
      <!-- Current Preset Info -->
      {#if $currentPreset}
        <div class="panel p-4 mb-4">
          <div class="flex items-center gap-3 mb-2">
            <span class="text-2xl">{$currentPreset.icon === 'shield' ? '🛡️' : $currentPreset.icon === 'flask' ? '🔬' : '📦'}</span>
            <div>
              <h4 class="font-semibold">{$currentPreset.name}</h4>
              <span class="text-xs text-synth-text-dim">{$currentPreset.difficulty}</span>
            </div>
          </div>
          <p class="text-sm text-synth-text-dim">{$currentPreset.description}</p>
        </div>
      {/if}

      <!-- Quick Invariant Status -->
      <div class="panel p-4 mb-4">
        <h4 class="font-medium mb-3">Invariants</h4>
        <div class="space-y-2">
          <div class="flex items-center justify-between text-sm">
            <span>I1: State Size</span>
            <span class="status-ok">✓</span>
          </div>
          <div class="flex items-center justify-between text-sm">
            <span>I2: Discount</span>
            <span class="status-ok">✓</span>
          </div>
          <div class="flex items-center justify-between text-sm">
            <span>I3: Learning Rate</span>
            <span class="status-ok">✓</span>
          </div>
          <div class="flex items-center justify-between text-sm">
            <span>I4: Exploration</span>
            <span class="status-ok">✓</span>
          </div>
          <div class="flex items-center justify-between text-sm">
            <span>I5: Budget</span>
            <span class="status-ok">✓</span>
          </div>
        </div>
      </div>

      <!-- Estimated Runtime -->
      <div class="panel p-4">
        <h4 class="font-medium mb-2">Estimated Runtime</h4>
        <div class="text-2xl font-mono text-synth-accent">~2.5 min</div>
        <p class="text-xs text-synth-text-dim mt-1">Preview mode • 8 agents</p>
      </div>
    </aside>
  </div>
</div>
