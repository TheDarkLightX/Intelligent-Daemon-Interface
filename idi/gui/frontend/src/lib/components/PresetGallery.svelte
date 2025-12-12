<script lang="ts">
  import { onMount } from 'svelte';
  import { loadPresets, selectPreset, type Preset } from '../stores/app';

  let presets: Preset[] = [];
  let loading = true;

  onMount(async () => {
    presets = await loadPresets();
    loading = false;
  });

  function getIcon(iconName: string): string {
    const icons: Record<string, string> = {
      'shield': '🛡️',
      'flask': '🔬',
      'zap': '⚡',
      'check-circle': '✅',
      'file': '📄',
    };
    return icons[iconName] || '📦';
  }

  function getDifficultyColor(difficulty: string): string {
    switch (difficulty) {
      case 'beginner': return 'bg-synth-success/20 text-synth-success';
      case 'intermediate': return 'bg-synth-warning/20 text-synth-warning';
      case 'advanced': return 'bg-synth-accent/20 text-synth-accent';
      default: return 'bg-synth-surface text-synth-text-dim';
    }
  }
</script>

<div class="fade-in">
  <div class="mb-8">
    <h2 class="text-2xl font-bold mb-2">Welcome to IDI Synth Studio</h2>
    <p class="text-synth-text-dim">Choose a preset to get started, or create a new configuration from scratch.</p>
  </div>

  {#if loading}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {#each [1, 2, 3, 4] as _}
        <div class="panel p-6 animate-pulse">
          <div class="w-12 h-12 bg-synth-surface rounded-lg mb-4" />
          <div class="h-6 bg-synth-surface rounded w-3/4 mb-2" />
          <div class="h-4 bg-synth-surface rounded w-full" />
        </div>
      {/each}
    </div>
  {:else}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {#each presets as preset}
        <button
          class="panel p-6 text-left card-hover group"
          on:click={() => selectPreset(preset)}
        >
          <div class="flex items-start justify-between mb-4">
            <div class="w-12 h-12 bg-synth-surface rounded-lg flex items-center justify-center text-2xl group-hover:bg-synth-accent/20 transition-colors">
              {getIcon(preset.icon)}
            </div>
            <span class="px-2 py-1 rounded text-xs font-medium {getDifficultyColor(preset.difficulty)}">
              {preset.difficulty}
            </span>
          </div>
          
          <h3 class="font-semibold text-lg mb-2 group-hover:text-synth-accent transition-colors">
            {preset.name}
          </h3>
          
          <p class="text-sm text-synth-text-dim mb-4">
            {preset.description}
          </p>
          
          <div class="flex flex-wrap gap-2">
            {#each preset.tags.slice(0, 3) as tag}
              <span class="px-2 py-1 bg-synth-surface rounded text-xs text-synth-text-dim">
                {tag}
              </span>
            {/each}
          </div>
        </button>
      {/each}

      <!-- Create New -->
      <button class="panel p-6 text-left card-hover border-dashed border-2 border-synth-border hover:border-synth-accent group">
        <div class="flex items-center justify-center h-full min-h-[200px]">
          <div class="text-center">
            <div class="w-12 h-12 mx-auto bg-synth-surface rounded-lg flex items-center justify-center text-2xl mb-4 group-hover:bg-synth-accent/20 transition-colors">
              ➕
            </div>
            <h3 class="font-semibold text-lg mb-2 group-hover:text-synth-accent transition-colors">
              Create New
            </h3>
            <p class="text-sm text-synth-text-dim">
              Start with a blank configuration
            </p>
          </div>
        </div>
      </button>
    </div>
  {/if}
</div>
