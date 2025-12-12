<script lang="ts">
  import { runStatus, isRunning, currentView, startRun } from '../stores/app';

  // Selected candidate for details view
  let selectedCandidate: any = null;
  let showDetails = false;

  function handleStop() {
    runStatus.update(r => ({
      ...r,
      status: 'stopped',
      logs: [...r.logs, { time: Date.now(), message: 'Run stopped by user.' }]
    }));
  }

  function handleViewDetails(candidate: any) {
    selectedCandidate = candidate;
    showDetails = true;
  }

  function handleCloseDetails() {
    showDetails = false;
    selectedCandidate = null;
  }

  function handleNewRun() {
    currentView.set('control-surface');
  }

  $: statusColor = {
    'idle': 'text-synth-text-dim',
    'starting': 'text-synth-warning',
    'running': 'text-synth-accent',
    'completed': 'text-synth-success',
    'failed': 'text-synth-danger',
    'stopped': 'text-synth-text-dim',
  }[$runStatus.status] || 'text-synth-text-dim';

  $: statusIcon = {
    'idle': '⏸️',
    'starting': '🔄',
    'running': '▶️',
    'completed': '✅',
    'failed': '❌',
    'stopped': '⏹️',
  }[$runStatus.status] || '⏸️';
</script>

<div class="fade-in">
  <div class="mb-8">
    <h2 class="text-2xl font-bold mb-2">Run Console</h2>
    <p class="text-synth-text-dim">Monitor synthesis progress and results in real-time.</p>
  </div>

  <!-- Status Header -->
  <div class="panel p-6 mb-6">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-4">
        <span class="text-3xl">{statusIcon}</span>
        <div>
          <h3 class="text-xl font-semibold {statusColor}">
            {$runStatus.status.charAt(0).toUpperCase() + $runStatus.status.slice(1)}
          </h3>
          {#if $runStatus.id}
            <p class="text-sm text-synth-text-dim font-mono">{$runStatus.id}</p>
          {/if}
        </div>
      </div>
      
      {#if $isRunning}
        <button 
          class="btn-secondary active:scale-95 transition-transform"
          on:click={handleStop}
        >
          ⏹️ Stop Run
        </button>
      {:else if $runStatus.status === 'completed' || $runStatus.status === 'stopped' || $runStatus.status === 'failed'}
        <button 
          class="btn-primary active:scale-95 transition-transform"
          on:click={handleNewRun}
        >
          🔄 New Run
        </button>
      {/if}
    </div>

    <!-- Progress Bar -->
    <div class="mb-4">
      <div class="flex justify-between text-sm mb-2">
        <span>Progress</span>
        <span class="font-mono">{$runStatus.progress}%</span>
      </div>
      <div class="h-3 bg-synth-surface rounded-full overflow-hidden">
        <div
          class="h-full bg-gradient-to-r from-synth-accent to-synth-accent-dim transition-all duration-300"
          class:animate-pulse={$isRunning}
          style="width: {$runStatus.progress}%"
        />
      </div>
    </div>

    <!-- Stats -->
    <div class="grid grid-cols-4 gap-4 text-center">
      <div>
        <div class="text-2xl font-mono text-synth-accent">
          {$runStatus.candidates.length}
        </div>
        <div class="text-xs text-synth-text-dim">Candidates</div>
      </div>
      <div>
        <div class="text-2xl font-mono text-synth-text">—</div>
        <div class="text-xs text-synth-text-dim">Pruned (KRR)</div>
      </div>
      <div>
        <div class="text-2xl font-mono text-synth-text">—</div>
        <div class="text-xs text-synth-text-dim">Evaluated</div>
      </div>
      <div>
        <div class="text-2xl font-mono text-synth-success">
          {$runStatus.status === 'completed' ? $runStatus.candidates.length : '—'}
        </div>
        <div class="text-xs text-synth-text-dim">Accepted</div>
      </div>
    </div>
  </div>

  <!-- Results -->
  {#if $runStatus.candidates.length > 0}
    <div class="panel p-6 mb-6">
      <h3 class="font-semibold mb-4">Results</h3>
      <div class="space-y-3">
        {#each $runStatus.candidates as candidate, i}
          <div class="flex items-center gap-4 p-4 bg-synth-surface rounded-lg">
            <div class="w-8 h-8 rounded-full bg-synth-accent/20 flex items-center justify-center font-mono text-sm">
              #{i + 1}
            </div>
            <div class="flex-1">
              <div class="font-mono text-sm mb-1">{candidate.id}</div>
              <div class="flex gap-4 text-sm">
                <span>
                  Reward: <span class="text-synth-success font-mono">
                    {(candidate.metrics.reward * 100).toFixed(1)}%
                  </span>
                </span>
                <span>
                  Risk: <span class="text-synth-warning font-mono">
                    {(candidate.metrics.risk * 100).toFixed(1)}%
                  </span>
                </span>
              </div>
            </div>
            <button 
              class="btn-ghost text-sm active:scale-95 transition-transform"
              on:click={() => handleViewDetails(candidate)}
            >
              View Details
            </button>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Logs -->
  <div class="panel p-6">
    <h3 class="font-semibold mb-4">Logs</h3>
    <div class="bg-synth-bg rounded-lg p-4 h-48 overflow-auto font-mono text-sm">
      {#each $runStatus.logs as log}
        <div class="text-synth-text-dim">
          <span class="text-synth-accent">
            [{new Date(log.time).toLocaleTimeString()}]
          </span>
          {log.message}
        </div>
      {/each}
      {#if $runStatus.logs.length === 0}
        <div class="text-synth-text-dim">No logs yet. Start a run to see output.</div>
      {/if}
    </div>
  </div>
</div>

<!-- Details Modal -->
{#if showDetails && selectedCandidate}
  <div 
    class="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
    on:click={handleCloseDetails}
    on:keydown={(e) => e.key === 'Escape' && handleCloseDetails()}
    role="dialog"
    tabindex="-1"
  >
    <div 
      class="panel p-6 max-w-lg w-full mx-4 shadow-2xl"
      on:click|stopPropagation
    >
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-xl font-semibold">Candidate Details</h3>
        <button 
          class="w-8 h-8 rounded-lg bg-synth-surface hover:bg-synth-border flex items-center justify-center transition-colors"
          on:click={handleCloseDetails}
        >
          ✕
        </button>
      </div>
      
      <div class="space-y-4">
        <div>
          <div class="text-sm text-synth-text-dim mb-1">ID</div>
          <div class="font-mono text-synth-accent">{selectedCandidate.id}</div>
        </div>
        
        <div class="grid grid-cols-2 gap-4">
          <div class="p-3 bg-synth-surface rounded-lg">
            <div class="text-sm text-synth-text-dim mb-1">Reward</div>
            <div class="text-2xl font-mono text-synth-success">
              {(selectedCandidate.metrics.reward * 100).toFixed(1)}%
            </div>
          </div>
          <div class="p-3 bg-synth-surface rounded-lg">
            <div class="text-sm text-synth-text-dim mb-1">Risk</div>
            <div class="text-2xl font-mono text-synth-warning">
              {(selectedCandidate.metrics.risk * 100).toFixed(1)}%
            </div>
          </div>
        </div>
        
        <div>
          <div class="text-sm text-synth-text-dim mb-2">Parameters</div>
          <pre class="bg-synth-bg rounded-lg p-3 text-sm font-mono overflow-auto max-h-48">{JSON.stringify(selectedCandidate.params || selectedCandidate.metrics, null, 2)}</pre>
        </div>
      </div>
      
      <div class="flex gap-3 mt-6">
        <button 
          class="btn-primary flex-1"
          on:click={handleCloseDetails}
        >
          Close
        </button>
      </div>
    </div>
  </div>
{/if}
