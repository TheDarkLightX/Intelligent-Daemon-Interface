<script lang="ts">
  import { invariants, allInvariantsOk } from '../stores/app';

  // Mock invariants for display
  const displayInvariants = $invariants.length > 0 ? $invariants : [
    { id: 'I1', label: 'State Size Bound', ok: true, message: '100 ≤ 2048', value: 100, threshold: 2048 },
    { id: 'I2', label: 'Discount Factor Bound', ok: true, message: '0.99 ≥ 0.5', value: 0.99, threshold: 0.5 },
    { id: 'I3', label: 'Learning Rate Bound', ok: true, message: '0.1 ≤ 0.5', value: 0.1, threshold: 0.5 },
    { id: 'I4', label: 'Exploration Decay Bound', ok: true, message: '1000 > 0', value: 1000, threshold: 1 },
    { id: 'I5', label: 'Budget Sanity', ok: true, message: '512 ≥ 64', value: 512, threshold: 64 },
  ];

  const invariantDescriptions: Record<string, string> = {
    'I1': 'Q-table cannot exceed 2048 states to ensure tractable learning. Larger state spaces increase memory usage and slow convergence.',
    'I2': 'Discount factor must be at least 0.5 for learning stability. Lower values cause the agent to be too short-sighted.',
    'I3': 'Learning rate must not exceed 0.5 to prevent divergence. Higher rates cause unstable value updates.',
    'I4': 'Exploration decay must be positive for convergence. Zero decay means exploration never reduces.',
    'I5': 'Training budget must allow meaningful exploration. Too few agent-episodes prevents learning.',
  };
</script>

<div class="fade-in">
  <div class="mb-8">
    <h2 class="text-2xl font-bold mb-2">Invariants & Safety</h2>
    <p class="text-synth-text-dim">
      These invariants ensure your configuration stays within safe, well-tested bounds.
    </p>
  </div>

  <!-- Overall Status -->
  <div class="panel p-6 mb-6">
    <div class="flex items-center gap-4">
      {#if $allInvariantsOk || displayInvariants.every(i => i.ok)}
        <div class="w-16 h-16 rounded-full bg-synth-success/20 flex items-center justify-center">
          <span class="text-3xl">✅</span>
        </div>
        <div>
          <h3 class="text-xl font-semibold text-synth-success">All Invariants Satisfied</h3>
          <p class="text-synth-text-dim">Your configuration is within safe bounds.</p>
        </div>
      {:else}
        <div class="w-16 h-16 rounded-full bg-synth-danger/20 flex items-center justify-center">
          <span class="text-3xl">⚠️</span>
        </div>
        <div>
          <h3 class="text-xl font-semibold text-synth-danger">Some Invariants Violated</h3>
          <p class="text-synth-text-dim">Review and adjust your configuration.</p>
        </div>
      {/if}
    </div>
  </div>

  <!-- Invariant Cards -->
  <div class="space-y-4">
    {#each displayInvariants as inv}
      <div class="panel p-6 {inv.ok ? '' : 'border-synth-danger'}">
        <div class="flex items-start gap-4">
          <div class="w-10 h-10 rounded-lg flex items-center justify-center {inv.ok ? 'bg-synth-success/20' : 'bg-synth-danger/20'}">
            <span class="text-xl">{inv.ok ? '✓' : '✗'}</span>
          </div>
          
          <div class="flex-1">
            <div class="flex items-center gap-3 mb-2">
              <span class="font-mono text-sm text-synth-accent">{inv.id}</span>
              <h4 class="font-semibold">{inv.label}</h4>
            </div>
            
            <p class="text-sm text-synth-text-dim mb-3">
              {invariantDescriptions[inv.id] || 'No description available.'}
            </p>
            
            <!-- Value Bar -->
            {#if inv.value !== undefined && inv.threshold !== undefined}
              <div class="mb-2">
                <div class="flex justify-between text-xs text-synth-text-dim mb-1">
                  <span>Current: <span class="font-mono">{inv.value}</span></span>
                  <span>Limit: <span class="font-mono">{inv.threshold}</span></span>
                </div>
                <div class="h-2 bg-synth-surface rounded-full overflow-hidden">
                  <div
                    class="h-full transition-all duration-300 {inv.ok ? 'bg-synth-success' : 'bg-synth-danger'}"
                    style="width: {Math.min(100, (inv.value / inv.threshold) * 100)}%"
                  />
                </div>
              </div>
            {/if}
            
            <div class="flex items-center gap-2">
              <span class="text-sm font-mono {inv.ok ? 'text-synth-success' : 'text-synth-danger'}">
                {inv.message}
              </span>
            </div>
          </div>
        </div>
      </div>
    {/each}
  </div>

  <!-- Tau Spec Preview -->
  <div class="panel p-6 mt-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-semibold">Tau Spec Preview</h3>
      <button class="btn-ghost text-sm">Export Full Spec</button>
    </div>
    <pre class="bg-synth-bg rounded-lg p-4 text-sm font-mono text-synth-text-dim overflow-x-auto"><code>// Generated Tau constraints
wff: state_count ≤ 2048
wff: discount ≥ 0.5
wff: learning_rate ≤ 0.5
wff: epsilon_decay > 0
wff: training_budget ≥ 64

// Safety invariants
always: bounded_state
always: stable_learning
always: convergent_exploration</code></pre>
  </div>
</div>
