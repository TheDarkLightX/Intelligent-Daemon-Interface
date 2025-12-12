<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let value: number = 0.5;
  export let min: number = 0;
  export let max: number = 1;
  export let size: number = 80;

  const dispatch = createEventDispatcher();

  let isDragging = false;
  let startY: number;
  let startValue: number;

  $: rotation = ((value - min) / (max - min)) * 270 - 135;
  $: displayValue = Math.round(value * 100);

  function handleMouseDown(e: MouseEvent) {
    isDragging = true;
    startY = e.clientY;
    startValue = value;
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }

  function handleMouseMove(e: MouseEvent) {
    if (!isDragging) return;
    
    const delta = (startY - e.clientY) / 200;
    let newValue = startValue + delta * (max - min);
    newValue = Math.max(min, Math.min(max, newValue));
    value = newValue;
    dispatch('change', value);
  }

  function handleMouseUp() {
    isDragging = false;
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
  }

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    const delta = -e.deltaY / 1000;
    let newValue = value + delta * (max - min);
    newValue = Math.max(min, Math.min(max, newValue));
    value = newValue;
    dispatch('change', value);
  }
</script>

<div
  class="relative cursor-pointer select-none"
  style="width: {size}px; height: {size}px;"
  on:mousedown={handleMouseDown}
  on:wheel={handleWheel}
  role="slider"
  aria-valuenow={value}
  aria-valuemin={min}
  aria-valuemax={max}
  tabindex="0"
>
  <!-- Outer ring -->
  <svg
    viewBox="0 0 100 100"
    class="absolute inset-0 w-full h-full"
  >
    <!-- Background arc -->
    <circle
      cx="50"
      cy="50"
      r="40"
      fill="none"
      stroke="currentColor"
      stroke-width="8"
      class="text-synth-surface"
      stroke-dasharray="188.5"
      stroke-dashoffset="47"
      transform="rotate(135 50 50)"
    />
    
    <!-- Value arc -->
    <circle
      cx="50"
      cy="50"
      r="40"
      fill="none"
      stroke="currentColor"
      stroke-width="8"
      class="text-synth-accent transition-all duration-100"
      stroke-dasharray="188.5"
      stroke-dashoffset={188.5 - (value - min) / (max - min) * 188.5 * 0.75}
      transform="rotate(135 50 50)"
      stroke-linecap="round"
    />
  </svg>

  <!-- Center circle with value -->
  <div
    class="absolute inset-0 flex items-center justify-center"
  >
    <div
      class="w-16 h-16 rounded-full bg-synth-panel border-2 border-synth-border flex items-center justify-center transition-all duration-100"
      class:border-synth-accent={isDragging}
      class:shadow-glow-sm={isDragging}
    >
      <span class="font-mono text-lg font-semibold">{displayValue}</span>
    </div>
  </div>

  <!-- Indicator dot -->
  <div
    class="absolute w-3 h-3 bg-synth-accent rounded-full shadow-glow-sm transition-transform duration-100"
    style="
      top: 50%;
      left: 50%;
      transform: rotate({rotation}deg) translateY(-28px) translateX(-50%);
      transform-origin: center center;
    "
  />
</div>
