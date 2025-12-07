#!/usr/bin/env python3
"""
Tau Process Integration - Connects Q-daemon to actual Tau execution
"""

import subprocess
import threading
import queue
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
import re


@dataclass 
class TauOutput:
    """Parsed output from Tau execution"""
    step: int
    inputs: dict  # {varname: value}
    outputs: dict # {varname: value}
    raw: str


class TauProcess:
    """Manages communication with Tau Docker container"""
    
    def __init__(self, spec_content: str):
        self.spec_content = spec_content
        self.process: Optional[subprocess.Popen] = None
        self.output_queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False
        
    def start(self):
        """Start Tau process"""
        self.process = subprocess.Popen(
            ["docker", "run", "--rm", "-i", "tau:latest"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.running = True
        
        # Start output reader thread
        self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self.reader_thread.start()
        
        # Send spec
        for line in self.spec_content.strip().split('\n'):
            self._send(line)
            time.sleep(0.05)  # Give Tau time to process
    
    def _read_output(self):
        """Background thread to read Tau output"""
        while self.running and self.process:
            try:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line.strip())
                elif self.process.poll() is not None:
                    break
            except Exception as e:
                print(f"Read error: {e}")
                break
    
    def _send(self, text: str):
        """Send input to Tau"""
        if self.process and self.process.stdin:
            self.process.stdin.write(text + '\n')
            self.process.stdin.flush()
    
    def send_inputs(self, *values: int) -> List[str]:
        """Send input values and collect outputs"""
        outputs = []
        
        for v in values:
            self._send(str(v))
            time.sleep(0.02)
        
        # Collect any outputs
        time.sleep(0.1)
        while not self.output_queue.empty():
            outputs.append(self.output_queue.get_nowait())
        
        return outputs
    
    def get_output(self, timeout: float = 0.5) -> Optional[str]:
        """Get next output line"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def parse_execution_step(self, lines: List[str]) -> Optional[TauOutput]:
        """Parse execution step from output lines"""
        step = -1
        inputs = {}
        outputs = {}
        
        for line in lines:
            # Match: Execution step: N
            if m := re.match(r'Execution step: (\d+)', line):
                step = int(m.group(1))
            
            # Match: varname[N] := value
            if m := re.match(r'(\w+)\[(\d+)\] := (\w+)', line):
                var, _, val = m.groups()
                if var.startswith('i'):
                    inputs[var] = val
                elif var.startswith('o'):
                    outputs[var] = val
        
        if step >= 0:
            return TauOutput(step, inputs, outputs, '\n'.join(lines))
        return None
    
    def step(self, *input_values: int) -> Optional[TauOutput]:
        """Send inputs and return parsed output"""
        outputs = self.send_inputs(*input_values)
        return self.parse_execution_step(outputs)
    
    def quit(self):
        """Stop Tau process"""
        self.running = False
        self._send('q')
        time.sleep(0.1)
        if self.process:
            self.process.terminate()
            self.process.wait()


class TauQController:
    """Connects Q-learning daemon to Tau execution"""
    
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        with open(spec_path, 'r') as f:
            self.spec_content = f.read()
        
        self.tau: Optional[TauProcess] = None
        self.history: List[TauOutput] = []
    
    def start(self):
        """Start Tau process"""
        self.tau = TauProcess(self.spec_content)
        self.tau.start()
        time.sleep(0.5)  # Wait for initialization
        
        # Drain initial output
        while True:
            out = self.tau.get_output(timeout=0.2)
            if not out:
                break
            print(f"[INIT] {out}")
    
    def step(self, price_up: bool, price_down: bool, 
             action_b0: int, action_b1: int) -> Optional[TauOutput]:
        """Execute one step with market data and Q-action"""
        if not self.tau:
            return None
        
        # Send all inputs
        result = self.tau.step(
            int(price_up), 
            int(price_down),
            action_b0,
            action_b1
        )
        
        if result:
            self.history.append(result)
        
        return result
    
    def run_episode(self, market_data: List[Tuple[bool, bool]],
                   action_selector) -> List[TauOutput]:
        """Run episode with market data and action selector function"""
        results = []
        
        for price_up, price_down in market_data:
            # Get action from selector (e.g., Q-table)
            action = action_selector(price_up, price_down)
            action_b0 = action & 1
            action_b1 = (action >> 1) & 1
            
            result = self.step(price_up, price_down, action_b0, action_b1)
            if result:
                results.append(result)
                print(f"Step {result.step}: in={result.inputs} out={result.outputs}")
        
        return results
    
    def stop(self):
        """Stop Tau process"""
        if self.tau:
            self.tau.quit()


def test_tau_integration():
    """Test direct Tau integration"""
    print("=" * 60)
    print("Tau Integration Test")
    print("=" * 60)
    
    spec = """
sbf i1 = console
sbf i2 = console
sbf o1 = console
sbf o2 = console
r o1[t] = i1[t] && o2[t] = i2[t]'
"""
    
    print("Starting Tau process...")
    tau = TauProcess(spec)
    tau.start()
    
    time.sleep(1)
    
    # Drain initial output
    print("\nInitial output:")
    while True:
        out = tau.get_output(timeout=0.3)
        if not out:
            break
        print(f"  {out}")
    
    # Send some inputs
    print("\nSending inputs...")
    for i, (v1, v2) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        print(f"\nStep {i}: i1={v1}, i2={v2}")
        outputs = tau.send_inputs(v1, v2)
        for out in outputs:
            print(f"  -> {out}")
    
    tau.quit()
    print("\nTest complete!")


if __name__ == "__main__":
    test_tau_integration()

