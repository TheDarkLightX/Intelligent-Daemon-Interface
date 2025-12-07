#!/usr/bin/env python3
"""
Q-Table Persistence and Checkpointing
- Save/load trained Q-tables
- Checkpoint during training
- Export for deployment
- Transfer learning support
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib


class QTablePersistence:
    """Save and load Q-tables with metadata"""
    
    SAVE_DIR = "checkpoints"
    
    def __init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
    
    def save(self, agent: Any, name: str, metadata: Optional[Dict] = None) -> str:
        """Save agent Q-tables and state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"
        filepath = os.path.join(self.SAVE_DIR, filename)
        
        # Collect Q-tables
        q_data = {}
        
        # Handle different agent types
        if hasattr(agent, 'ensemble'):
            # Master/Advanced agent with ensemble
            q_data['type'] = 'ensemble'
            q_data['tables'] = {}
            for tbl_name, tbl in agent.ensemble.tables.items():
                q_data['tables'][tbl_name] = tbl.tolist()
            if hasattr(agent.ensemble, 'visits'):
                q_data['visits'] = {k: v.tolist() for k, v in agent.ensemble.visits.items()}
        elif hasattr(agent, 'q_table'):
            # Simple agent with single Q-table
            q_data['type'] = 'single'
            if hasattr(agent.q_table, 'q1'):
                # Double Q-learning
                q_data['q1'] = agent.q_table.q1.tolist()
                q_data['q2'] = agent.q_table.q2.tolist()
            elif hasattr(agent.q_table, 'q'):
                q_data['q'] = agent.q_table.q.tolist()
        elif hasattr(agent, 'q_tables'):
            # Volatility specialist with multiple tables
            q_data['type'] = 'specialist'
            q_data['tables'] = {k: v.tolist() for k, v in agent.q_tables.items()}
        
        # Add stats
        stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
        
        # Build save data
        save_data = {
            'q_data': q_data,
            'stats': self._serialize_stats(stats),
            'metadata': metadata or {},
            'timestamp': timestamp,
            'checksum': self._compute_checksum(q_data)
        }
        
        # Save as JSON
        json_path = filepath + '.json'
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Also save numpy arrays for efficiency
        np_path = filepath + '.npz'
        np_arrays = {}
        if q_data['type'] == 'ensemble':
            for name, tbl in q_data['tables'].items():
                np_arrays[f'table_{name}'] = np.array(tbl)
        elif q_data['type'] == 'single':
            if 'q1' in q_data:
                np_arrays['q1'] = np.array(q_data['q1'])
                np_arrays['q2'] = np.array(q_data['q2'])
            else:
                np_arrays['q'] = np.array(q_data['q'])
        
        np.savez_compressed(np_path, **np_arrays)
        
        print(f"✅ Saved to {filepath}")
        return filepath
    
    def load(self, filepath: str) -> Dict:
        """Load Q-tables from file"""
        # Try JSON first
        json_path = filepath if filepath.endswith('.json') else filepath + '.json'
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Checkpoint not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Verify checksum
        computed = self._compute_checksum(data['q_data'])
        if computed != data['checksum']:
            print("⚠️ Warning: Checksum mismatch, data may be corrupted")
        
        # Convert lists back to numpy arrays
        q_data = data['q_data']
        if q_data['type'] == 'ensemble':
            q_data['tables'] = {k: np.array(v) for k, v in q_data['tables'].items()}
            if 'visits' in q_data:
                q_data['visits'] = {k: np.array(v) for k, v in q_data['visits'].items()}
        elif q_data['type'] == 'single':
            if 'q1' in q_data:
                q_data['q1'] = np.array(q_data['q1'])
                q_data['q2'] = np.array(q_data['q2'])
            else:
                q_data['q'] = np.array(q_data['q'])
        elif q_data['type'] == 'specialist':
            q_data['tables'] = {k: np.array(v) for k, v in q_data['tables'].items()}
        
        print(f"✅ Loaded from {filepath}")
        print(f"   Type: {q_data['type']}")
        print(f"   Stats: {data['stats']}")
        
        return data
    
    def list_checkpoints(self) -> list:
        """List available checkpoints"""
        checkpoints = []
        for f in os.listdir(self.SAVE_DIR):
            if f.endswith('.json'):
                path = os.path.join(self.SAVE_DIR, f)
                with open(path, 'r') as fp:
                    data = json.load(fp)
                checkpoints.append({
                    'name': f.replace('.json', ''),
                    'timestamp': data.get('timestamp', 'unknown'),
                    'type': data['q_data']['type'],
                    'stats': data.get('stats', {})
                })
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def _serialize_stats(self, stats: Dict) -> Dict:
        """Make stats JSON-serializable"""
        result = {}
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (np.int64, np.float64)):
                result[k] = float(v)
            elif isinstance(v, dict):
                result[k] = self._serialize_stats(v)
            elif hasattr(v, 'name'):  # Enum
                result[k] = v.name
            else:
                try:
                    json.dumps(v)
                    result[k] = v
                except:
                    result[k] = str(v)
        return result
    
    def _compute_checksum(self, q_data: Dict) -> str:
        """Compute checksum for data integrity"""
        data_str = json.dumps(q_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]


def apply_to_agent(agent: Any, loaded_data: Dict):
    """Apply loaded Q-tables to an agent"""
    q_data = loaded_data['q_data']
    
    if q_data['type'] == 'ensemble' and hasattr(agent, 'ensemble'):
        for name, tbl in q_data['tables'].items():
            if name in agent.ensemble.tables:
                agent.ensemble.tables[name] = tbl
        if 'visits' in q_data and hasattr(agent.ensemble, 'visits'):
            for name, v in q_data['visits'].items():
                if name in agent.ensemble.visits:
                    agent.ensemble.visits[name] = v
        print(f"✅ Applied ensemble tables: {list(q_data['tables'].keys())}")
    
    elif q_data['type'] == 'single' and hasattr(agent, 'q_table'):
        if 'q1' in q_data and hasattr(agent.q_table, 'q1'):
            agent.q_table.q1 = q_data['q1']
            agent.q_table.q2 = q_data['q2']
        elif 'q' in q_data and hasattr(agent.q_table, 'q'):
            agent.q_table.q = q_data['q']
        print("✅ Applied single Q-table")
    
    elif q_data['type'] == 'specialist' and hasattr(agent, 'q_tables'):
        for name, tbl in q_data['tables'].items():
            if name in agent.q_tables:
                agent.q_tables[name] = tbl
        print(f"✅ Applied specialist tables: {list(q_data['tables'].keys())}")


# Demo
if __name__ == "__main__":
    from master_agent import MasterAgent, gen_prices
    
    print("🧠 Training agent...")
    agent = MasterAgent(n_states=200)
    
    # Quick training
    for _ in range(20):
        prices = gen_prices(200, "mixed")
        agent.run_episode(prices)
    
    print(f"\n📊 Stats after training:")
    stats = agent.get_stats()
    print(f"  PnL: {stats['total_pnl']*100:+.2f}%")
    print(f"  Trades: {stats['trades']}")
    
    # Save
    persistence = QTablePersistence()
    path = persistence.save(agent, "master_demo", {"note": "Quick demo"})
    
    # List checkpoints
    print("\n📁 Available checkpoints:")
    for cp in persistence.list_checkpoints():
        print(f"  • {cp['name']} ({cp['type']})")
    
    # Create new agent and load
    print("\n🔄 Creating new agent and loading checkpoint...")
    new_agent = MasterAgent(n_states=200)
    data = persistence.load(path)
    apply_to_agent(new_agent, data)
    
    print("\n✅ Transfer complete!")

