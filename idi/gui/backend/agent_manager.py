import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class AgentMetadata(BaseModel):
    name: str
    strategy: str
    created_at: float
    path: str

class AgentManager:
    def __init__(self, practice_dir: str = "idi/practice"):
        # Resolve to absolute path relative to CWD if necessary, 
        # but relying on simple relative path for now as we run from root.
        self.practice_dir = Path(practice_dir)
        self.practice_dir.mkdir(parents=True, exist_ok=True)

    def list_agents(self) -> List[AgentMetadata]:
        agents = []
        if not self.practice_dir.exists():
            return []
            
        for item in self.practice_dir.iterdir():
            if item.is_dir():
                # Check for metadata
                meta_file = item / "wizard_data.json"
                if meta_file.exists():
                    try:
                        data = json.loads(meta_file.read_text())
                        agents.append(AgentMetadata(
                            name=item.name,
                            strategy=data.get("strategy", "unknown"),
                            created_at=meta_file.stat().st_mtime,
                            path=str(item)
                        ))
                    except Exception:
                        continue
        return sorted(agents, key=lambda x: x.created_at, reverse=True)

    def save_agent(self, name: str, spec: str, wizard_data: Dict[str, Any]) -> str:
        # Sanitize name
        safe_name = "".join([c for c in name if c.isalnum() or c in ('-', '_')]).strip()
        if not safe_name:
            raise ValueError("Invalid agent name")
            
        agent_dir = self.practice_dir / safe_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files
        (agent_dir / "agent.tau").write_text(spec)
        (agent_dir / "wizard_data.json").write_text(json.dumps(wizard_data, indent=2))
        
        return str(agent_dir)

    def get_agent_data(self, name: str) -> Optional[Dict[str, Any]]:
        safe_name = "".join([c for c in name if c.isalnum() or c in ('-', '_')]).strip()
        agent_dir = self.practice_dir / safe_name
        meta_file = agent_dir / "wizard_data.json"
        
        if meta_file.exists():
            return json.loads(meta_file.read_text())
        return None

    def delete_agent(self, name: str) -> bool:
        safe_name = "".join([c for c in name if c.isalnum() or c in ('-', '_')]).strip()
        agent_dir = self.practice_dir / safe_name
        if agent_dir.exists() and agent_dir.is_dir():
            shutil.rmtree(agent_dir)
            return True
        return False
