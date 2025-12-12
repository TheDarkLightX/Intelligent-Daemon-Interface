//! Agent Development CLI - Rust version.
//!
//! High-performance tool for creating Tau Language intelligent agents.
//! Leverages Rust's type safety and performance for production use.

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

#[derive(Parser)]
#[command(name = "create-agent")]
#[command(about = "Create a new Tau Language intelligent agent", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new agent
    New {
        /// Agent name (e.g., 'momentum_agent')
        #[arg(short, long)]
        name: String,
        
        /// Strategy template to use
        #[arg(short, long, default_value = "momentum")]
        strategy: String,
        
        /// Output directory
        #[arg(short, long, default_value = "../practice")]
        out: PathBuf,
        
        /// Custom template JSON file
        #[arg(long)]
        custom_template: Option<PathBuf>,
    },
    
    /// List available templates
    List,
}

#[derive(Debug, Serialize, Deserialize)]
struct AgentTemplate {
    description: String,
    tau_spec: String,
    training_config: TrainingConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingConfig {
    episodes: u32,
    episode_length: u32,
    discount: f64,
    learning_rate: f64,
    exploration_decay: f64,
    quantizer: QuantizerConfig,
    rewards: RewardWeights,
}

#[derive(Debug, Serialize, Deserialize)]
struct QuantizerConfig {
    price_buckets: u8,
    volume_buckets: u8,
    trend_buckets: u8,
    scarcity_buckets: u8,
    mood_buckets: u8,
}

#[derive(Debug, Serialize, Deserialize)]
struct RewardWeights {
    pnl: f64,
    scarcity_alignment: f64,
    ethics_bonus: f64,
    communication_clarity: f64,
}

fn load_templates() -> std::collections::HashMap<String, AgentTemplate> {
    let mut templates = std::collections::HashMap::new();
    
    // Momentum template
    templates.insert("momentum".to_string(), AgentTemplate {
        description: "Momentum following strategy".to_string(),
        tau_spec: include_str!("../../templates/momentum.tau.template").to_string(),
        training_config: TrainingConfig {
            episodes: 256,
            episode_length: 128,
            discount: 0.95,
            learning_rate: 0.15,
            exploration_decay: 0.996,
            quantizer: QuantizerConfig {
                price_buckets: 8,
                volume_buckets: 4,
                trend_buckets: 8,
                scarcity_buckets: 4,
                mood_buckets: 4,
            },
            rewards: RewardWeights {
                pnl: 1.0,
                scarcity_alignment: 0.3,
                ethics_bonus: 0.2,
                communication_clarity: 0.1,
            },
        },
    });
    
    templates
}

fn create_agent_directory(
    name: &str,
    strategy: &str,
    output_dir: &Path,
    template: &AgentTemplate,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let agent_dir = output_dir.join(name);
    fs::create_dir_all(&agent_dir)?;
    
    // Create subdirectories
    fs::create_dir_all(agent_dir.join("inputs"))?;
    fs::create_dir_all(agent_dir.join("outputs"))?;
    fs::create_dir_all(agent_dir.join("tests"))?;
    
    // Write Tau spec
    let spec_content = template.tau_spec.replace("{name}", name).replace("{description}", &template.description);
    fs::write(agent_dir.join(format!("{}.tau", name)), spec_content)?;
    
    // Write training config
    let config_json = serde_json::to_string_pretty(&template.training_config)?;
    fs::write(agent_dir.join("training_config.json"), config_json)?;
    
    // Write README
    let readme = format!(
        "# {}\n\n**Strategy**: {}\n\n## Quick Start\n\n1. Train Q-table:\n   ```bash\n   python3 train_agent.py\n   ```\n\n2. Run agent:\n   ```bash\n   ./run_agent.sh\n   ```\n\n## Configuration\n\nTraining config: `training_config.json`\n\nTau spec: `{}.tau`\n\n## Development Notes\n\n- Created with IDI Agent Development CLI (Rust)\n- Strategy: {}\n",
        name, template.description, name, strategy
    );
    fs::write(agent_dir.join("README.md"), readme)?;
    
    Ok(agent_dir)
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::List => {
            let templates = load_templates();
            println!("Available templates:");
            for (name, template) in templates.iter() {
                println!("  {:15} - {}", name, template.description);
            }
        }
        Commands::New { name, strategy, out, custom_template } => {
            let templates = load_templates();
            
            let template = if let Some(custom_path) = custom_template {
                let content = fs::read_to_string(&custom_path)
                    .map_err(|e| format!("Failed to read template: {}", e))?;
                serde_json::from_str(&content)
                    .map_err(|e| format!("Failed to parse template: {}", e))?
            } else {
                templates.get(&strategy)
                    .ok_or_else(|| format!("Unknown strategy: {}", strategy))?
                    .clone()
            };
            
            match create_agent_directory(&name, &strategy, &out, &template) {
                Ok(agent_dir) => {
                    println!("✅ Created agent '{}' at {}", name, agent_dir.display());
                    println!("\nNext steps:");
                    println!("  1. cd {}", agent_dir.display());
                    println!("  2. python3 train_agent.py");
                    println!("  3. ./run_agent.sh");
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}

