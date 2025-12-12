use clap::{Parser, Subcommand};
use idi_iann::{config::TrainingConfig, trainer::QTrainer};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "train")]
#[command(about = "IDI/IANN Q-learning trainer CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[arg(long, global = true)]
    seed: Option<u64>,

    #[arg(long, global = true)]
    episodes: Option<usize>,

    #[arg(long, global = true)]
    out: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(long)]
        episodes: Option<usize>,
    },
    Export {
        #[arg(long)]
        policy: PathBuf,
    },
}

fn load_config(path: Option<&PathBuf>) -> anyhow::Result<TrainingConfig> {
    match path {
        Some(p) => {
            let content = fs::read_to_string(p)?;
            let config: TrainingConfig = serde_json::from_str(&content)?;
            Ok(config)
        }
        None => Ok(TrainingConfig::default()),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut config = load_config(cli.config.as_ref())?;

    if let Some(episodes) = cli.episodes {
        config.episodes = episodes;
    }

    match cli.command {
        Commands::Train { episodes } => {
            if let Some(eps) = episodes {
                config.episodes = eps;
            }
            let mut trainer = QTrainer::new(config);
            let trace = trainer.run();
            let out_dir = cli.out.unwrap_or_else(|| PathBuf::from("outputs"));
            fs::create_dir_all(&out_dir)?;
            let trace_file = out_dir.join("trace.json");
            let trace_json = serde_json::to_string_pretty(&trace)?;
            fs::write(&trace_file, trace_json)?;
            println!("Training complete. {} ticks written to {}", trace.len(), trace_file.display());
        }
        Commands::Export { policy } => {
            println!("Export command not yet implemented. Policy: {}", policy.display());
        }
    }

    Ok(())
}

