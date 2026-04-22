use clap::{Parser, Subcommand};

use cascade_agent::agent::AgentLoop;
use cascade_agent::config::AgentConfig;

#[derive(Parser)]
#[command(name = "cascade-agent", version, about = "Async LLM agentic engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the agent with a prompt
    Run {
        /// The prompt to send to the agent
        prompt: String,
        /// Path to config file
        #[arg(long, default_value = "config.toml")]
        config: String,
    },
    /// Initialize a new config file
    Init {
        /// Output path for config file
        #[arg(long, default_value = "config.toml")]
        output: String,
    },
    /// List discovered skills
    Skills {
        #[arg(long, default_value = "config.toml")]
        config: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cascade_agent=info,tokio=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { prompt, config } => {
            let config = AgentConfig::load(std::path::Path::new(&config))?;
            let mut agent = AgentLoop::new(config).await?;
            let result = agent.run(prompt).await?;
            println!("{}", result);
        }
        Commands::Init { output } => {
            let default_config = include_str!("../config.example.toml");
            std::fs::write(&output, default_config)?;
            println!("Config file written to {}", output);
        }
        Commands::Skills { config } => {
            let config = AgentConfig::load(std::path::Path::new(&config))?;
            let mut skill_manager = cascade_agent::skills::SkillManager::new(
                std::path::PathBuf::from(&config.paths.skills_dir),
            )?;
            let discovered = skill_manager.discover()?;
            if discovered.is_empty() {
                println!("No skills discovered in {}", config.paths.skills_dir);
            } else {
                println!("Discovered {} skill(s):", discovered.len());
                for name in &discovered {
                    if let Some(skill) = skill_manager.get(name) {
                        println!(
                            "  - {} (v{})",
                            skill.metadata.name,
                            skill.metadata.version.as_deref().unwrap_or("?")
                        );
                        println!("    {}", skill.metadata.description);
                    }
                }
            }
        }
    }

    Ok(())
}
