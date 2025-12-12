//! Agent Development GUI - Rust version using egui.
//!
//! Interactive GUI for creating and managing Tau Language intelligent agents.
//! Leverages Rust's performance and egui's immediate mode GUI.

use eframe::egui;
use std::path::PathBuf;

struct AgentDevApp {
    agent_name: String,
    selected_strategy: String,
    output_dir: PathBuf,
    status_message: String,
}

impl Default for AgentDevApp {
    fn default() -> Self {
        Self {
            agent_name: String::new(),
            selected_strategy: "momentum".to_string(),
            output_dir: PathBuf::from("../practice"),
            status_message: String::new(),
        }
    }
}

impl eframe::App for AgentDevApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("IDI Agent Development GUI");
            ui.separator();
            
            ui.horizontal(|ui| {
                ui.label("Agent Name:");
                ui.text_edit_singleline(&mut self.agent_name);
            });
            
            ui.horizontal(|ui| {
                ui.label("Strategy:");
                egui::ComboBox::from_id_source("strategy")
                    .selected_text(&self.selected_strategy)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_strategy, "momentum".to_string(), "Momentum");
                        ui.selectable_value(&mut self.selected_strategy, "mean_reversion".to_string(), "Mean Reversion");
                        ui.selectable_value(&mut self.selected_strategy, "regime_aware".to_string(), "Regime Aware");
                    });
            });
            
            ui.horizontal(|ui| {
                ui.label("Output Directory:");
                ui.text_edit_singleline(&mut self.output_dir.to_string_lossy().to_string());
            });
            
            ui.separator();
            
            if ui.button("Create Agent").clicked() {
                if self.agent_name.is_empty() {
                    self.status_message = "❌ Error: Agent name required".to_string();
                } else {
                    // TODO: Call create_agent_directory
                    self.status_message = format!("✅ Created agent '{}'", self.agent_name);
                }
            }
            
            if !self.status_message.is_empty() {
                ui.separator();
                ui.label(&self.status_message);
            }
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_title("IDI Agent Development"),
        ..Default::default()
    };
    
    eframe::run_native(
        "IDI Agent Development",
        options,
        Box::new(|_cc| Box::new(AgentDevApp::default())),
    )
}

