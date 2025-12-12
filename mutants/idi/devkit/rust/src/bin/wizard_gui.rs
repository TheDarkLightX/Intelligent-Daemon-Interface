//! Wizard GUI - child-friendly egui interface for agent generation.

use eframe::egui;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WizardStep {
    Strategy,
    Inputs,
    Layers,
    Safety,
    Review,
}

struct WizardApp {
    step: WizardStep,
    agent_name: String,
    strategy: String,
    selected_inputs: std::collections::HashMap<String, bool>,
    num_layers: u32,
    include_safety: bool,
    include_communication: bool,
    preview_spec: String,
}

impl Default for WizardApp {
    fn default() -> Self {
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("q_buy".to_string(), true);
        inputs.insert("q_sell".to_string(), true);
        inputs.insert("price_up".to_string(), true);
        inputs.insert("price_down".to_string(), true);
        inputs.insert("trend".to_string(), false);
        inputs.insert("volume".to_string(), false);
        inputs.insert("regime".to_string(), false);
        inputs.insert("risk_budget_ok".to_string(), false);
        
        Self {
            step: WizardStep::Strategy,
            agent_name: String::new(),
            strategy: "momentum".to_string(),
            selected_inputs: inputs,
            num_layers: 1,
            include_safety: true,
            include_communication: false,
            preview_spec: String::new(),
        }
    }
}

impl WizardApp {
    fn step_number(&self) -> usize {
        match self.step {
            WizardStep::Strategy => 1,
            WizardStep::Inputs => 2,
            WizardStep::Layers => 3,
            WizardStep::Safety => 4,
            WizardStep::Review => 5,
        }
    }
    
    fn total_steps() -> usize {
        5
    }
    
    fn can_go_back(&self) -> bool {
        self.step != WizardStep::Strategy
    }
    
    fn can_go_next(&self) -> bool {
        match self.step {
            WizardStep::Strategy => !self.strategy.is_empty(),
            WizardStep::Inputs => {
                !self.agent_name.trim().is_empty()
                    && self.selected_inputs.values().any(|&v| v)
            }
            WizardStep::Layers => self.num_layers >= 1,
            WizardStep::Safety => true,
            WizardStep::Review => true,
        }
    }
    
    fn generate_preview(&mut self) {
        // Simple preview generation
        self.preview_spec = format!(
            "# {} Agent (Auto-generated)\n# Strategy: {}\n\n# Inputs:\n",
            self.agent_name, self.strategy
        );
        
        for (name, selected) in &self.selected_inputs {
            if *selected {
                self.preview_spec.push_str(&format!("i_{}:sbf = in file(\"inputs/{}.in\").\n", name, name));
            }
        }
        
        self.preview_spec.push_str("\n# Outputs:\no0:sbf = out file(\"outputs/position.out\").\n");
        self.preview_spec.push_str("o1:sbf = out file(\"outputs/buy_signal.out\").\n");
        self.preview_spec.push_str("o2:sbf = out file(\"outputs/sell_signal.out\").\n");
    }
    
    fn render_progress_bar(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!("Step {} of {}", self.step_number(), Self::total_steps()));
            
            let progress = self.step_number() as f32 / Self::total_steps() as f32;
            ui.add(egui::ProgressBar::new(progress).show_percentage());
        });
        ui.separator();
    }
    
    fn render_strategy_step(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pick Your Agent's Strategy");
        ui.add_space(10.0);
        ui.label("How should your agent trade?");
        ui.add_space(20.0);
        
        ui.radio_value(&mut self.strategy, "momentum".to_string(), "📈 Follow the Trend - Buy when prices go up");
        ui.add_space(5.0);
        ui.radio_value(&mut self.strategy, "mean_reversion".to_string(), "📊 Buy Low, Sell High - Buy dips, sell spikes");
        ui.add_space(5.0);
        ui.radio_value(&mut self.strategy, "regime_aware".to_string(), "🧠 Smart Adaptation - Change strategy with market");
    }
    
    fn render_inputs_step(&mut self, ui: &mut egui::Ui) {
        ui.heading("What Should Your Agent Watch?");
        ui.add_space(10.0);
        ui.label("Select the market signals your agent needs:");
        ui.add_space(10.0);
        
        ui.horizontal(|ui| {
            ui.label("Agent Name:");
            ui.text_edit_singleline(&mut self.agent_name);
        });
        ui.add_space(10.0);
        
        ui.separator();
        ui.label("Market Signals:");
        
        let input_labels = [
            ("q_buy", "Q-Learning Buy Signal"),
            ("q_sell", "Q-Learning Sell Signal"),
            ("price_up", "Price Going Up"),
            ("price_down", "Price Going Down"),
            ("trend", "Price Trend"),
            ("volume", "Trading Volume"),
            ("regime", "Market Regime"),
            ("risk_budget_ok", "Risk Budget OK"),
        ];
        
        for (key, label) in input_labels.iter() {
            let value = self.selected_inputs.get(*key).copied().unwrap_or(false);
            let mut checked = value;
            ui.checkbox(&mut checked, *label);
            self.selected_inputs.insert(key.to_string(), checked);
        }
    }
    
    fn render_layers_step(&mut self, ui: &mut egui::Ui) {
        ui.heading("How Many Layers?");
        ui.add_space(10.0);
        ui.label("Layers help your agent make better decisions:");
        ui.add_space(20.0);
        
        ui.horizontal(|ui| {
            ui.label("Number of Layers:");
            ui.add(egui::Slider::new(&mut self.num_layers, 1..=5));
            ui.label(format!("{}", self.num_layers));
        });
    }
    
    fn render_safety_step(&mut self, ui: &mut egui::Ui) {
        ui.heading("Safety Settings");
        ui.add_space(10.0);
        
        ui.checkbox(&mut self.include_safety, "Include Risk Management");
        ui.add_space(5.0);
        ui.checkbox(&mut self.include_communication, "Enable Communication Outputs");
    }
    
    fn render_review_step(&mut self, ui: &mut egui::Ui) {
        ui.heading("Review Your Agent");
        ui.add_space(10.0);
        
        if self.preview_spec.is_empty() {
            self.generate_preview();
        }
        
        ui.label("Generated Tau Spec Preview:");
        ui.add_space(5.0);
        
        egui::ScrollArea::vertical()
            .max_height(400.0)
            .show(ui, |ui| {
                ui.code_editor(&mut self.preview_spec);
            });
    }
    
    fn render_navigation(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.horizontal(|ui| {
            let back_enabled = self.can_go_back();
            if ui.add_enabled(back_enabled, egui::Button::new("← Back")).clicked() {
                if back_enabled {
                    self.step = match self.step {
                        WizardStep::Inputs => WizardStep::Strategy,
                        WizardStep::Layers => WizardStep::Inputs,
                        WizardStep::Safety => WizardStep::Layers,
                        WizardStep::Review => WizardStep::Safety,
                        _ => self.step,
                    };
                }
            }
            
            if self.step == WizardStep::Review {
                if ui.button("Finish & Save").clicked() {
                    // TODO: Save spec to file
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            } else {
                let next_enabled = self.can_go_next();
                if ui.add_enabled(next_enabled, egui::Button::new("Next →")).clicked() {
                    if next_enabled {
                        self.step = match self.step {
                            WizardStep::Strategy => WizardStep::Inputs,
                            WizardStep::Inputs => WizardStep::Layers,
                            WizardStep::Layers => WizardStep::Safety,
                            WizardStep::Safety => WizardStep::Review,
                            _ => self.step,
                        };
                        if self.step == WizardStep::Review {
                            self.generate_preview();
                        }
                    }
                }
            }
        });
    }
}

impl eframe::App for WizardApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_progress_bar(ui);
            
            match self.step {
                WizardStep::Strategy => self.render_strategy_step(ui),
                WizardStep::Inputs => self.render_inputs_step(ui),
                WizardStep::Layers => self.render_layers_step(ui),
                WizardStep::Safety => self.render_safety_step(ui),
                WizardStep::Review => self.render_review_step(ui),
            }
            
            self.render_navigation(ui);
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 700.0])
            .with_title("Tau Agent Factory - Create Your Agent"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Tau Agent Factory",
        options,
        Box::new(|_cc| Box::new(WizardApp::default())),
    )
}

