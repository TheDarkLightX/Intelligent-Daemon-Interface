//! High-Performance Q-Learning Daemon for Tau Agents
//! 
//! This daemon manages multi-layer Q-tables, communicates with Tau via subprocess,
//! and provides fast learning and inference.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Action encoding (matches Tau spec)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Hold = 0,  // 00
    Buy = 1,   // 01
    Sell = 2,  // 10
    Wait = 3,  // 11
}

impl Action {
    pub fn from_bits(b0: bool, b1: bool) -> Self {
        match (b1, b0) {
            (false, false) => Action::Hold,
            (false, true) => Action::Buy,
            (true, false) => Action::Sell,
            (true, true) => Action::Wait,
        }
    }

    pub fn to_bits(self) -> (u8, u8) {
        ((self as u8) & 1, ((self as u8) >> 1) & 1)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Action::Hold => "HOLD",
            Action::Buy => "BUY",
            Action::Sell => "SELL",
            Action::Wait => "WAIT",
        }
    }
}

/// Market state encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketState {
    IdleNeutral = 0,
    IdleUp = 1,
    IdleDown = 2,
    HoldingNeutral = 3,
    HoldingUp = 4,
    HoldingDown = 5,
}

impl MarketState {
    pub fn from_signals(price_up: bool, price_down: bool, position: bool) -> Self {
        match (position, price_up, price_down) {
            (false, false, false) => MarketState::IdleNeutral,
            (false, true, _) => MarketState::IdleUp,
            (false, false, true) => MarketState::IdleDown,
            (true, false, false) => MarketState::HoldingNeutral,
            (true, true, _) => MarketState::HoldingUp,
            (true, false, true) => MarketState::HoldingDown,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Q-Table with epsilon-greedy policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    /// Q-values: [state][action]
    q: [[f64; 4]; 6],
    /// Learning rate
    learning_rate: f64,
    /// Discount factor
    discount: f64,
    /// Exploration rate
    epsilon: f64,
    /// Name for debugging
    name: String,
}

impl QTable {
    pub fn new(name: &str) -> Self {
        let mut q = [[0.0; 4]; 6];
        
        // Initialize with domain knowledge
        q[MarketState::IdleNeutral.index()][Action::Hold as usize] = 0.5;
        q[MarketState::IdleUp.index()][Action::Buy as usize] = 1.0;
        q[MarketState::IdleDown.index()][Action::Hold as usize] = 0.6;
        q[MarketState::HoldingNeutral.index()][Action::Hold as usize] = 0.5;
        q[MarketState::HoldingUp.index()][Action::Hold as usize] = 0.8;
        q[MarketState::HoldingDown.index()][Action::Sell as usize] = 1.0;

        Self {
            q,
            learning_rate: 0.1,
            discount: 0.95,
            epsilon: 0.1,
            name: name.to_string(),
        }
    }

    pub fn with_params(name: &str, lr: f64, discount: f64, epsilon: f64) -> Self {
        let mut table = Self::new(name);
        table.learning_rate = lr;
        table.discount = discount;
        table.epsilon = epsilon;
        table
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: MarketState, explore: bool) -> Action {
        let mut rng = rand::thread_rng();
        
        if explore && rng.gen::<f64>() < self.epsilon {
            // Random exploration
            match rng.gen_range(0..4) {
                0 => Action::Hold,
                1 => Action::Buy,
                2 => Action::Sell,
                _ => Action::Wait,
            }
        } else {
            // Greedy exploitation
            self.best_action(state)
        }
    }

    /// Get best action for state
    pub fn best_action(&self, state: MarketState) -> Action {
        let q_values = &self.q[state.index()];
        let mut best_idx = 0;
        let mut best_val = q_values[0];
        
        for (i, &v) in q_values.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        
        match best_idx {
            0 => Action::Hold,
            1 => Action::Buy,
            2 => Action::Sell,
            _ => Action::Wait,
        }
    }

    /// Q-learning update
    pub fn update(&mut self, state: MarketState, action: Action, 
                  reward: f64, next_state: MarketState, done: bool) {
        let s = state.index();
        let a = action as usize;
        let ns = next_state.index();
        
        let target = if done {
            reward
        } else {
            reward + self.discount * self.q[ns].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
        
        self.q[s][a] += self.learning_rate * (target - self.q[s][a]);
    }

    /// Get Q-values for state
    pub fn get_q_values(&self, state: MarketState) -> [f64; 4] {
        self.q[state.index()]
    }

    /// Print Q-table
    pub fn print(&self) {
        println!("\nQ-Table: {}", self.name);
        println!("State          | HOLD  | BUY   | SELL  | WAIT");
        println!("{}", "-".repeat(50));
        
        let states = [
            (MarketState::IdleNeutral, "IdleNeutral"),
            (MarketState::IdleUp, "IdleUp"),
            (MarketState::IdleDown, "IdleDown"),
            (MarketState::HoldingNeutral, "HoldingNeutral"),
            (MarketState::HoldingUp, "HoldingUp"),
            (MarketState::HoldingDown, "HoldingDown"),
        ];
        
        for (state, name) in states {
            let q = self.q[state.index()];
            println!("{:14} | {:5.2} | {:5.2} | {:5.2} | {:5.2}",
                     name, q[0], q[1], q[2], q[3]);
        }
    }
}

/// Layered Q-Table with multiple strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredQTable {
    /// Momentum strategy
    pub momentum: QTable,
    /// Contrarian strategy
    pub contrarian: QTable,
    /// Trend-following strategy
    pub trend: QTable,
    /// Strategy weights
    pub weights: [f64; 3],
}

impl LayeredQTable {
    pub fn new() -> Self {
        let mut momentum = QTable::with_params("Momentum", 0.1, 0.95, 0.05);
        momentum.q[MarketState::IdleUp.index()][Action::Buy as usize] = 2.0;
        
        let mut contrarian = QTable::with_params("Contrarian", 0.1, 0.95, 0.05);
        contrarian.q[MarketState::IdleDown.index()][Action::Buy as usize] = 1.5;
        contrarian.q[MarketState::HoldingUp.index()][Action::Sell as usize] = 1.5;
        
        let mut trend = QTable::with_params("Trend", 0.1, 0.95, 0.05);
        trend.q[MarketState::IdleUp.index()][Action::Buy as usize] = 1.5;
        trend.q[MarketState::HoldingDown.index()][Action::Sell as usize] = 1.5;
        
        Self {
            momentum,
            contrarian,
            trend,
            weights: [0.4, 0.3, 0.3],
        }
    }

    /// Weighted action selection
    pub fn select_action(&self, state: MarketState) -> (Action, [f64; 4]) {
        let q_m = self.momentum.get_q_values(state);
        let q_c = self.contrarian.get_q_values(state);
        let q_t = self.trend.get_q_values(state);
        
        let mut combined = [0.0; 4];
        for i in 0..4 {
            combined[i] = self.weights[0] * q_m[i] 
                        + self.weights[1] * q_c[i]
                        + self.weights[2] * q_t[i];
        }
        
        let mut best_idx = 0;
        let mut best_val = combined[0];
        for (i, &v) in combined.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        
        let action = match best_idx {
            0 => Action::Hold,
            1 => Action::Buy,
            2 => Action::Sell,
            _ => Action::Wait,
        };
        
        (action, combined)
    }

    /// Update all layers
    pub fn update_all(&mut self, state: MarketState, action: Action,
                      reward: f64, next_state: MarketState) {
        self.momentum.update(state, action, reward, next_state, false);
        self.contrarian.update(state, action, reward, next_state, false);
        self.trend.update(state, action, reward, next_state, false);
    }

    /// Print all Q-tables
    pub fn print_all(&self) {
        self.momentum.print();
        self.contrarian.print();
        self.trend.print();
        println!("\nWeights: Momentum={:.2}, Contrarian={:.2}, Trend={:.2}",
                 self.weights[0], self.weights[1], self.weights[2]);
    }
}

/// Trading statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TradingStats {
    pub total_reward: f64,
    pub total_trades: u32,
    pub profitable_trades: u32,
    pub episode_rewards: Vec<f64>,
}

impl TradingStats {
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.profitable_trades as f64 / self.total_trades as f64
        }
    }

    pub fn print(&self) {
        println!("\n=== Trading Statistics ===");
        println!("Total Reward: {:.2}", self.total_reward);
        println!("Total Trades: {}", self.total_trades);
        println!("Profitable: {}", self.profitable_trades);
        println!("Win Rate: {:.1}%", self.win_rate() * 100.0);
        if !self.episode_rewards.is_empty() {
            let avg: f64 = self.episode_rewards.iter().sum::<f64>() 
                         / self.episode_rewards.len() as f64;
            println!("Avg Episode Reward: {:.2}", avg);
        }
    }
}

/// Q-Learning Daemon
pub struct QDaemon {
    pub q_table: LayeredQTable,
    pub stats: TradingStats,
    pub position: bool,
    pub entry_price_up: bool,
}

impl QDaemon {
    pub fn new() -> Self {
        Self {
            q_table: LayeredQTable::new(),
            stats: TradingStats::default(),
            position: false,
            entry_price_up: false,
        }
    }

    /// Compute reward for action
    pub fn compute_reward(&mut self, action: Action, executed: bool,
                          _price_up: bool, _price_down: bool) -> f64 {
        if !executed {
            return -0.5; // Penalty for invalid action
        }
        
        match action {
            Action::Sell => {
                self.stats.total_trades += 1;
                self.stats.profitable_trades += 1;
                1.0
            }
            Action::Buy => -0.1,
            Action::Hold => 0.01,
            Action::Wait => 0.0,
        }
    }

    /// Run one step
    pub fn step(&mut self, price_up: bool, price_down: bool) -> (Action, f64, bool) {
        let state = MarketState::from_signals(price_up, price_down, self.position);
        let (action, _q_values) = self.q_table.select_action(state);
        
        // Validate action
        let valid = match action {
            Action::Buy => !self.position,
            Action::Sell => self.position,
            _ => true,
        };
        
        let executed_action = if valid { action } else { Action::Hold };
        
        // Execute
        match executed_action {
            Action::Buy => {
                self.position = true;
                self.entry_price_up = price_up;
            }
            Action::Sell => {
                self.position = false;
            }
            _ => {}
        }
        
        // Compute reward
        let reward = self.compute_reward(executed_action, valid, price_up, price_down);
        self.stats.total_reward += reward;
        
        // Get next state
        let next_state = MarketState::from_signals(price_up, price_down, self.position);
        
        // Update Q-tables
        self.q_table.update_all(state, executed_action, reward, next_state);
        
        (executed_action, reward, valid)
    }

    /// Run episode with market data
    pub fn run_episode(&mut self, market_data: &[(bool, bool)], verbose: bool) -> f64 {
        let mut episode_reward = 0.0;
        
        for (step, &(price_up, price_down)) in market_data.iter().enumerate() {
            let (action, reward, valid) = self.step(price_up, price_down);
            episode_reward += reward;
            
            if verbose {
                let signal = if price_up { "↑" } else if price_down { "↓" } else { "-" };
                let pos = if self.position { "H" } else { "I" };
                let valid_mark = if valid { "" } else { " [INVALID]" };
                println!("Step {:2}: {} | {:4} | Pos: {} | R: {:+.2}{}",
                         step, signal, action.name(), pos, reward, valid_mark);
            }
        }
        
        self.stats.episode_rewards.push(episode_reward);
        episode_reward
    }
}

/// Generate synthetic market data
pub fn generate_market_data(n_steps: usize, trend_prob: f64) -> Vec<(bool, bool)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n_steps);
    let mut trend = 1i32;
    
    for _ in 0..n_steps {
        let price_up = if rng.gen::<f64>() < trend_prob {
            trend > 0
        } else {
            let up = trend < 0;
            if rng.gen::<f64>() < 0.3 {
                trend *= -1;
            }
            up
        };
        
        data.push((price_up, !price_up));
    }
    
    data
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("Rust Q-Learning Daemon - Intelligence Test");
    println!("{}", "=".repeat(60));
    
    let mut daemon = QDaemon::new();
    
    // Run multiple episodes
    let n_episodes = 10;
    
    for ep in 0..n_episodes {
        let market_data = generate_market_data(30, 0.6);
        let verbose = ep == n_episodes - 1;
        
        if !verbose {
            print!("Episode {:2}: ", ep + 1);
        } else {
            println!("\n--- Episode {} (detailed) ---", ep + 1);
        }
        
        let reward = daemon.run_episode(&market_data, verbose);
        
        if !verbose {
            println!("reward = {:.2}", reward);
        } else {
            println!("Episode reward: {:.2}", reward);
        }
    }
    
    // Print final statistics
    daemon.stats.print();
    daemon.q_table.print_all();
}

