#[derive(Default, Clone, Copy)]
pub struct EmoteBits {
    pub positive: u8,
    pub alert: u8,
    pub persistence: u8,
}

pub struct EmotionEngine {
    linger: u8,
    counter: u8,
}

impl EmotionEngine {
    pub fn new(linger: u8) -> Self {
        Self { linger, counter: 0 }
    }

    pub fn render(&mut self, mood_bucket: u8) -> EmoteBits {
        let positive = u8::from(mood_bucket >= 2);
        let alert = u8::from(mood_bucket == 0);
        if positive == 1 || alert == 1 {
            self.counter = self.linger;
        } else if self.counter > 0 {
            self.counter -= 1;
        }
        EmoteBits {
            positive,
            alert,
            persistence: u8::from(self.counter > 0),
        }
    }
}
