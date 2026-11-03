//! Ideas in the search box, from the examples (query.rs) and assets/search_ideas.txt (~3,000 phrases written as searches). While
//! the box is empty its placeholder is one at random; while typing, the rest of one that starts with what was typed shows after
//! it, faint. Up and down switch to another idea, Tab takes it. When nothing finishes what was typed, or the box holds an idea
//! taken whole, up and down put a whole other search in the box instead. After a NOT only the part after it is completed
//! ("street NOT ca" -> "street NOT cars").

const IDEAS: &str = include_str!("../assets/search_ideas.txt");

pub struct Suggest {
    pool: Vec<String>,
    /// What the ideas below are for (the part being typed, lowercase), and the ideas: places in `pool`, in a random order.
    typed: String,
    ideas: Vec<usize>,
    at: usize,
    /// Every idea in a random order, for whole searches, and the one last put in the box whole (or taken with Tab).
    all: Vec<usize>,
    all_at: usize,
    given: String,
    seed: u64,
}

impl Suggest {
    pub fn new() -> Self {
        let mut seen = std::collections::HashSet::new();
        let ideas = crate::query::EXAMPLES.iter().map(|(text, _)| *text).chain(IDEAS.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')));
        let pool = ideas.map(String::from).filter(|p| seen.insert(p.to_lowercase())).collect();
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64) | 1;
        Self::with(pool, seed)
    }

    fn with(pool: Vec<String>, seed: u64) -> Self {
        let mut s = Suggest { pool, typed: String::new(), ideas: Vec::new(), at: 0, all: Vec::new(), all_at: 0, given: String::new(), seed };
        let mut all: Vec<usize> = (0..s.pool.len()).collect();
        s.shuffle(&mut all);
        s.all = all;
        s.refill("");
        s
    }

    /// The whole box's text with the idea for `text` (what is in the box) filled in, if there is one. Empty box: a search to try.
    pub fn idea(&mut self, text: &str) -> Option<String> {
        let (head, part) = split(text);
        let part_lc = part.to_lowercase();
        if part_lc != self.typed {
            self.refill(&part_lc);
        }
        let idea = &self.pool[*self.ideas.get(self.at)?];
        Some(format!("{head}{part}{}", idea.chars().skip(part.chars().count()).collect::<String>()))
    }

    /// Whether up / down put a whole other search in the box for `text`: nothing finishes it, or it is the idea last given whole.
    pub fn replaces(&mut self, text: &str) -> bool {
        !text.is_empty() && (text.eq_ignore_ascii_case(&self.given) || self.idea(text).is_none())
    }

    /// Up / down (1 / -1): another ending for `text`, or, when `replaces`, a whole other search: the box's new text.
    pub fn switch(&mut self, text: &str, by: i32) -> Option<String> {
        if self.replaces(text) {
            let n = self.all.len() as i64;
            if n == 0 {
                return None;
            }
            self.all_at = (self.all_at as i64 + by as i64).rem_euclid(n) as usize;
            let whole = self.pool[self.all[self.all_at]].clone();
            self.took(&whole);
            return Some(whole);
        }
        if !self.ideas.is_empty() {
            self.at = (self.at as i64 + by as i64).rem_euclid(self.ideas.len() as i64) as usize;
        }
        None
    }

    /// The box now holds this idea whole (Tab took it): up / down go on to other whole searches.
    pub fn took(&mut self, text: &str) {
        self.given = text.to_string();
    }

    /// The ideas for what is being typed, in a new random order; the one showing stays first while it still fits.
    fn refill(&mut self, typed: &str) {
        let showing = self.ideas.get(self.at).copied();
        let mut ideas: Vec<usize> = (0..self.pool.len()).filter(|&i| fits(&self.pool[i], typed)).collect();
        self.shuffle(&mut ideas);
        if let Some(k) = showing.filter(|_| !typed.is_empty()).and_then(|s| ideas.iter().position(|&i| i == s)) {
            ideas.swap(0, k);
        }
        self.typed = typed.to_string();
        self.ideas = ideas;
        self.at = 0;
    }

    fn shuffle(&mut self, list: &mut [usize]) {
        for i in (1..list.len()).rev() {
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 7;
            self.seed ^= self.seed << 17;
            list.swap(i, (self.seed % (i as u64 + 1)) as usize);
        }
    }
}

/// An idea fits what is typed if it starts with it (any case) and adds something.
fn fits(idea: &str, typed: &str) -> bool {
    idea.chars().count() > typed.chars().count() && idea.to_lowercase().starts_with(typed)
}

/// The box's text before the part being typed (up to and with the last "NOT "), and that part.
fn split(text: &str) -> (&str, &str) {
    match text.rfind(" NOT ") {
        Some(k) => text.split_at(k + 5),
        None => ("", text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> Vec<String> {
        ["red car", "red barn", "river", "cars"].map(String::from).to_vec()
    }

    #[test]
    fn empty_box_shows_an_idea() {
        assert!(Suggest::with(pool(), 7).idea("").is_some());
    }

    #[test]
    fn typing_completes_and_keeps_the_typed_case() {
        let mut s = Suggest::with(pool(), 7);
        let idea = s.idea("Red ").unwrap();
        assert!(idea == "Red car" || idea == "Red barn", "{idea}");
        assert_eq!(s.switch("Red ", 1), None);
        let other = s.idea("Red ").unwrap();
        assert_ne!(idea, other);
        s.switch("Red ", 1);
        assert_eq!(s.idea("Red ").unwrap(), idea); // round again
        assert_eq!(s.idea("red car"), None); // nothing longer starts so
    }

    #[test]
    fn the_idea_stays_while_it_still_fits() {
        let mut s = Suggest::with(pool(), 7);
        let first = s.idea("r").unwrap();
        assert_eq!(s.idea(&first[..2]).unwrap(), first);
    }

    #[test]
    fn after_not_only_the_last_part_is_completed() {
        let mut s = Suggest::with(pool(), 7);
        assert_eq!(s.idea("street NOT ca").unwrap(), "street NOT cars");
        assert_eq!(s.idea("street NOT x"), None);
    }

    #[test]
    fn with_nothing_to_finish_up_and_down_give_whole_searches() {
        let mut s = Suggest::with(pool(), 7);
        let a = s.switch("a long query nothing starts with", 1).unwrap();
        assert!(pool().contains(&a));
        // a whole idea given stays whole: the next switch replaces it again
        let b = s.switch(&a, 1).unwrap();
        assert_ne!(a, b);
        assert_eq!(s.switch(&b, -1).unwrap(), a);
        assert_eq!(s.switch("", 1), None); // the empty box switches its placeholder
    }

    #[test]
    fn the_real_pool_has_the_examples_and_the_ideas() {
        let s = Suggest::new();
        assert!(s.pool.len() > 2500);
        assert!(s.pool.iter().any(|p| p == "foggy forest") && s.pool.iter().any(|p| p == "misty mountains at sunrise"));
    }
}
