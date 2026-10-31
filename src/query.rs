//! Words with NOT (in capitals, so that "not" in a phrase stays a plain word): "street at night NOT cars NOT people". Each part
//! is searched on its own, and every NOT takes away half of its cosine (the strongest NOT counts). A NOT weighed 0.5 kept
//! overlapping words working on Flickr30k ("people NOT children"). AND and OR were tried and dropped: an AND followed the weaker
//! word and an OR let one word crowd out the others, as the words' cosines don't share a scale; a plain phrase already means
//! "and" ("woman on a bike in winter"). Several photos are averaged.

/// How much of a NOT term's cosine is taken away.
const NOT_WEIGHT: f32 = 0.5;

pub struct Logic {
    /// The terms to search for: what to look for first, then what to push down.
    pub terms: Vec<String>,
    /// Photos, not words: every term counts the same (the average cosine ranks like the cosine to the photos' average vector).
    average: bool,
}

impl Logic {
    /// A search for one thing (a photo, or words without NOT).
    pub fn one() -> Self {
        Logic { terms: vec![String::new()], average: false }
    }

    /// A search for what several photos share.
    pub fn photos(n: usize) -> Self {
        Logic { terms: vec![String::new(); n], average: true }
    }

    /// Read the box's text. An error says what's wrong.
    pub fn parse(text: &str) -> Result<Self, String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Err("nothing to search for".into());
        }
        let mut terms: Vec<String> = Vec::new();
        for (i, part) in words.split(|w| *w == "NOT").enumerate() {
            let term = part.join(" ");
            if i == 0 && term.is_empty() {
                return Err("NOT needs something to look for before it, like \"street NOT cars\"".into());
            }
            if !term.is_empty() && !terms.contains(&term) {
                terms.push(term);
            }
        }
        Ok(Logic { terms, average: false })
    }

    /// The photos' scores from each term's cosines (in the order of `terms`).
    pub fn combine(&self, cos: &[Vec<f32>]) -> Vec<f32> {
        let (yes, no) = cos.split_first().expect("a search has a term");
        if self.average {
            return (0..yes.len()).map(|i| cos.iter().map(|c| c[i]).sum::<f32>() / cos.len() as f32).collect();
        }
        (0..yes.len())
            .map(|i| match no.iter().map(|c| c[i]).reduce(f32::max) {
                Some(no) => yes[i] - NOT_WEIGHT * no,
                None => yes[i],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_words_are_one_term() {
        let l = Logic::parse("salt and  pepper, not too much").unwrap();
        assert_eq!(l.terms, ["salt and pepper, not too much"]);
        assert_eq!(l.combine(&[vec![0.1, 0.3]]), [0.1, 0.3]);
    }

    #[test]
    fn not_takes_half_of_the_strongest_away() {
        let l = Logic::parse("street at night NOT cars NOT  people NOT").unwrap();
        assert_eq!(l.terms, ["street at night", "cars", "people"]);
        let cos = [vec![0.5, 0.25], vec![0.25, 0.0], vec![0.0, 0.125]];
        assert_eq!(l.combine(&cos), [0.375, 0.1875]);
    }

    #[test]
    fn photos_average() {
        assert_eq!(Logic::photos(2).combine(&[vec![0.5, 0.25], vec![0.25, 0.0]]), [0.375, 0.125]);
    }

    #[test]
    fn not_alone_is_an_error() {
        assert!(Logic::parse("NOT people").is_err());
        assert!(Logic::parse("  ").is_err());
    }
}
