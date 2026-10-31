//! Words joined with AND, OR and NOT (in capitals, so that "salt and pepper" stays plain words). Each term is searched on its own
//! and the cosines are combined: an AND takes the lower of its terms (a photo must match them all), an OR the higher, and a NOT
//! takes half of its term's cosine away. AND goes before OR: "beach AND dog OR cat NOT snow" is (beach and dog) or (cat, no snow).
//! Raw cosines did better than normalised ones on Flickr30k (with normalising, a rare term took over an OR); a NOT weighed 0.5
//! kept overlapping terms working ("people NOT children").

/// How much of a NOT term's cosine is taken away.
const NOT_WEIGHT: f32 = 0.5;

/// One AND group: its terms (numbers in `Logic::terms`) that must match, and those that must not.
struct All {
    yes: Vec<usize>,
    no: Vec<usize>,
}

pub struct Logic {
    /// The terms to search for, each once.
    pub terms: Vec<String>,
    /// The OR's alternatives.
    any: Vec<All>,
}

impl Logic {
    /// A search for one thing (a photo, or words without AND, OR or NOT).
    pub fn one() -> Self {
        Logic { terms: vec![String::new()], any: vec![All { yes: vec![0], no: Vec::new() }] }
    }

    /// Read the box's text. Plain words are one term. An error says what's wrong.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut terms: Vec<String> = Vec::new();
        let mut any = vec![All { yes: Vec::new(), no: Vec::new() }];
        let mut words: Vec<&str> = Vec::new();
        let mut not = false;
        let mut end = |words: &mut Vec<&str>, not: bool, any: &mut Vec<All>| {
            if words.is_empty() {
                return;
            }
            let term = words.join(" ");
            words.clear();
            let k = terms.iter().position(|t| *t == term).unwrap_or_else(|| {
                terms.push(term);
                terms.len() - 1
            });
            let all = any.last_mut().unwrap();
            if not { all.no.push(k) } else { all.yes.push(k) }
        };
        for word in text.split_whitespace() {
            match word {
                "AND" | "NOT" => {
                    end(&mut words, not, &mut any);
                    not = word == "NOT";
                }
                "OR" => {
                    end(&mut words, not, &mut any);
                    not = false;
                    any.push(All { yes: Vec::new(), no: Vec::new() });
                }
                _ => words.push(word),
            }
        }
        end(&mut words, not, &mut any);
        any.retain(|a| !a.yes.is_empty() || !a.no.is_empty());
        if any.is_empty() {
            return Err("nothing to search for".into());
        }
        if any.iter().any(|a| a.yes.is_empty()) {
            return Err("NOT needs something to look for before it, like \"street NOT cars\"".into());
        }
        Ok(Logic { terms, any })
    }

    /// The photos' scores from each term's cosines (in the order of `terms`).
    pub fn combine(&self, cos: &[Vec<f32>]) -> Vec<f32> {
        let n = cos.first().map_or(0, Vec::len);
        (0..n)
            .map(|i| {
                self.any
                    .iter()
                    .map(|all| {
                        let yes = all.yes.iter().map(|&k| cos[k][i]).fold(f32::INFINITY, f32::min);
                        let no = all.no.iter().map(|&k| cos[k][i]).fold(f32::NEG_INFINITY, f32::max);
                        if all.no.is_empty() { yes } else { yes - NOT_WEIGHT * no }
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_words_are_one_term() {
        let l = Logic::parse("salt and  pepper").unwrap();
        assert_eq!(l.terms, ["salt and pepper"]);
        assert_eq!(l.combine(&[vec![0.1, 0.3]]), [0.1, 0.3]);
    }

    #[test]
    fn and_before_or_and_not_takes_half_away() {
        let l = Logic::parse("beach AND dog OR cat NOT snow").unwrap();
        assert_eq!(l.terms, ["beach", "dog", "cat", "snow"]);
        let cos = [vec![0.3, 0.1], vec![0.2, 0.1], vec![0.0, 0.25], vec![0.0, 0.1]];
        assert_eq!(l.combine(&cos), [0.2, 0.2]);
    }

    #[test]
    fn not_alone_is_an_error() {
        assert!(Logic::parse("NOT people").is_err());
        assert!(Logic::parse("cat OR NOT dog").is_err());
        assert!(Logic::parse("AND OR").is_err());
    }
}
