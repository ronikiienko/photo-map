//! Words with NOT (in capitals, so that "not" in a phrase stays a plain word): "street at night NOT cars NOT people". Each part
//! is searched on its own, and every NOT takes away half of its cosine (the strongest NOT counts). A NOT weighed 0.5 kept
//! overlapping words working on Flickr30k ("people NOT children"). AND and OR were tried and dropped: an AND followed the weaker
//! word and an OR let one word crowd out the others, as the words' cosines don't share a scale; a plain phrase already means
//! "and" ("woman on a bike in winter"). Several photos are averaged.
//!
//! Filters pick photos by what is written about them, as plain text (any case), before the rest ranks them by meaning: name:
//! by the file and folder names (`beach name:greece`, `NOT name:screenshot`), and camera:, date: (year:) and place: by what
//! the photo's EXIF said (data::Facts): `dog camera:pixel date:2019-07 place:"new york"`. Commas mean any of them
//! (`year:2016,2018`, `place:kyiv,lviv`), and two dots a range of dates (`year:2016..2018`, `date:2019-06..`).

/// Searches to try (the Examples menu beside the box), each with what it shows the model can do.
pub const EXAMPLES: [(&str, &str); 12] = [
    ("foggy forest", "weather and places together"),
    ("street at night NOT cars", "NOT: photos with cars go lower"),
    ("red car", "a colour that belongs to a thing"),
    ("dog in the snow", "a thing in a setting"),
    ("people laughing", "what people do and how they feel"),
    ("food on a plate, close up", "how near the camera is"),
    ("view from above", "where the photo was taken from"),
    ("black and white photo", "style, not content"),
    ("reflection in water", "light and what it does"),
    ("cozy room with warm light", "mood"),
    ("sign with text", "writing in the picture"),
    ("mountains NOT snow", "NOT takes away what usually comes along"),
];

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

/// What a filter looks at.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Key {
    /// The file and folder names (below the folder all the map's photos share): name:
    Name,
    /// data::Facts: camera:pixel, date:2019-07 (or year:2019), place:lviv.
    Camera,
    Date,
    Place,
}

/// One filter: `camera:pixel`, `name:greece`, `NOT place:kyiv`. Typed in the box, it leaves it for a chip of its own.
#[derive(Debug, PartialEq, Clone)]
pub struct Filter {
    pub key: Key,
    /// Lowercase; any of them picks a photo (`year:2016,2018`). A date can be a range, both ends in: "2016..2018", "2019-06..".
    pub values: Vec<String>,
    /// After NOT: the photos it picks are left out.
    pub not: bool,
}

impl Key {
    /// Its name on a chip, and what to write in one.
    pub fn label(self) -> &'static str {
        match self {
            Key::Name => "Name",
            Key::Camera => "Camera",
            Key::Date => "Date",
            Key::Place => "Place",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Key::Name => "greece, screenshot",
            Key::Camera => "pixel, redmi",
            Key::Date => "2019-07, 2016..2018",
            Key::Place => "lviv, kyiv",
        }
    }
}

impl Filter {
    /// A filter from what was written in a chip: values split by commas (none: no filter).
    pub fn new(key: Key, value: &str, not: bool) -> Option<Filter> {
        let values: Vec<String> = value.split(',').map(|v| v.trim().to_lowercase()).filter(|v| !v.is_empty()).collect();
        (!values.is_empty()).then_some(Filter { key, values, not })
    }

    /// Its values as a chip shows them: "pixel, redmi".
    pub fn value_text(&self) -> String {
        self.values.join(", ")
    }

    /// As it is typed: "NOT place:kyiv,lviv", a value with spaces in quotes.
    pub fn text(&self) -> String {
        let key = match self.key {
            Key::Name => "name",
            Key::Camera => "camera",
            Key::Date => "date",
            Key::Place => "place",
        };
        let value = self.values.join(",");
        let value = if value.contains(' ') { format!("\"{value}\"") } else { value };
        format!("{}{key}:{value}", if self.not { "NOT " } else { "" })
    }

    /// Whether a photo with this name and these facts passes (a NOT passes what it doesn't pick). A date picks by its start.
    pub fn passes(&self, name: &str, facts: Option<&crate::data::Facts>) -> bool {
        let picks = |value: &str| match (self.key, facts) {
            (Key::Name, _) => has_name(name, value),
            (Key::Camera, Some(f)) => has_name(&f.camera, value),
            (Key::Date, Some(f)) => in_dates(&f.taken, value),
            (Key::Place, Some(f)) => has_name(&f.place, value),
            (_, None) => false,
        };
        self.values.iter().any(|v| picks(v)) != self.not
    }
}

/// Whether a photo taken at `taken` ("2019-07-04 14:00") is in `dates`: one that starts it ("2019-07"), or a range of them
/// ("2016..2018": from the start of the first to the end of the last; either end may be left out).
fn in_dates(taken: &str, dates: &str) -> bool {
    if taken.is_empty() {
        return false;
    }
    match dates.split_once("..") {
        Some((from, to)) => taken >= from.trim() && (to.trim().is_empty() || taken.get(..to.trim().len()).unwrap_or(taken) <= to.trim()),
        None => taken.starts_with(dates),
    }
}

/// The key a word starts with ("camera:", any case), and the rest of the word.
fn key_of(word: &str) -> Option<(Key, &str)> {
    let (key, value) = word.split_once(':')?;
    let key = match key.to_lowercase().as_str() {
        "name" => Key::Name,
        "camera" => Key::Camera,
        "date" | "year" => Key::Date,
        "place" => Key::Place,
        _ => return None,
    };
    Some((key, value))
}

/// The filters taken out of the box's text, and what is left to search by meaning. A filter is key:value, the value in quotes if it has spaces (`camera:"pixel 7"`); a NOT right before one belongs to it. An unclosed quote
/// runs to the end; an unknown key ("http:") is plain words.
pub fn filters(text: &str) -> (String, Vec<Filter>) {
    let mut rest: Vec<&str> = Vec::new();
    let mut found = Vec::new();
    let mut add = |rest: &mut Vec<&str>, key: Key, value: &str| {
        // commas: any of these
        let not = rest.last() == Some(&"NOT");
        let Some(filter) = Filter::new(key, value, not) else { return };
        if not {
            rest.pop();
        }
        found.push(filter);
    };
    let mut at = text.trim_start();
    while !at.is_empty() {
        // a quoted value: to the closing quote, or the end
        let quoted = |from: &str| -> (usize, usize) {
            let body = &from[1..];
            let end = body.find('"').unwrap_or(body.len());
            (1 + end, (1 + end + 1).min(from.len()))
        };
        let word_end = at.find(char::is_whitespace).unwrap_or(at.len());
        let word = &at[..word_end];
        match key_of(word) {
            Some((key, value)) if value.starts_with('"') => {
                let start = word.len() - value.len();
                let (end, next) = quoted(&at[start..]);
                add(&mut rest, key, &at[start + 1..start + end]);
                at = at[start + next..].trim_start();
            }
            Some((key, value)) if !value.is_empty() => {
                add(&mut rest, key, value);
                at = at[word_end..].trim_start();
            }
            _ => {
                rest.push(word);
                at = at[word_end..].trim_start();
            }
        }
    }
    // a NOT left at the end (its words were all filters) has nothing after it
    while rest.last() == Some(&"NOT") {
        rest.pop();
    }
    (rest.join(" "), found)
}

/// While typing: the filters finished in the box's text (a space after them, no quote left open), taken out of it, and the
/// words left (with a space after them, to go on typing). None while there are none.
pub fn finished(text: &str) -> Option<(String, Vec<Filter>)> {
    if !text.ends_with(char::is_whitespace) || text.matches('"').count() % 2 == 1 {
        return None;
    }
    let (rest, found) = filters(text);
    if found.is_empty() {
        return None;
    }
    Some((if rest.is_empty() { rest } else { rest + " " }, found))
}

/// Whether `name` (already lowercase) is in `hay`, in any case.
pub fn has_name(hay: &str, name: &str) -> bool {
    if hay.is_ascii() && name.is_ascii() {
        hay.as_bytes().windows(name.len().max(1)).any(|w| w.eq_ignore_ascii_case(name.as_bytes()))
    } else {
        hay.to_lowercase().contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_come_out_of_the_text() {
        let f = |key, value: &str, not| Filter { key, values: value.split(',').map(String::from).collect(), not };
        assert_eq!(filters("beach name:Greece sunset"), ("beach sunset".into(), vec![f(Key::Name, "greece", false)]));
        assert_eq!(filters("street NOT name:screenshot NOT cars"), ("street NOT cars".into(), vec![f(Key::Name, "screenshot", true)]));
        assert_eq!(filters(r#"dog name:"trip to lviv"#), ("dog".into(), vec![f(Key::Name, "trip to lviv", false)]));
        assert_eq!(filters(r#"a "quoted" word"#), (r#"a "quoted" word"#.into(), vec![]), "plain quotes are just words");
        assert_eq!(filters("no filters here"), ("no filters here".into(), vec![]));
        assert_eq!(
            filters(r#"dog Camera:Pixel year:2019 NOT place:"new york" park"#),
            ("dog park".into(), vec![f(Key::Camera, "pixel", false), f(Key::Date, "2019", false), f(Key::Place, "new york", true)])
        );
        assert_eq!(filters("see http://x.org and camera:"), ("see http://x.org and camera:".into(), vec![]));
        assert_eq!(filters("NOT camera:canon"), ("".into(), vec![f(Key::Camera, "canon", true)]));
        assert_eq!(filters("year:2016, 2017 x"), ("2017 x".into(), vec![f(Key::Date, "2016", false)]), "a space ends the value");
        assert_eq!(filters(r#"place:"kyiv, lviv" year:2016,,2018"#), ("".into(), vec![f(Key::Place, "kyiv,lviv", false), f(Key::Date, "2016,2018", false)]));
    }

    #[test]
    fn finished_filters_leave_the_box() {
        assert_eq!(finished("dog camera:pi"), None, "still typing it");
        assert_eq!(finished(r#"place:"new "#), None, "the quote is open");
        assert_eq!(finished("dog NOT "), None);
        let (rest, found) = finished(r#"dog NOT place:"New York" "#).unwrap();
        assert_eq!(rest, "dog ");
        assert_eq!(found[0].text(), r#"NOT place:"new york""#);
        assert_eq!(finished("year:2016..2018,2020 ").unwrap(), ("".into(), vec![filters("year:2016..2018,2020").1[0].clone()]));
        assert_eq!(filters("year:2016..2018,2020").1[0].text(), "date:2016..2018,2020");
    }

    #[test]
    fn dates_any_of_and_ranges() {
        let at = |taken: &str| crate::data::Facts { taken: taken.into(), ..Default::default() };
        let passes = |text: &str, taken: &str| filters(text).1[0].passes("", Some(&at(taken)));
        assert!(passes("year:2016,2018", "2018-01-01 10:00"));
        assert!(!passes("year:2016,2018", "2017-01-01 10:00"));
        assert!(passes("year:2016..2018", "2016-01-01"));
        assert!(passes("year:2016..2018", "2018-12-31 23:59"));
        assert!(!passes("year:2016..2018", "2019-01-01"));
        assert!(!passes("year:2016..2018", "2015-12-31 23:59"));
        assert!(passes("date:2019-06..2019-08", "2019-08-31 12:00"));
        assert!(!passes("date:2019-06..2019-08", "2019-09-01"));
        assert!(passes("date:2019-06..", "2024-01-01"));
        assert!(passes("date:..2010", "2003-05-05"));
        assert!(!passes("date:..2010", ""));
        assert!(passes("NOT year:2016,2017", "2018-01-01"));
    }

    #[test]
    fn filters_pass_photos() {
        let facts = crate::data::Facts { camera: "Google Pixel 7".into(), taken: "2019-07-04 14:00".into(), place: "Lviv, Lviv Oblast, Ukraine".into() };
        let (_, fs) = filters(r#"camera:pixel date:2019-07 place:lviv "trip""#);
        assert!(fs.iter().all(|f| f.passes("Trip 2019/a.jpg", Some(&facts))));
        let (_, fs) = filters("date:2019-08");
        assert!(!fs[0].passes("a.jpg", Some(&facts)));
        let (_, fs) = filters("NOT camera:canon camera:pixel");
        assert!(fs[0].passes("a.jpg", None), "nothing known: a NOT lets it through");
        assert!(!fs[1].passes("a.jpg", None), "nothing known: not picked");
    }

    #[test]
    fn names_in_any_case() {
        assert!(has_name("Takeout/Trip to Lviv 2021/IMG_1.jpg", "trip to lviv"));
        assert!(has_name("Фото/Київ/1.jpg", "київ"));
        assert!(!has_name("a/b.jpg", "c"));
    }

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
