# Wikivoyage shard 0005 — what's inside

File: `index/wikivoyage/shards/0005.npy` — a table of **10,000 rows × 768 numbers** (float32, 30.7 MB).
Row *i* is the vector of chunk number 50,000 + *i* in `chunks.json` (chunks are sorted by length, so this shard holds mid-length chunks).

## The numbers

- Every row has length ≈ 1 (normalized): min 0.9963, max 1.0041. That's why a dot product = cosine similarity.
- Values range from -0.342 to 0.316; mean 0.0005, typical size ±0.036.
- No single number means anything on its own ("dimension 17 = beaches" doesn't exist) — meaning is in the whole pattern.
- The first 128 / 256 / 512 numbers matter most (Matryoshka): they hold 34% / 50% / 77% of each vector's length².

## Sample rows and their nearest neighbours in this shard

Similarity = cosine between two vectors (1.0 = same direction). For scale: a typical random pair in this shard scores 0.58.

### Row 9,500 — [Riau](https://en.wikivoyage.org/wiki/Riau)

> Riau is an oil-rich province on the east coast of Sumatra. In 2004, the eastern islands of the province near Singapore were carved off into the new province of Riau Islands (Kepulauan Riau), covered separately. Cities Pekanbaru - the capital of the province. Bagansiapiapi - a little fishermen's town. Famous for Chinese festivals like "Bakar To …

Vector starts: `[-0.012, -0.010, +0.007, -0.039, +0.020, -0.021, +0.047, +0.024, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.805 | **Riau** — . Get around By boat Pekanbaru and Selatpanjang are connected by ferries. See Muara Takus Temple, an anc … |
| 0.764 | **Rawa** — Rawa (Malay: Pulau Rawa) is a small, beautiful island located 16 km (10 mi) off the east coast of peninsular M … |
| 0.756 | **Rayong** — Rayong (ระยอง) is a city on the Eastern Gulf Coast of Thailand. Modest beaches dotted around Rayong Province a … |
| 0.752 | **Raub** — Raub, is one of the oldest towns in the State of Pahang. It began as a gold mining town in the early 19th cent … |
| 0.730 | **Rewa** — Rewa is a city in northeastern Madhya Pradesh, India, with a rich history and cultural significance. The city … |

### Row 4,728 — [Marburg](https://en.wikivoyage.org/wiki/Marburg)

> Get around As a small town it's best explored on foot. Other possibilities include the public bus services and bicycle. A touristic tour is offered by the Schlossbahn. If you need, ask the inhabitants. Most people are very friendly and will go out of their way to point out the sights. By bus Schedules and connections should be checked with the …

Vector starts: `[-0.088, -0.044, +0.043, +0.036, -0.006, -0.069, +0.005, -0.014, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.883 | **Marburg** — Oberstadt" you can also use one of the two lifts, which make it easily accessible for the disabled. Both lifts … |
| 0.807 | **Marburg** — Marburg is a small university town north of Frankfurt in the Lahn valley, Hesse. The castle ("Landgrafenschlos … |
| 0.772 | **Maribor** — . On north, Maribor is embraced with the town (wine-growing) hills, and on the south-western part of the city, … |
| 0.761 | **Magdeburg** — all trams and buses (in the front part of the vehicle). Most box offices are closed on weekends. Outside vend … |
| 0.755 | **Marbella** — modern and air conditioned. Make sure to sit on the left side of the bus if you want to see the beautiful coa … |

### Row 348 — [Hoboken](https://en.wikivoyage.org/wiki/Hoboken)

> Liberty State Park (with short inland section). In summer/fall, water is available from water fountains. If you ride skateboards or rollerblades, there is a small skatepark along the Hudson River (with spectacular Manhattan views) about 1 mile north of the PATH station. Indoor climb at Gravity Vault (Clinton St and 14th). Pier A Park: Offers grea …

Vector starts: `[+0.017, -0.021, +0.006, -0.035, -0.004, -0.011, -0.042, +0.037, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.846 | **Hoboken** — car. Finding a parking spot is so hard in Hoboken that they sell parking spaces for up to $40,000. There ar … |
| 0.806 | **Hoboken** — 3PM. Price: Sandwiches $8-11. Leo's Grandevous: Hoboken's oldest restaurant, serving the area for more than 8 … |
| 0.756 | **Kobe** — 09:30-16:30, closed Mon. Price: Free entry, free sake. Hamafukutsuru-Ginjo Brewery and Shop (浜福鶴吟醸工房): Offers … |
| 0.753 | **Kobe** — Nunobiki Falls (布引の滝, Nunobiki no taki): Four waterfalls that empty into the Ikuta River. The largest is 43m … |
| 0.747 | **Manhattan** — square (most of the grass is fenced off for security) which makes an excellent spot to rest after walking ove … |

### Row 9,486 — [Rhine](https://en.wikivoyage.org/wiki/Rhine)

> Uhldingen-Mühlhofen, prehistoric lakeside pile dwellings, reconstructed as part of an archaeological open-air museum; Baroque pilgrimage church of Birnau. Ferry to Mainau. Überlingen Bodman-Ludwigshafen: given that Tuttlingen on the Danube is just 30 km from here, this is the most convenient point to change from the Rhine to the Danube itinerary …

Vector starts: `[-0.025, -0.026, +0.023, +0.012, +0.038, -0.051, +0.052, -0.014, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.894 | **Rhine** — Starting point of the Rheintaler Höhenweg (hiking trail). Two-hourly buses from here to Balzers, Vaduz, Schaa … |
| 0.887 | **Rhine** — in today's town of Augst Upper Rhine (Basel–Karlsruhe) Basel. There is a northbound rail route to Karlsru … |
| 0.848 | **Rhine** — Koblenz to Emmerich cross the Rhine and continue on its right-hand bank (via Duisburg, Oberhausen, Wesel), by … |
| 0.833 | **Rhine** — by train and bus). If you do not plan to cruise the whole length of the Rhine, you can use the Euroairport … |
| 0.822 | **Rhine** — to Elst, then change to an ordinary train to Tiel Tiel. The rail link to Gorinchem runs quite distant from th … |

### Row 8,228 — [Peru](https://en.wikivoyage.org/wiki/Peru)

> to take a bus, just signal the driver to stop. If the bus is not completely overfilled (and sometimes when it is, too), it will stop to pick you up. During the ride, the ticket collector will ask you for the fee or, if there is not a ticket collector, you pay the driver when you get off. The latter is more common when taking longer trips where mos …

Vector starts: `[-0.018, +0.013, -0.010, +0.019, +0.038, -0.045, +0.003, +0.024, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.853 | **Peru** — in advance so that you can be relatively sure of finding a seat. If you come directly before the bus leaves, … |
| 0.846 | **Peru** — The city of Iquitos in the Amazonas region has connections by boat to Leticia in Colombia and Tabatinga in Br … |
| 0.836 | **Peru** — of foreign credit cards to buy the cheapest ticket class. You can purchase flights online. Make sure to conf … |
| 0.834 | **Peru** — the exit stamp. Inmigración and migración are found on all border crossing-points. Travelling to and from nei … |
| 0.813 | **Peru** — Peru Hop is a hop-on, hop-off bus tour. Peru Hop allow you to stop along the way from Lima to Cusco at Paraca … |

### Row 1,441 — [Japan](https://en.wikivoyage.org/wiki/Japan)

> are not available on Android devices sold outside Japan. Digital IC cards cannot be refunded to a non-Japanese bank account. Should you still have leftover balance on your card by the time you are leaving Japan or the card is about to expire, you can spend the money at restaurants, convenience stores, or at the duty-free shops at the airport. M …

Vector starts: `[+0.008, -0.049, -0.025, -0.008, +0.090, +0.047, -0.025, +0.030, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.867 | **Japan** — international contactless logo and say "touch" (タッチ tacchi) to pay with your foreign credit card. IC cards … |
| 0.845 | **Japan** — a few restaurants, small shops, and even some lodgings only accept cash, particularly in smaller towns and mo … |
| 0.825 | **Japan** — Japanese numerals only occasionally spotted in places like high-end restaurants. The main exception is that l … |
| 0.820 | **Japan** — envelope. Exceptions to the no-tipping culture are high-end ryokan (see § Sleep) and interpreters or tour gui … |
| 0.819 | **Japan** — of the funny ways of foreigners, but there's one rule where no exceptions are made: you have to wash yourself … |

### Row 7,547 — [Oslo](https://en.wikivoyage.org/wiki/Oslo)

> 37 to Tøyengata or bus 60 to Norbygata) and the city centre especially Youngstorget or Grensen. Norwegian beer is worth exploring. In addition to the major breweries in Oslo, Ringnes and Frydenlund, there are a number of local, smaller breweries that brew high-quality craft beers. Sleep There are several hotels, hostels in Oslo. Normally it is o …

Vector starts: `[-0.008, +0.010, +0.004, -0.050, +0.055, -0.033, +0.013, -0.022, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.862 | **Oslo** — right of way. Oslo has a web of tram lines downtown and as the trams are fast and heavy, you will certainly l … |
| 0.853 | **Oslo** — find several parks, eateries, industrial heritage and the main campus of the university. Inner East: The more … |
| 0.851 | **Oslo** — areas, such as Grünerløkka (tram 11-12-13 to Nybrua, Schous plass, Olaf Ryes plass or Birkelunden), Majorstue … |
| 0.847 | **Oslo** — on kitchens, kitchenware, interior designs, exclusive Norwegian furniture, light design and others. Eat T … |
| 0.847 | **Oslo** — views differ, and most people tolerate this. There are even far-left opinions in the everyday crowd, but true … |

### Row 5,115 — [Memphis](https://en.wikivoyage.org/wiki/Memphis)

> The Oasis Lounge: A great place to come relax and have a cup of coffee and enjoy a nice hookah. This is a private club due to the smoking factor so be prepared to pay a $6 membership fee and to be carded (this is an 18 and up establishment). It's got a very nice, laid back atmosphere and also has free Wi-Fi. It's on South Highland next to McDonald …

Vector starts: `[-0.023, +0.017, -0.001, -0.049, +0.036, +0.000, -0.007, -0.005, …]` (768 numbers)

| similarity | nearest chunks in this shard |
|---|---|
| 0.899 | **Memphis** — noisy. If you want to party then this is the place, but for a quieter getaway, stay a few blocks away. Addres … |
| 0.837 | **Memphis** — Round pizza, square cut, unique sauce and cheese. Also excellent hamburgers. Directions: 2 locations: Fox Mea … |
| 0.836 | **Memphis** — 10-15. Earnestine and Hazel's: Eclectic, unique atmosphere, a staff that defines cool and of course the Soul … |
| 0.835 | **Memphis** — Not categorized by price Holiday Inn - Downtown: Not flashy, but a decent place to stay. Holiday Inn was fo … |
| 0.830 | **Memphis** — night. Dry-rub ribs are the trademark, but also give the sausage plate and BBQ nachos a try. Pricey given the … |
