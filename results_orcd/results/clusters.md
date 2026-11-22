# Subject clusters: sanity report

- featurizer: **idf_bag_rp256** (fallback: MiniLM projected 43.2 min > 30 min to assign all blocks)
- k-means mean cosine per restart: [0.2713, 0.2707, 0.2709]
- balance converged: True after 9 iterations; bias = [-0.021, 0.021, -0.024, 0.058, 0.033, -0.029, -0.001, -0.036]
- I(cluster; block majority source): train **0.435 bits**, val 0.417 bits (max possible log2(7) = 2.807)
- time: 0.8 min (36,007 blocks/s on the sample)

## Cluster sizes

| cluster | train blocks | train % | val blocks | val % |
|---:|---:|---:|---:|---:|
| 0 | 2,649,713 | 14.1 | 21,841 | 14.8 |
| 1 | 1,987,890 | 10.6 | 14,784 | 10.0 |
| 2 | 2,548,399 | 13.6 | 20,169 | 13.6 |
| 3 | 1,801,631 | 9.6 | 14,200 | 9.6 |
| 4 | 2,147,371 | 11.5 | 16,505 | 11.1 |
| 5 | 2,631,376 | 14.0 | 21,207 | 14.3 |
| 6 | 2,373,587 | 12.7 | 18,848 | 12.7 |
| 7 | 2,610,033 | 13.9 | 20,565 | 13.9 |

## Cluster vs. block majority source (train, row %)

| cluster | CC | C4 | Github | Book | ArXiv | Wiki | StackEx |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 57.9 | 32.6 | 0.1 | 6.2 | 0.2 | 0.7 | 2.3 |
| 1 | 12.0 | 5.5 | 51.4 | 2.3 | 3.9 | 4.4 | 20.5 |
| 2 | 63.7 | 29.8 | 0.1 | 3.2 | 0.6 | 2.1 | 0.5 |
| 3 | 43.9 | 24.1 | 1.1 | 2.1 | 0.6 | 26.4 | 1.7 |
| 4 | 35.7 | 15.4 | 2.4 | 2.1 | 36.4 | 4.2 | 3.8 |
| 5 | 49.3 | 45.6 | 0.2 | 2.7 | 0.3 | 0.4 | 1.5 |
| 6 | 57.9 | 16.6 | 10.9 | 4.1 | 1.0 | 5.0 | 4.6 |
| 7 | 66.3 | 27.2 | 0.1 | 3.7 | 0.8 | 1.2 | 0.6 |

## Sample blocks

**cluster 0**

- `Everything is in the details: Irena turning on a light in a darkened room as she murmurs, "I like the dark, it's friendly"; her encounter with the unknown catlike woman who stares at her and greets her as a sister (an exchange of glances th…`
- `which she serves as chairperson, to fight for employees' benefits, earning her much admiration from the staff. She always recalls what Dr. Ho has continually stressed: "When one drinks water, one must not forget where it comes from", which …`
- `such an overwhelming reading experience that I had to literally pause for a moment after I finished reading the issue, to sit back and take it all in. ⏎ The density of the story and art is staggering. ⏎ Fraction's plot is straightforward ye…`

**cluster 1**

- `SUFFIX; ⏎ public static final String ESC_FG_YELLOW = ESC_PREFIX ⏎`
- `+ SGR_FG_WHITE ⏎`
- `+ ESC_SUFFIX; ⏎ public static final String ESC_BGB_RED = ESC_PREFIX ⏎`

**cluster 2**

- `, and cinematographer, Nicholas Musuraca. ⏎ The huge success of Cat People enabled Lewton to keep making, under the guise of B-movie thrillers, movies that are often more like symbolist poems or obscure fetishistic rituals. They are not so …`
- `, when they are considering how well children's rights are protected in the UK. ⏎ In 2016, the UN Committee on the Rights of the Child issued recommendations (known as Concluding Observations) telling the Government how it should improve th…`
- `au Facebook Page screenshot captured Sept. 9, 2019 ⏎ Photos and an announcement regarding Mielke's political plans briefly appeared on the city's Facebook page, but the post was taken down after critics spoke up. Mielke, in September, said …`

**cluster 3**

- `не. Режирала га је Мирјана Самарџић, а сценарио је писала`
- `⏎ Marcella Vitali, Bice Montuschi Simboli, Berti Antonio, pittore (20.9.1830 Faenza, † 14.07.1912 Faenza) in Manfrediana. Bollettino della biblioteca comunale`
- `works well regardless of the boundary specification. ⏎ The plot above shows a comparison of surge volumes calculated according to equations (2) and (7), labeled "OLGA Method" and "Proposed Method" respectively. We can see that filtering out…`

**cluster 4**

- `go to government hospitals because services have been upgraded. So far 1,008 people have received the institutional delivery incentive. ⏎ Nawalparasi district borders India, and earlier this year it became a hotspot for Covid-19, as Nepalis…`
- `analysis of the sampled accounts on Twitter and users' perceptions regarding information services. Findings: The results of the study revealed that crowdsourcing was not used in the development of information services per se, but it has bee…`
- `radiation delivers the optimal dose of radiation to the tumor target volume and minimal or no radiation to normal tissues and organs. As with the IMRT radiation, there would still be an exit beam hitting healthy parts of Justin's brain. ⏎ F…`

**cluster 5**

- `have found to get Amazon Gift Cards. You earn sSagbucks by completing searches through the Swagbucks search engine (you can download a toolbar if you want, or set your search engine to it) — not for every search, but a decent amount. Admitt…`
- `More often the not, countries that exploited their natural resources have seen their economies grow, new modern infrastructure built and poverty alleviated. We want to chart the path of prudent management of our natural resources to bring a…`
- `In Brierwood Estates On A Large .8 Acre Lot With No Hoa! Separate Living Space With Private Entrance. Built As Builder's Personal Home. The Beautiful Wrap-around Front Porch And Circular Driveway Are Just The Beginning. Interior Features In…`

**cluster 6**

- `2018. Please email SurfaceWater@SnoCo.org for more information.<\|endoftext\|>MainAll NewsCoup ⏎ Turkey extends state of emergency ⏎ Arutz Sheva Staff , 1/4/2017, 5:35 AM (Thinkstock) ⏎ Turkey arrests brother of Gulen ⏎ Ben Ariel, Canada ,1…`
- `not truth and therefore such separation will go, even though it will take a long time' ⏎ PBThesisPt1.pdf (application/pdf Object) ⏎ dont know if you are familiar with paul brunton, but he did engage with theosophy for a while, and this thes…`
- `who worked with us to test this community trust idea to the panel who guided selection of our steward organizations. This is a milestone moment for the Bush Foundation, and we are thankful to have had the wisdom and insights of many Black a…`

**cluster 7**

- `insurance policies on everyone in his family and they ended up dead; the subsequent trial of the man who murdered Reverend Miller at the funeral home mourning his latest victim; Harper Lee's involvement as a last gasp effort to produce a bo…`
- `astic online slot version from Microgaming, developed through its licensing agreement with DC Entertainment. It's the software developer's second foray into the Batman franchise, following the hugely successful The Dark Knight™, unveiled in…`
- `For the most part, the SOA initiatives I've seen have only led to extra complexity and reduced productivity in development without ever coming close to the orchestration nirvana envisioned by the architects. ⏎ As I see it, the positive goal…`

