# Tier 2 Data Audit

## Summary

- Total ticket count: 140
- Missing or blank exactDate: 60
- Placeholder / ingestion-style extendedNotes cleared in Tier 2: 137
- Placeholder / ingestion-style extendedNotes remaining after verification: 0
- Weak / generic copy flagged: 87
- Strong / preserved entries: 3

## Automated Cleanup Applied

- Cleared 137 placeholder `extendedNotes` values that were deterministic ingestion/OCR/workflow text.
- Normalized factual archival copy and aligned `shareDescription` for `blur-liberty-lunch-1997`, `wilco-stubbs-bbq-2004`, and `yaz-terminal-5-2008`.
- Restored stronger existing backup wording for `the-wallflowers-frank-erwin-center-1997`.
- Did not change any schema fields outside `copy`, `extendedNotes`, and matching `shareDescription` where needed.

## Tier 2 Verification Corrections Applied

- Fixed mojibake in `tickets.js` for `beyonc-barclays-center-2013` (`Beyoncé`), `sigur-rs-madison-square-garden-2013` (`Sigur Rós`), `sinad-oconnor-hogg-memorial-auditorium-2007` (`Sinéad O'Connor`), `various-artists-villa-rufolo-2013` (`€35.66` price), and `the-fall-indigo2-at-the-o2-2011` (`£37.50` price).
- Re-checked the cleared `extendedNotes` values against the backup and restored none, because the reviewed candidates remained clearly placeholder/workflow text rather than personal/editorial notes.
- Re-checked the Tier 2 copy/shareDescription edits and kept them as-is for `blur-liberty-lunch-1997`, `wilco-stubbs-bbq-2004`, and `yaz-terminal-5-2008`; preserved the stronger restored backup wording for `the-wallflowers-frank-erwin-center-1997`.
- Intentionally left mechanically written but non-placeholder records unchanged when the backup did not contain clearly stronger trustworthy wording.

## A. Missing Or Blank `exactDate` Entries

- `aimee-mann-hogg-memorial-auditorium-2005` | Aimee Mann | 2005 | Hogg Memorial Auditorium
- `animal-collective-antones-2007` | Animal Collective | 2007 | Antone's
- `animal-collective-prospect-park-bandshell-2011` | Animal Collective | 2011 | Prospect Park Bandshell
- `arcade-fire-the-o2-arena-2010` | Arcade Fire | 2010 | The O2 Arena
- `beach-house-bowery-ballroom-2008` | Beach House | 2008 | Bowery Ballroom
- `beyonc-barclays-center-2013` | Beyoncé | 2013 | Barclays Center
- `the-brian-jonestown-massacre-the-parish-2005` | The Brian Jonestown Massacre | 2005 | The Parish
- `britney-spears-madison-square-garden-2009` | Britney Spears | 2009 | Madison Square Garden
- `cass-mccombs-the-public-theater-2009` | Cass Mccombs | 2009 | The Public Theater
- `courtney-barnett-and-kurt-vile-beacon-theatre-2017` | Courtney Barnett and Kurt Vile | 2017 | Beacon Theatre
- `courtney-barnett-bowery-ballroom-2015` | Courtney Barnett | 2015 | Bowery Ballroom
- `courtney-barnett-music-hall-of-williamsburg-2014` | Courtney Barnett | 2014 | Music Hall of Williamsburg
- `courtney-barnett-terminal-5-2015` | Courtney Barnett | 2015 | Terminal 5
- `eels-terminal-5-2010` | Eels | 2010 | Terminal 5
- `fable-with-porcelain-raft-knockdown-center-2015` | Fable with Porcelain Raft | 2015 | Knockdown Center
- `fat-boy-slim-austin-music-hall-1999` | Fat Boy Slim | 1999 | Austin Music Hall
- `first-aid-kit-beacon-theatre-2018` | First Aid Kit | 2018 | Beacon Theatre
- `george-strait-prudential-center-2014` | George Strait | 2014 | Prudential Center
- `government-mule-beacon-theatre-2017` | Government Mule | 2017 | Beacon Theatre
- `incredible-caffeine-world-rave-arks-ranch-2000` | Incredible Caffeine World Rave | 2000 | Ark's Ranch
- `jason-isbell-beacon-theatre-2015` | Jason Isbell | 2015 | Beacon Theatre
- `john-prine-kings-theatre-2016` | John Prine | 2016 | Kings Theatre
- `justin-timberlake-barclays-center-2013` | Justin Timberlake | 2013 | Barclays Center
- `justin-timberlake-madison-square-garden-2018` | Justin Timberlake | 2018 | Madison Square Garden
- `leon-bridges-beacon-theatre-2016` | Leon Bridges | 2016 | Beacon Theatre
- `lucinda-williams-la-zona-rosa-2004` | Lucinda Williams | 2004 | La Zona Rosa
- `madonna-madison-square-garden-2004` | Madonna | 2004 | Madison Square Garden
- `modest-mouse-stubbs-bbq-2003` | Modest Mouse | 2003 | Stubb's BBQ
- `modest-mouse-stubbs-bbq-2004` | Modest Mouse | 2004 | Stubb's BBQ
- `modest-mouse-webster-hall-2015` | Modest Mouse | 2015 | Webster Hall
- `neil-michael-hagerty-and-the-howling-hex-knitting-factory-2008` | Neil Michael Hagerty and the Howling Hex | 2008 | Knitting Factory
- `peaches-emos-2004` | Peaches | 2004 | Emo's
- `pixies-stubbs-bbq-2004` | Pixies | 2004 | Stubb's BBQ
- `pj-harvey-irving-plaza-2009` | PJ Harvey | 2009 | Irving Plaza
- `pj-harvey-the-fillmore-philadelphia-2018` | PJ Harvey | 2018 | The Fillmore Philadelphia
- `pj-harvey-terminal-5-2016` | PJ Harvey | 2016 | Terminal 5
- `porcelain-raft-mercury-lounge-2012` | Porcelain Raft | 2012 | Mercury Lounge
- `sharon-jones-beacon-theatre-2014` | Sharon Jones | 2014 | Beacon Theatre
- `sigur-rs-madison-square-garden-2013` | Sigur Rós | 2013 | Madison Square Garden
- `sinad-oconnor-hogg-memorial-auditorium-2007` | Sinéad O'Connor | 2007 | Hogg Memorial Auditorium
- `sonic-youth-hammersmith-apollo-2010` | Sonic Youth | 2010 | Hammersmith Apollo
- `spoon-white-eagle-hall-2024` | Spoon | 2024 | White Eagle Hall
- `sturgill-simpson-kings-theatre-2016` | Sturgill Simpson | 2016 | Kings Theatre
- `sturgill-simpson-music-hall-of-williamsburg-2015` | Sturgill Simpson | 2015 | Music Hall of Williamsburg
- `the-black-angels-bowery-ballroom-2010` | The Black Angels | 2010 | Bowery Ballroom
- `the-breeders-webster-hall-2013` | The Breeders | 2013 | Webster Hall
- `the-chicks-frank-erwin-center-2000` | The Chicks | 2000 | Frank Erwin Center
- `the-faint-la-zona-rosa-2004` | The Faint | 2004 | La Zona Rosa
- `the-fall-indigo2-at-the-o2-2011` | The Fall | 2011 | indigO2 at The O2
- `the-rapture-and-black-rebel-motorcycle-club-stubbs-bbq-2004` | The Rapture and Black Rebel Motorcycle Club | 2004 | Stubb's BBQ
- `thee-oh-sees-irving-plaza-2013` | Thee Oh Sees | 2013 | Irving Plaza
- `tim-mcgraw-and-faith-hill-barclays-center-2017` | Tim McGraw & Faith Hill | 2017 | Barclays Center
- `tobias-jesso-jr-music-hall-of-williamsburg-2015` | Tobias Jesso jr | 2015 | Music Hall of Williamsburg
- `tony-bennett-90th-birthday-radio-city-music-hall-2016` | Tony Bennett 90th Birthday | 2016 | Radio City Music Hall
- `wilco-stubbs-bbq-2004` | Wilco | 2004 | Stubb's BBQ
- `wilco-united-palace-2022` | Wilco | 2022 | United Palace
- `willie-nelson-and-family-prospect-park-bandshell-2015` | Willie Nelson & Family | 2015 | Prospect Park Bandshell
- `women-mercury-lounge-2010` | Women | 2010 | Mercury Lounge
- `yaz-terminal-5-2008` | Yaz | 2008 | Terminal 5
- `yeah-yeah-yeahs-stubbs-bbq-2006` | Yeah Yeah Yeahs | 2006 | Stubb's BBQ

## B. Placeholder / Ingestion-Style `extendedNotes`

Matched patterns included: `Image filename suggests supplementary material`, `Exact date not cleanly legible from OCR`, `This draft keeps the tone concise`, `Update this with personal context`, `ticket anchors`, `detail view`, `OCR`, and similar process language.

All entries listed below were cleaned to `extendedNotes: ""` in `tickets.js`:

- `gris-gris-emos-2006`
- `le-tigre-emos-2005`
- `adele-madison-square-garden-2016`
- `adele-radio-city-music-hall-2015`
- `aimee-mann-hogg-memorial-auditorium-2005`
- `airport-2-rave-arks-ranch-2000`
- `animal-collective-antones-2007`
- `animal-collective-prospect-park-bandshell-2011`
- `anohni-antony-and-the-johnsons-alice-tully-hall-2010`
- `arcade-fire-air-canada-centre-2017`
- `arcade-fire-the-o2-arena-2010`
- `arcade-fire-stubbs-bbq-2005`
- `ariel-pink-webster-hall-2010`
- `band-of-horses-ed-sullivan-theater-2012`
- `beach-house-bowery-ballroom-2008`
- `belle-and-sebastian-forest-hills-stadium-2018`
- `ben-harper-the-backyard-2003`
- `beyonc-barclays-center-2013`
- `bikini-kill-terminal-5-2019`
- `blonde-redhead-emos-2004`
- `blur-liberty-lunch-1997`
- `blur-madison-square-garden-2015`
- `the-brian-jonestown-massacre-terminal-5-2008`
- `the-brian-jonestown-massacre-the-parish-2005`
- `britney-spears-madison-square-garden-2009`
- `caleb-hawley-mercury-lounge-2014`
- `caribou-bowery-ballroom-2008`
- `cass-mccombs-the-public-theater-2009`
- `cat-power-white-eagle-hall-2022`
- `clientele-emos-2007`
- `clinic-gramercy-theatre-2007`
- `courtney-barnett-and-kurt-vile-beacon-theatre-2017`
- `courtney-barnett-bowery-ballroom-2015`
- `courtney-barnett-lucindas-2025`
- `courtney-barnett-music-hall-of-williamsburg-2014`
- `courtney-barnett-terminal-5-2015`
- `crocodiles-music-hall-of-williamsburg-2011`
- `dead-and-company-madison-square-garden-2015`
- `deerhunter-webster-hall-2019`
- `eels-terminal-5-2010`
- `fable-with-porcelain-raft-knockdown-center-2015`
- `fat-boy-slim-austin-music-hall-1999`
- `first-aid-kit-beacon-theatre-2018`
- `fleet-foxes-williamsburg-waterfront-2011`
- `fleetwood-mac-madison-square-garden-2015`
- `gayngs-music-hall-of-williamsburg-2010`
- `george-strait-prudential-center-2014`
- `government-mule-beacon-theatre-2017`
- `imogen-heap-stubbs-bbq-2006`
- `incredible-caffeine-world-rave-arks-ranch-2000`
- `jason-isbell-beacon-theatre-2015`
- `john-prine-kings-theatre-2016`
- `justin-timberlake-barclays-center-2013`
- `justin-timberlake-madison-square-garden-2018`
- `kanye-west-madison-square-garden-2008`
- `kings-of-leon-madison-square-garden-2010`
- `leon-bridges-beacon-theatre-2016`
- `les-savy-fav-emos-2008`
- `little-richard-paramount-theatre-2008`
- `lucinda-williams-la-zona-rosa-2004`
- `lucinda-williams-stubbs-bbq-2002`
- `m83-razzmatazz-2012`
- `madonna-madison-square-garden-2004`
- `madonna-madison-square-garden-2012`
- `modest-mouse-austin-music-hall-2005`
- `modest-mouse-stubbs-bbq-2003`
- `modest-mouse-stubbs-bbq-2004`
- `modest-mouse-webster-hall-2015`
- `my-morning-jacket-ed-sullivan-theater-2010`
- `my-morning-jacket-terminal-5-2010`
- `my-morning-jacket-forest-hills-stadium-2021`
- `my-morning-jacket-stubbs-bbq-2006`
- `neil-michael-hagerty-and-the-howling-hex-knitting-factory-2008`
- `patti-smith-la-zona-rosa-2004`
- `peaches-emos-2004`
- `pixies-brooklyn-steel-2017`
- `pixies-brooklyn-steel-2018`
- `pixies-hammerstein-ballroom-2009`
- `pixies-reliant-arena-2004`
- `pixies-stubbs-bbq-2004`
- `pj-harvey-and-john-parish-beacon-theatre-2009`
- `pj-harvey-brooklyn-steel-2017`
- `pj-harvey-irving-plaza-2009`
- `pj-harvey-the-fillmore-philadelphia-2018`
- `pj-harvey-stubbs-bbq-2017`
- `pj-harvey-terminal-5-2011`
- `pj-harvey-terminal-5-2016`
- `porcelain-raft-mercury-lounge-2012`
- `robin-thicke-highline-ballroom-2013`
- `sharon-jones-beacon-theatre-2014`
- `she-and-him-terminal-5-2008`
- `sigur-rs-madison-square-garden-2013`
- `sinad-oconnor-hogg-memorial-auditorium-2007`
- `sonic-youth-hammersmith-apollo-2010`
- `sonic-youth-stubbs-bbq-2006`
- `sparklehorse-emos-2006`
- `spoon-white-eagle-hall-2024`
- `sturgill-simpson-kings-theatre-2016`
- `sturgill-simpson-music-hall-of-williamsburg-2015`
- `supergrass-the-parish-2006`
- `tame-impala-beacon-theatre-2014`
- `the-bad-plus-with-bill-frisell-jazz-at-lincoln-center-2013`
- `the-black-angels-bowery-ballroom-2010`
- `the-black-lips-bowery-ballroom-2008`
- `the-breeders-webster-hall-2008`
- `the-breeders-webster-hall-2013`
- `the-breeders-webster-hall-2013-12-20`
- `the-breeders-webster-hall-2013-12-19`
- `the-chicks-frank-erwin-center-2000`
- `the-clean-the-bell-house-2010`
- `the-faint-emos-2002`
- `the-faint-la-zona-rosa-2004`
- `the-fall-indigo2-at-the-o2-2011`
- `the-fall-stubbs-bbq-2006`
- `the-knife-terminal-5-2014`
- `the-pretenders-austin-music-hall-2006`
- `the-rapture-and-black-rebel-motorcycle-club-stubbs-bbq-2004`
- `the-vaselines-southpaw-2008`
- `the-vaselines-webster-hall-2010`
- `the-white-stripes-stubbs-bbq-2003`
- `thee-oh-sees-bowery-ballroom-2016`
- `thee-oh-sees-irving-plaza-2013`
- `tim-mcgraw-and-faith-hill-barclays-center-2017`
- `tobias-jesso-jr-music-hall-of-williamsburg-2015`
- `tony-bennett-90th-birthday-radio-city-music-hall-2016`
- `true-colors-radio-city-music-hall-2008`
- `various-artists-villa-rufolo-2013`
- `wilco-beacon-theatre-2017`
- `wilco-stubbs-bbq-2004`
- `wilco-united-palace-2022`
- `willie-nelson-and-family-prospect-park-bandshell-2015`
- `women-mercury-lounge-2010`
- `yaz-terminal-5-2008`
- `yeah-yeah-yeahs-emos-2003`
- `yeah-yeah-yeahs-stubbs-bbq-2006`
- `yeah-yeah-yeahs-webster-hall-2013`
- `yuck-music-hall-of-williamsburg-2011`

## C. Weak / Generic Copy Flagged

These entries still use factual archival copy that is safe to keep live, but they remain mechanically written and would benefit from later human-authored revision.

- `gris-gris-emos-2006` | Gris Gris at Emo's in Austin in 2006
- `le-tigre-emos-2005` | Le Tigre at Emo's in Austin in 2005
- `aimee-mann-hogg-memorial-auditorium-2005` | Aimee Mann at Hogg Memorial Auditorium in Austin in 2005
- `airport-2-rave-arks-ranch-2000` | Airport 2 Rave at Ark's Ranch in Driftwood in 2000
- `animal-collective-antones-2007` | Animal Collective at Antone's in Austin in 2007
- `animal-collective-prospect-park-bandshell-2011` | Animal Collective at Prospect Park Bandshell in Brooklyn in 2011
- `anohni-antony-and-the-johnsons-alice-tully-hall-2010` | ANOHNI / Antony and the Johnsons at Alice Tully Hall in New York in 2010
- `band-of-horses-ed-sullivan-theater-2012` | Band of Horses at Ed Sullivan Theater in New York in 2012
- `ben-harper-the-backyard-2003` | Ben Harper at The Backyard in Austin in 2003
- `bikini-kill-terminal-5-2019` | Bikini Kill at Terminal 5 in New York in 2019
- `blonde-redhead-emos-2004` | Blonde Redhead at Emo's in Austin in 2004
- `blur-liberty-lunch-1997` | Blur at Liberty Lunch in Austin in 1997.
- `the-brian-jonestown-massacre-terminal-5-2008` | The Brian Jonestown Massacre at Terminal 5 in New York in 2008
- `the-brian-jonestown-massacre-the-parish-2005` | The Brian Jonestown Massacre at The Parish in Austin in 2005
- `caleb-hawley-mercury-lounge-2014` | Caleb Hawley at Mercury Lounge in New York in 2014
- `cass-mccombs-the-public-theater-2009` | Cass Mccombs at The Public Theater in New York in 2009
- `clientele-emos-2007` | Clientele at Emo's in Austin in 2007
- `clinic-gramercy-theatre-2007` | Clinic at Gramercy Theatre in New York in 2007
- `courtney-barnett-and-kurt-vile-beacon-theatre-2017` | Courtney Barnett and Kurt Vile at Beacon Theatre in New York in 2017
- `courtney-barnett-music-hall-of-williamsburg-2014` | Courtney Barnett at Music Hall of Williamsburg in Brooklyn in 2014
- `crocodiles-music-hall-of-williamsburg-2011` | Crocodiles at Music Hall of Williamsburg in Brooklyn in 2011
- `eels-terminal-5-2010` | Eels at Terminal 5 in New York in 2010
- `fable-with-porcelain-raft-knockdown-center-2015` | Fable with Porcelain Raft at Knockdown Center in Queens in 2015
- `fat-boy-slim-austin-music-hall-1999` | Fat Boy Slim at Austin Music Hall in Austin in 1999
- `gayngs-music-hall-of-williamsburg-2010` | Gayngs at Music Hall of Williamsburg in Brooklyn in 2010
- `government-mule-beacon-theatre-2017` | Government Mule at Beacon Theatre in New York in 2017
- `imogen-heap-stubbs-bbq-2006` | Imogen Heap at Stubb's BBQ in Austin in 2006
- `incredible-caffeine-world-rave-arks-ranch-2000` | Incredible Caffeine World Rave at Ark's Ranch in Driftwood in 2000
- `jason-isbell-beacon-theatre-2015` | Jason Isbell at Beacon Theatre in New York in 2015
- `kings-of-leon-madison-square-garden-2010` | Kings of Leon at Madison Square Garden in New York in 2010
- `les-savy-fav-emos-2008` | Les Savy Fav at Emo's in Austin in 2008
- `little-richard-paramount-theatre-2008` | Little Richard at Paramount Theatre in Austin in 2008
- `lucinda-williams-la-zona-rosa-2004` | Lucinda Williams at La Zona Rosa in Austin in 2004
- `lucinda-williams-stubbs-bbq-2002` | Lucinda Williams at Stubb's BBQ in Austin in 2002
- `modest-mouse-austin-music-hall-2005` | Modest Mouse at Austin Music Hall in Austin in 2005
- `modest-mouse-stubbs-bbq-2003` | Modest Mouse at Stubb's BBQ in Austin in 2003
- `modest-mouse-stubbs-bbq-2004` | Modest Mouse at Stubb's BBQ in Austin in 2004
- `modest-mouse-webster-hall-2015` | Modest Mouse at Webster Hall in New York in 2015
- `my-morning-jacket-ed-sullivan-theater-2010` | My Morning Jacket at Ed Sullivan Theater in New York in 2010
- `my-morning-jacket-terminal-5-2010` | My Morning Jacket at Terminal 5 in New York in 2010
- `my-morning-jacket-stubbs-bbq-2006` | My Morning Jacket at Stubb's BBQ in Austin in 2006
- `neil-michael-hagerty-and-the-howling-hex-knitting-factory-2008` | Neil Michael Hagerty and the Howling Hex at Knitting Factory in New York in 2008
- `patti-smith-la-zona-rosa-2004` | Patti Smith at La Zona Rosa in Austin in 2004
- `peaches-emos-2004` | Peaches at Emo's in Austin in 2004
- `pixies-brooklyn-steel-2017` | Pixies at Brooklyn Steel in Brooklyn in 2017
- `pixies-brooklyn-steel-2018` | Pixies at Brooklyn Steel in Brooklyn in 2018
- `pixies-hammerstein-ballroom-2009` | Pixies at Hammerstein Ballroom in New York in 2009
- `pj-harvey-and-john-parish-beacon-theatre-2009` | PJ Harvey and John Parish at Beacon Theatre in New York in 2009
- `pj-harvey-irving-plaza-2009` | PJ Harvey at Irving Plaza in New York in 2009
- `pj-harvey-terminal-5-2011` | PJ Harvey at Terminal 5 in New York in 2011
- `porcelain-raft-mercury-lounge-2012` | Porcelain Raft at Mercury Lounge in New York in 2012
- `sharon-jones-beacon-theatre-2014` | Sharon Jones at Beacon Theatre in New York in 2014
- `she-and-him-terminal-5-2008` | She and Him at Terminal 5 in New York in 2008
- `sinad-oconnor-hogg-memorial-auditorium-2007` | Sinéad O'Connor at Hogg Memorial Auditorium in Austin in 2007
- `sonic-youth-hammersmith-apollo-2010` | Sonic Youth at Hammersmith Apollo in London in 2010
- `sonic-youth-stubbs-bbq-2006` | Sonic Youth at Stubb's BBQ in Austin in 2006
- `sparklehorse-emos-2006` | Sparklehorse at Emo's in Austin in 2006
- `sturgill-simpson-music-hall-of-williamsburg-2015` | Sturgill Simpson at Music Hall of Williamsburg in Brooklyn in 2015
- `supergrass-the-parish-2006` | Supergrass at The Parish in Austin in 2006
- `the-bad-plus-with-bill-frisell-jazz-at-lincoln-center-2013` | The Bad Plus with Bill Frisell at Jazz at Lincoln Center in New York in 2013
- `the-black-angels-bowery-ballroom-2010` | The Black Angels at Bowery Ballroom in New York in 2010
- `the-black-lips-bowery-ballroom-2008` | The Black Lips at Bowery Ballroom in New York in 2008
- `the-breeders-webster-hall-2008` | The Breeders at Webster Hall in New York in 2008
- `the-clean-the-bell-house-2010` | The Clean at The Bell House in Brooklyn in 2010
- `the-faint-emos-2002` | The Faint at Emo's in Austin in 2002
- `the-faint-la-zona-rosa-2004` | The Faint at La Zona Rosa in Austin in 2004
- `the-fall-indigo2-at-the-o2-2011` | The Fall at indigO2 at The O2 in London in 2011
- `the-fall-stubbs-bbq-2006` | The Fall at Stubb's BBQ in Austin in 2006
- `the-pretenders-austin-music-hall-2006` | The Pretenders at Austin Music Hall in Austin in 2006
- `the-rapture-and-black-rebel-motorcycle-club-stubbs-bbq-2004` | The Rapture and Black Rebel Motorcycle Club at Stubb's BBQ in Austin in 2004
- `the-vaselines-southpaw-2008` | The Vaselines at Southpaw in Brooklyn in 2008
- `the-vaselines-webster-hall-2010` | The Vaselines at Webster Hall in New York in 2010
- `the-white-stripes-stubbs-bbq-2003` | The White Stripes at Stubb's BBQ in Austin in 2003
- `thee-oh-sees-bowery-ballroom-2016` | Thee Oh Sees at Bowery Ballroom in New York in 2016
- `thee-oh-sees-irving-plaza-2013` | Thee Oh Sees at Irving Plaza in New York in 2013
- `tim-mcgraw-and-faith-hill-barclays-center-2017` | Tim McGraw & Faith Hill at Barclays Center in Brooklyn in 2017
- `tobias-jesso-jr-music-hall-of-williamsburg-2015` | Tobias Jesso jr at Music Hall of Williamsburg in Brooklyn in 2015
- `tony-bennett-90th-birthday-radio-city-music-hall-2016` | Tony Bennett 90th Birthday at Radio City Music Hall in New York in 2016
- `true-colors-radio-city-music-hall-2008` | True Colors at Radio City Music Hall in New York in 2008
- `various-artists-villa-rufolo-2013` | Various Artists at Villa Rufolo in Ravello in 2013
- `wilco-stubbs-bbq-2004` | Wilco at Stubb's BBQ in Austin in 2004.
- `willie-nelson-and-family-prospect-park-bandshell-2015` | Willie Nelson & Family at Prospect Park Bandshell in Brooklyn in 2015
- `women-mercury-lounge-2010` | Women at Mercury Lounge in New York in 2010
- `yaz-terminal-5-2008` | Yaz at Terminal 5 in New York in 2008.
- `yeah-yeah-yeahs-emos-2003` | Yeah Yeah Yeahs at Emo's in Austin in 2003
- `yeah-yeah-yeahs-stubbs-bbq-2006` | Yeah Yeah Yeahs at Stubb's BBQ in Austin in 2006
- `yuck-music-hall-of-williamsburg-2011` | Yuck at Music Hall of Williamsburg in Brooklyn in 2011

## D. Strong Legacy Entries Worth Preserving

These entries contain clearly personal/editorial writing and were intentionally protected from cleanup beyond deterministic placeholder-note removal elsewhere.

- `aerosmith-frank-erwin-center-1997` | A masterclass in stage presence. The scarves on the mic stand were iconic.
- `third-eye-blind-austin-music-hall-1998` | The venue is gone now, but the heat of that April night remains vivid.
- `the-wallflowers-frank-erwin-center-1997` | First arena show. The scale of the sound felt massive at 14.

## Manual Follow-Up Recommended

These records still need human research and/or human writing. They were left conservative on purpose:

- Research needed for missing `exactDate`: all entries in section A.
- Human writing needed for mechanical archival copy: all entries in section C.
- Highest-priority follow-up slugs:
- `aimee-mann-hogg-memorial-auditorium-2005`
- `animal-collective-antones-2007`
- `animal-collective-prospect-park-bandshell-2011`
- `arcade-fire-the-o2-arena-2010`
- `beach-house-bowery-ballroom-2008`
- `beyonc-barclays-center-2013`
- `fat-boy-slim-austin-music-hall-1999`
- `pj-harvey-irving-plaza-2009`
- `pj-harvey-the-fillmore-philadelphia-2018`
- `pj-harvey-terminal-5-2016`
- `tim-mcgraw-and-faith-hill-barclays-center-2017`
- `wilco-stubbs-bbq-2004`
- `women-mercury-lounge-2010`
- `yaz-terminal-5-2008`

## Notes

- No exact dates were guessed or added.
- No companions, openers, memories, or editorial details were invented.
- Backup text was only reused where the slug clearly matched and the older wording was already stronger.
- The cleanup intentionally favors factual stability over aggressive rewriting.
