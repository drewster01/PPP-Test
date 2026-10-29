# PPP-Test

PrizePicks lines compared with sharp books (Pinnacle and Kalshi), plus researched leans on the niche lines the books don't price, for the NBA and NFL. The model learns from graded results.

## Run the board

```
python live.py
```

Open http://127.0.0.1:8766. Use the NBA and NFL tabs to switch leagues. Each league has three pages:

- **Picks (Pinnacle):** every PrizePicks line Pinnacle also prices, showing Pinnacle's posted odds and Kalshi's price. A side is listed only when PrizePicks' line is as easy as Pinnacle's line or easier.
- **Leans:** research picks on niche lines. Each lean's confidence blends the research estimate with the track record of that lean type.
- **Results:** graded leans for each day, showing hit or miss and what the player actually put up, plus the record by lean type.

Both leagues refresh every 10 minutes from 9 AM ET to midnight ET.

## Leans workflow

```
python tools/leans.py new   nba                        # start today's board (archives the old one)
python tools/leans.py add   nba batch.json research    # add researched leans (checked against the live board)
python tools/leans.py grade nba 20261028               # grade a day from ESPN box scores
python tools/leans.py model nba                        # print what the record says
```

Grading adds every lean to `data/<league>/leans_history.json`. New leans are scored as follows:

```
model = 0.6 x research confidence + 0.4 x shrunk hit rate of that lean type
```

A lean type is a stat group plus a side, for example "low-count Less" or "threes More".
