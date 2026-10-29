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

## How leans are made

1. **Screen** (`tools/screen.py`): every line on the board, both sides, goes through the player-data model in `tools/hitrate.py`. That model builds his per-minute rate from this season's and last season's game logs (ESPN; offensive and defensive rebounds come from Basketball-Reference), shrinks small samples toward league rates and multiplies by tonight's minutes. It then turns that into a chance with a negative binomial, averaged over minutes plus or minus 5. Lines at 60%+ become candidates.
2. **Verify**: research checks the inputs the data can't know: expected minutes, usage shifts from injuries or new roles (`usage_adj`), back-to-backs, blowout risk and late news. The model reruns with those inputs.
3. **Score**: the final number blends the data model (70%), the lean type's graded record (20%) and research judgment (10%). Low lines (2.5 or less) must pass a one-event stress test, and high-volatility lines get pulled toward 50%.
4. **Grade and refit**: each morning's grading stores every component. `calibrate` scores each component on the history and refits the blend once 150+ leans are graded.

## Leans workflow

```
python tools/leans.py new   nba                        # start today's board (archives the old one)
python tools/leans.py add   nba batch.json research    # add researched leans (checked against the live board)
python tools/leans.py grade nba 20261028               # grade a day from ESPN box scores
python tools/leans.py model nba                        # print what the record says
```

Grading adds every lean to `data/<league>/leans_history.json`. New leans are scored as follows:

A lean type is a stat group plus a side, for example "low-count Less" or "threes More". The model also keeps records by cushion (projection vs line), research confidence, payout and minutes risk.
