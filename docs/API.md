# BRAVS API Reference

## Player Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/search?q=name` | Search players by name |
| GET | `/api/player/<id>/bravs/<season>` | Get BRAVS for a player-season |
| GET | `/api/player/<id>/seasons` | All seasons for a player |
| GET | `/api/player/<id>/career` | Career summary |
| GET | `/api/player/<id>/projection` | Future projection |
| GET | `/api/player/<id>/similar/<season>` | Similar players |
| GET | `/api/export/<id>/<season>` | CSV export |
| GET | `/player/<id>/<season>` | Permalink page |

## Team Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/team/<abbrev>/<season>` | Full team roster with BRAVS |
| GET | `/api/team-rankings/<year>` | Team power rankings |
| GET | `/api/leaderboard/<season>/<league>` | Season leaderboard |
| GET | `/api/leaderboard-all/<year>` | Positional leaderboard |

## Analysis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/awards/<award>/<season>/<league>` | Award race results |
| GET | `/api/mvp/live/<league>` | Live MVP race (2026) |
| GET | `/api/dynasty/<id>` | Dynasty analysis |
| GET | `/api/dreamteam` | All-time dream team |
| GET | `/api/whatif` | Position swap what-if |
| POST | `/api/compare` | Multi-player comparison |
| GET | `/api/projections/<id>` | 2026 projection |
| GET | `/api/hof/<id>` | Hall of Fame probability |
| GET | `/api/aging-curve/<position>` | Empirical aging curve |

## Lineup Optimizer

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/lineup/optimize` | Optimal batting order (GPU) |
| POST | `/api/lineup/trade` | Trade impact simulation |
| POST | `/api/lineup/season` | 162-game allocation |
| GET | `/api/lineup/teams/<year>` | Available teams |

## MiLB Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/milb/player/<id>` | Minor league career |
| GET | `/api/milb/leaderboard/<year>/<level>` | MiLB leaderboard |
| GET | `/api/milb/prospects` | Top prospect rankings |

## Video Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/video/search/<query>` | Search highlight clips |
| GET | `/api/video/player/<id>/<season>` | Player highlights |
| GET | `/api/video/game/<gamePk>` | Game highlights |
| GET | `/api/video/pitches/<gamePk>/<pitcherId>` | Pitch-by-pitch video |
| GET | `/api/video/pitch/<playId>` | Single pitch video |

## Live & Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/live/scoreboard` | Today's live scores |
| GET | `/api/live/game/<gamePk>` | Live game detail |
| GET | `/api/dashboard/2026` | Full 2026 dashboard |
| POST | `/api/simulate` | Monte Carlo game simulation |
| GET | `/api/trade-value/<name>` | Player trade value |
| GET | `/api/roster-optimizer?budget=200` | Optimal roster for budget |
| GET | `/api/stats` | System statistics |
