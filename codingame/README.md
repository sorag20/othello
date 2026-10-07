# Othello bot for CodinGame

> **Note:** The bot source (`othello_bot.cpp`) and the code under
> [`archive/`](archive/) are intentionally **not included** in this public
> repository, in line with CodinGame's request that players not share
> ready-to-paste bot code. Only this write-up is published; the same search
> techniques are implemented in Python in
> [`backend/ai/minimax_engine.py`](../backend/ai/minimax_engine.py).

`othello_bot.cpp` is a single, self-contained C++17 file with **zero external
dependencies** (standard library only) that plays CodinGame's "Othello"
bot-programming game.

> An earlier attempt at this submission used a distilled neural network
> (PUCT MCTS guided by a small CNN) instead of classical search. It's kept
> for reference in [`archive/`](archive/) — see
> [`archive/README.md`](archive/README.md) for why this bitboard/alpha-beta
> approach replaced it.

## How to submit

1. Open the "Othello" puzzle/game on [CodinGame](https://www.codingame.com)
   and start (or open) a Web IDE session for it.
2. Set the language selector to **C++** (this file targets the C++17
   standard; CodinGame's C++ compiler supports it).
3. Select all the code in `othello_bot.cpp` and paste it into the Web IDE,
   replacing the starter code entirely.
4. Click **Play**/**Submit**. There is nothing else to configure: no build
   files, no libraries, no extra input — the whole bot is this one file.

That's it. The bot reads the board and legal moves from stdin and writes a
single move coordinate (e.g. `d3`) to stdout each turn, exactly per the
Othello protocol, and never prints anything else to stdout.

## How it works

### Board representation

The board is two `uint64_t` bitboards, `own` (the side to move's stones) and
`opp` (the opponent's), with bit index `row*8 + col` (row 0 = the top row
given in the input, col 0 = column `a`). `own`/`opp` are rebuilt from scratch
from the board string every turn — the bot never tries to incrementally
track state itself — which makes it naturally robust to the engine silently
skipping a player's turn (no legal moves): we simply never need to notice a
pass ourselves.

### Move generation

Legal moves and flip sets are computed with the classic "kogge-stone"/
direction-shift bitboard technique: from every one of the mover's stones,
walk each of the 8 ray directions through a run of 1-6 consecutive opponent
stones and see where it lands. This is O(1) per direction (a handful of
shift/and/or operations, no per-square loop), which is what makes bitboard
Othello engines fast enough to search deeply within a tiny time budget.

### Search

- **Negamax with alpha-beta pruning**, iterative deepening (depth 1, 2, 3,
  ...), and a persistent Zobrist transposition table.
- **NegaScout / Principal Variation Search (PVS)**: both the heuristic
  midgame search and the exact endgame solver search their (best-ordered)
  first move with the full `(-beta, -alpha)` window, then every subsequent
  move with a cheap null (zero-width) window `(-alpha-1, -alpha)` first,
  re-searching with the full window `(-beta, -value)` only if that null-
  window probe suggests the move might actually beat alpha (i.e.
  `alpha < value < beta`). This is a strict improvement when move ordering
  is already decent (which it is here, given the TT/killer/corner/
  fewest-replies ordering below): most moves get cheaply refuted by the
  null window without ever paying for a full-width search.
- A **killer-move table** (heuristic search only): whenever a move causes a
  beta cutoff, it's remembered keyed by remaining search depth (up to 2
  moves per depth, most-recent non-duplicate first), and tried right after
  the TT move in move ordering on subsequent nodes at that depth — a
  well-known cheap way to catch strong moves that recur across sibling
  branches before falling back to the generic ordering heuristics. (The
  exact endgame solver does not use killers — its own
  fewest-opponent-replies ordering is already strong, and this matches the
  verified reference implementation it was ported from.)
- The transposition table is keyed by an **absolute** hash (actual
  black/white occupancy + side to move, not "own/opp" which flips meaning
  every ply), and the table is **never cleared between turns** — it stays
  populated for the whole match, so positions explored on earlier turns
  (including ones only reached hypothetically during search) keep paying
  off later, especially once the game reaches the endgame where
  transpositions are common. It's a fixed-size array (2^22 entries, ~64MB,
  two tables for a total of ~128MB — comfortably inside CodinGame's ~256MB
  budget) with a depth-preferred replacement policy.
- **Move ordering**: TT move first, then (heuristic search only) killer
  moves for the current remaining depth, then corners, then by flip count
  and a static positional weight table (all cheap, no extra search) for the
  midgame search; fewest-opponent-replies-first (a well-known strong
  ordering heuristic) for the exact endgame solver.
- **Evaluation** (used whenever the exact endgame solver isn't active) is a
  phase-aware composite of:
  - **coin parity**, phase-weighted (`max(0, 30 - emptyCount)`, so material
    matters more as the board fills up),
  - **mobility** (`100 * (ownMoves - oppMoves) / (ownMoves + oppMoves)`),
  - a small-weight **static positional table** (residual signal only now
    that real stability is computed directly),
  - **real stable-disc counting** (`stableDiscs()`, weight 20): a disc is
    stable once, for all 4 axes through it (horizontal, vertical, both
    diagonals), at least one direction either reaches the board edge or its
    neighbor in that direction is same-colored *and itself already stable*
    — computed as a single coupled fixed point over one shared bitset
    across all 4 axes simultaneously (see the big doc comment above
    `stableDiscs()` in the source: an earlier per-axis-independent version
    was tried and found **wrong** by exhaustive brute-force game-tree
    enumeration — an anchor disc down one axis isn't a real anchor unless
    it's itself immune to flips from some *other* axis),
  - **frontier-disc counting** (`frontierDiscs()`, weight 2, inverted sign:
    fewer of our own is better): discs adjacent to an empty square are more
    exposed/flippable soon, a cheap "potential mobility" proxy.

  This replaces the previous version's corner-occupancy / X-C "danger
  square" terms outright — real stability is a strictly more accurate
  safety signal than that static proxy. Weights are still a reasonable,
  clearly-structured starting point, not finely tuned — see `evaluate()` in
  the source.

### Endgame exact solver

Once at most **12 empty squares** remain, the bot switches entirely to an
exact alpha-beta negamax search that ignores the heuristic and maximizes the
final disc-count differential (empty squares left when neither side can move
again are awarded to the leader, the standard convention for exact solving).

**Why 12:** measured locally (see `bench_endgame.cpp`, a throwaway benchmark
harness built for this change, not part of the submission) by timing
from-scratch exact solves of random-legal-play positions at each empty-count
depth, with the same TT + fewest-opponent-replies move ordering the real
solver uses (now also NegaScout/PVS, see "Search" above):

| empty squares | median solve time | p90 | worst observed (of 150 trials) |
|---:|---:|---:|---:|
| 6  | 0.04 ms | 0.10 ms | 0.21 ms |
| 8  | 0.12 ms | 0.25 ms | 0.40 ms |
| 10 | 0.45 ms | 0.98 ms | 1.9 ms |
| 12 | 2.4 ms | 6.1 ms | 9.2 ms |
| 14 | 16.1 ms | 37.7 ms | 65.3 ms |
| 16 | 77.3 ms | 225.4 ms | 404.9 ms |

(150 random-legal-play trials per empty-count bucket; measured fresh on one
development machine as part of re-verifying timing after adding stability +
frontier evaluation and NegaScout/PVS, see "Verification performed" below.
Expect some machine-to-machine variance — this is exactly why the threshold
is picked with a healthy margin rather than at the point of first failure.)

**Important finding from re-measuring:** the exact solver's own per-node
cost is *unaffected* by the stability/frontier evaluation changes (it never
calls `evaluate()` — it maximizes real disc differential directly), and the
added NegaScout/PVS pruning actually made it measurably *faster* than the
original benchmark at every bucket (e.g. 12 empties: 9.2ms worst now vs.
38.5ms worst previously; 14 empties: 65.3ms vs. 137.1ms previously). Despite
this improvement, **the threshold is kept at 12, unchanged** — not raised —
on purpose: CodinGame's judge hardware is unknown and could be slower than
this development machine, solve time still grows steeply (roughly an order
of magnitude every 2-4 empties), and the actual risk this change needed to
manage was the *heuristic* search getting slower (per-leaf-node stability +
frontier computation), not the exact solver. Since the exact solver turned
out to need no safety margin adjustment at all, there was no measured reason
to move the threshold in either direction. At 12 empties the worst case
observed remains well under a tenth of the 100ms internal target for a
normal turn; at 14 empties the worst case (65ms) looks safer than it used to
but is still closer to the budget than warranted given unknown judge
hardware. If the exact solve ever can't finish in the time available on a
given turn regardless (e.g. a slower judge machine, or a pathological
position even below the threshold), the bot doesn't just bail out — see
"Time budget and never-crash design" below for exactly how it falls back
gracefully instead of returning nothing.

### Time budget and never-crash design

This is the part the task cares about most, so it's deliberately
over-engineered for safety over playing strength:

- **Hard limits**: 2000ms wall-clock on the very first turn the process
  ever handles, 150ms on every turn after. The bot times itself with
  `std::chrono::steady_clock` starting from when that turn's board-reading
  finishes (not from process start), and targets internal deadlines of
  1650ms / 100ms respectively — pulled in a bit further than the
  "obvious" 1700ms/120ms split to bank extra margin, based on what dry-run
  testing actually showed (see "Verification" below). Re-measured after
  adding stability + frontier evaluation (see below): turn-1 latency across
  60 fresh process launches/games stayed at 1657-1661ms median/worst (2000ms
  budget), and regular-turn latency across 749 real-protocol turns (34
  full games) stayed at a consistent ~100.0-100.1ms (150ms budget) — both
  indistinguishable from the pre-change numbers, because `checkTime()`
  bounds wall-clock search time structurally regardless of per-node cost.
  Measured raw node throughput (`bench_search.cpp`, fixed-depth-9 searches
  from 12 representative midgame positions) actually came out roughly flat
  end to end (~7.9M nodes/sec new vs. ~8.1M nodes/sec old, the heavier
  `evaluate()` mostly offset by NegaScout/PVS + killer-move pruning needing
  far fewer total nodes for the same depth — total nodes across the same 12
  positions dropped from 15.47M to 8.63M, i.e. same-depth searches finished
  in roughly half the wall-clock time on this benchmark). In isolation,
  `evaluate()` itself is only ~10% slower on average (69ns -> 76ns per call)
  though its worst observed single-call cost rose more (125ns -> 6000ns,
  presumably the stability fixed-point needing more iterations on sparser
  boards) — still utterly negligible against the 128-node `checkTime()`
  sampling interval (worst-case bound: ~0.76ms of possible overshoot per
  128-node burst, versus a ~50ms cushion between the 100ms internal target
  and the 150ms hard cap).
- **Time checks**: a global node counter is checked every 128 nodes during
  search; if the deadline has passed, a lightweight exception unwinds the
  search cleanly. 128 was chosen (rather than a coarser interval) after
  dry-run testing showed a coarser interval could let a single burst of
  expensive nodes carry the search past the hard deadline.
- **Never a corrupted move**: iterative deepening only ever commits
  `bestMove` after a depth's search finishes *completely*; if the time-up
  exception fires mid-depth, that whole (partial, unreliable) iteration is
  discarded and the move from the last fully-completed iteration is kept.
  The endgame exact solver applies the same idea per root move instead of
  per depth: it commits progress after every fully-solved root child, so an
  abort mid-solve still leaves the best fully-verified move so far, never a
  move from a half-evaluated subtree.
- **Never an illegal move**: the actual move returned is always looked up
  from the engine-provided legal-move list for that turn, not synthesized
  from the bot's own move generator. Even a latent bug in the bitboard move
  generator could only make the search weaker — it structurally cannot
  produce an illegal output move. There's also a final defensive fallback in
  `main()`: if search logic somehow failed to produce a square from that
  list (should be unreachable), it falls back to the first legal move
  given that turn.
- **Single legal move**: if the engine gives us exactly one legal move, the
  bot plays it immediately without searching at all, saving the whole
  budget for later turns.
- **Startup cost kept out of the timed window**: the ~128MB transposition
  table storage is explicitly touched (memset) once at program start,
  before any turn's clock begins, so that first-touch page faults happen as
  a one-time untimed cost rather than landing unpredictably during a timed
  search later.
- **No stray stdout**: the bot never prints `MSG`, `EXPERT`, or debug output
  on stdout — only ever the chosen move coordinate, once per turn, followed
  by a flush.

## Verification performed before submission

1. **Bitboard correctness cross-check**: a deliberately naive, obviously
   correct 8x8-array reference implementation (brute-force 8-direction scan)
   was self-played against itself with uniformly random legal moves for
   **20,000 full games**. At every single position, both the *set* of legal
   moves and the exact flip set for *every* legal move were compared between
   the naive reference and the bitboard code; all matched (1,248,141
   positions, 10,238,151 individual legal-move flip-sets checked, zero
   mismatches).
2. **End-to-end protocol dry run**: a Python test driver spawned the actual
   compiled `othello_bot` binary as a real subprocess and drove it through
   the exact stdin/stdout protocol (one process per game, started once and
   kept alive for the whole game, exactly like the real judge) against a
   second, independent naive random-legal-move reference implementation, for
   140 full games total across both colors. Confirmed: the bot always
   returned a coordinate from that turn's legal-move list (zero illegal
   moves across all games), it never crashed or produced malformed/empty
   output, and per-turn latency stayed within budget after the timing fixes
   below.
3. **Compiler warnings**: builds warning-clean under both
   `g++ -std=c++17 -O2 -Wall -Wextra` (real GNU g++ 15) and
   `clang++ -std=c++17 -O2 -Wall -Wextra` (Apple clang), zero warnings on
   either.

### Re-verification after adding stability / frontier / NegaScout / killers

The three changes above (real stable-disc detection, frontier-disc
counting, and NegaScout/PVS + a killer-move table replacing the plain
alpha-beta loop) were ported from an already-verified Python reference
(`backend/ai/minimax_engine.py`) and then independently re-verified in C++,
since real added per-node cost on the hottest path (`evaluate()`) changes
the timing math the original thresholds above were tuned against:

4. **Stability correctness, exhaustively, not sampled**: a throwaway C++
   harness (`test_stability_cpp.cpp`) `#include`s the real
   `othello_bot.cpp` directly (so it tests the actual shipped function, not
   a reimplementation) and, mirroring the methodology in
   `backend/tests/test_minimax_engine.py::TestStableDiscs` (the test that
   originally caught the per-axis stability bug in the Python version),
   generates many small random-legal-play positions with 6-10 empty squares
   remaining, computes `stableDiscs()` for both colors, then exhaustively
   enumerates **every** possible continuation to every terminal leaf,
   asserting no disc marked stable ever changes color. Result: **240
   positions checked exhaustively (25 more skipped for having too large a
   game tree to fully enumerate within the node budget), zero violations**.
   Hand-crafted sanity cases (a corner-anchored 3-disc run is fully stable;
   an isolated disc touching no edge is not stable at all) also passed.
5. **Node-throughput / timing re-verification**: three more throwaway C++
   harnesses (`bench_eval.cpp`, `bench_search.cpp`, `bench_endgame.cpp`),
   each built once against the old (pre-change) source and once against the
   new source for a true before/after comparison on this machine, covering:
   raw `evaluate()` call cost, fixed-depth full-search node throughput, and
   exact endgame solve time per empty-count bucket. See "Why 12" above and
   the "Time budget" section above for the actual numbers; bottom line: the
   exact solver is unaffected (actually faster, from NegaScout/PVS), and the
   heuristic search's raw nodes/sec stayed roughly flat because better
   pruning (NegaScout + killers) offset the heavier per-leaf-node
   evaluation. No timing regression was found, so no threshold needed to
   move in the "more conservative" direction; the endgame threshold was
   nonetheless left unchanged at 12 (see "Why 12" above for the reasoning).
6. **Full end-to-end re-run** of the protocol dry run above against the
   *updated* binary: 20 full games vs. a random-legal-move opponent (both
   colors, alternating), zero illegal moves, zero crashes, zero malformed
   output, zero budget violations — turn-1 latency 1657.5-1661.1ms (2000ms
   budget), regular-turn latency 99.8-100.1ms across 515 turns (150ms
   budget).
7. **Strength sanity checks**:
   - vs. random-legal-move opponent (the required check): **20/20 decisive
     wins**, both colors, margins ranging roughly 40-20 up to 64-0 (a
     shutout). The pre-change bot was re-benchmarked the same way for
     comparison and also wins 10/10 vs. random (expected — beating random
     play is a low bar for either version; see the head-to-head below for a
     more meaningful comparison).
   - **new vs. old, head-to-head** (bonus check, not required but more
     informative than "beats random" alone): the updated bot vs. the
     pre-change bot, both as real subprocesses via the real protocol,
     alternating colors. **10/10 games won by the new version** (42-22 as
     black, 45-19 as white — both deterministic given deterministic search
     and no RNG in the game itself, hence the repeated identical scores per
     color). This is the most direct evidence the 3 ported improvements
     measurably increased playing strength rather than just adding cost.

### A real bug the dry run caught

The first end-to-end pass surfaced a genuine timing problem: one turn took
212ms against the 150ms hard budget (a would-be forfeit). Root-caused to two
compounding effects: (a) the original 2048-node check interval was too
coarse relative to actual per-node cost, and (b) each search node was
allocating `std::vector`s for move lists/ordering, adding allocator-latency
variance. Fixed by switching the hot recursive search path to fixed-capacity
stack arrays (zero heap allocation per node), tightening the time-check
interval to every 128 nodes, pre-touching the transposition table's memory
once at startup (outside any timed window), and pulling the internal target
deadlines in further (1650ms/100ms instead of 1700ms/120ms). Re-running the
dry run after the fix showed no further hard-budget violations across
multiple stress-test batches.

## Things worth a human double-check

- **Evaluation weights** (`evaluate()`) are a reasonable, clearly-structured
  starting point per the task brief, not tuned by self-play/tournament
  testing against other engines. Stability and frontier counting are now
  real (not proxies), so the next iteration worth trying is probably
  reweighting the 5 terms (coin parity / mobility / static table / stability
  / frontier) rather than adding new ones.
- **Endgame threshold (12 empties)** is based on solve-time benchmarks on
  one development machine; CodinGame's judge hardware is unknown and could
  be slower or faster. Freshly re-measured after this change (see "Why 12"
  above): the exact solver turned out *faster* than the original benchmark
  (NegaScout/PVS pruning), with the worst observed solve time at 12 empties
  now under a tenth of the internal turn budget. The threshold was
  deliberately left at 12 regardless, rather than raised to take advantage
  of the new margin — the risk this change needed to manage was the
  heuristic search getting slower, not the exact solver, so there was no
  measured reason to move it. If you have the ability to test on CodinGame's
  actual judge, it would be worth confirming the margin holds there too, and
  only then considering raising it (e.g. to 14).
- **Killer-move table is heuristic-search-only**, not used by the exact
  endgame solver. This matches the verified Python reference it was ported
  from (`backend/ai/minimax_engine.py`'s `_solve`/`_solve_endgame` use
  NegaScout but never call `_record_killer` or pass a depth to
  `_order_moves`), rather than an oversight — worth a second look if you
  want to push endgame solve times down further, since the exact solver's
  existing fewest-opponent-replies ordering may still leave something on
  the table that killers could catch.
- The **stability exhaustive check** (240 positions, 6-10 empties, zero
  violations) is strong evidence but not a formal proof; if you make any
  further change to `stableDiscs()` or the bitboard direction tables it was
  built on, re-run that check (or extend it to more empty-count buckets)
  before trusting the result.
- **Transposition table sizing** (2^22 entries per table, ~128MB total) was
  chosen to comfortably fit CodinGame's commonly-cited ~256MB budget with
  margin for the rest of the process, but wasn't pushed to find the exact
  ceiling. If CodinGame's actual memory limit for this game is tighter than
  assumed, this would be the first thing to shrink.
- The **empty-square-to-leader scoring convention** used by the exact
  endgame solver (when neither side can move again with empty squares still
  on the board) follows the standard tournament-scoring convention; this
  situation is rare in practice but worth being aware of as a modeling
  choice.
