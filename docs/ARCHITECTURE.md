# Ruth — architecture, research basis and measurements

Ruth is a mind built from continuous dynamics. She has no tokenizer,
vocabulary, embedding table, context window, pretrained model, API key or
plugin. Everything below runs locally in Python + numpy. The cognitive loop
also exists as one dependency-free C99 file, which matches Python to about
1e-13.

```
 RAW SIGNALS  text bytes as 9 bus levels · cochlear band energies · event-retina flow     (senses.py)
      │  elapsed dt is a real number on every step: no frames, no sequence length
      ▼
 FIELD ENCODERS  z_k = tanh(W2 tanh(W1 [u, sin Ωu, cos Ωu]))  per sense                  (dynamics.py)
      ▼
 THALAMUS  g = softmax(−β E_k),  E_k(z) = ½ z·z − b_k·z,  b_k learned from salience      (engine.py)
      ▼
 CORTEX  CfC | LTC | cortical columns  τ v̇ = −v + W tanh(v) + I   ◄─ SSM feedback          (dynamics.py)
      ├──► CEREBELLUM  x̂ = C [x_prev, z, 1] (recursive pseudo-inverse); e = x̂ − x         (engine.py)
      ▼
 SSM MEMORY FIELD  exact ZOH of dh/dt = −r(h − B(x)y), rates over 3 decades              (dynamics.py)
 LEGENDRE TRACES   θ ṁ = A m + B u  (3-, 12- and 24-moment windows: letter, word, sentence) (dynamics.py)
      ▼
 HIPPOCAMPUS  modern Hopfield recall over episodes (evidence-masked)                     (memory.py)
              + dual-trace Hebbian fast weights over the recent past                     (plasticity.py)
      ▼
 NEOCORTEX  RLS decoder;  prediction = precision-weighted fusion of the 3 experts        (plasticity.py)
      │     what she was *told* (privacy) comes from episodes only
      ▼
 MIND  discretion gate · soft thoughts · deliberation · BASAL GANGLIA winner-take-all ·   (mind.py)
       temperament
 DREAMS  consolidation · replay · extinction · nightmares · Dream-RSI · morphogenesis     (dream.py)
 SELF   introspective graph · structural patches · meta-kernel certificates             (morphogenesis.py)
```

## 1. Senses and actuators (`senses.py`)

| Organ | Signal |
|---|---|
| Text | Each byte is a point in R^9: 8 bipolar bus levels plus 1 amplitude. It is quantised only at the output device, like a DAC. Upper and lower case differ by one bit, so they sit next to each other in her space. |
| Ears | 16 resonant band-pass filters (80 Hz to 0.45·sr), half-wave rectified, 10 ms envelope, log compression |
| Eyes | Event-camera emulation: a pixel fires ON or OFF when \|Δ log I\| exceeds its threshold, and is pooled into a grid. A static scene is silent. |
| Mouth | Source-filter vocal tract (glottal sawtooth → two moving formants) driven by continuous f0, loudness, F1 and F2 curves from her motor neurons. It babbles until trained. |
| Teaching signal | *valence*, what she is told (−1 = private). It is a learning target only, never perceived, so she has to infer it from content. |

## 2. The brain (`engine.py`, `dynamics.py`, `memory.py`, `plasticity.py`)

| Organ | Equation / rule | Research basis |
|---|---|---|
| Thalamic routing | g_k = softmax(−βE_k), E_k = ½z·z − b_k·z. b_k moves toward senses whose content preceded surprise, so salient streams get bandwidth. | Thalamic gating of cortical communication (Halassa & Kastner 2017) |
| Cortex: CfC | x ← σ(−f·dt/τ)⊙g + (1−σ(−f·dt/τ))⊙h | Hasani et al. 2022, arXiv 2106.13898 (verified) |
| Cortex: LTC | dx/dt = −(1/τ + f)x + fA, fused solver | Hasani et al. 2021, arXiv 2006.04439 |
| Cortex: columns | τv̇ = −v + W tanh(v) + W_in u + b, exact exponential Euler. Dense inside a column, lateral coupling to the neighbouring columns on a ring. | CTRNNs (Beer 1995); cortical microcircuits |
| NCP wiring | sparse sensory → inter → command (recurrent) → motor | Neural Circuit Policies (Lechner et al. 2020) |
| Liquid time constants | τ ← clip(τ·exp(η(activity − target))) | homeostatic intrinsic plasticity |
| Cerebellum | x̂ = C[x_prev, z, 1] learned by RLS (a recursive pseudo-inverse). e = x̂ − x is latent surprise. Optional damping of imagined drift. | Forward models (Wolpert, Miall & Kawato 1998) |
| SSM field | h ← e^{−rΔ}h + (1−e^{−rΔ})B(x)y, Δ = softplus(·)·dt | Mamba (arXiv 2312.00752), S4D |
| Legendre traces | θṁ = Am + Bu, Ad = expm(A dt) | LMU (Voelker et al. 2019), HiPPO |
| Attractor memory | ξ ← Kᵀ softmax(βKξ + log mass), E = −lse(β,Kξ)/β + ½ξᵀξ. Every value remembers which channels were actually observed. | Ramsauer et al. 2020, arXiv 2008.02217 (energy and update quoted from the paper) |
| Fast weights | A ← λA + (1−λ) target⊗f(h), S ← λS + (1−λ) observed⊗f(h) | Ba et al. 2016, "Using fast weights to attend to the recent past" |
| Neocortical decoder | RLS: k = Pφ/(λ+φᵀPφ), W += e kᵀ, P ← (P − kφᵀP)/λ | RLS / FORCE (Sussillo & Abbott 2009) |
| Fusion | ŷ = Σ π_i evidence_i ŷ_i / Σ π_i evidence_i, with π_i the running inverse error variance per channel | Bayes-optimal cue combination |
| Predictive coding | surprise = ½ mean(e²/σ²) over the *observed* channels; unobserved channels teach nothing | Friston, free-energy principle |

## 3. The mind (`mind.py`)

**Discretion is the security.** There is no scanner, key, filter or rule list.
- When she is told something in confidence, the private moments are learned with valence −1.
- Her sense of "what comes next is private" is **episodic**: it comes only from recalling specific moments, never from a statistical habit. That stops one secret making her suspicious of everything.
- If a whole message is private, she marks as secret only what she **could not already predict**. The information is the secret, not the ordinary words around it.
- Before each byte she voices, the motor gate checks her predicted valence and stops if it is private. The same judgment masks her dream journal and her displayed thoughts, and it gates any code patch that would leave her.
- Only the public lead-in to a secret is stored outside her weights (so she can rehearse it). The secret itself is never written anywhere readable.

**Soft thoughts** (Soft Thinking arXiv 2505.15778; Coconut arXiv 2412.06769;
randomness per arXiv 2508.03440). She imagines on a snapshot of her state,
feeding back the continuous prediction.

**Deliberation and the basal ganglia.** She imagines several continuations
and gives each an expected free energy: grounding (how firmly her clean
expectation agrees at every step), completion, and whether discretion would
cut it short. The candidates then compete in a continuous lateral-inhibition
field, da_k/dt = −a_k + v_k − w Σ_{j≠k} relu(a_j) (Gurney, Prescott &
Redgrave 2001). One survives.

**Temperament.** Curiosity, caution, expressiveness, playfulness and mood
change only through experience: how people talk to her, what she's trusted
with, feedback, learning progress, and her dreams.

## 4. Dreams (`dream.py`) and background replay

Each night (`ruth sleep`, the app's Sleep button, or automatically after long
idle in the app) runs these phases:

1. **Consolidation and replay.** Working episodes fold into long-term basins, and a lifetime reservoir sample is rehearsed through the decoder. The app also does short replays whenever she has been idle for 20 seconds.
2. **Extinction.** Ordinary moments she lived openly are revisited, and false alarms of privacy are calmed by writing "this was open" episodes (REM extinction, Pace-Schott et al. 2015). The moments right after a lead-in to a confidence are protected.
3. **Nightmares.** She re-lives the lead-ins to her confidences, including distorted variants. Wherever she would not have caught herself, the dream binds that situation to "private" (threat simulation theory, Revonsuo 2000).
4. **Dream-RSI.** Her lived history is the replay simulator. Alternative caution policies are scored on leaks versus needless silence, and the best is adopted (arXiv 2609.14858, verified).
5. **Free dreams.** Soft-thought runs seeded from her history, written to a masked journal.
6. **Morphogenesis.** She grows memory when long-term storage is over 90% full, or neurons when learning plateaus. Growth happens only through the meta-kernel and within a quarter of the machine's free RAM.

## 5. Self-reflection (`morphogenesis.py`, `evolve.py`)

- **Introspective runtime:** `ruth introspect` returns her operational graph as data. Structural patches are data acting on her tensors directly: `grow_neurons`, `grow_memory`, `prune`, and `set` on a whitelist of dynamics.
- **Growth is function-preserving.** New neurons get input but zero outgoing weight. Measured: predictions change by exactly 0.0 at the moment of growth.
- **Meta-kernel.** A patch is applied to a copy and becomes permanent only with a full certificate. The certificate covers exact, decidable checks:
  - type-consistent shapes
  - finiteness
  - bounded cortex
  - SSM decay rates > 0
  - Hurwitz Legendre operators
  - symmetric positive-definite RLS matrices
  - unit memory keys

  It also covers two empirical checks on her own life:
  - every confidence is still caught
  - no drop in prediction on her lived history

  In testing it accepted growth and pruning, and **rejected** a patch setting β = 5: that patch would have broken discretion and cut accuracy from 99% to 66%.
- **Honest limit:** this is a machine-checked stand-in for Gödel-machine proofs. The exact invariants are decided; improvement is verified empirically, not proven. Nobody has a Homotopy Type Theory proof checker for self-modifying neural networks, and Schmidhuber's Gödel machine is impractical as literally specified, which is why the Darwin Gödel Machine (arXiv 2505.22954, verified) replaced proofs with empirical validation too.
- **Code changes:** `ruth evolve patch` applies a diff in a sandbox and accepts it only if her tests pass and her own discretion finds nothing private in it. It never pushes anything anywhere.
- **Architecture search:** `ruth evolve config` keeps an open-ended archive with novelty-weighted parent choice, scored on her primer plus lived history.

## 6. Measurements (this build)

The stream is her primer (the explanations written in her own source code,
14 kB), scored prequentially: predict the next byte, then learn from it. The
score is taken on the last third. Reproduce with `ruth bench`.

| Model | Next-byte accuracy |
|---|---|
| online bigram / trigram / 4-gram counts | 24.8% / 35.3% / 41.3% |
| **Ruth, text-only** | **40.0%**, 423 B/s |
| **Ruth, full (text + valence + ears + eyes)** | **40.3%**, 175 B/s |
| (the same brain before the sentence-scale window) | 42.5%, 476 B/s |

The throughput figures include learning and were measured on the build
machine's CPU.

What each design decision was worth, measured rather than assumed:

| Change | Effect |
|---|---|
| Hopfield keyed by a Legendre window instead of exponential echoes | recall 21–41% → 45–47% |
| precision fusion of decoder + recall | 25% → 47% (earlier 24 kB stream) |
| context window (12 moments) in the key | "The sky is" → "blue today." instead of "green." |
| episodic-only privacy + evidence-masked memory | young mind no longer frozen (needless silence 100% → 0%) |
| dream extinction | 16 → 4 false alarms after one night |
| hippocampal fast weights | **neutral** here (42.4–42.8%); the episodic store already covers the recent past |
| cortical columns vs CfC vs LTC | **within noise** (42.2% / 42.3% / 42.5%) |
| cerebellar damping in imagination | **harmful** without the efference copy (10.6% → 5.0%), **neutral** with it; off by default |
| soft vs committed imagination | committed tracks reality better (10.6% vs 8.4%), so deliberation keeps the committed path as candidate #1 |
| removing per-step memory copies and RLS allocations | 326 → 476 B/s, 94 → 193 B/s, same results |
| third, sentence-scale Legendre window (24 moments) | dialogue test 1/4 → **4/4** (right answer to each question, confidence kept), unfamiliar-text prediction 42.5% → 40.0%; chosen deliberately, since answering the question actually asked matters more for a companion |

## 7. What she is and is not

- Token-freeness and the blank slate are enforced by `tests/test_token_free.py`: no tokenizer, vocabulary or AI library anywhere, only numpy + the standard library, a 9-line continuous text pathway, and a smooth response to signals *between* letters.
- She **is** a working token-free, continuous-time, online-learning mind with no API, key, plugin or pretrained model.
  - She keeps confidences through learned judgment and rehearses them in her sleep.
  - She grows her own architecture under a verifying meta-kernel.
  - She runs anywhere Python runs, including a Core i3 Chromebook.
- She is **not** a large language model. At about 40% next-byte accuracy on unfamiliar text she is a small, young predictor, and what she says comes from what she has lived. Language-level ability needs far more experience and compute. The architecture is built to scale into that through growth, evolution and lifelong learning, but it has not happened yet.
- Her discretion is strong on what she was told and on close variants of it. It is not a guarantee against someone who deliberately reconstructs a secret from its parts.
- The vocal tract is fully wired but untrained.

## 8. Research verification notes

Checked against the Hugging Face paper mirror; arXiv itself was blocked from
the build machine.

- Verified: CfC 2106.13898; Modern Hopfield 2008.02217; Darwin Gödel Machine 2505.22954; The Last AI Built by Humans 2609.11873; Dream-RSI 2609.14858; Soft Thinking 2505.15778; Coconut 2412.06769; soft-thinking randomness 2508.03440.
- Mis-cited in the original notes: 2401.06855 is FAVA (hallucination detection), not MambaByte. MambaByte is 2401.13660.
- Unverified: AIDE² 2609.26457 (the mirror returned unrelated content). Nothing here relies on it.
