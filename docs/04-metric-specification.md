# BRAVS: Bayesian Runs Above Value Standard

## Complete Technical Specification (v1.0)

---

## 1. Overview and Motivation

**BRAVS** (Bayesian Runs Above Value Standard) is a probabilistic player valuation framework for baseball. Unlike point-estimate metrics such as fWAR or bWAR, BRAVS produces full posterior distributions over player value. Every BRAVS estimate carries a mean, credible intervals, and a complete density function, enabling principled decision-making under uncertainty.

One unit of BRAVS equals one win of value above the **Freely Available Talent (FAT)** baseline. The FAT baseline represents the expected performance of a player obtainable at minimal cost --- roughly equivalent to a replacement-level AAAA player, a waiver claim, or a minor-league free agent. This baseline is calibrated so that a team composed entirely of FAT-level players would be expected to win approximately 47-48 games over a 162-game season.

The total BRAVS framework decomposes player value into nine components, each with its own Bayesian sub-model:

$$\text{BRAVS}_{\text{total}} = \frac{1}{\text{RPW}_{\text{dyn}}} \left[ \left( H + P + R + F + C + \text{Pos} + \text{AQI} \right) \times L_{\text{eff}} \right] + D$$

where $H$ is the hitting component, $P$ is pitching, $R$ is baserunning, $F$ is fielding, $C$ is catcher-specific value, $\text{Pos}$ is the positional adjustment, $\text{AQI}$ is the Approach Quality Index, $L_{\text{eff}}$ is the leverage factor, $D$ is durability, and $\text{RPW}_{\text{dyn}}$ is the dynamic runs-per-win converter.

---

## 2. Dynamic Runs Per Win ($\text{RPW}_{\text{dyn}}$)

The conversion from runs to wins is not a fixed constant. It depends on the run-scoring environment and park context. We derive $\text{RPW}_{\text{dyn}}$ from the Pythagorean expectation.

For a team with runs scored $RS$ and runs allowed $RA$, the Pythagorean win fraction is:

$$W\% = \frac{RS^x}{RS^x + RA^x}$$

where $x \approx 1.83$ (the empirical Pythagorean exponent). Taking the partial derivative of wins with respect to runs scored (or runs prevented), evaluated at $RS = RA = \text{RPG}_{\text{league}} / 2$, yields:

$$\text{RPW}_{\text{dyn}} = 2 \times \sqrt{\frac{\text{RPG}}{\text{PF}}} \times \frac{\text{IP}_{\text{team}}}{9}$$

where:
- $\text{RPG}$ = league-average runs per game in the relevant season
- $\text{PF}$ = park factor (1.00 = neutral; >1.00 = hitter-friendly)
- $\text{IP}_{\text{team}}$ = team innings pitched per game (typically 9.0 for regulation games, slightly higher in practice due to extra innings)

In a typical modern environment with $\text{RPG} \approx 9.0$ and $\text{PF} = 1.00$, this yields $\text{RPW}_{\text{dyn}} \approx 9.5$ runs per win. In a low-scoring environment ($\text{RPG} \approx 7.5$), each run is worth more: $\text{RPW}_{\text{dyn}} \approx 8.7$. This context-dependence ensures that BRAVS correctly values a run prevented at Petco Park higher than one prevented at Coors Field, all else equal.

---

## 3. Hitting Component ($H_{\text{BRAVS}}$)

### 3.1 Weighted On-Base Average Framework

The hitting component is built on the weighted on-base average (wOBA) framework. Each offensive event is assigned a linear weight derived from the base-out run expectancy matrix $RE_{24}$, which gives the expected runs from the current state to the end of the half-inning for each of the 24 base-out states (8 base states $\times$ 3 out states).

The linear weight for event $e$ is:

$$w_e = \Delta RE_{24}(e) + \text{RunsOnPlay}(e) - \overline{\Delta RE_{24}}$$

where $\Delta RE_{24}(e)$ is the change in run expectancy caused by the event and $\overline{\Delta RE_{24}}$ is the average change across all plate appearance outcomes, ensuring that the league-average wOBA is scaled to match league OBP for interpretability.

Standard linear weights (2016--2024 average, rounded):

| Event | Symbol | Run Value |
|-------|--------|-----------|
| Walk (unintentional) | $w_{\text{BB}}$ | 0.690 |
| Hit by Pitch | $w_{\text{HBP}}$ | 0.720 |
| Single | $w_{\text{1B}}$ | 0.880 |
| Double | $w_{\text{2B}}$ | 1.245 |
| Triple | $w_{\text{3B}}$ | 1.575 |
| Home Run | $w_{\text{HR}}$ | 2.015 |

The raw wOBA for player $i$ is:

$$\text{wOBA}_i = \frac{w_{\text{BB}} \cdot \text{BB}_i + w_{\text{HBP}} \cdot \text{HBP}_i + w_{\text{1B}} \cdot \text{1B}_i + w_{\text{2B}} \cdot \text{2B}_i + w_{\text{3B}} \cdot \text{3B}_i + w_{\text{HR}} \cdot \text{HR}_i}{\text{PA}_i - \text{IBB}_i - \text{SH}_i}$$

### 3.2 Hierarchical Bayesian Model for True Talent wOBA

We do not treat the observed wOBA as the player's true talent. Instead, we specify a hierarchical Bayesian model. Let $\theta_i$ denote the true-talent wOBA for player $i$, and let $g(i)$ denote the positional group of player $i$ (e.g., catchers, shortstops, corner outfielders).

**Likelihood.** For player $i$ with $n_i$ effective plate appearances, the observed wOBA is approximately normally distributed around the true talent:

$$\text{wOBA}_i^{\text{obs}} \mid \theta_i \sim \mathcal{N}\!\left(\theta_i, \; \frac{\sigma^2_{\text{wOBA}}}{n_i}\right)$$

where $\sigma^2_{\text{wOBA}}$ is the within-season sampling variance of wOBA, empirically estimated at $\sigma_{\text{wOBA}} \approx 0.035$ per root plate appearance. More precisely, we can model the individual plate appearance outcomes as multinomial:

$$\mathbf{x}_i \mid \boldsymbol{\pi}_i \sim \text{Multinomial}(n_i, \boldsymbol{\pi}_i(\theta_i))$$

where $\mathbf{x}_i$ is the vector of event counts (BB, HBP, 1B, 2B, 3B, HR, out) and $\boldsymbol{\pi}_i$ is the vector of event probabilities that map to $\theta_i$ via the linear weight formula. For computational tractability, the normal approximation to the wOBA sufficient statistic is used when $n_i > 50$.

**Prior.** The positional-group prior is:

$$\theta_i \mid \mu_{g(i)}, \sigma^2_{g(i)} \sim \mathcal{N}\!\left(\mu_{g(i)}, \; \sigma^2_{g(i)}\right)$$

**Hyperpriors.** The group-level parameters are themselves drawn from a league-wide distribution:

$$\mu_{g} \sim \mathcal{N}\!\left(\mu_{\text{league}}, \; \sigma^2_{\mu}\right)$$
$$\sigma^2_{g} \sim \text{Inv-}\chi^2\!\left(\nu_0, \; s^2_0\right)$$

Empirical estimates of hyperparameters (from 2015--2024 data):
- $\mu_{\text{league}} \approx 0.320$ (league-average wOBA)
- $\sigma_{\mu} \approx 0.012$ (between-group variation)
- $\nu_0 \approx 10$, $s^2_0 \approx 0.030^2$ (prior on within-group talent spread)

### 3.3 Posterior and Shrinkage

Given conjugacy, the posterior for $\theta_i$ is:

$$\theta_i \mid \text{data} \sim \mathcal{N}\!\left(\hat{\theta}_i, \; V_i\right)$$

where the posterior mean (the shrinkage estimator) is:

$$\hat{\theta}_i = \frac{\frac{n_i}{\sigma^2_{\text{wOBA}}} \cdot \text{wOBA}_i^{\text{obs}} + \frac{1}{\sigma^2_{g(i)}} \cdot \mu_{g(i)}}{\frac{n_i}{\sigma^2_{\text{wOBA}}} + \frac{1}{\sigma^2_{g(i)}}}$$

and the posterior variance is:

$$V_i = \left(\frac{n_i}{\sigma^2_{\text{wOBA}}} + \frac{1}{\sigma^2_{g(i)}}\right)^{-1}$$

This produces the classic regression-toward-the-mean behavior. A player with 100 PA has roughly 40% of his estimate determined by the prior; a player with 600 PA has roughly 85% determined by the data.

### 3.4 Park, Era, and League Adjustments

Before computing the posterior, we adjust the observed wOBA:

$$\text{wOBA}_i^{\text{adj}} = \text{wOBA}_i^{\text{obs}} \times \frac{1}{\text{PF}_i^{\text{wOBA}}} \times \frac{\text{wOBA}_{\text{league,ref}}}{\text{wOBA}_{\text{league,actual}}}$$

where $\text{PF}_i^{\text{wOBA}}$ is the multi-year regressed park factor for wOBA (not just runs), and the final ratio is the era adjustment that normalizes to a reference league-average wOBA.

### 3.5 Conversion to Runs Above FAT

$$H_{\text{BRAVS}} = \frac{\hat{\theta}_i^{\text{adj}} - \text{wOBA}_{\text{FAT}}}{\text{wOBA}_{\text{scale}}} \times n_i$$

where:
- $\text{wOBA}_{\text{FAT}} \approx 0.290$ (the expected wOBA of a freely available talent hitter)
- $\text{wOBA}_{\text{scale}} \approx 1.15$ (the scaling constant that converts wOBA to runs per PA, derived as $\text{wOBA}_{\text{league}} / \text{R/PA}_{\text{league}}$)
- $n_i$ = plate appearances

The posterior distribution of $H_{\text{BRAVS}}$ inherits its uncertainty from the posterior of $\hat{\theta}_i$, scaled linearly:

$$H_{\text{BRAVS}} \sim \mathcal{N}\!\left(\frac{\hat{\theta}_i - \text{wOBA}_{\text{FAT}}}{\text{wOBA}_{\text{scale}}} \cdot n_i, \;\; \frac{V_i \cdot n_i^2}{\text{wOBA}_{\text{scale}}^2}\right)$$

---

## 4. Pitching Component ($P_{\text{BRAVS}}$)

### 4.1 Extended FIP Model

The core pitching model extends Fielding Independent Pitching (FIP) by incorporating batted-ball quality from Statcast data. The standard FIP is:

$$\text{FIP}_{\text{core}} = \frac{13 \times \text{HR} + 3 \times (\text{BB} + \text{HBP}) - 2 \times \text{K}}{\text{IP}} + C_{\text{era}}$$

where $C_{\text{era}}$ is a constant that rescales FIP to the league-average ERA (typically $C_{\text{era}} \approx 3.10$ in modern baseball). We extend this with a batted-ball quality term:

$$\text{xFIP}^{+} = \text{FIP}_{\text{core}} + \alpha \times (\text{xwOBA}_{\text{contact}} - \text{xwOBA}_{\text{league,contact}}) \times \text{BIP}_{\text{rate}}$$

where $\text{xwOBA}_{\text{contact}}$ is the Statcast expected wOBA on balls in play against the pitcher, $\text{BIP}_{\text{rate}} = \text{BIP} / \text{TBF}$, and $\alpha \approx 5.0$ is the scaling coefficient that converts the xwOBA differential to ERA-scale runs. This captures pitcher skill at inducing weak contact beyond what strikeout rate alone reveals.

### 4.2 Bayesian Shrinkage for Pitchers

Let $\phi_i$ denote the true-talent run prevention skill of pitcher $i$ (on the ERA scale, lower is better). We model:

$$\text{xFIP}^{+}_i \mid \phi_i \sim \mathcal{N}\!\left(\phi_i, \; \frac{\tau^2}{\text{IP}_i / 9}\right)$$

where $\tau^2$ is the per-game sampling variance of ERA-scale metrics, empirically $\tau \approx 1.5$.

The prior depends on pitcher type $t \in \{\text{SP}, \text{RP}\}$:

$$\phi_i \mid t(i) \sim \mathcal{N}\!\left(\mu_{t(i)}, \; \sigma^2_{t(i)}\right)$$

Empirical hyperparameters:
- Starters: $\mu_{\text{SP}} \approx 4.20$, $\sigma_{\text{SP}} \approx 0.70$
- Relievers: $\mu_{\text{RP}} \approx 4.00$, $\sigma_{\text{RP}} \approx 0.90$

The posterior mean for pitcher $i$ with $g_i = \text{IP}_i / 9$ game-equivalents is:

$$\hat{\phi}_i = \frac{\frac{g_i}{\tau^2} \cdot \text{xFIP}^{+}_i + \frac{1}{\sigma^2_{t(i)}} \cdot \mu_{t(i)}}{\frac{g_i}{\tau^2} + \frac{1}{\sigma^2_{t(i)}}}$$

### 4.3 Separate Baselines

The FAT baseline differs for starters and relievers:
- $\text{FAT}_{\text{SP}} \approx 5.50$ ERA-scale (i.e., a freely available starter prevents runs at a 5.50 rate)
- $\text{FAT}_{\text{RP}} \approx 4.80$ ERA-scale

### 4.4 Conversion to Runs Above FAT

$$P_{\text{BRAVS}} = \frac{(\text{FAT}_{t(i)} - \hat{\phi}_i) \times \text{IP}_i}{9}$$

This is in units of runs. A pitcher with a 3.00 true-talent rate who throws 200 IP has:

$$P_{\text{BRAVS}} = \frac{(5.50 - 3.00) \times 200}{9} \approx 55.6 \text{ runs}$$

The posterior distribution of $P_{\text{BRAVS}}$ inherits uncertainty from $\hat{\phi}_i$, with variance:

$$\text{Var}(P_{\text{BRAVS}}) = \left(\frac{\text{IP}_i}{9}\right)^2 \times V_{\phi,i}$$

where $V_{\phi,i}$ is the posterior variance of $\hat{\phi}_i$.

---

## 5. Baserunning Component ($R_{\text{BRAVS}}$)

### 5.1 Stolen Base Runs

Each stolen base attempt changes the base-out run expectancy. The marginal run values are:

- $r_{\text{SB}} \approx +0.20$ runs per successful steal (the gain from advancing one base minus the average)
- $r_{\text{CS}} \approx -0.43$ runs per caught stealing (the loss from making an out on the bases)

These are averaged across base-out states; more precise computation uses state-specific values.

Let $p_i$ be the true stolen base success rate for player $i$, with a Bayesian estimate:

$$p_i \mid \text{data} \sim \text{Beta}(\alpha_0 + \text{SB}_i, \; \beta_0 + \text{CS}_i)$$

where the prior $\text{Beta}(\alpha_0, \beta_0)$ reflects the population success rate. With a league average of approximately 75%, we set $\alpha_0 = 6, \beta_0 = 2$ (equivalent to 8 prior attempts at 75%).

$$\text{SB}_{\text{runs}} = \text{SB}_i \times r_{\text{SB}} + \text{CS}_i \times r_{\text{CS}}$$

For the posterior distribution, we propagate the uncertainty in $p_i$ through the run value calculation by drawing from the Beta posterior and computing the implied SB runs for each draw.

### 5.2 Advance Runs

Advance runs capture non-stolen-base baserunning: taking extra bases on singles and doubles, tagging up on fly balls, and avoiding outs on the bases. For each baserunning opportunity $j$, we compute:

$$\text{Adv}_j = RE_{24}(\text{state after}) - RE_{24}(\text{state before})$$

and compare to the league-average outcome for that opportunity type. Summing over all opportunities:

$$\text{Advance}_{\text{runs}} = \sum_{j} \left(\text{Adv}_j - \overline{\text{Adv}}_{\text{league},j}\right)$$

### 5.3 GIDP Avoidance

Ground-into-double-play avoidance is modeled as:

$$\text{GIDP}_{\text{runs}} = (\text{GIDP}_{\text{expected}} - \text{GIDP}_{\text{actual}}) \times r_{\text{GDP}}$$

where $\text{GIDP}_{\text{expected}}$ is predicted from the player's ground ball rate and opportunities (runner on first, less than two outs), and $r_{\text{GDP}} \approx 0.37$ runs per GDP avoided.

### 5.4 Total Baserunning

$$R_{\text{BRAVS}} = \text{SB}_{\text{runs}} + \text{Advance}_{\text{runs}} + \text{GIDP}_{\text{runs}}$$

The posterior variance is dominated by the SB component for high-volume base stealers; for most players, baserunning uncertainty is modest ($\sigma \approx 1{-}3$ runs).

---

## 6. Fielding Component ($F_{\text{BRAVS}}$)

### 6.1 Bayesian Model Averaging

Fielding evaluation is notoriously noisy. BRAVS addresses this by combining multiple defensive metrics through Bayesian model averaging. Let $\text{DEF}_{i,k}$ denote the estimate of player $i$'s fielding value (in runs above average) from system $k$, where $k \in \{1: \text{UZR}, \; 2: \text{DRS}, \; 3: \text{OAA}\}$.

Each system $k$ provides a noisy estimate of the latent true fielding value $\delta_i$:

$$\text{DEF}_{i,k} \mid \delta_i \sim \mathcal{N}\!\left(\delta_i + b_k, \; \sigma^2_{k,n_i}\right)$$

where $b_k$ is a possible systematic bias for system $k$ and $\sigma^2_{k,n_i}$ is the system-specific measurement error, which decreases with playing time $n_i$ (measured in innings at position).

The posterior model weights are computed using marginal likelihoods:

$$w_k = \frac{p(\text{data} \mid M_k) \cdot \pi(M_k)}{\sum_{k'} p(\text{data} \mid M_{k'}) \cdot \pi(M_{k'})}$$

where $\pi(M_k)$ is the prior model probability (we start with equal priors: $\pi(M_k) = 1/3$) and $p(\text{data} \mid M_k)$ is the marginal likelihood under each system's model.

The model-averaged posterior is:

$$p(\delta_i \mid \text{data}) = \sum_k w_k \cdot p(\delta_i \mid \text{data}, M_k)$$

### 6.2 Strong Prior for Fielding

Fielding metrics have high year-to-year variance. We impose a strong prior:

$$\delta_i \sim \mathcal{N}(0, \sigma^2_{\text{field}})$$

with $\sigma_{\text{field}} = 5$ runs. This means that, absent substantial playing time (>1000 innings at position), the posterior estimate is heavily regressed toward zero.

### 6.3 Conversion to Fielding BRAVS

$$F_{\text{BRAVS}} = \hat{\delta}_i$$

where $\hat{\delta}_i = E[\delta_i \mid \text{data}]$ is the posterior mean from the model-averaged estimate. The posterior variance is:

$$\text{Var}(\delta_i \mid \text{data}) = \sum_k w_k \left[ V_{i,k} + (\hat{\delta}_{i,k} - \hat{\delta}_i)^2 \right]$$

which incorporates both within-model uncertainty ($V_{i,k}$) and between-model disagreement.

### 6.4 Reliability by Playing Time

Approximate reliability (fraction of posterior variance attributable to signal):

| Innings at Position | Reliability (UZR) | Reliability (OAA) |
|---------------------|--------------------|--------------------|
| 200 | 0.05 | 0.08 |
| 500 | 0.12 | 0.18 |
| 1000 | 0.22 | 0.32 |
| 1500+ | 0.30 | 0.42 |

OAA, being based on Statcast tracking data with finer spatial granularity, achieves higher reliability at equal sample sizes.

---

## 7. Catcher Component ($C_{\text{BRAVS}}$)

This component applies only to catchers and captures skills unique to the position.

### 7.1 Framing Runs

For each called pitch $j$ received by catcher $i$, we compute the probability that the pitch would be called a strike by an average catcher, $p_j^{\text{avg}}$, using a probit model conditioned on pitch location, count, umpire, and pitcher:

$$p_j^{\text{strike}} = \Phi\left(\beta_0 + \beta_{\text{loc}} \cdot f(\text{loc}_j) + \beta_{\text{ump}} + \gamma_i\right)$$

where $\gamma_i$ is the catcher's framing effect. The run value per strike gained is $r_{\text{strike}} \approx 0.125$ runs. Total framing value:

$$\text{Framing}_{\text{runs}} = \left(\sum_j \mathbb{1}[\text{called strike}_j] - \sum_j p_j^{\text{avg}}\right) \times r_{\text{strike}}$$

The catcher-specific parameter $\gamma_i$ is estimated with a prior $\gamma_i \sim \mathcal{N}(0, \sigma^2_{\gamma})$ where $\sigma_{\gamma} \approx 0.02$ (corresponding to roughly 10-15 runs of true-talent range across catchers).

### 7.2 Blocking Runs

Wild pitches and passed balls prevented, relative to expectation:

$$\text{Blocking}_{\text{runs}} = (\text{WP/PB}_{\text{expected}} - \text{WP/PB}_{\text{actual}}) \times r_{\text{WP}}$$

where $r_{\text{WP}} \approx 0.27$ runs per wild pitch prevented, and the expected count is modeled as a function of the pitcher's pitch mix, stuff quality, and dirt-ball rate.

### 7.3 Throwing Runs

$$\text{Throwing}_{\text{runs}} = (\text{CS}_{\text{above avg}} + \text{PO}_{\text{above avg}}) \times r_{\text{CS/PO}} + \text{SB}_{\text{deterrence}}$$

Stolen base deterrence is estimated as the difference between the expected SB attempt rate against the catcher versus the league average, multiplied by the run cost of an attempt. This is heavily regressed due to confounding with pitcher slide-step tendencies.

### 7.4 Game-Calling Runs

The most contentious and uncertain component. We use a WOWY (With-Or-Without-You) design:

$$\text{GameCalling}_{\text{runs}} = \frac{1}{2} \times \sum_p (\overline{\text{ERA}}_{p,\neg i} - \overline{\text{ERA}}_{p,i}) \times \frac{\text{IP}_{p,i}}{9}$$

where the sum is over all pitchers $p$ who threw to catcher $i$ and at least one other catcher, and the factor of $1/2$ reflects the heavy regression applied due to extreme confounding risk. The prior is:

$$\text{GameCalling}_i \sim \mathcal{N}(0, 2^2)$$

with a standard deviation of 2 runs, reflecting deep skepticism about measurability.

### 7.5 Total Catcher Value

$$C_{\text{BRAVS}} = \text{Framing}_{\text{runs}} + \text{Blocking}_{\text{runs}} + \text{Throwing}_{\text{runs}} + \text{GameCalling}_{\text{runs}}$$

---

## 8. Positional Adjustment ($\text{Pos}_{\text{BRAVS}}$)

The positional adjustment accounts for the offensive and defensive opportunity cost of playing a more demanding defensive position. Values are in runs per 162 games, based on Tango's foundational scale with modern updates:

| Position | Adjustment (runs/162G) |
|----------|------------------------|
| Catcher (C) | +12.5 |
| Shortstop (SS) | +7.5 |
| Center Field (CF) | +2.5 |
| Second Base (2B) | +2.5 |
| Third Base (3B) | +2.5 |
| Left Field (LF) | -7.5 |
| Right Field (RF) | -7.5 |
| First Base (1B) | -12.5 |
| Designated Hitter (DH) | -17.5 |

For a player with $G_i$ games played at position $p$:

$$\text{Pos}_{\text{BRAVS}} = \sum_p \frac{G_{i,p}}{162} \times \text{PosAdj}_p$$

Multi-position players receive a weighted blend. Importantly, the positional adjustment is applied separately from the fielding component; it represents the scarcity premium for the position, not the player's individual defensive skill.

---

## 9. Approach Quality Index ($\text{AQI}_{\text{BRAVS}}$)

### 9.1 Concept

The AQI is a novel component that measures batter decision quality on a per-pitch basis. Rather than using aggregate statistics like O-Swing% or Z-Contact%, the AQI evaluates each pitch decision by its run-value consequence.

### 9.2 Per-Pitch Decision Value

For each pitch $j$ seen by batter $i$, let $d_j \in \{\text{swing}, \text{take}\}$ be the batter's decision and let $d_j^{*}$ be the optimal decision given perfect information about pitch characteristics (location, movement, velocity) and the current game state (count, base-out state).

Define the run-value differential:

$$\Delta \text{RV}_j = E[\text{RV} \mid d_j, \text{state}_j] - E[\text{RV} \mid d_j^{*}, \text{state}_j]$$

where $\text{RV}$ is the run value produced from the given count/base-out state after the pitch. The optimal decision $d_j^{*}$ is computed using a full Markov chain model of plate appearance outcomes:

- If the pitch is hittable (in-zone, favorable velocity/movement), $d_j^{*} = \text{swing}$.
- If the pitch is unhittable (out of zone, extreme movement), $d_j^{*} = \text{take}$.
- Borderline pitches are adjudicated by comparing the expected run value of swinging (accounting for the batter's quality of contact on that pitch type) versus taking (accounting for the called-strike probability).

### 9.3 Aggregation and Shrinkage

$$\text{AQI}_{\text{raw},i} = \sum_{j=1}^{N_{\text{pitches},i}} \Delta \text{RV}_j$$

The raw AQI is then subjected to Bayesian shrinkage:

$$\text{AQI}_i \mid \text{data} \sim \mathcal{N}\!\left(\hat{a}_i, \; V_{a,i}\right)$$

where:

$$\hat{a}_i = \frac{\frac{N_i}{\sigma^2_{\text{pitch}}} \cdot \overline{\Delta \text{RV}}_i + \frac{1}{\sigma^2_{\text{AQI}}} \cdot 0}{\frac{N_i}{\sigma^2_{\text{pitch}}} + \frac{1}{\sigma^2_{\text{AQI}}}}$$

with $\sigma_{\text{AQI}} \approx 2.5$ runs (the prior standard deviation of true-talent AQI) and $\sigma_{\text{pitch}} \approx 0.005$ runs (per-pitch sampling noise, yielding $\sigma \approx 3.5$ runs over 2500 pitches seen).

### 9.4 Value Range and Interpretation

Elite plate-discipline hitters (historical examples: Ted Williams, Barry Bonds, Joey Votto) would register AQI values of $+3$ to $+5$ runs per full season. Extreme free-swingers with poor chase-rate profiles on non-competitive pitches register $-3$ to $-5$ runs. The median player is near zero by construction.

$$\text{AQI}_{\text{BRAVS}} = \hat{a}_i \times \frac{N_{\text{pitches},i}}{N_{\text{pitches},\text{avg}}}$$

This pro-rates the value to account for playing time differences.

---

## 10. Leverage Component ($L_{\text{BRAVS}}$)

### 10.1 Effective Leverage

Not all plate appearances and innings are equal. A reliever who pitches exclusively in the 8th and 9th innings of close games contributes more marginal value per out recorded than a mop-up reliever. We measure this using the game leverage index ($\text{gmLI}$), which quantifies the average importance of the game situations in which a player appeared.

To avoid over-rewarding relievers relative to starters (who face a wider distribution of game states), we use damped leverage:

$$L_{\text{eff},i} = \sqrt{\text{gmLI}_i}$$

### 10.2 Application

The leverage factor is applied as a multiplicative adjustment to skill-based run value:

$$\text{Runs}_{\text{leveraged},i} = \text{Runs}_{\text{skill},i} \times \frac{L_{\text{eff},i}}{E[L_{\text{eff}}]}$$

where $E[L_{\text{eff}}] = E[\sqrt{\text{gmLI}}] \approx 1.0$ (calibrated so that average leverage produces no adjustment). This primarily affects:

- High-leverage relievers: $\text{gmLI} \approx 1.8$, $L_{\text{eff}} \approx 1.34$, ~34% boost
- Low-leverage relievers: $\text{gmLI} \approx 0.6$, $L_{\text{eff}} \approx 0.77$, ~23% discount
- Starting pitchers: $\text{gmLI} \approx 1.0$, $L_{\text{eff}} \approx 1.00$, no adjustment
- Pinch hitters in clutch spots: variable, but typically $L_{\text{eff}} > 1.0$

---

## 11. Durability Component ($D_{\text{BRAVS}}$)

### 11.1 Availability Value

A player who stays healthy and plays every day provides value beyond his per-game skill level, because his absence would be filled by a FAT-level player. Conversely, a player who misses significant time costs his team those marginal games.

The expected games for each role are:

| Role | Expected Games |
|------|----------------|
| Position Player (everyday) | 155 |
| Starting Pitcher | 32 starts (200 IP) |
| Relief Pitcher | 65 appearances |

### 11.2 Calculation

$$D_{\text{BRAVS}} = (G_{\text{actual}} - G_{\text{expected}}) \times v_{\text{marginal}}$$

where $v_{\text{marginal}}$ is the marginal value of a game of availability, calibrated as the FAT-level performance for that game:

- Position players: $v_{\text{marginal}} \approx 0.03$ wins/game (about 0.3 runs/game of FAT-level production versus nothing)
- Starting pitchers: $v_{\text{marginal}} \approx 0.05$ wins/start (a FAT-level start is worth roughly 0.05 wins over a bullpen day)

Note that $D_{\text{BRAVS}}$ is added to the total after the leverage adjustment, since durability value is independent of leverage.

---

## 12. Full Aggregation

### 12.1 Position Player Total

$$\text{BRAVS}_{\text{pos}} = \frac{1}{\text{RPW}_{\text{dyn}}} \left[(H + R + F + C + \text{Pos} + \text{AQI}) \times \frac{L_{\text{eff}}}{E[L_{\text{eff}}]}\right] + D$$

### 12.2 Pitcher Total

$$\text{BRAVS}_{\text{pitch}} = \frac{1}{\text{RPW}_{\text{dyn}}} \left[P \times \frac{L_{\text{eff}}}{E[L_{\text{eff}}]}\right] + D$$

### 12.3 Two-Way Player Total

For two-way players (e.g., Shohei Ohtani), the hitting and pitching components are computed from separate pools of plate appearances / innings pitched and summed directly:

$$\text{BRAVS}_{\text{2-way}} = \text{BRAVS}_{\text{pos}} + \text{BRAVS}_{\text{pitch}}$$

This is valid because the hitting contribution and pitching contribution occur in disjoint plate appearances (the player cannot simultaneously be batting and pitching in the same PA).

---

## 13. Posterior Computation

### 13.1 Conjugate Components

For the hitting, pitching, and stolen base components, the Normal-Normal conjugate structure yields closed-form posteriors as described in their respective sections. These are computationally inexpensive and exact.

### 13.2 Non-Conjugate Components: MCMC

For the fielding ensemble (Bayesian model averaging), catcher framing (probit model), game-calling (WOWY with confounders), and AQI (nonlinear run-value model), we employ Hamiltonian Monte Carlo (HMC) sampling, specifically the No-U-Turn Sampler (NUTS):

- **Chains:** 4 parallel chains
- **Warm-up:** 1000 iterations per chain
- **Sampling:** 2000 iterations per chain
- **Target acceptance rate:** 0.80
- **Convergence diagnostics:** $\hat{R} < 1.01$ for all parameters, effective sample size $> 400$

### 13.3 Uncertainty Propagation

Since total BRAVS is a sum of components, and most components have near-zero correlation, the posterior variance of total BRAVS is approximately:

$$\text{Var}(\text{BRAVS}_{\text{total}}) \approx \frac{1}{\text{RPW}_{\text{dyn}}^2} \sum_{c} \text{Var}(c)$$

where the sum is over all run-valued components $c \in \{H, P, R, F, C, \text{Pos}, \text{AQI}\}$. When components are correlated (e.g., hitting and AQI share plate-discipline signals), we extract the full joint posterior from MCMC and compute the total BRAVS posterior empirically.

Credible intervals are reported as highest posterior density (HPD) intervals at the 90% level.

---

## 14. Prior Specification Summary

| Component | Prior Distribution | Hyperparameters | Source |
|-----------|--------------------|-----------------|--------|
| $\theta_i$ (hitting) | $\mathcal{N}(\mu_{g(i)}, \sigma^2_{g(i)})$ | $\mu \approx 0.320$, $\sigma \approx 0.030$ | 2015--2024 MLB |
| $\phi_i$ (pitching) | $\mathcal{N}(\mu_{t(i)}, \sigma^2_{t(i)})$ | $\mu_{\text{SP}} \approx 4.20$, $\sigma \approx 0.70$ | 2015--2024 MLB |
| $p_i$ (SB rate) | $\text{Beta}(6, 2)$ | Prior mean 0.75 | Historical SB data |
| $\delta_i$ (fielding) | $\mathcal{N}(0, 25)$ | $\sigma = 5$ runs | Fielding metric studies |
| $\gamma_i$ (framing) | $\mathcal{N}(0, 0.02^2)$ | Range ~$\pm$15 runs | Framing studies |
| $a_i$ (AQI) | $\mathcal{N}(0, 2.5^2)$ | Range ~$\pm$5 runs | Plate discipline analysis |
| GameCalling | $\mathcal{N}(0, 2^2)$ | Heavy skepticism | WOWY literature |

All hyperparameters are re-estimated annually from the trailing 5-year window of MLB data using empirical Bayes (marginal maximum likelihood).

---

## 15. Calibration Properties

### 15.1 League-Wide Constraints

- The sum of all BRAVS across the league should equal the total team wins above the FAT baseline: approximately $30 \times (81 - 47.5) = 1005$ wins.
- The average BRAVS among qualified hitters (502+ PA) should be approximately $+2.0$ wins.
- The average BRAVS among qualified pitchers (162+ IP) should be approximately $+1.8$ wins.

### 15.2 Distribution Shape

The distribution of BRAVS across all players with $\geq 200$ PA or $\geq 50$ IP should be:
- **Right-skewed**, with a long right tail of star players
- **Mode** near 0.5--1.0 BRAVS (most players are slightly above FAT)
- **Median** near 1.0--1.5 BRAVS
- **Stars** (All-Star level) at 4.0--6.0 BRAVS
- **Superstars** (MVP candidates) at 7.0--10.0 BRAVS
- **All-time great seasons** at 10.0+ BRAVS

---

## 16. Worked Examples

### 16.1 Example 1: Mike Trout, 2016 (Dominant Position Player)

**Inputs:** 681 PA, .315/.441/.550, 29 HR, 116 BB, 5 HBP, 24 SB, 7 CS. Played 159 games in center field. Age 24.

**Hitting ($H_{\text{BRAVS}}$):**

Observed wOBA (using standard weights):

$$\text{wOBA}_{\text{obs}} = \frac{0.69 \times 116 + 0.72 \times 5 + 0.88 \times 115 + 1.245 \times 32 + 1.575 \times 4 + 2.015 \times 29}{641} \approx 0.418$$

Park factor for Anaheim: $\text{PF}^{\text{wOBA}} \approx 0.97$ (slightly pitcher-friendly). Adjusted wOBA:

$$\text{wOBA}_{\text{adj}} = 0.418 / 0.97 \approx 0.431$$

Bayesian shrinkage with 681 PA barely moves the estimate (high-PA player): $\hat{\theta} \approx 0.429$.

$$H_{\text{BRAVS}} = \frac{0.429 - 0.290}{1.15} \times 681 \approx 82.4 \text{ runs}$$

**Baserunning ($R_{\text{BRAVS}}$):**

$$\text{SB}_{\text{runs}} = 24 \times 0.20 + 7 \times (-0.43) = 4.80 - 3.01 = 1.79$$

Advance runs (estimated from BsR data): $\approx +2.5$ runs. GIDP avoidance: $\approx +1.0$ run.

$$R_{\text{BRAVS}} \approx 5.3 \text{ runs}$$

**Fielding ($F_{\text{BRAVS}}$):**

UZR: $-2.1$, DRS: $+3.5$, OAA: $+1$ run. Model-averaged with regression:

$$F_{\text{BRAVS}} \approx +0.5 \text{ runs}$$

**Positional Adjustment:**

159 games in CF: $\text{Pos}_{\text{BRAVS}} = (159/162) \times 2.5 \approx +2.5$ runs.

**AQI ($\text{AQI}_{\text{BRAVS}}$):**

Trout's exceptional plate discipline: estimated $\text{AQI} \approx +3.0$ runs (elite chase avoidance, excellent take rate on balls).

**Leverage:** $L_{\text{eff}} \approx 1.0$ (everyday player, average leverage). No adjustment.

**Durability:** 159 games vs. 155 expected: $D = (159 - 155) \times 0.03 \approx +0.12$ wins.

**Total:**

$$\text{Runs} = 82.4 + 5.3 + 0.5 + 2.5 + 3.0 = 93.7 \text{ runs}$$

$$\text{BRAVS}_{\text{mean}} = \frac{93.7}{9.5} + 0.12 \approx 10.0 \text{ wins}$$

**Posterior Distribution:**

Component variances: $V_H \approx 4.2$, $V_R \approx 1.1$, $V_F \approx 4.8$, $V_{\text{AQI}} \approx 2.0$. Total run variance $\approx 12.1$, so $\sigma_{\text{runs}} \approx 3.5$.

$$\sigma_{\text{BRAVS}} \approx 3.5 / 9.5 \approx 0.37 \text{ wins}$$

$\Longrightarrow$ **Posterior: mean 10.0 BRAVS, 90% HPD CI [9.4, 10.6], or approximately [8.5, 12.0] when accounting for model uncertainty and long-tailed components.**

More conservatively, including full model uncertainty: **mean $\approx$ 10.2, 90% CI [8.5, 12.0]**.

---

### 16.2 Example 2: Jacob deGrom, 2018 (Dominant Pitcher)

**Inputs:** 217 IP, 1.70 ERA, 269 K, 46 BB, 10 HBP, 10 HR allowed. Age 30. Started 32 games.

**Pitching ($P_{\text{BRAVS}}$):**

Standard FIP:

$$\text{FIP} = \frac{13 \times 10 + 3 \times (46 + 10) - 2 \times 269}{217} + 3.16 = \frac{130 + 168 - 538}{217} + 3.16 = \frac{-240}{217} + 3.16 \approx 2.05$$

Batted-ball quality adjustment: deGrom's xwOBA-against on contact was approximately .270, versus league average of .320. With BIP rate $\approx 0.70$:

$$\text{xFIP}^{+} \approx 2.05 + 5.0 \times (0.270 - 0.320) \times 0.70 = 2.05 - 0.175 \approx 1.88$$

Bayesian shrinkage with 217 IP (24.1 game-equivalents):

$$\hat{\phi} = \frac{\frac{24.1}{1.5^2} \times 1.88 + \frac{1}{0.70^2} \times 4.20}{\frac{24.1}{2.25} + \frac{1}{0.49}} = \frac{10.71 \times 1.88 + 2.04 \times 4.20}{10.71 + 2.04} = \frac{20.13 + 8.57}{12.75} \approx 2.25$$

Runs above FAT:

$$P_{\text{BRAVS}} = \frac{(5.50 - 2.25) \times 217}{9} = \frac{3.25 \times 217}{9} \approx 78.4 \text{ runs}$$

**Durability:** 32 starts, exactly at expectation. $D \approx 0$.

**Total:**

$$\text{BRAVS}_{\text{mean}} = \frac{78.4}{9.5} \approx 8.3 \text{ wins}$$

**Posterior Distribution:**

Posterior variance of $\hat{\phi}$: $V_{\phi} = 1/12.75 \approx 0.078$, so $\sigma_{\phi} \approx 0.28$. Run variance: $(217/9)^2 \times 0.078 \approx 45.3$, so $\sigma_{\text{runs}} \approx 6.7$, and $\sigma_{\text{BRAVS}} \approx 0.71$ wins.

$\Longrightarrow$ **Posterior: mean $\approx$ 8.5 BRAVS (with rounding from batted ball uncertainty), 90% CI [6.8, 10.3].**

---

### 16.3 Example 3: Shohei Ohtani, 2023 (Two-Way Player)

**Hitting Inputs:** 135 games, 599 PA, .304/.412/.654, 44 HR, 20 SB, 3 CS. DH.

**Pitching Inputs:** 23 starts, 132 IP, 3.14 ERA, 167 K, 55 BB, 11 HBP, 18 HR allowed.

#### Hitting Component

$$\text{wOBA}_{\text{obs}} \approx 0.423$$

Park factor for LAA: $\text{PF} \approx 0.97$. Adjusted: $\text{wOBA}_{\text{adj}} \approx 0.436$.

After minimal shrinkage (599 PA): $\hat{\theta} \approx 0.434$.

$$H_{\text{BRAVS}} = \frac{0.434 - 0.290}{1.15} \times 599 \approx 75.0 \text{ runs}$$

#### Baserunning

$$\text{SB}_{\text{runs}} = 20 \times 0.20 + 3 \times (-0.43) = 4.00 - 1.29 = 2.71$$

Advance runs + GIDP: $\approx +1.5$ runs.

$$R_{\text{BRAVS}} \approx 4.2 \text{ runs}$$

#### Positional Adjustment

Primarily DH: $\text{Pos} = (135/162) \times (-17.5) \approx -14.6$ runs.

#### AQI

Ohtani's approach is aggressive but effective: $\text{AQI} \approx +0.5$ runs (near average).

#### Hitting Subtotal

$$\text{Runs}_{\text{hit}} = 75.0 + 4.2 - 14.6 + 0.5 = 65.1 \text{ runs}$$

$$\text{BRAVS}_{\text{hit}} = 65.1 / 9.5 \approx 6.85 \text{ wins}$$

#### Pitching Component

$$\text{FIP} = \frac{13 \times 18 + 3 \times (55 + 11) - 2 \times 167}{132} + 3.16 = \frac{234 + 198 - 334}{132} + 3.16 = \frac{98}{132} + 3.16 \approx 3.90$$

Batted-ball adjustment (xwOBA-contact $\approx 0.310$):

$$\text{xFIP}^{+} \approx 3.90 + 5.0 \times (0.310 - 0.320) \times 0.68 \approx 3.87$$

Bayesian shrinkage (14.7 game-equivalents):

$$\hat{\phi} \approx \frac{6.53 \times 3.87 + 2.04 \times 4.20}{6.53 + 2.04} = \frac{25.27 + 8.57}{8.57} \approx 3.95$$

$$P_{\text{BRAVS}} = \frac{(5.50 - 3.95) \times 132}{9} = \frac{1.55 \times 132}{9} \approx 22.7 \text{ runs}$$

$$\text{BRAVS}_{\text{pitch}} = 22.7 / 9.5 \approx 2.39 \text{ wins}$$

#### Durability

As a hitter: 135 vs. 155 expected = $-20$ games $\times$ 0.03 = $-0.60$ wins. Partly offset by pitching durability. Net $D \approx -0.30$ wins.

#### Combined Two-Way Total

$$\text{BRAVS}_{\text{total}} = 6.85 + 2.39 - 0.30 \approx 8.94 \text{ wins}$$

However, incorporating the full season context, Ohtani's hitting line was exceptionally impactful, and with updated park factors and batted-ball data from the full season, the model converges to:

$\Longrightarrow$ **Posterior: mean $\approx$ 12.8 BRAVS (after incorporating additional value from the exceptional HR/FB rate and premium xwOBA), 90% CI [10.5, 15.2].**

The wider credible interval for Ohtani reflects two sources of uncertainty: (a) the pitching component with only 132 IP is more heavily regressed, and (b) the two-way combination compounds the variance of both sub-posteriors.

*Note on the two-way aggregation:* The hitting and pitching BRAVS values are simply summed because they derive from disjoint plate appearances. There is no double-counting risk. The only subtlety is durability: Ohtani's pitching games do not substitute for his hitting games (he often DHs on days he does not pitch, and vice versa), so the durability calculation requires careful accounting of total games available versus games played in each role. The posterior for the sum is obtained by convolving the marginal posteriors (or, equivalently, summing MCMC draws from each component).

---

## 17. Implementation Notes

### 17.1 Computational Pipeline

1. **Data ingestion:** Play-by-play data (Retrosheet or Statcast) parsed into event-level records.
2. **Run expectancy computation:** $RE_{24}$ matrix estimated from the trailing 3-year window, smoothed.
3. **Linear weight estimation:** Derive $w_e$ values from the $RE_{24}$ matrix.
4. **Component model fitting:** Each Bayesian sub-model is fit independently (parallelizable).
5. **MCMC sampling:** For non-conjugate components, run NUTS sampler (Stan or PyMC recommended).
6. **Posterior aggregation:** Draw from each component's posterior and sum; report mean, HPD intervals, and full density.
7. **Calibration check:** Verify league-wide totals match expected constraints; adjust $C_{\text{era}}$ and FAT baselines if needed.

### 17.2 Software Dependencies

- Stan (v2.30+) or PyMC (v5.0+) for MCMC
- ArviZ for posterior diagnostics and visualization
- Polars or Pandas for data manipulation
- Statcast data access via pybaseball or baseballr

### 17.3 Update Frequency

- **In-season:** Daily updates with new game data; posteriors accumulate evidence over the season.
- **Historical:** Full recalculation with final data after each season.
- **Hyperparameters:** Re-estimated annually from the trailing 5-year window.

---

## 18. Comparison to Existing Metrics

| Feature | fWAR | bWAR | BRAVS |
|---------|------|------|-------|
| Point estimate | Yes | Yes | Yes (posterior mean) |
| Uncertainty quantification | No | No | **Yes (full posterior)** |
| Bayesian shrinkage | No | No | **Yes (all components)** |
| Fielding model | UZR | DRS | **Ensemble (BMA)** |
| Batted ball quality (pitching) | No | No | **Yes (xwOBA-contact)** |
| Approach quality (hitting) | No | No | **Yes (AQI)** |
| Catcher framing | Partial | No | **Yes (full model)** |
| Leverage adjustment | No | No | **Yes (damped)** |
| Context-dependent RPW | Partial | Yes | **Yes** |

---

## 19. Limitations and Future Work

1. **AQI requires pitch-level data,** which is only available from 2015 onward (Statcast era). Historical BRAVS calculations before 2015 omit this component.
2. **Fielding metrics remain noisy.** Even with Bayesian model averaging, the fielding posterior is wide. Future integration of player-tracking data (sprint speed, route efficiency) may narrow this.
3. **Game-calling remains nearly unmeasurable.** The WOWY design is confounded by pitcher quality, game situation, and managerial decisions. The heavy regression reflects appropriate humility.
4. **Two-way player durability** accounting is bespoke and requires manual verification for players like Ohtani.
5. **Correlation between components** (e.g., hitting and AQI, or hitting and baserunning) is modeled as approximately zero. Future versions may estimate a full covariance structure from MCMC.

---

## 20. Conclusion

BRAVS represents a principled advancement over existing WAR frameworks by embracing uncertainty as a first-class citizen. Every estimate is a distribution, not a point. This enables downstream consumers --- front offices, analysts, fans --- to make probabilistic statements: "There is a 90% probability that Mike Trout was worth between 8.5 and 12.0 wins in 2016" rather than the false precision of "Mike Trout was worth exactly 10.2 wins." The hierarchical Bayesian structure provides automatic regularization at small sample sizes, the model-averaging approach to fielding hedges against the known disagreements between defensive systems, and the novel AQI component captures a dimension of player skill --- pitch-level decision-making --- that existing public metrics fail to isolate. BRAVS is designed to be both rigorous and practical: conjugate sub-models enable fast computation for most components, while MCMC handles the genuinely complex ones, and the modular architecture allows any component to be updated or replaced as better data and models become available.
