# HabitVLA-2 v10 → v11 원고 수정 패치 (2026-08-28)

> 근거: 검증 5건 실측 확정(2026-08-28, tmux `paper1`) + 전 리뷰(v10) 승인 항목.
> 원칙: 수동 숫자 입력 제로 유지. 수치 변경은 **P12 단 1건**(434→431 s, seed 0 단독값의 3-seed 정정 →
> 매크로 소스 교체로만 적용). 0.5945·73.47%는 기존 수치 유지(사용자·검증 확정). 나머지는 문장 의미·수식 표기·정의 명시에 한정.
> 적용 순서: **본문 수정 → `build_numbers.py`/`verify_numbers.py` 재실행 → [16] 제거·재배열 →
> RAS conference format 재컴파일 → 페이지·플로트 확인 → preflight.**

> ⚠ **실험 환경 검증 결과 이 패치보다 우선하는 정정이 있음** — `00_INSTRUCTION.md` §A 참조.
> 요약: (1) P3의 0.5945는 실측 0.594444로 **0.5944** 정정 필요 (2) P14 신규 항목(181.8 s가 E5 조건 아님).

---

## A. 검증으로 확정된 수정 (P1–P5)

### P1 [V-C] ever-matured 문장 의미 수정 — 검증 1 확정
수치 20.3 ± 2.1은 정확(ever-matured = 22/18/21). 문장의 "ended mature"가 오류.

**현재:**
> With 50.7±0.6 retrainings per seed, 20.3 ± 2.1 of 33 clusters ended mature; round-1/2 probe pass rates were 0.3838/0.5262, and the median exposure count of clusters reaching maturity was 22.7±0.6.

**수정:**
> With 50.7±0.6 retrainings per seed, 20.3 ± 2.1 of 33 clusters reached maturity at some point during the stream (final mature counts per seed are given in Table II); round-1/2 probe pass rates were 0.3838/0.5262, and the median exposure count of clusters reaching maturity was 22.7±0.6.

- 괄호 구절은 Table II(M/I/X = 최종 상태)와의 역할 분리를 명시해 리뷰어 교차 대조를 유도.
- 매크로 교체 불필요 — `numbers.json`의 해당 매크로가 ever-matured 산출인지 출처 필드로 재확인만.

### P2 [III-B·III-C, Eq. (4)–(5), Fig. 1] τ per-cluster 수식 수정 — 검증 2 확정 ★핵심
코드 실측: `ClusterState`마다 `ACIRiskController` 독립 보유, 타 클러스터 τ 변화 0건.
리뷰의 "global 공유 문장 추가" 권고는 **폐기**하고 수식을 실제 구현에 맞춘다.

**Eq. (4) 현재:**
$$A_{\mathrm{mat}}(k) = \mathbf{1}\!\left[\Pr(s_k \ge \tau_t \mid \mathcal{D}_k) \ge 1-\delta\right]$$

**Eq. (4) 수정:**
$$A_{\mathrm{mat}}(k) = \mathbf{1}\!\left[\Pr(s_k \ge \tau_k \mid \mathcal{D}_k) \ge 1-\delta\right]$$

**Eq. (5) 현재:**
$$\tau_{t+1} = \operatorname{clip}\!\left(\tau_t + \gamma\,(y_t - \varepsilon),\ \tau_{\min},\ \tau_{\max}\right)$$

**Eq. (5) 수정:**
$$\tau_{k,\,j+1} = \operatorname{clip}\!\left[\tau_{k,\,j} + \gamma\,(y_{k,\,j} - \varepsilon),\ \tau_{\min},\ \tau_{\max}\right]$$

**Eq. (5) 도입 문장 현재:**
> We continuously track the conditional failure of real habit firings and adjust the success threshold τ_t according to an allowed failure rate ε. With firing-failure indicator y_t ∈ {0,1}, the ACI-inspired update [26] is

**수정:**
> We continuously track the conditional failure of real habit firings and adjust each cluster's success threshold $\tau_k$ according to an allowed failure rate $\varepsilon$. With $y_{k,j} \in \{0,1\}$ the failure indicator of cluster $k$'s $j$-th habit firing, the ACI-inspired update [26] is

**Eq. (5) 직후 추가 문장 (설계 강조로 삽입):**
> The threshold is maintained per cluster, so realized failures of one habit tighten only that habit's certification criterion, keeping risk control local to the responsible policy.

**연쇄 교체 (III-C 본문 전체):**
- "A failure raises $\tau_t$, making subsequent certification stricter" → "A failure raises $\tau_k$, ..."
- "the threshold never falls below its initial value" → 유지 (τ_min = τ_0 = 0.8이므로 그대로 성립)
- "(τ₀, δ) = (0.8, 0.1)" (III-B) → "(each cluster's initial threshold and confidence are set to $(\tau_{k,0},\delta)=(0.8,0.1)$)" 또는 최소 수정으로 τ₀ 유지 + "each cluster is initialized at τ_{k,0}=τ_0" 한 구절.
- **Fig. 1 (TikZ)**: 게이트 박스 "maturity gate A_mat(k), Pr(s_k ≥ τ | D_k) ≥ 1−δ" → "Pr(s_k ≥ τ_k | D_k) ≥ 1−δ". 점선 라벨 "gate update: ACI on fired outcomes" → "gate update: per-cluster ACI on fired outcomes" (공간 부족 시 캡션에서 처리).

### P3 [V-E] cold-start 라우팅 수식어 수정 — 검증 3 확정
실측: full-stream episode-weighted에서 일치.
⚠ **0.5945 → 0.5944 정정 필요** (실험 환경 재검증: pooled 6420/10800 = 0.594444, seed별 평균도 동일).

**현재:**
> Late-stream routing remained higher for the cold-start clusters (0.8017 versus 0.5945), consistent with the greater teacher reliance expected before repeated experience accumulates; because that pool comes from a single suite, the contrast is descriptive and does not isolate novelty from suite identity.

**수정:**
> Over the full stream, routing remained higher for the cold-start clusters (0.8017 versus 0.5944), consistent with the greater teacher reliance expected before repeated experience accumulates; because that pool comes from a single suite, the contrast is descriptive and does not isolate novelty from suite identity.

### P4 [III-D] split-conformal 분리 보정 명시 — 검증 4 (수정 불필요, 선제 방어 1문장)
코드 확인: 결정적 셔플 후 반반 분할, μ·Σ는 fit 절반 / 분위수는 disjoint calibration 절반, ⌈(n+1)(1−α)⌉ 유한표본 보정. 명칭·인용([31][32]) 유지.

**Eq. (7) 도입부 현재:**
> The familiarity decision is defined by comparison with a split-conformal quantile q_{k,0.9} [31], [32], calibrated to cover 90% of successful scenes:

**수정:**
> The familiarity decision is defined by comparison with a split-conformal quantile $q_{k,0.9}$ [31], [32]: a disjoint calibration subset of successful scenes, held out from the mean–covariance fit, is used to compute the finite-sample-corrected quantile covering 90% of successful scenes:

### P5 [V-B, IV-B, Eq. (8)] 통계량 정의 + 절단 모집단 명시 — 검증 5 확정
정의 확인: censored ranks 위 $SS_{\mathrm{between}}/SS_{\mathrm{total}}$ (순위 기반 $\eta^2$). 절단 1개 = long 스위트, 22셀 밖.

**Eq. (8) 현재:**
$$N^{\star} = \min\{n : \hat{s}(n) \ge 0.8\}$$

**Eq. (8) 수정:**
$$N^{\star} = \min\left\{\, n \in \{10, 20, 40, 80\} : \hat{s}(n) \ge 0.8 \,\right\}$$

**Eq. (8) 직후 추가:**
> A cluster that fails the criterion at $n=80$ is recorded as right-censored ($N^{\star}>80$) and treated as tied at the top rank in the rank-based analyses below.

**V-B between-category share 현재:**
> In Fig. 2(b), the between-category share of N⋆ variation is only 0.0246 and category differences are not significant (H = 0.5176, p = 0.772).

**수정:**
> In Fig. 2(b), the between-category share of $N^{\star}$ variation—defined as $SS_{\mathrm{between}}/SS_{\mathrm{total}}$ on the censored ranks, a rank-based $\eta^2$—is only 0.0246, and category differences are not significant (Kruskal–Wallis $H = 0.5176$, $p = 0.772$).

**V-A 절단 클러스터 모집단 명시 — 현재:**
> Median N⋆ is 10 for object and 20 for goal/spatial, and only one cluster fails to reach the criterion by n = 80.

**수정:**
> Median $N^{\star}$ is 10 for object and 20 for goal/spatial, and only one cluster—a long-suite cluster, hence outside the 22-cell category analysis of Fig. 2(b)—fails to reach the criterion by $n = 80$.

---

## B. 인덱스 표기 정비 (리뷰 §2, τ 수정과 함께 일괄)

에피소드 $e$ / 에피소드 내 제어 스텝 $t$ / 클러스터별 발화 위험 갱신 $j$ 분리. Eq. (5)는 P2에서 처리 완료.

### P6 [III-A, Eq. (1)–(2)]
**현재:**
> Episode t presents a situation consisting of an initial observation I₀ and a language instruction ℓ_t: x_t = (I₀, ℓ_t) (1)

**수정:**
> Episode $e$ presents a situation consisting of an initial observation $I_{e,0}$ and a language instruction $\ell_e$:
> $$x_e = (I_{e,0}, \ell_e) \tag{1}$$

**Eq. (2) 현재:** $a_{t:t+K-1}$, 조건 $A_{\mathrm{mat}}(k(x_t))=1$, $\pi_V(o_t,\ell_t)$

**수정:**
$$a_{e,\,t:t+K-1} = \begin{cases} \pi_H^{k(x_e)}(o_{e,t}), & A_{\mathrm{mat}}(k(x_e)) = 1,\\[2pt] \pi_V(o_{e,t}, \ell_e), & \text{otherwise}, \end{cases} \tag{2}$$

> where $o_{e,t}$ is the in-flight observation at requery step $t$ of episode $e$.

**연쇄:** "A deterministic clustering function k(x_t)" → $k(x_e)$; III-B "(o_t, a_{t:t+K−1})" → $(o_{e,t}, a_{e,t:t+K-1})$; Fig. 1 TikZ "situation x_t = (I₀, ℓ)" → "situation $x_e=(I_{e,0},\ell_e)$".

---

## C. 통계 서술·용어 수정 (리뷰 승인 항목)

### P7 [III-B] "counts" → "effective evidence" + 재인증 문장
**현재:**
> Let σ_k and ϕ_k be the habit's success and failure counts in D_k and s_k the true success rate.

**수정:**
> Let $\sigma_k$ and $\phi_k$ denote the effective success and failure evidence maintained in the habit's ledger $\mathcal{D}_k$—before any retraining these coincide with the raw counts—and $s_k$ the true success rate.

**감쇠 문장 현재:**
> To avoid discarding history entirely, the ledger's sufficient statistics are attenuated by c = 0.25 immediately after retraining and carried over as a weak prior.

**수정:**
> To avoid discarding history entirely, evidence accumulated under the previous policy is discounted by $c = 0.25$ immediately after retraining and retained as weak prior evidence for the updated policy.

**III-C 말미 현재:**
> Teacher successes create policies, but firing authority must always be re-earned by the new policy's own outcomes.

**수정:**
> Teacher successes create policies, but after retraining, firing authority is re-certified from the discounted carry-over evidence together with the new policy's own probe outcomes.

### P8 [IV-C] 비열등성 표현 + estimand 확정 + mean ± s.d. 정의
**현재:**
> success uses B = 10,000 paired bootstrap with a −3 percentage-point non-inferiority margin, pre-specified as the maximum acceptable absolute degradation in task success; the risk budget is ε = 0.2.

**수정:**
> success uses $B = 10{,}000$ paired bootstrap with a $-3$ percentage-point non-inferiority margin, pre-specified as the maximum acceptable decrease in task success; non-inferiority is evaluated on habit-fired episodes for which paired teacher replays are available; the risk budget is $\varepsilon = 0.2$. Unless otherwise stated, values reported as mean $\pm$ s.d. are computed across the three seeds.

### P9 [V-B] 검정 명명 + horizon 서술 완화
**현재:**
> Horizon, however, acts as a constraint: ŝ(80) drops from 0.9750 for one-stage to 0.8833 for three-stage tasks, though not significantly (p = 0.0968); of the two controlled two-stage chains, one is significantly below the independence baseline—the square of its single-stage success rate, 0.96² = 0.9216 (0.80, 40/50, p = 0.0049) while the other composes without loss (0.98, 49/50).

**수정:**
> Longer horizon, however, showed a downward trend in habit success: $\hat{s}(80)$ drops from 0.9750 for one-stage to 0.8833 for three-stage tasks, though not significantly ($p = 0.0968$, one-sided Fisher exact test); of the two controlled two-stage chains, one achieved $40/50 = 0.80$, significantly below the independence baseline—the square of its single-stage success rate, $0.96^2 = 0.9216$ (one-sided exact binomial $p = 0.0049$)—whereas the other composed without loss ($49/50 = 0.98$).

※ 검정명 확정 (2026-08-28 코드 실측): `fisher_exact([[k_lo, n_lo−k_lo],[k_hi, n_hi−k_hi]], alternative="less")` → one-sided Fisher exact test. §H의 미해결 항목 종결.

### P10 [V-C] "not merely nominal"
**현재:** "Risk control was not merely nominal: across the three seeds it demoted 11 mature habits, 5 of which later re-matured."
**수정:** "The adaptive controller was exercised in practice: across the three seeds it demoted 11 mature habits, 5 of which later re-matured."

---

## D. 문장·단어 교체 (리뷰 §15·26–29 승인분)

| 위치 | 현재 | 수정 |
|---|---|---|
| Abstract | provide broad manipulation generalization | provide broad generalization across manipulation tasks |
| Abstract | repetitive situations they have already mastered | repetitive situations they already solve reliably |
| Abstract | distills the successful trajectories ... into per-cluster lightweight visuomotor policies | converts the successful trajectories ... into per-cluster lightweight visuomotor policies |
| Abstract | while task success remained non-inferior to paired full-VLA replay | while success on habit-fired episodes remained non-inferior to paired full-VLA replay |
| I | comes with a fixed inference cost. A large backbone must be executed for every action decision, and this cost remains essentially unchanged even when ... | comes with a substantial recurring inference cost. In standard VLA deployment, the large backbone is repeatedly invoked at each requery, even when the robot revisits tasks it has already solved many times. |
| I | In other words, current VLAs learn what to do from experience, but they do not learn when the large model no longer needs to be called during deployment. | In other words, standard VLA deployment does not exploit repeated successful experience to learn when the large model need no longer be invoked. |
| I | a computational resource that can be progressively reduced | a computational resource whose use can be progressively reduced |
| I | its success-rate Beta–Bernoulli posterior | a Beta–Bernoulli posterior over the habit's success rate |
| I (기여 3) | we evaluate habit formation and online inference amortization over 27 formation clusters and three seeds (12,000 online episodes) | we evaluate batch habit formation over 27 clusters and online inference amortization over three 4,000-episode deployment streams (12,000 episodes in total) |
| II-A | have broadened generalist manipulation policies | have advanced generalist manipulation policies |
| II-B | yielding a hierarchy formed by deployment experience rather than fixed before deployment | yielding an experience-induced hierarchy rather than one fixed before deployment |
| II-C | adaptive conformal inference tracks long-run risk under distribution shift [26] | adaptive conformal inference adjusts miscoverage online under distribution shift [26] |
| III-B | Training is lazy: a first habit is trained when \|B_k\| = 20 | Training is event-triggered: a first habit is trained when $\|\mathcal{B}_k\| = 20$ |
| III-B | A trained policy does not receive control by virtue of existing. | Training alone does not grant control authority. |
| III(도입)·Fig.1 캡션 | probe and real firing outcomes | probe and habit-execution outcomes |
| IV-C | compared to separate representation from score-aggregation bottlenecks | compared to disentangle representation quality from score-aggregation effects |
| V-A | The slowest pilot cluster rises | The slowest-forming pilot cluster rises |
| V-D | while in the collapse regime at w = 0.08 m | while in the largest-perturbation regime at $w = 0.08$ m |
| V-D (Fig.4 캡션 포함) | under-rejects in the collapse regime | under-rejects in the largest-perturbation regime |
| V-D | Fig. 4(b) indicates a representation rather than an aggregation bottleneck. | Fig. 4(b) suggests a representation bottleneck more than an aggregation bottleneck. |
| V-D | In this closed workspace, delegation was carried by ... | In this closed-set benchmark setting, delegation was carried by ... |

### P12 [V-F] 형성 이벤트당 시간 정정 — ★ 유일한 수치 변경 (전 검증 발견)
원고 434 s는 seed 0 단독값(433.9)의 반올림. 3-seed 집계는 seed별 평균 431.1 / 총합·총횟수 431.2.
같은 문장의 50.7이 3-seed 평균이므로 434만 seed 0 기준이면 단위 혼합 → 전 검증 패턴("한 시드 사실의 일반화") 재발.

**현재:**
> Formation time covers each retraining event end to end—habit training, the P = 20 off-stream certification probes, and data preparation—averaging about 434 s over 50.7 events; training alone is far cheaper, about 181.8 s for a warm-started n=40 fit (≈2,137 VLA-call equivalents).

**수정:**
> Formation time covers each retraining event end to end—habit training, the $P = 20$ off-stream certification probes, and data preparation—averaging about 431 s over 50.7 events; training alone is far cheaper, about 181.8 s for a warm-started $n{=}40$ fit ($\approx$2,137 VLA-call equivalents).

**⚠ 적용 방법 주의:** 본문 직접 수정 금지 (수동 숫자 제로 원칙). `build_numbers.py`에서 해당 매크로의 집계 소스를 seed 0 원장 → 3-seed 평균으로 교체 후 재생성. `numbers.json` 출처 필드로 seed 0 단독 산출이었는지 함께 확인. 181.8 s와 2,137도 같은 문장 → 동일 소스 여부 점검 필요.

> ⚠ **실험 환경 후속 검증**: 181.8 s / 2,137은 E1 anchor5 = **E2 배치 8,000스텝 warm-start n=40 fit**으로
> E5(B-2 scratch, 10,000/28,000 스텝) 조건이 아님이 확인됨. 실측은 이벤트 431.2 s 중 **학습이 369.9 s(86%)**,
> VLA-호출 등가 4,348(학습만)/5,068(이벤트 전체). → `00_INSTRUCTION.md` §A-3 (P14) 참조, 판정 필요.

### P13 [V-F] 73.47% 집계 방식 확정 구절 (수치 유지)
73.47% = seed별 비율(0.7277, 0.7643, 0.7120)의 비가중 평균으로 확정 → 원고가 맞음. 다만 pooled 집계(73.66%)와 0.2%p 갈리므로 한 구절로 방어.

**현재:**
> and 73.47% of late VLA traffic flows through 7–10 X clusters per seed.

**수정:**
> and 73.47% of late VLA traffic (unweighted mean of per-seed shares) flows through 7–10 X clusters per seed.

### P11 [VI] 결론 마지막 문장 (두 문장 분리안 채택)
**현재:**
> Total-cost break-even including formation, open-world recurrence recognition, long-horizon hierarchies, and real-robot validation remain as future work.

**수정:**
> We do not claim total-cost break-even once habit-formation cost is included. Open-world recurrence recognition, long-horizon hierarchies, and real-robot validation remain important directions for future work.

---

## E. 참고문헌 (리뷰 §19–23 승인분)

| Ref | 수정 |
|---|---|
| [5] | Y. Chen et al., "RoboRouter: ..." (7인 이상 → et al.) |
| [8] | ... in Proc. 8th Conf. Robot Learning (CoRL), PMLR, vol. 270, pp. 2679–2713, **2025**. |
| [23] | J. Zhao et al., "Retrieve-then-Steer: ..." |
| [24] | Y. Wang et al., "Learning while deploying: ..." |
| [26] | I. Gibbs and E. J. Candès, "Adaptive conformal inference under distribution shift," in Advances in Neural Information Processing Systems, vol. 34, pp. 1660–1672, 2021. (LaTeX: `Cand\`es`) |
| [16] | **제거** (Viola–Jones): II-B를 "Knowledge distillation [15] and speculative decoding [17] move expensive computation onto cheaper paths, ..."로. **모든 본문 수정·검증 완료 후 마지막에** 번호 재배열과 함께 수행. |

---

## F. 포맷·파이프라인

1. "Manuscript received August 27, 2026." **제거** (초기 투고본).
2. 본문 수정 완료 후 `build_numbers.py` → `verify_numbers.py` 재실행 (수치 변경 1건이므로 전건 통과 확인이 목적).
3. [16] 제거 + 참고문헌 재배열 → 재검증.
4. RAS conference format(`ieeeconf`) 재컴파일 → 페이지 수·Fig 2/3/4/5 배치 재확인 → preflight.

---

## G. 한국어 원고 전환 체크리스트 (`HabitVLA2_manuscript_ko.md` + `ko_template.md`)

- [ ] P2: Eq. (4)–(5) $\tau_k$·$y_{k,j}$ 표기 + per-cluster 국소성 문장 번역 반영
- [ ] P6: 인덱스 $e/t/j$ 분리 (Eq. 1–2)
- [ ] P1: "성숙 상태로 종료" → "스트림 중 한 번이라도 성숙 도달" 의미 수정
- [ ] P3: "후반 스트림" → "전체 스트림" 수식어 수정 + 0.5944 반영
- [ ] P5: Eq. (8) 후보 집합·우측절단 처리 + 순위 기반 $\eta^2$ 정의 + long 스위트 모집단 구절
- [ ] P4·P7·P8·P9·P10·P11: 해당 문장 한국어 문체로 반영 (직역 금지 — 독립 작성 원칙)
- [ ] P12: 431 s 매크로 재생성분이 `render_ko.py` 토큰 치환으로 자동 전파되는지 확인 (수동 수정 금지)
- [ ] P13: 73.47% 확정 구절("seed별 몫의 비가중 평균") 반영
- [ ] `render_ko.py` 재실행 → `verify_ko.py` 전건 통과 확인

---

## H. 미해결 확인 사항 — 전건 종결 (2026-08-28)

- ~~P9의 $p=0.0968$ 검정명~~ → **종결**: one-sided Fisher exact test (`fisher_less`) 코드 실측 확정, P9 반영 완료.
- 원고 수치 전수 검증 결과: 12개 항목 중 11건 일치, 1건 정정(P12: 434→431 s), 1건 확정 구절 추가(P13: 73.47%).
- P12 적용 시 `build_numbers.py` 매크로 소스 교체 → 재생성 → `verify_numbers.py` 전건 통과가 절대적 선결 조건.

> ⚠ **실험 환경 후속 검증(2026-08-28)에서 추가된 항목**: P3의 0.5945 → 0.5944 정정, P14(181.8 s 조건 불일치).
> 상세는 `00_INSTRUCTION.md` §A. 두 건은 **연구원 판정 후** 적용.
