"""2단 gate — 관할(Mahalanobis) + 성숙도(Beta-Bernoulli) + ACI 위험 통제.

정식화 (설계서 §2.5; Beta-Bernoulli·conformal 수학은 Paper 1 공유 — 인용 처리):
  1단 관할:   A_jur(x) = 1[ d_M(h(I₀); μ_k, Σ_k) ≤ q_{k,1−α_j} ],  α_j = 0.1
              μ_k, Σ_k = 클러스터 성공 에피소드 초기 특징의 표본 평균 + Ledoit–Wolf 수축 공분산
              q = calibration 분할 Mahalanobis 거리의 (1−α_j) 분위수
  2단 성숙도: A_mat(k) = 1[ Pr(s_k ≥ τ | 𝒟_k) ≥ 1−δ ],  s_k ~ Beta(1+σ_k, 1+φ_k),  (τ,δ)=(0.8,0.1)
  발화:       g(x) = Habit ⟺ A_jur ∧ A_mat
  갱신 재초기화: π_k 재학습 직후 Beta(1 + c·σ_k, 1 + c·φ_k), c = 0.25
  위험 통제:  Pr(fail|fire) ≤ ε = 0.2 를 ACI로 추적, 위반 시 τ 상향

E5 개정 (preregistration §4h — 통합 지시서, 연구원 승인 확정):
  lifecycle 4상태 = {미지, 기지-미성숙, 기지-성숙, 부적격(habit-ineligible)}.
  probe 리허설(P=20)은 정책 버전당 1라운드·총 2라운드 상한 — 2라운드 모두
  성숙 판정 미달 시 부적격 전이(E5 범위 내 VLA 고정, 재도전 없음).
  성숙 판정은 사후 확률 Pr(s ≥ τ | 이월 posterior + probe) ≥ 1−δ — 고정 통과선 없음.

  ★ A_mat 원장 분리 (§4h, 통합 지시서 §2): A_mat 사후(σ, φ)는 **습관 출처
  (probe + fire) 결과만** 산입한다. teacher/VLA 결과는 history(𝒟_k 기록·보고)에만
  남고 A_mat에 불산입 — 산입 시 n=80 재학습 이월만으로 probe 0회 무검증 성숙
  통과가 가능해 게이트가 s_H가 아닌 s_V를 측정하는 모델링 오류가 된다.
  원장 갱신은 source 태그(teacher|probe|fire) 필수; probe·fire는 ACI observe_fire와
  별개 계정(probe는 r_V·fired 집계 제외). 회귀 테스트: experiments/gate_regression.py.

모든 수치는 configs/preregistration.md §1이 원본이다.
"""
import numpy as np
from scipy.stats import beta as beta_dist

# preregistration §1 (하드코딩 아님 — 원본은 configs/preregistration.md; 불일치 시 그쪽이 이김)
ALPHA_J = 0.1
TAU = 0.8
DELTA = 0.1
EPSILON = 0.2
REINIT_C = 0.25
# ACI 자유 파라미터 — preregistration §4c 등재 (τ 궤적·E5 발화율에 실질 영향)
ACI_GAMMA = 0.02
ACI_TAU_MAX = 0.99


class JurisdictionGate:
    """클러스터별 관할 판정 (1단)."""

    def __init__(self, alpha_j=ALPHA_J):
        self.alpha_j = alpha_j
        self.mu = None
        self.prec = None  # Σ^{-1}
        self.q = None

    MIN_N = 20  # fit 10 + calib 10 최소 (소표본 크래시/쓰레기 정밀도 방지 — 검증 워크플로우 실측)

    def fit(self, feats, calib_feats=None, split_seed=0):
        """feats: (N, d) 성공 에피소드 초기 특징. calib_feats 미지정 시 feats를 분할.

        - 분할은 **결정적 셔플 후 반반** — 짝/홀 분할은 수집 스트림의 base_idx 순환(40 = 짝수)과
          parity가 정렬되어 fit/calib이 서로 다른 초기상태 모드만 보는 계통 편향 발생 (검증 발견).
        - 분위수는 split conformal 유한표본 보정: ⌈(n+1)(1−α)⌉번째 순서통계량 (method='higher').
        """
        from sklearn.covariance import LedoitWolf

        feats = np.asarray(feats)
        if calib_feats is None:
            if len(feats) < self.MIN_N:
                raise ValueError(f"JurisdictionGate needs >= {self.MIN_N} feats, got {len(feats)}")
            perm = np.random.default_rng(split_seed).permutation(len(feats))
            half = len(feats) // 2
            fit_f, calib_feats = feats[perm[:half]], feats[perm[half:]]
        else:
            fit_f = feats
            calib_feats = np.asarray(calib_feats)
            if len(fit_f) < 10 or len(calib_feats) < 10:
                raise ValueError("need >= 10 fit and >= 10 calibration samples")
        self.mu = fit_f.mean(0)
        lw = LedoitWolf().fit(fit_f)
        self.prec = np.linalg.inv(lw.covariance_)
        d = np.sort(self._dist(np.asarray(calib_feats)))
        n = len(d)
        k = min(int(np.ceil((n + 1) * (1 - self.alpha_j))), n)  # 유한표본 보정 순서통계량
        self.q = float(d[k - 1])
        return self

    def _dist(self, X):
        diff = X - self.mu
        return np.sqrt(np.einsum("ni,ij,nj->n", diff, self.prec, diff))

    def accepts(self, feat):
        return bool(self._dist(np.asarray(feat)[None])[0] <= self.q)

    def score(self, feat):
        """거리 (낮을수록 관할 내) — 분석·AUC용."""
        return float(self._dist(np.asarray(feat)[None])[0])


class MaturityGate:
    """클러스터별 성숙도 판정 (2단) — Beta-Bernoulli (Paper 1 수학 이월)."""

    PROBE_MAX_ROUNDS = 2  # preregistration §4h: 1차 n=20 학습 후 + 2차 n=80 재학습 후
    SOURCES = ("teacher", "probe", "fire")
    A_MAT_SOURCES = ("probe", "fire")  # A_mat 사후에 산입되는 습관 출처 (§4h 장부 분리)

    def __init__(self, tau=TAU, delta=DELTA):
        self.tau = tau
        self.delta = delta
        self.succ = 0            # A_mat 계수 — 습관 출처(probe+fire)만 (§4h 장부 분리)
        self.fail = 0
        self.history = []        # [(source, success)] — 𝒟_k 기록·probe 이력 전문 보고용 (§4h)
        self.probe_rounds = 0    # 소진한 probe 라운드 수 (정책 버전당 1, 총 2 상한)
        self.ineligible = False  # 습관 부적격 — E5 범위 내 VLA 고정 (§4h)

    def update(self, success, source):
        """𝒟_k 원장 갱신. source ∈ {teacher, probe, fire} 필수 (§4h·통합 지시서 §2).

        A_mat 계수(succ/fail)에는 **습관 출처(probe·fire)만** 산입 — teacher 결과는
        history(기록·보고)에만 남는다. probe 결과는 ACI observe_fire·r_V 계정에
        절대 넣지 않는다(발화 시 driver가 fire로 update + observe_fire를 별도 호출)."""
        if source not in self.SOURCES:
            raise ValueError(f"unknown ledger source: {source!r} (must be teacher|probe|fire)")
        if source in self.A_MAT_SOURCES:
            if success:
                self.succ += 1
            else:
                self.fail += 1
        self.history.append((source, bool(success)))

    def record_probe_round(self, outcomes):
        """probe 리허설 1라운드(P=20) 기록 + 성숙 판정 (§4h).

        재학습 → reinit_after_retrain() → 본 메서드 순서가 전제 (이월 posterior 위에
        probe 결과 누적 — 고정 통과선 없음). 반환 = 성숙 여부. 미달이 총 2라운드에
        도달하면 습관 부적격으로 전이한다 (재도전 없음 — future work)."""
        if self.ineligible:
            raise RuntimeError("cluster is habit-ineligible; no further probes (§4h)")
        for s in outcomes:
            self.update(s, source="probe")
        self.probe_rounds += 1
        if self.accepts():
            return True
        if self.probe_rounds >= self.PROBE_MAX_ROUNDS:
            self.ineligible = True
        return False

    def prob_ge_tau(self):
        # Pr(s ≥ τ | D) = 1 − CDF_Beta(τ; 1+σ, 1+φ)
        return float(1.0 - beta_dist.cdf(self.tau, 1 + self.succ, 1 + self.fail))

    def accepts(self):
        return self.prob_ge_tau() >= 1 - self.delta

    def reinit_after_retrain(self, c=REINIT_C):
        """정책 갱신 시 사후 재초기화: Beta(1+c·σ, 1+c·φ) — 구 증거를 약한 prior로 이월.

        σ·φ는 A_mat 계수(습관 출처만)이므로, 습관 이력이 없는 첫 재학습 직후에는
        정확히 Beta(1,1)에서 probe가 시작된다 (§4h)."""
        self.succ = c * self.succ
        self.fail = c * self.fail


class ACIRiskController:
    """발화 실패율 Pr(fail|fire) ≤ ε 추적 (adaptive conformal; Paper 1 C4 이월).

    위반 시 성숙도 τ를 상향해 발화를 보수화. 갱신식(ACI 표준형):
      τ ← clip(τ + γ·(loss − ε)),  loss = 1[발화했는데 실패]  (발화 시에만 갱신)
    """

    def __init__(self, epsilon=EPSILON, gamma=ACI_GAMMA, tau_min=TAU, tau_max=ACI_TAU_MAX):
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.fired = 0
        self.fired_fail = 0

    def observe_fire(self, success, maturity_gate):
        self.fired += 1
        loss = 0.0 if success else 1.0
        if not success:
            self.fired_fail += 1
        new_tau = maturity_gate.tau + self.gamma * (loss - self.epsilon)
        maturity_gate.tau = float(np.clip(new_tau, self.tau_min, self.tau_max))

    def empirical_risk(self):
        return self.fired_fail / self.fired if self.fired else 0.0


class TwoStageGate:
    """발화 규칙 g(x) + lifecycle 4상태 (설계서 §2.5 + preregistration §4h 개정)."""

    def __init__(self):
        self.jurisdiction = {}  # cluster_id -> JurisdictionGate
        self.maturity = {}      # cluster_id -> MaturityGate
        self.aci = {}           # cluster_id -> ACIRiskController

    def ensure_cluster(self, cid):
        if cid not in self.maturity:
            self.maturity[cid] = MaturityGate()
            self.aci[cid] = ACIRiskController()
            return True  # 신설 (미지 → 기지)
        return False

    def decide(self, cid, feat):
        """반환: ("habit"|"vla", 사유). lifecycle 4상태: 미지/기지-미성숙/기지-성숙/부적격.

        사유 구분: "jurisdiction_unfit"(게이트 미적합 — 표본 부족)과
        "out_of_jurisdiction"(실제 Mahalanobis 기각)을 분리 — E5 관할 오류 양방향 회계에서
        미적합 기각이 false reject로 오집계되는 것을 방지 (검증 발견).
        "habit_ineligible" = probe 2라운드 소진 클러스터의 VLA 고정 (§4h)."""
        if cid not in self.maturity:
            return "vla", "unknown_cluster"
        mat = self.maturity[cid]
        if mat.ineligible:
            return "vla", "habit_ineligible"
        if not mat.accepts():
            return "vla", "immature"
        jur = self.jurisdiction.get(cid)
        if jur is None:
            return "vla", "jurisdiction_unfit"
        if not jur.accepts(feat):
            return "vla", "out_of_jurisdiction"
        return "habit", "fire"
