"""수집 스트림·held-out 평가 세트의 결정적 명세 생성기.

설계 근거 (설계서 §4.1–§4.2 + E0-6 실측):
  - 클러스터 = (스위트, 태스크). 클러스터 내 변이 = (env seed, 공식 init state 인덱스,
    섭동 폭 w, 노이즈 seed) 4원소 — 전부 결정적.
  - **대역 분리 (4중 disjoint — 수집 / held-out / novel / probe):**
      (1) base_idx: 수집 = 0..39, held-out = 40..49  (미본 초기상태 평가 — 가장 강한 분리)
          novel·probe는 base 신규 대역이 없어(공식 init state 50개 전부 배정) 0..39를 재사용하고
          seed·noise 대역으로 분리한다 — held-out(40..49)의 미노출성 보존 (§4b).
      (2) seed 대역: 수집 = 10000+i, held-out = 20000+j, novel = 30000+j,
          probe = 40000+1000·r+j (r = probe 라운드), novel-2(타 태스크 차용, 예약) = 50000+j
      (3) noise_seed 대역: 수집 = i, held-out = 1e6+j, novel = 2e6+j,
          probe = 3e6+1000·r+j, novel-2 예약 = 4e6+j
          (연쇄 래퍼 relocate noise 5e5+/1.5e6+는 별도 필드 — 값 공간 구분 유지)
  - **w_id (in-distribution 변이 폭) = 0.01 m** — 전 스위트 가용 폭(≥0.02) 내부.
    novel(E4)은 w > w_id ~ usable_w_max 대역과 타 태스크 초기상태 차용으로 생성.

★ 본 파일의 수치(w_id=0.01, base_idx 분할 40/10, seed·noise 대역)는 사전등록 대상 —
  configs/preregistration.md §4b에 등재. 변경 시 반드시 §5 이력 기록.
"""
from .libero_env import EpisodeSpec, USABLE_W_MAX

W_ID = 0.01               # in-distribution 섭동 폭 (전 스위트 가용)
COLLECT_BASE_RANGE = range(0, 40)   # 공식 init state 50개 중 수집용 40
HELDOUT_BASE_RANGE = range(40, 50)  # held-out 전용 10
COLLECT_SEED_BASE = 10_000
HELDOUT_SEED_BASE = 20_000
NOVEL_SEED_BASE = 30_000
PROBE_SEED_BASE = 40_000            # E5 probe 리허설 (§4h) — 라운드당 +1000
NOVEL2_SEED_BASE = 50_000           # E4 타-태스크 init 차용 novel (예약)
HELDOUT_NOISE_BASE = 1_000_000
NOVEL_NOISE_BASE = 2_000_000
PROBE_NOISE_BASE = 3_000_000        # 라운드당 +1000
NOVEL2_NOISE_BASE = 4_000_000       # 예약


def collection_specs(suite_name, task_id, n_episodes=120):
    """클러스터당 teacher 수집 스트림 (preregistration §1: 120 ep)."""
    specs = []
    bases = list(COLLECT_BASE_RANGE)
    for i in range(n_episodes):
        specs.append(
            EpisodeSpec(
                suite_name,
                task_id,
                seed=COLLECT_SEED_BASE + i,
                base_idx=bases[i % len(bases)],
                w=W_ID,
                noise_seed=i,
            )
        )
    return specs


def heldout_specs(suite_name, task_id, n_episodes):
    """held-out 고정 평가 세트 (E2=50, E3=20). 체크포인트 간 동일 → paired 비교."""
    specs = []
    bases = list(HELDOUT_BASE_RANGE)
    for j in range(n_episodes):
        specs.append(
            EpisodeSpec(
                suite_name,
                task_id,
                seed=HELDOUT_SEED_BASE + j,
                base_idx=bases[j % len(bases)],
                w=W_ID,
                noise_seed=HELDOUT_NOISE_BASE + j,
            )
        )
    return specs


def probe_specs(suite_name, task_id, round_idx, n_episodes=20):
    """E5 probe 리허설 세트 (preregistration §4h): 재학습 직후 P=20 오프-스트림 rollout.

    round_idx ∈ {0, 1} — 총 2라운드 상한(§4h). 라운드 간에도 seed·noise가 분리되어
    동일 스펙 재실행이 없다. base_idx는 수집 대역(0..39) 재사용 — held-out 미노출성 보존.
    결과는 𝒟_k(성숙도 원장, source="probe")에만 기록하고 BC 풀·ACI 계정에는 넣지 않는다.
    """
    if round_idx not in (0, 1):
        raise ValueError(f"round_idx must be 0 or 1 (§4h probe 2-round cap), got {round_idx}")
    specs = []
    bases = list(COLLECT_BASE_RANGE)
    for j in range(n_episodes):
        specs.append(
            EpisodeSpec(
                suite_name,
                task_id,
                seed=PROBE_SEED_BASE + 1000 * round_idx + j,
                base_idx=bases[j % len(bases)],
                w=W_ID,
                noise_seed=PROBE_NOISE_BASE + 1000 * round_idx + j,
            )
        )
    return specs


def novel_specs(suite_name, task_id, n_episodes, w_novel=None, seed_base=NOVEL_SEED_BASE, noise_base=NOVEL_NOISE_BASE):
    """E4 관할 파일럿용 novel: 변이 폭 확대 (w_id < w ≤ usable_w_max).
    타 태스크 초기상태 차용 novel은 태스크 쌍이 필요해 e4 스크립트에서 별도 구성."""
    w = w_novel if w_novel is not None else USABLE_W_MAX[suite_name]
    if not (W_ID < w <= USABLE_W_MAX[suite_name]):
        raise ValueError(f"w_novel={w} must be in ({W_ID}, {USABLE_W_MAX[suite_name]}]")
    specs = []
    for j in range(n_episodes):
        specs.append(
            EpisodeSpec(
                suite_name,
                task_id,
                seed=seed_base + j,
                base_idx=list(COLLECT_BASE_RANGE)[j % 40],
                w=w,
                noise_seed=noise_base + j,
            )
        )
    return specs


# ---- E5 스트림 대역 (§4b 등재 2026-08-16, 설계서 v0.3 §2) ----
E5_STREAM_SEED_BASE, E5_STREAM_NOISE_BASE = 70_000, 6_000_000
E5_NOVEL_SEED_BASE, E5_NOVEL_NOISE_BASE = 80_000, 7_000_000
E5_CLUSTERS = (
    [("libero_object", t) for t in range(10)]
    + [("libero_goal", t) for t in range(10)]
    + [("libero_spatial", 0), ("libero_spatial", 1)]
    + [("libero_10", 0), ("libero_10", 2), ("libero_10", 5)]
)                                                    # 25 distinct (결함2 반영)
E5_NOVEL_POOL = [("libero_spatial", t) for t in range(2, 10)]  # Spatial-b 8 태스크
E5_NOVEL_RATE = 0.10                                 # §4 파생상수


def e5_stream_specs(seed_idx, n_episodes=4000, novel_rate=E5_NOVEL_RATE, clusters=None):
    """E5 온라인 스트림 명세 (seed_idx ∈ {0,1,2}).

    - 비-novel = 25 클러스터 균등 순환, novel 주입 = Spatial-b 풀에서 균등 (설계서 §2·§0).
    - 순서는 seed_idx로 결정적 셔플. base_idx는 수집 대역(0–39) 재사용(배포 재발 모사),
      held-out(40–49) 사용 금지. w = w_id (전 스위트 usable_w_max 이내 — long 0.02 포함).
    반환: [(spec, cluster_key, is_novel_injection)]
    """
    import numpy as np

    # clusters=None → 본실행 25종. 지정 시 그 부분집합만 순환(**스모크 전용** — 50 ep가
    # 25종에 흩어지면 |B_k| 트리거가 발화하지 않아 재학습·probe 경로를 검증할 수 없다).
    E5_CLUSTERS_ = clusters or E5_CLUSTERS
    n_novel = int(round(n_episodes * novel_rate))
    plan = ([(E5_CLUSTERS_[i % len(E5_CLUSTERS_)], False) for i in range(n_episodes - n_novel)]
            + [(E5_NOVEL_POOL[i % len(E5_NOVEL_POOL)], True) for i in range(n_novel)])
    order = np.random.default_rng(1000 + seed_idx).permutation(len(plan))
    bases = list(COLLECT_BASE_RANGE)
    out = []
    for i, idx in enumerate(order):
        (suite, task), is_novel = plan[idx]
        sbase = E5_NOVEL_SEED_BASE if is_novel else E5_STREAM_SEED_BASE
        nbase = E5_NOVEL_NOISE_BASE if is_novel else E5_STREAM_NOISE_BASE
        w = min(W_ID, USABLE_W_MAX[suite])  # 스위트별 상한 준수 (결함2 조건 (a))
        out.append((EpisodeSpec(suite, task,
                                seed=sbase + 10_000 * seed_idx + i,
                                base_idx=bases[i % len(bases)], w=w,
                                noise_seed=nbase + 1_000_000 * seed_idx + i),
                    f"{suite}_task{task}", is_novel))
    return out


def assert_six_bands_disjoint(seed_idx=0, sample=400):
    """드라이버 기동 시 1회 강제 (렌즈 6). 실패 시 예외 → 기동 거부."""
    from .chained_env import chained_collection_specs, chained_heldout_specs

    suite, task = "libero_object", 0
    bands = {
        "collect": collection_specs(suite, task),
        "heldout": heldout_specs(suite, task, 50),
        "novel": novel_specs(suite, task, 40),
        "probe": probe_specs(suite, task, 0) + probe_specs(suite, task, 1),
        "chained": chained_collection_specs(suite, task) + chained_heldout_specs(suite, task, 50),
        "e5_stream": [s for s, cl, _ in e5_stream_specs(seed_idx, sample) if cl == f"{suite}_task{task}"],
    }
    names = list(bands)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ua, ub = {s.uid for s in bands[a]}, {s.uid for s in bands[b]}
            inter = ua & ub
            assert not inter, f"대역 충돌 {a}↔{b}: {sorted(inter)[:3]}"
    return {k: len(v) for k, v in bands.items()}


def assert_disjoint(a_specs, b_specs):
    ua, ub = {s.uid for s in a_specs}, {s.uid for s in b_specs}
    inter = ua & ub
    assert not inter, f"spec overlap: {sorted(inter)[:5]}"
    return True
