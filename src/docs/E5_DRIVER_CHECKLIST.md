# E5 드라이버 사전 함정 점검 체크리스트 (연구원 지시 2026-08-16 §5 — 착수 전 검증 렌즈)

> C-T2 3막 사가(log.md 2026-08-15/16)의 교훈을 E5 스트림 드라이버 설계에 선반영한다.
> 드라이버 코드 리뷰·검증 워크플로우는 본 체크리스트를 렌즈로 사용한다.

| # | 렌즈 | C-T2 교훈 원천 | E5 드라이버 점검 항목 |
|---|---|---|---|
| 1 | **경계 처리** | v1: 시연 분포 밖 상태에서 teacher 동결 | 에피소드 경계 = E0-6 3단 프로토콜 완전 준수 (seed→reset→set_init_state→settle). 재학습 일시정지 후 재개 지점의 상태 재현성 단언 |
| 2 | **컨트롤러 상태 이월** | v2: reset 생략 → OSC stale goal 0.48 rad 이탈 | 모든 에피소드 전이에서 env.reset 경유 확인 — set_init_state 단독 호출 금지 (정적 grep + 런타임 단언) |
| 3 | **실행기 대칭성** | v3: stale chunk tail — 수집·평가·스트림 실행 경로 불일치 위험 | 스트림 실행기 = `execute_chunk_with_boundary` 단일 경로. VLA/습관/probe 롤아웃 전부 동일 헬퍼 — 별도 루프 작성 금지. 단위 테스트로 고정 |
| 4 | **출력 경로 유일성** | t2_diag.json 덮어쓰기 사고 (R2) | 에피소드 로그·체크포인트·재학습 산출물에 seed·시점 suffix 의무화. 기존 파일 존재 시 덮어쓰기 대신 FAIL |
| 5 | **source 태그 강제** | A_mat 장부 분리 (§4h·통합 §2) | 원장 갱신 전수에 source ∈ {teacher, probe, fire} 필수. probe → observe_fire·r_V 유입 금지 (회귀 테스트 t5·t6). 에피소드 로그에 lifecycle 상태 동봉 (counterfactual 회계) |
| 6 | **대역 disjoint** | §4b 5대역 규약 | 스트림 스펙 vs 수집/held-out/novel/probe/연쇄 uid 전수 `assert_disjoint` — 드라이버 기동 시 1회 강제 실행 후 시작 |

추가 (E5 고유):
- 재학습 트리거·probe 라운드·부적격 전이의 상태 기계를 드라이버 시작 전 dry-run으로 검증
  (§4h 규칙: lazy {20,80}, P=20, 2라운드, Beta 이월 — `gate_regression.py` 통과 전제).
- 진행 heartbeat 동반 필수 (`tools/with_heartbeat.sh`) — seed당 수 시간 작업, seed 단위 중간 요약 1회.
- 호출률·비열등 회계는 counterfactual completion (§4h) — 발화 스펙 재실행 결정성 사전 검증 포함.
