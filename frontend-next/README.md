# gyul-frontend-next

`frontend/`(기존 UI)를 복사한 **디자인 실험용 프론트엔드**입니다.
새 디자인을 시험하는 동안 기존 화면을 그대로 두고 나란히 비교하기 위해 분기했습니다.

| | 디렉터리 | 포트 |
|---|---|---|
| 기존 UI | `frontend/` | 5173 |
| 실험 UI | `frontend-next/` | 5174 |

두 앱 모두 Vite 프록시로 **같은 FastAPI(127.0.0.1:8000)** 를 봅니다. 백엔드는 하나만 띄우면 됩니다.

## 실행

```bash
./start.sh                              # API(8000) + 기존 UI(5173)

cd frontend-next && npm install         # 최초 1회
npm run dev                             # 실험 UI(5174)
```

## 주의

- **포트가 다르면 origin이 달라 `sessionStorage`가 분리됩니다.** 5173에서 만든 리포트는
  5174에서 보이지 않습니다 (`report:*` 캐시 키가 origin 단위로 저장되기 때문).
  같은 세션을 두 UI로 번갈아 보려면 캐시를 서버나 `localStorage` 공유 방식으로 바꿔야 합니다.
- `src/hooks/`, `src/services/`, `src/types/`, `src/utils/` 는 `frontend/` 와 **복제 관계**입니다.
  WS 프로토콜·감정 매핑·리포트 스키마가 바뀌면 **양쪽 모두 고쳐야 합니다.**
  둘 다 장기 유지하기로 정해지면 공통 로직을 한 벌로 합치는 편이 안전합니다.
- 디자인 목업은 `frontend/mockups/` 에 있습니다 (복사하지 않았습니다).

## 실험 결과 정리

- 새 디자인 채택 → `frontend-next/` 내용을 `frontend/` 로 옮기고 이 디렉터리 삭제
- 폐기 → 이 디렉터리만 삭제 (`frontend/` 는 손대지 않았으므로 영향 없음)
