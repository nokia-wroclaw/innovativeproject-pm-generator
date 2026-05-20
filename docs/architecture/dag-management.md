# DAG Management — kontrakt architektoniczny

> Status: **DRAFT — Faza 2a**, do akceptacji.
> Źródło prawdy dla modeli danych i kontraktu API między naszym frontendem (Vue 3),
> naszym backendem (FastAPI) i silnikiem Apache Airflow 3.2 (REST API v2).
>
> Dyscyplina: każda zmiana modelu lub endpointu musi być najpierw zapisana tutaj,
> dopiero potem implementowana w Pydantic (`apps/backend/app/models/dags.py`)
> i odwzorowana jako JSDoc `@typedef` w (`apps/frontend/src/features/dags/types.js`).

---

## 1. Zakres i nie-zakres v1

**W zakresie v1:**

- Lista DAG-ów (dashboard) z aktualnym statusem ostatniego runa i metrykami.
- Detal DAG-a (graf Vue Flow) z węzłami pokazującymi status task instances z najnowszego runa.
- Wybór historycznego DAG runa (dropdown) i podmiana statusów na grafie.
- Panel boczny po kliknięciu węzła: overview, logi (live + historia + filtr), XCom (placeholder).
- Akcje: **Trigger DAG**, **Clear task**, **Clear run**, **Stop run**.
- RBAC: `viewer` (czyta wszystko) vs `admin` (mutuje).

**Poza zakresem v1 (do rozważenia w v2):**

- Pause / Unpause DAG.
- Mark task success / failed (manualne nadpisanie stanu).
- Edycja zmiennych Airflow / connections / pools.
- Historia akcji (audit log) w naszej bazie.
- Notyfikacje WebSocket o zmianach statusu między widokami.

---

## 2. Mapowanie statusów Airflow → nasz enum

Airflow ma ponad 10 stanów task-instance i 4 stany DAG-runa. Dla **czytelności**
i ograniczenia liczby wariantów wizualnych redukujemy je do siedmiu kategorii.

### 2.1 `TaskStatus` (task instance)

| Nasz status   | Airflow states                                            | Kolor (Tailwind)             | Ikona (lucide)    | Animacja |
| ------------- | --------------------------------------------------------- | ---------------------------- | ----------------- | -------- |
| `success`     | `success`                                                 | `emerald-500`                | `check-circle-2`  | —        |
| `running`     | `running`                                                 | `sky-500`                    | `loader-2`        | spin     |
| `failed`      | `failed`, `upstream_failed`                               | `rose-500`                   | `x-circle`        | —        |
| `up_for_retry`| `up_for_retry`, `up_for_reschedule`, `restarting`         | `amber-500`                  | `refresh-cw`      | —        |
| `queued`      | `queued`, `scheduled`, `deferred`                         | `violet-400`                 | `clock`           | —        |
| `skipped`     | `skipped`                                                 | `slate-400`                  | `skip-forward`    | —        |
| `none`        | `none`, `removed`, `null`/brak                            | `slate-300`                  | `circle-dashed`   | —        |

> Backend zwraca **tylko** naszą kategorię w polu `status`, a **oryginalny stan Airflow**
> w `raw_state` (string) — dzięki temu UI nigdy nie operuje na surowych enumach Airflow,
> ale w razie potrzeby debug-mode może go pokazać w tooltipie.

### 2.2 `DagRunStatus` (DAG run)

| Nasz status   | Airflow states            |
| ------------- | ------------------------- |
| `success`     | `success`                 |
| `running`     | `running`                 |
| `failed`      | `failed`                  |
| `queued`      | `queued`                  |

> Nie maskujemy `queued` jako `running` — różnica jest istotna dla operatora.

---

## 3. Modele danych (DTO)

Zapis w stylu Pydantic, ale **języko-neutralny** — będzie odwzorowany 1:1 w `models/dags.py`
oraz w JSDoc `@typedef` na froncie. ID-ki czasów są `datetime` w ISO 8601 (UTC),
identyfikatory tekstowe — `str`.

### 3.1 `DagSummary` — wiersz tabeli na dashboardzie

```text
DagSummary
├── dag_id            : str                  # unikalne id, np. "generate_synthetic_pm"
├── display_name      : str                  # human-readable, fallback = dag_id
├── description       : str | null
├── owners            : list[str]            # właściciele z Airflow ("owner")
├── tags              : list[str]
├── is_paused         : bool
├── is_active         : bool                 # czy DAG jest „aktywny" w Airflow
├── schedule          : str | null           # cron lub interval, "@daily" itp.
├── next_run_at       : datetime | null      # planowane następne uruchomienie
├── last_run          : DagRunSummary | null
└── stats_24h         : DagStats             # success/failed/running w ostatnich 24h
```

### 3.2 `DagStats`

```text
DagStats
├── success           : int
├── failed            : int
├── running           : int
└── total             : int
```

### 3.3 `DagRunSummary`

```text
DagRunSummary
├── run_id            : str                  # Airflow dag_run_id
├── logical_date      : datetime             # data logiczna (dawne execution_date)
├── start_date        : datetime | null
├── end_date          : datetime | null
├── duration_ms       : int | null
├── status            : DagRunStatus
├── raw_state         : str                  # surowy stan Airflow
├── run_type          : Literal["manual","scheduled","backfill","asset_triggered"]
└── triggered_by      : str | null           # username z naszego JWT (jeśli nasz)
```

### 3.4 `DagDetails` — payload widoku grafu

```text
DagDetails
├── summary           : DagSummary
├── graph             : DagGraph
└── recent_runs       : list[DagRunSummary]  # ostatnie 10 runów do dropdownu
```

### 3.5 `DagGraph`

```text
DagGraph
├── nodes             : list[TaskNode]
└── edges             : list[TaskEdge]
```

### 3.6 `TaskNode`

```text
TaskNode
├── task_id           : str                  # unikalne w obrębie DAG-a
├── label             : str                  # display name, fallback = task_id
├── operator          : str                  # np. "PythonOperator", "BashOperator"
├── is_group          : bool                 # TaskGroup (collapsable na froncie)
├── trigger_rule      : str                  # "all_success", "one_failed", itp.
├── retries_max       : int
└── depends_on_past   : bool
```

### 3.7 `TaskEdge`

```text
TaskEdge
├── source            : str   # task_id źródła
└── target            : str   # task_id celu
```

### 3.8 `TaskInstance` — stan w konkretnym DAG runie

```text
TaskInstance
├── task_id           : str
├── run_id            : str
├── status            : TaskStatus
├── raw_state         : str
├── try_number        : int        # aktualny numer próby
├── max_tries         : int
├── start_date        : datetime | null
├── end_date          : datetime | null
├── duration_ms       : int | null
├── operator          : str
├── pool              : str
├── queue             : str
├── executor_config   : dict       # surowe
└── note              : str | null
```

### 3.9 `TaskTry` — pojedyncza próba (dla dropdownu „Try 1 / Try 2 / ...")

```text
TaskTry
├── try_number        : int
├── status            : TaskStatus
├── start_date        : datetime | null
├── end_date          : datetime | null
└── duration_ms       : int | null
```

### 3.10 `LogChunk` — pojedyncza partia logu (SSE event lub odpowiedź historyczna)

```text
LogChunk
├── try_number        : int
├── seq               : int                  # rosnący numer chunka w obrębie próby
├── lines             : list[LogLine]
├── has_more          : bool                 # czy są kolejne chunki do pobrania
└── continuation      : str | null           # opaque token do nextpage (Airflow „token")
```

```text
LogLine
├── timestamp         : datetime | null      # null jeśli nie udało się sparsować
├── level             : Literal["DEBUG","INFO","WARNING","ERROR","CRITICAL"] | null
├── source            : str | null           # np. "scheduler", "task", "trigger"
└── message           : str
```

### 3.11 `TriggerRequest` / `ActionResponse`

```text
TriggerRequest
├── conf              : dict          | null    # opcjonalny payload do DAG-a
├── logical_date      : datetime      | null    # domyślnie now()
└── note              : str           | null    # widoczna w Airflow

ActionResponse
├── run_id            : str | null              # nowy run_id (dla trigger)
├── message           : str                     # czytelny komunikat dla UI
└── airflow_status    : int                     # propagowany kod HTTP z Airflow
```

### 3.12 `ApiError`

Spójna struktura błędu w całym API v1.

```text
ApiError
├── error             : str               # stable code, np. "AIRFLOW_UNAVAILABLE"
├── message           : str               # human-readable, EN
├── details           : dict | null
└── request_id        : str               # X-Request-ID
```

---

## 4. Kontrakt API v1

Prefix: `/api/v1`. Wszystkie endpointy wymagają nagłówka `Authorization: Bearer <keycloak-token>`.
Kolumna **RBAC**: `R` = `require_auth` (viewer + admin), `W` = `require_admin`.

| Metoda | Ścieżka                                                                  | RBAC | Opis                                                       | Response          |
| ------ | ------------------------------------------------------------------------ | ---- | ---------------------------------------------------------- | ----------------- |
| GET    | `/dags`                                                                  | R    | Lista DAG-ów z ostatnim runem i statsami 24h               | `list[DagSummary]`|
| GET    | `/dags/{dag_id}`                                                         | R    | Detale DAG-a: graf + ostatnie 10 runów                     | `DagDetails`      |
| GET    | `/dags/{dag_id}/runs`                                                    | R    | Paginowana historia runów (`?limit=`, `?offset=`)          | `list[DagRunSummary]` |
| GET    | `/dags/{dag_id}/runs/{run_id}/tasks`                                     | R    | Stan wszystkich tasków w danym runie                       | `list[TaskInstance]` |
| GET    | `/dags/{dag_id}/runs/{run_id}/tasks/{task_id}`                           | R    | Pojedynczy task instance                                   | `TaskInstance`    |
| GET    | `/dags/{dag_id}/runs/{run_id}/tasks/{task_id}/tries`                     | R    | Lista prób (do dropdownu w logach)                         | `list[TaskTry]`   |
| GET    | `/dags/{dag_id}/runs/{run_id}/tasks/{task_id}/logs`                      | R    | **Historyczne** logi (paginowane). Query: `?try_number=N&token=...&limit=...` | `LogChunk`        |
| GET    | `/dags/{dag_id}/runs/{run_id}/tasks/{task_id}/logs/stream`               | R    | **Live SSE** logu. Query: `?try_number=N` (domyślnie latest) | `text/event-stream` |
| POST   | `/dags/{dag_id}/runs`                                                    | W    | Trigger DAG. Body: `TriggerRequest`                        | `ActionResponse`  |
| POST   | `/dags/{dag_id}/runs/{run_id}/stop`                                      | W    | Zatrzymanie runa (Airflow: PATCH state=failed)             | `ActionResponse`  |
| POST   | `/dags/{dag_id}/runs/{run_id}/clear`                                     | W    | Re-run całego DAG runa                                     | `ActionResponse`  |
| POST   | `/dags/{dag_id}/runs/{run_id}/tasks/{task_id}/clear`                     | W    | Re-run pojedynczego taska. Query: `?downstream=true`       | `ActionResponse`  |
| GET    | `/airflow/health`                                                        | R    | Healthcheck Airflow przepuszczony przez backend            | `{status:str}`    |

### 4.1 Konwencje paginacji

- `limit`: domyślnie 50, max 200.
- `offset`: domyślnie 0.
- Logi: kursor `token` (opaque, pochodzi z Airflow). Klient woła ponownie z tym tokenem.

### 4.2 Cache-Control / ETag

- `GET /dags` zwraca `ETag` na podstawie hash listy + statusów. Frontend respektuje 304.
- `GET /dags/{dag_id}` — bez ETaga w v1 (graf zmienia się rzadko, ale statusy często).

---

## 5. SSE — format streamu logów

**Endpoint:** `GET /api/v1/dags/{dag_id}/runs/{run_id}/tasks/{task_id}/logs/stream?try_number=N`
**Content-Type:** `text/event-stream`

### 5.1 Typy zdarzeń

```text
event: chunk
data:  <LogChunk JSON>

event: heartbeat
data:  {"ts": "2026-05-20T18:30:00Z"}

event: end
data:  {"reason": "task_finished" | "user_disconnect" | "max_duration"}

event: error
data:  {"error": "AIRFLOW_UNAVAILABLE", "message": "..."}
```

### 5.2 Zachowanie

- Pierwszy `chunk` zawiera **cały dotychczasowy log** dla danej próby (catch-up).
- Kolejne `chunk`-i to przyrostowe linie.
- `heartbeat` co 15s, żeby proxy/CDN nie zamknęło połączenia.
- Po zakończeniu taska backend wysyła `end` i zamyka połączenie.
- Frontend reaguje na `EventSource.onerror` → automatyczny reconnect z eksponencjalnym backoffem (max 30s).
- Wyszukiwanie/filtr logu (level/regex) **odbywa się po stronie klienta** na materializowanym buforze — w MVP nie filtrujemy server-side.

### 5.3 Limity

- Max długość połączenia SSE: **2 godziny** (`LOG_STREAM_MAX_DURATION_SECONDS=7200`). Po tym czasie backend wysyła `event: end` z `reason: "max_duration"` i klient automatycznie otwiera nowy stream (kontynuacja jest transparentna dla użytkownika).
- Max chunk: 500 linii lub 256 KB (cokolwiek pierwsze).

---

## 6. Kody błędów

| `error` (stable code)      | HTTP | Kiedy                                                          |
| -------------------------- | ---- | -------------------------------------------------------------- |
| `UNAUTHENTICATED`          | 401  | Brak/nieważny token Keycloak (już realizuje `auth.py`)         |
| `FORBIDDEN`                | 403  | Brak roli `admin` przy akcji mutującej                         |
| `DAG_NOT_FOUND`            | 404  | Airflow zwraca 404 dla DAG-a                                   |
| `RUN_NOT_FOUND`            | 404  | Airflow zwraca 404 dla DAG runa                                |
| `TASK_NOT_FOUND`           | 404  | Airflow zwraca 404 dla task instance                           |
| `VALIDATION_ERROR`         | 422  | Niepoprawny payload (Pydantic)                                 |
| `AIRFLOW_UNAVAILABLE`      | 502  | Airflow nieosiągalny / timeout / 5xx                           |
| `AIRFLOW_AUTH_FAILED`      | 502  | Nasz service-account JWT odrzucony przez Airflow               |
| `AIRFLOW_CONFLICT`         | 409  | Airflow zwraca 409 (np. trigger z istniejącym `run_id`)        |
| `RATE_LIMITED`             | 429  | Wewnętrzny limit per-user dla akcji mutujących (do dorobienia) |
| `INTERNAL_ERROR`           | 500  | Cokolwiek nieoczekiwanego                                      |

Każda odpowiedź błędu zawiera nagłówek `X-Request-ID` i `ApiError` w body.

---

## 7. Auth Backend → Airflow

Browser **nigdy** nie komunikuje się z Airflow bezpośrednio.

```text
Browser  ──[Keycloak access token]──▶  Backend  ──[Airflow service-account JWT]──▶  Airflow
```

### 7.1 Konfiguracja (zmienne środowiskowe, do `.env`)

```text
AIRFLOW_URL                    = http://airflow-apiserver:9005
AIRFLOW_API_PREFIX             = /api/v2
AIRFLOW_JWT_SECRET             = ${AIRFLOW__API_AUTH__JWT_SECRET}      # ten sam co Airflow
AIRFLOW_JWT_ISSUER             = genpm-backend
AIRFLOW_SERVICE_ACCOUNT_SUB    = genpm-backend
AIRFLOW_JWT_TTL_SECONDS        = 600
AIRFLOW_HTTP_TIMEOUT_SECONDS   = 15
```

### 7.2 Strategia tokenu

- `AirflowAuth` (klasa w `app/integrations/airflow/auth.py`) generuje JWT HS256 z `iss`, `sub`, `aud="airflow"`, `iat`, `exp`.
- Cache w pamięci procesu z `asyncio.Lock`; odświeżamy 60s przed `exp`.
- W razie 401 z Airflow — invalidate cache + jednorazowy retry.

### 7.3 Mapowanie tożsamości

Akcje mutujące przekazują `triggered_by=<keycloak preferred_username>` w polu `note` Airflow DAG-runa, tak żeby w UI Airflow widać było, kto co zrobił.

---

## 8. Strategia polling / cache

| Widok                     | Refetch interval | Uwagi                                                                |
| ------------------------- | ---------------- | -------------------------------------------------------------------- |
| Dashboard (`GET /dags`)   | 5000 ms          | Vue Query + ETag. Pauza polling przy `visibilitychange = hidden`.    |
| Detal DAG-a (`GET /dags/{id}`) | 3000 ms     | Tylko gdy istnieje DAG run w stanie `running`. W innym wypadku 15000 ms. |
| Task instance details     | 2000 ms          | Tylko gdy `status ∈ {running, queued, up_for_retry}`.                |
| Logi historyczne          | brak polling     | Pobierane na żądanie (klik „Refresh").                               |
| Logi live                 | SSE              | Patrz §5.                                                            |

Po każdej akcji mutującej (trigger/clear/stop) frontend wykonuje **optymistyczną** invalidację cache (Vue Query `invalidateQueries`) dla:

- `["dags"]`
- `["dags", dag_id]`
- `["dags", dag_id, "runs"]`

---

## 9. Czego ten dokument **nie** definiuje

- Wyglądu poszczególnych komponentów Vue (to Faza 2b).
- Layoutu auto-grafu (dagre konfiguracja) — to detal implementacyjny w `composables/useDagLayout.js`.
- Wewnętrznych endpointów Airflow w pełnej szczegółowości — patrz oficjalna dokumentacja Airflow 3.2 REST API v2.
- Migracji bazy GenPM — w v1 nic nie persystujemy lokalnie (Airflow jest single source of truth).

---

## 10. Decyzje przyjęte (zamknięte 2026-05-20)

1. **`stats_24h`**: liczone server-side per request `GET /dags`. Drożej dla BE, taniej dla UX (brak osobnego round-tripa). Backend cache'uje na 30s w pamięci procesu.
2. **`display_name`**: używane gdy zdefiniowane w DAG-u, fallback `dag_id`.
3. **Max długość SSE**: **2 godziny** (zaktualizowano §5). Frontend transparentnie kontynuuje stream po `reason: "max_duration"`.
