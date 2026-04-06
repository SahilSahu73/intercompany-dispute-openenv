# Intercompany Dispute OpenEnv Build Plan

## Goal

Build a real-world OpenEnv environment for deterministic intercompany financial dispute resolution. The single acting agent is the Enterprise Consolidation Orchestrator. The Legal and Tax/Treasury specialists are modeled in V1 as deterministic MCP-style services exposed through the environment so the benchmark stays reproducible, gradeable, and safe for future RL training.

## OpenEnv Constraints Pulled From The Docs

- The environment should follow the standard OpenEnv shape: `__init__.py`, `models.py`, `client.py`, `openenv.yaml`, `README.md`, `pyproject.toml`, `uv.lock`, and `server/` with `app.py` and `Dockerfile`.
- `server/app.py` should expose the API with `create_app(...)` and pass a class or factory, not a singleton instance, so each session gets isolated episode state.
- `openenv.yaml` should minimally declare `spec_version`, `name`, `type`, `runtime`, `app`, and `port`.
- `pyproject.toml` should provide a server entry point so local execution and validation work with `uv run --project . server`.
- Validation should be done with `uv run openenv validate --verbose`.
- Deployment should target a Docker-backed Hugging Face Space. `openenv push` handles frontmatter and upload.
- `openenv serve` is not the primary local path right now; use `uv run --project . server` or Docker.

## Naming Decisions

- OpenEnv name: `intercompany_dispute_env`
- Python package name: `intercompany_dispute_env`
- Suggested HF Space slug: `intercompany-dispute-env`
- Suggested project title in README: `Intercompany Dispute Environment for OpenEnv`

## Repository Shape To Build Toward

The repo root should become the OpenEnv environment root so local validation and HF deployment stay simple.

```text
.
├── build_plan/
│   └── intercompany_dispute_env_build_plan.md
├── requirements_n_scheme/
│   ├── hackathon_details.md
│   └── idea.md
├── __init__.py
├── models.py
├── client.py
├── openenv.yaml
├── README.md
├── pyproject.toml
├── uv.lock
├── domain/
│   ├── __init__.py
│   ├── enums.py
│   ├── money.py
│   ├── ledger_models.py
│   ├── document_models.py
│   ├── scenario_models.py
│   └── task_models.py
├── services/
│   ├── __init__.py
│   ├── ledger_service.py
│   ├── document_service.py
│   ├── treasury_service.py
│   ├── legal_service.py
│   ├── matching_service.py
│   └── audit_service.py
├── graders/
│   ├── __init__.py
│   ├── base.py
│   ├── easy_grader.py
│   ├── medium_grader.py
│   └── hard_grader.py
├── tasks/
│   ├── __init__.py
│   ├── registry.py
│   ├── easy_batch_matching.py
│   ├── medium_fx_variance.py
│   └── hard_liability_dispute.py
├── seed_data/
│   ├── easy/
│   ├── medium/
│   └── hard/
├── scripts/
│   ├── baseline_inference.py
│   ├── smoke_eval.py
│   └── generate_scenarios.py
├── tests/
│   ├── test_models.py
│   ├── test_services.py
│   ├── test_environment.py
│   ├── test_graders.py
│   └── test_api_contract.py
└── server/
    ├── __init__.py
    ├── app.py
    ├── intercompany_dispute_environment.py
    ├── session_factory.py
    └── Dockerfile
```

## Agent To Environment Interaction Model

1. `reset(task_id=..., scenario_id=..., seed=...)` creates a clean deterministic episode and returns the initial observation.
2. The orchestrator agent receives a summary of unresolved disputes, visible open items, accessible documents, and allowed tool-like actions.
3. Each `step()` takes exactly one typed action request. The environment routes that request to one internal deterministic service.
4. Read-only services return evidence. Write services mutate the simulated ERP ledger and append to the audit log.
5. The environment computes dense step reward after every action and emits a final normalized task score on completion.
6. `state()` returns only public episode state. Hidden grader truth must never leak through `state()` or the observation.

## Architectural Principle

The environment is the source of truth. The specialist "agents" are internal service adapters now, not separate LLM workers. Their APIs should still look like future MCP tools so they can later be swapped to real agent-backed services without changing the orchestrator-facing contract.

## Core Public Schemas

All finance amounts should be stored as `Decimal`, not `float`. Public JSON serialization should keep decimal precision intact.

```python
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


class Money(BaseModel):
    amount: Decimal
    currency: Literal["USD", "GBP", "EUR"]


class OpenItemView(BaseModel):
    txn_id: str
    entity_id: str
    counterparty_entity_id: str
    account_code: str
    side: Literal["debit", "credit"]
    money: Money
    booking_date: date
    description: str
    status: Literal["open", "matched", "adjusted", "eliminated"]
    document_ids: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    document_type: Literal["invoice", "email", "contract", "shipment_report"]
    title: str
    snippet: str


class RewardBreakdown(BaseModel):
    progress: float
    evidence: float
    accuracy: float
    efficiency_penalty: float
    invalid_action_penalty: float
    terminal_bonus: float
    total: float


class ViolationRecord(BaseModel):
    code: str
    message: str
    severity: Literal["warning", "error"]


class QueryOpenItemsArgs(BaseModel):
    action_type: Literal["query_open_items"]
    entity_id: str | None = None
    counterparty_entity_id: str | None = None
    status: Literal["open", "matched", "adjusted", "eliminated"] = "open"
    limit: int = Field(default=100, ge=1, le=1000)


class QueryLedgerBalanceArgs(BaseModel):
    action_type: Literal["query_ledger_balance"]
    entity_id: str
    account_code: str
    as_of_date: date | None = None


class FetchDocumentArgs(BaseModel):
    action_type: Literal["fetch_document"]
    document_id: str


class AskLegalAnalystArgs(BaseModel):
    action_type: Literal["ask_legal_analyst"]
    document_id: str
    question: str


class CalculateFxArgs(BaseModel):
    action_type: Literal["calculate_fx"]
    source_currency: Literal["USD", "GBP", "EUR"]
    target_currency: Literal["USD", "GBP", "EUR"]
    conversion_date: date
    amount: Decimal


class ExecuteMatchArgs(BaseModel):
    action_type: Literal["execute_match"]
    debit_txn_id: str
    credit_txn_id: str


class PostAdjustmentArgs(BaseModel):
    action_type: Literal["post_adjustment"]
    entity_id: str
    debit_account_code: str
    credit_account_code: str
    money: Money
    reason_code: Literal["fx_variance", "liability_recognition", "inventory_loss", "manual_true_up"]
    evidence_refs: list[str] = Field(default_factory=list)


class ExecuteEliminationArgs(BaseModel):
    action_type: Literal["execute_elimination"]
    entity_id: str
    matched_pair_id: str


FinanceActionPayload = Annotated[
    QueryOpenItemsArgs
    | QueryLedgerBalanceArgs
    | FetchDocumentArgs
    | AskLegalAnalystArgs
    | CalculateFxArgs
    | ExecuteMatchArgs
    | PostAdjustmentArgs
    | ExecuteEliminationArgs,
    Field(discriminator="action_type"),
]


class FinanceDisputeAction(Action):
    request: FinanceActionPayload


class FinanceDisputeObservation(Observation):
    task_id: str
    scenario_id: str
    difficulty: Literal["easy", "medium", "hard"]
    summary: str
    last_action_status: Literal["ok", "rejected", "invalid", "noop"]
    last_result: dict[str, object] | None = None
    visible_open_items: list[OpenItemView] = Field(default_factory=list)
    accessible_documents: list[DocumentSummary] = Field(default_factory=list)
    completed_objectives: list[str] = Field(default_factory=list)
    remaining_objectives: list[str] = Field(default_factory=list)
    reward_breakdown: RewardBreakdown
    violations: list[ViolationRecord] = Field(default_factory=list)
    terminal_task_score: float | None = Field(default=None, ge=0.0, le=1.0)


class AuditEvent(BaseModel):
    timestamp: datetime
    actor: Literal["orchestrator", "ledger_service", "treasury_service", "legal_service", "environment"]
    action_type: str
    status: Literal["ok", "rejected", "invalid", "noop"]
    reference_id: str | None = None


class FinanceDisputeState(State):
    task_id: str
    scenario_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_limit: int
    unresolved_case_ids: list[str] = Field(default_factory=list)
    completed_objectives: list[str] = Field(default_factory=list)
    evidence_cache_ids: list[str] = Field(default_factory=list)
    audit_log: list[AuditEvent] = Field(default_factory=list)
```

## Internal-Only Schemas

These models must live in the server/domain layer and must never be exposed through the public state.

```python
from dataclasses import dataclass, field


class GroundTruthChecklist(BaseModel):
    required_matches: list[tuple[str, str]]
    required_adjustments: list[str]
    required_eliminations: list[str]
    required_liable_entity_id: str | None = None
    required_fx_rate: Decimal | None = None


class ScenarioBundle(BaseModel):
    scenario_id: str
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    ledger_lines: list[dict]
    documents: list[dict]
    fx_table: list[dict]
    legal_truth: dict | None = None
    objectives: list[str]
    step_limit: int


@dataclass
class EpisodeInternalContext:
    scenario: ScenarioBundle
    ground_truth: GroundTruthChecklist
    ledger_index: dict[str, object] = field(default_factory=dict)
    match_index: dict[str, object] = field(default_factory=dict)
    evidence_cache: dict[str, object] = field(default_factory=dict)
    issued_adjustments: list[object] = field(default_factory=list)
    hidden_flags: set[str] = field(default_factory=set)
```

## Tool Contract Matrix

| Tool / action | Mutates ledger | Expected input | Expected output |
| --- | --- | --- | --- |
| `query_open_items` | No | entity filters, status, limit | visible open-item batch |
| `query_ledger_balance` | No | `entity_id: str`, `account_code: str`, `as_of_date: date | None` | account balance snapshot |
| `fetch_document` | No | `document_id: str` | raw document text plus metadata |
| `ask_legal_analyst` | No | `document_id: str`, `question: str` | liability decision JSON with incoterm and rationale |
| `calculate_fx` | No | currency pair, date, amount | deterministic historical FX conversion result |
| `execute_match` | Yes | debit txn id, credit txn id | match record or rejection |
| `post_adjustment` | Yes | entity, debit account, credit account, money, evidence refs | created journal entry |
| `execute_elimination` | Yes | entity, matched pair id | elimination record and updated exposure |

## Reward Design Blueprint

- Step reward stays dense and local to the latest action. Use the `reward` field on `FinanceDisputeObservation`.
- Final episode score stays normalized in `[0.0, 1.0]` and is emitted in `terminal_task_score` on the terminal observation.
- Do not make the terminal score identical to the step reward. The step reward is shaping. The terminal score is the benchmark grade.
- Use small penalties for inefficiency such as `-0.01` per step and stronger penalties for invalid or destructive actions.
- Reward evidence-first behavior. A write action without preceding supporting evidence should score worse even if it accidentally helps the ledger.
- Penalize hallucinated accounts, unknown transaction IDs, mismatched currencies, and unbalanced journals.

Suggested shaping formula:

```text
step_reward = progress + evidence + accuracy - efficiency_penalty - invalid_action_penalty + terminal_bonus
terminal_task_score = weighted_checklist_score clamped to [0.0, 1.0]
```

## Phase 1: OpenEnv Skeleton And Local Runtime Foundation

Intent: Turn the repo root into a valid uv-managed OpenEnv environment skeleton that can be served and validated before any domain complexity is added.

Expected outcome: `uv run --project . server` starts a valid FastAPI app, `openenv.yaml` is present, package metadata is aligned, and the repo is ready for environment logic.

Implementation blueprint:

- Replace the placeholder app structure with OpenEnv files at repo root.
- Set `openenv.yaml` to use `spec_version: 1`, `name: intercompany_dispute_env`, `type: space`, `runtime: fastapi`, `app: server.app:app`, `port: 8000`.
- Update `pyproject.toml` for an OpenEnv environment package and add `[project.scripts] server = "intercompany_dispute_env.server.app:main"`.
- Keep the package rooted at `.` using `tool.setuptools.package-dir` so the repo root acts as the environment root.
- Create a minimal `server/app.py` with `create_app(...)` and a clean `main()` entry point.
- Use `README.md` as a real environment README from day one because OpenEnv loads it into metadata and the default `/web` UI.

Validation gate:

- `uv sync`
- `uv run --project . server`
- `uv run openenv validate --verbose`

## Phase 2: Domain Models, Scenario DSL, And Public/Private State Split

Intent: Model the accounting world as deterministic typed data before building agent actions.

Expected outcome: We have stable Pydantic domain contracts for money, ledger entries, documents, contracts, FX tables, objectives, and ground-truth checklists. Public state is explicitly separated from hidden grader truth.

Implementation blueprint:

- Create `domain/money.py` with `Money` and serialization helpers.
- Create `domain/ledger_models.py` with ledger line, journal entry, match record, and elimination record types.
- Create `domain/document_models.py` with invoice, email, contract, and shipment report models.
- Create `domain/scenario_models.py` with `ScenarioBundle`, `ScenarioManifest`, and dataset validation.
- Create `domain/task_models.py` with `TaskSpec`, `ObjectiveSpec`, and grader checklist models.
- Define a strict rule: anything used for grading but not meant for the acting agent must remain in internal-only server state.
- Seed-data files should be deterministic, file-backed, and versionable rather than generated randomly at runtime.

Validation gate:

- Unit tests for schema validation, decimal handling, and serialization round-trips.
- Snapshot tests for public-state serialization to ensure no hidden fields leak.

## Phase 3: Deterministic Service Layer With MCP-Shaped Contracts

Intent: Build the internal services that behave like future MCP tools while remaining deterministic and grader-friendly now.

Expected outcome: The environment can route tool-like actions to isolated services for ledger access, document retrieval, treasury FX, legal interpretation, and auditing.

Implementation blueprint:

- `ledger_service.py` should answer open-item and balance queries and apply write operations.
- `document_service.py` should return canonical raw text and snippets from seeded documents.
- `treasury_service.py` should expose historical FX lookup based only on seeded rate tables.
- `legal_service.py` should return structured liability decisions from seeded contract truth and explicit rule logic around Incoterms.
- `matching_service.py` should validate whether a match is legal, complete, and currency-consistent.
- `audit_service.py` should record action order, evidence usage, invalid actions, and loop patterns for reward shaping.
- Keep service interfaces narrow so later replacement by real MCP servers changes only adapters, not environment behavior.

Validation gate:

- Service-level tests for correct FX lookup, correct liability selection, and correct journal balancing logic.
- Determinism test: same `scenario_id` and `seed` must always yield the same outputs.

## Phase 4: OpenEnv Models, Environment Class, And Typed Client

Intent: Wire the domain and services into the actual OpenEnv API surface.

Expected outcome: The environment exposes typed `reset()`, `step()`, and `state()` behavior through OpenEnv, and a typed client can consume it locally or over a containerized Space.

Implementation blueprint:

- Implement `models.py` with `FinanceDisputeAction`, `FinanceDisputeObservation`, and `FinanceDisputeState`.
- Implement `server/intercompany_dispute_environment.py` as the only stateful episode controller.
- The environment class should own the public state plus an internal context object carrying hidden grader truth.
- `reset()` should support `task_id`, `scenario_id`, and `seed` through reset kwargs.
- `step()` should validate the discriminated action request, dispatch to the correct service, compute reward, and produce the next observation.
- `state` should return only public data and recent audit events.
- `client.py` should map typed action payloads to JSON and parse `StepResult[FinanceDisputeObservation]` back from the server.
- Mark `SUPPORTS_CONCURRENT_SESSIONS = True` only if each session gets a completely isolated in-memory episode context.

Validation gate:

- API contract tests for reset, step, state, schema, and health endpoints.
- One local end-to-end smoke run using the client against `uv run --project . server`.

## Phase 5: Easy Task Pack - Baseline Batch Matching

Intent: Ship the first real benchmark task with clean 1-to-1 intercompany matching and elimination.

Expected outcome: Task 1 is complete, gradeable, and reproducible. The agent can clear explicit invoice pairs and execute eliminations in a high-throughput flow.

Implementation blueprint:

- Seed the easy task with explicit parent/subsidiary transaction references.
- Include a benchmark-size dataset of around 1,000 transactions, but also keep a much smaller smoke dataset for tests and local iteration.
- Reward correct high-volume matching, low-latency procedural behavior, and complete elimination.
- Grader checks should include match coverage, match correctness, elimination correctness, invalid write count, and step efficiency.
- Keep the first task intentionally free of legal ambiguity and FX uncertainty so the environment contract stabilizes early.

Validation gate:

- Grader returns deterministic score in `[0.0, 1.0]`.
- Regression tests on easy scenarios with known expected scores.

## Phase 6: Medium Task Pack - Noisy Text And FX Variance

Intent: Add the first evidence-heavy task where the agent must read documents, infer linkage, and retrieve the exact historical FX conversion before posting adjustments.

Expected outcome: Task 2 forces the orchestrator to gather evidence before writing to the ledger. The treasury service becomes mandatory, not optional.

Implementation blueprint:

- Seed noisy invoice text and delayed settlement dates.
- Document retrieval should expose the vendor/entity clues needed to align transactions.
- Treasury service should compute conversion using scenario-seeded historical FX tables only.
- `post_adjustment` should require evidence refs and should be rejected or heavily penalized when unsupported.
- The grader should score document usage, correct FX rate usage, correct adjustment amount, correct match, and final elimination.

Validation gate:

- Golden tests where a known FX rate and known adjustment amount must be produced exactly.
- Negative tests where guessed FX, wrong date usage, or missing evidence refs get penalized.

## Phase 7: Hard Task Pack - Liability And Multi-Hop Dispute Resolution

Intent: Add the adversarial dispute where legal liability determines which entity must recognize the payable and inventory loss.

Expected outcome: Task 3 becomes a genuine reasoning benchmark. The agent must fetch contract evidence, obtain a liability decision, post the correct adjustments, and then eliminate the resolved exposure.

Implementation blueprint:

- Seed supply-chain contracts with explicit Incoterms like `CIF` and `FOB`.
- Legal service should return structured output such as `incoterm`, `liable_entity_id`, `liable_event`, and `rationale`.
- The environment should track prerequisite order. If the agent posts liability adjustments without legal evidence first, reward should reflect incorrect process even if the end state improves.
- Grader checks should cover liable entity correctness, correct account usage, correct loss recognition, final exposure cleanup, and invalid action count.
- The hard task should involve multiple dependent steps so it cannot be solved by one lucky ledger mutation.

Validation gate:

- End-to-end tests for both `CIF` and `FOB` branches.
- Anti-shortcut tests ensuring the hard task cannot be passed by skipping legal retrieval.

## Phase 8: Graders, Episode Termination, And Anti-Exploit Hardening

Intent: Make the benchmark trustworthy for hackathon judging and later RL use.

Expected outcome: Each task has a deterministic grader, sensible done conditions, partial progress signals, and exploit-resistant penalties.

Implementation blueprint:

- Each task gets its own grader module plus a shared grading base class.
- Episode termination should fire on success, hard failure, or step-limit exhaustion.
- Add loop detection through the audit service so repeated useless reads or repeated invalid writes decay reward.
- Reject unsupported account codes, cross-entity writes that violate scenario rules, and impossible matches.
- Emit a structured final score report in observation metadata or terminal result fields so baseline scripts can record it directly.

Validation gate:

- Determinism tests across repeated runs.
- Adversarial tests for invalid IDs, wrong currencies, and spammy action loops.

## Phase 9: Baseline Inference, Documentation, Docker, And Hugging Face Deployment

Intent: Finish the hackathon-required packaging and reproducible evaluation path.

Expected outcome: The environment is runnable locally, validates under OpenEnv, builds into a container, and is ready for `openenv push` to the HF Space you already created.

Implementation blueprint:

- Add `scripts/baseline_inference.py` that runs all three tasks with deterministic seeds and reads `OPENAI_API_KEY` from environment variables.
- Add `scripts/smoke_eval.py` for a quick non-OpenAI local pass.
- Write `README.md` with motivation, environment description, action/observation/state definitions, task descriptions, reward design, setup, baseline instructions, and expected scores.
- Use the OpenEnv-compatible multi-stage `server/Dockerfile` pattern and keep `uv` as the dependency manager.
- Prefer the default OpenEnv `/web` interface in V1. A custom Gradio tab is optional and should only be added if it helps explain ledger state clearly.
- Validate locally with `uv run openenv validate --verbose`, then build with Docker, then push with `uv run openenv push --repo-id <your-username>/intercompany-dispute-env`.

Validation gate:

- `uv run openenv validate --verbose`
- `docker build` and `docker run`
- Remote validation against the deployed Space URL

## Recommended Build Order For The Next Turns

1. Phase 1 and Phase 2 first. That gives us the environment skeleton and typed domain model foundation.
2. Phase 3 next. That makes the OpenEnv API real.
3. Phase 5 next. Easy task should be the first fully playable benchmark.
4. Phase 6 and Phase 7 after that. Medium and hard should layer onto the same environment contract, not fork it.
5. Phase 8 and Phase 9 last. Hardening and deployment should happen only after all three tasks are stable.

## Non-Negotiable Guardrails While Building

- Do not expose hidden grader truth through `state()` or observation fields.
- Do not use floating-point accounting amounts for ledger math.
- Do not make specialist services nondeterministic in V1.
- Do not let write actions bypass evidence requirements in medium and hard tasks.
- Do not overload the repo with separate microservices now. Keep services in-process behind replaceable adapters.
- Do not add a custom web UI until the environment API, tasks, and graders are correct.

## First Concrete Implementation Slice

The first coding turn should build only the minimal OpenEnv skeleton plus the shared domain models and deterministic service interfaces. That slice is enough to make the package shape real without prematurely locking task logic.
